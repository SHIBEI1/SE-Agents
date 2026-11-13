from __future__ import annotations

from typing import Annotated, ClassVar, Literal

from pydantic import Field, StringConstraints, field_validator

from .models import ModelError, ModelRunner
from .schemas import Blackboard, Record, TechnicalReport, Text
from .storage import BudgetLimit

Section = Annotated[str, StringConstraints(strip_whitespace=True, min_length=160, max_length=4800)]
RequirementId = Annotated[str, StringConstraints(pattern=r"^R[0-9]{2}$")]

SCOPE = {
    "目标、角色与版本范围": "描述业务问题、角色与权限边界、首版与后续范围，以及可衡量的目标。",
    "关键流程与业务规则": "至少写一条完整主流程及异常路径，明确状态、进入条件、操作者、失败与补偿。",
}
ARCHITECTURE = {
    "系统架构与技术选型": "明确部署边界、模块职责和调用关系。给出技术选型理由、备选方案和取舍。可用 Mermaid。",
    "数据模型与状态流转": "逐表列字段、类型、主外键、唯一约束和索引。描述核心状态转换、历史快照与事务边界。",
    "接口与集成契约": "列出关键接口的方法、路径、角色、请求字段、响应、错误码和幂等规则。给出一个请求与响应示例。",
    "安全、并发与故障处理": "落实权限校验、数据保护、并发控制、重试、超时和失败恢复。说明如何避免重复写入。",
}
DELIVERY = {
    "工程组织与部署": "建议目录、环境配置、部署拓扑、数据库迁移、日志监控、备份恢复与回滚，给出可执行的操作顺序。",
    "开发阶段与交付验收": "按依赖顺序拆分任务，每阶段给出可交付物和完成条件。时间或人力以建议估算表达。",
    "风险、待核查事项与替代方案": "按影响排序，写明验证动作、启用条件、负责人角色和验证失败时的替代做法。",
}


class Chapters(Record):
    directory: ClassVar[dict[str, str]] = DELIVERY
    sections: dict[str, Section]

    @field_validator("sections", mode="before")
    @classmethod
    def keep_report_chapters(cls, sections):
        if isinstance(sections, dict):
            return {key: value for key, value in sections.items() if key in cls.directory}
        return sections


class Chapter(Record):
    content: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=160, max_length=12_000)
    ]


class Assumption(Record):
    topic: Text
    recommendation: Text
    rationale: Text
    verify: Text
    alternative: Text


class Requirement(Record):
    id: RequirementId
    priority: Literal["必须", "建议", "可选"]
    actor: Text
    behavior: Text
    acceptance: Text


class Scope(Chapters):
    directory: ClassVar[dict[str, str]] = SCOPE
    assumptions: list[Assumption] = Field(min_length=1, max_length=10)
    requirements: list[Requirement] = Field(min_length=5, max_length=14)


class TestCase(Record):
    requirements: list[RequirementId] = Field(min_length=1)
    scenario: Text
    action: Text
    expected: Text


class Delivery(Record):
    tests: list[TestCase] = Field(min_length=5)


class Finding(Record):
    severity: Literal["重大", "一般"]
    issue: Text
    fix: Text


class Review(Record):
    findings: list[Finding] = Field(max_length=8)


def require_sections(part: Chapters, expected: dict[str, str]) -> None:
    missing = [key for key in expected if key not in part.sections]
    if missing:
        raise ValueError("缺少必填章节：" + "、".join(missing) + "。")


def validate_scope(part: Scope) -> None:
    require_sections(part, SCOPE)
    if len({row.id for row in part.requirements}) != len(part.requirements):
        raise ValueError("需求编号不得重复。")
    if not any(row.priority == "必须" for row in part.requirements):
        raise ValueError("请明确首版必须实现的需求。")


def validate_delivery(part: Delivery, requirements: list[Requirement]) -> None:
    referenced = {key for case in part.tests for key in case.requirements}
    available = {item.id for item in requirements}
    required = {item.id for item in requirements if item.priority == "必须"}
    if referenced - available:
        raise ValueError("测试引用了未知需求：" + "、".join(sorted(referenced - available)))
    if required - referenced:
        raise ValueError("以下必须需求缺少测试：" + "、".join(sorted(required - referenced)))
    if len(part.tests) > 18:
        selected = set()
        uncovered = required.copy()
        for index, case in enumerate(part.tests):
            if uncovered.intersection(case.requirements):
                selected.add(index)
                uncovered.difference_update(case.requirements)
        for index in range(len(part.tests)):
            if len(selected) >= 18:
                break
            selected.add(index)
        part.tests = [part.tests[index] for index in sorted(selected)]


def context(board: Blackboard) -> dict:
    return {
        "用户原始想法": board.idea,
        "用户提供的补充背景": (
            board.frame_hints.model_dump(exclude_none=True) if board.frame_hints else {}
        ),
        "澄清问题与建议默认值": [
            item.model_dump(exclude_defaults=True) for item in board.design_questions
        ],
        "用户实际回复": [
            item.value["reply"] for item in board.decisions if item.kind == "design_clarify"
        ],
    }


def build_report(model: ModelRunner, board: Blackboard) -> tuple[TechnicalReport, bool]:
    source = context(board)
    model.store.set_stage(model.run_id, "report_scope")
    scope = model.run("report_scope", "pm", {"输入": source, "目录": SCOPE}, Scope, validate_scope)
    model.store.set_stage(model.run_id, "report_architecture")
    architecture = {}
    for title, instruction in ARCHITECTURE.items():
        chapter = model.run(
            "report_architecture",
            "tech",
            {
                "已定需求": scope.model_dump(),
                "已有架构章节": architecture,
                "当前章节": title,
                "章节要求": instruction,
            },
            Chapter,
        )
        architecture[title] = chapter.content
    model.store.set_stage(model.run_id, "report_delivery")
    plan = model.run(
        "report_delivery",
        "user",
        {"已定需求": scope.model_dump(), "架构": architecture, "目录": DELIVERY},
        Chapters,
        lambda part: require_sections(part, DELIVERY),
    )
    model.store.set_stage(model.run_id, "report_tests")
    delivery = model.run(
        "report_tests",
        "user",
        {"需求": [item.model_dump() for item in scope.requirements], "架构": architecture},
        Delivery,
        lambda part: validate_delivery(part, scope.requirements),
    )
    model.store.set_stage(model.run_id, "report_review")
    review_error = None
    try:
        review = model.run(
            "report_review",
            "devil",
            {
                "用户想法": board.idea,
                "需求": scope.model_dump(),
                "架构": architecture,
                "实施与测试": {"计划": plan.model_dump(), "测试": delivery.model_dump()},
            },
            Review,
        )
    except (ModelError, BudgetLimit) as exc:
        review_error = str(exc)
        review = Review(findings=[])

    sections = {key: scope.sections[key] for key in SCOPE}
    sections["默认设计决策与待验证假设"] = "\n\n".join(
        f"- **{item.topic}**。建议采用 {item.recommendation}。"
        f"依据是 {item.rationale}。验证方式为 {item.verify}。备选方案为 {item.alternative}。"
        for item in scope.assumptions
    )
    sections["详细需求与验收标准"] = "\n\n".join(
        f"- **{item.id} · {item.priority} · {item.actor}**。{item.behavior}\n"
        f"  - 验收标准：{item.acceptance}"
        for item in scope.requirements
    )
    sections.update(architecture)
    sections.update((key, plan.sections[key]) for key in DELIVERY)
    sections["需求与测试追踪"] = "\n\n".join(
        f"- **测试 {index:02} · {'、'.join(case.requirements)}**。{case.scenario}\n"
        f"  - 操作：{case.action}\n  - 预期：{case.expected}"
        for index, case in enumerate(delivery.tests, 1)
    )
    sections["独立审查与遗留问题"] = (
        f"独立审查尚未完成。{review_error}"
        if review_error
        else "\n\n".join(
            f"- **{item.severity}**。{item.issue}\n  - 建议修正：{item.fix}"
            for item in review.findings
        )
        or "本轮静态审查未报告问题。请在实现后执行上面的验收测试。"
    )
    model.store.set_stage(model.run_id, "technical_report")
    return (
        TechnicalReport(model=model.config.model, sections=sections),
        review_error is not None or any(item.severity == "重大" for item in review.findings),
    )
