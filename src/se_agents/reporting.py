from __future__ import annotations

from . import deliberation as rules
from .presentation import (
    EVIDENCE,
    ISSUE_KIND,
    ISSUE_STATUS,
    SIGNOFF,
    STAGES,
    STATUS,
    decision_text,
    zh,
)
from .schemas import ROLE_NAMES, Blackboard, RunConfig, canonical


def technical_body(title: str, body: str) -> str:
    first, _, rest = body.strip().partition("\n")
    if first.startswith("#") and first.lstrip("#").strip() == title:
        body = rest.lstrip()
    if title == "默认设计决策与待验证假设":
        body = body.replace("。。", "。")
    return body


def export_json(board: Blackboard, config: RunConfig, run: dict, calls: list[dict]) -> str:
    return canonical(
        {
            "run_id": run["id"],
            "name": run["name"],
            "revision": run["revision"],
            "status": run["status"],
            "config": config.model_dump(),
            "blackboard": board.model_dump(),
            "rankings": rules.rankings(board, config.weights),
            "calls": calls,
            "user_review": "not_reviewed",
        }
    )


def export_markdown(board: Blackboard, config: RunConfig, run: dict, calls: list[dict]) -> str:
    if board.technical_report:
        report = board.technical_report
        status = "待复核草案" if board.outcome == "partial" else "设计报告"
        lines = [
            f"# {run['name']} · 软件开发技术报告",
            f"{status} · 生成模型 `{report.model}`",
            "## 用户原始想法",
            board.idea,
        ]
        replies = [row.value["reply"] for row in board.decisions if row.kind == "design_clarify"]
        if replies:
            lines.extend(["## 用户对关键问题的答复", *replies])
        lines.extend(
            f"## {title}\n\n{technical_body(title, body)}"
            for title, body in report.sections.items()
        )
        return "\n\n".join(lines) + "\n"
    mode = "模拟演示" if config.mode == "mock" else config.model
    status = "可供用户审阅" if run["status"] == "completed" else "部分结果"
    lines = [
        f"# {run['name']} · {'论证报告' if board.proposal else '论证摘要'}",
        f"{mode} · 第 {run['revision']} 版记录 · {status}",
        f"任务状态为{STATUS[run['status']]}，等待用户审阅。",
    ]
    if board.stop_reason:
        lines.append(zh(board.stop_reason))
    if run.get("error"):
        lines.append(zh(run["error"]))
    if config.mode == "mock":
        lines.append("模拟模式使用固定场景响应，用于演示协作流程。")
    if board.proposal:
        lines.extend(
            [
                f"选定{zh(board.proposal.option_id)} · 论证方案第 {board.proposal.revision + 1} 版",
                *(
                    f"## {zh(title)}\n\n{zh(section)}"
                    for title, section in board.proposal.sections.items()
                ),
            ]
        )
    else:
        lines.extend(["## 软件想法", board.idea])

    lines.append("## 候选比较")
    ranks = {item["option_id"]: item for item in rules.rankings(board, config.weights)}
    for option in board.options:
        row = ranks.get(option.id)
        score = (
            "未进入评分"
            if row is None
            else ("评分不足" if row["score"] is None else f"总分 {row['score']:.3f}")
        )
        support = "" if row is None else f"，有效支持 {row['consensus']:.0%}"
        lines.append(
            f"- **{zh(option.id)} {zh(option.title)}** · {score}{support} · {zh(option.approach)}"
        )
        for item in option.assumptions:
            lines.append(
                f"  - {zh(item.id)}（{EVIDENCE[item.status]}）"
                f"{zh(item.statement)}。{zh(item.verification_plan)}"
            )

    lines.append("## 保留分歧与证据")
    for issue in board.objections:
        severity = "重大" if issue.severity == "major" else "一般"
        lines.append(
            f"- {zh(issue.id)} → {zh(issue.option_id)} · {ISSUE_KIND[issue.kind]} · {severity}"
            f" · {ISSUE_STATUS[issue.status]}。{zh(issue.reason)}。后续处理：{zh(issue.action)}"
        )
    for item in board.evidence:
        lines.append(
            f"- {zh(item.id)}（{EVIDENCE[item.judgment]}）用户材料 {item.source} · {zh(item.excerpt)}"
        )

    lines.append("## 观点与决定")
    for claim in board.claims:
        lines.append(
            f"- {zh(claim.id)} → {zh(claim.option_id)} · {ROLE_NAMES[claim.author]} · {zh(claim.rationale)}"
        )
    for decision in board.decisions:
        actor = "演示" if decision.actor == "demo" else "用户"
        lines.append(
            f"- {zh(decision.id)} · {STAGES[decision.kind]} · {actor} · {decision_text(decision)}"
        )
        if decision.reason != decision_text(decision):
            lines.append(f"  - {zh(decision.reason)}")

    lines.append("## 专家签署")
    for role, label in ROLE_NAMES.items():
        item = board.signoffs.get(role)
        if item:
            conditions = (
                f" · 条件 {'、'.join(map(zh, item.condition_ids))}" if item.condition_ids else ""
            )
            lines.append(f"- {label} · {SIGNOFF[item.verdict]} · {zh(item.reason)}{conditions}")
        else:
            lines.append(f"- {label} · 尚未签署")

    lines.append("## 用量")
    known = sum(row["tokens"] or 0 for row in calls)
    unknown = sum(row["tokens"] is None for row in calls)
    lines.append(f"请求 {len(calls)} 次，已知词元 {known} 个，用量未知请求 {unknown} 次。")
    return "\n\n".join(lines) + "\n"
