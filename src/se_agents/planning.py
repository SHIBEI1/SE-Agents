from __future__ import annotations

import hashlib
import html
import json
import re
from decimal import Decimal
from graphlib import CycleError, TopologicalSorter
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Line = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1200)]
Reference = Annotated[str, StringConstraints(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,39}$")]
TaskStatus = Literal["pending", "ready", "running", "review", "blocked", "done", "failed"]
STATUS_NAMES = {
    "pending": "等待依赖", "ready": "可启动", "running": "执行中", "review": "待验收",
    "blocked": "阻塞", "done": "已验收", "failed": "执行失败",
}


class PlanningRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SpecRequirement(PlanningRecord):
    id: Reference
    description: Line
    priority: Literal["Must", "Should", "Could"]
    acceptance: Line


class DesignModule(PlanningRecord):
    id: Reference
    name: Line
    responsibility: Line
    requirement_ids: list[Reference] = Field(min_length=1)


class DesignInterface(PlanningRecord):
    id: Reference
    name: Line
    source_module: Reference
    target_module: Reference | None = None
    contract: Line


class DesignSpecification(PlanningRecord):
    title: Line
    overview: Line
    requirements: list[SpecRequirement] = Field(min_length=1, max_length=32)
    modules: list[DesignModule] = Field(min_length=1, max_length=24)
    interfaces: list[DesignInterface] = Field(default_factory=list, max_length=48)
    constraints: list[Line] = Field(default_factory=list, max_length=16)
    assumptions: list[Line] = Field(default_factory=list, max_length=16)


class DevelopmentTask(PlanningRecord):
    id: Reference
    title: Line
    kind: Literal["code", "test", "doc", "research", "human"]
    owner: Line
    requirement_ids: list[Reference] = Field(min_length=1)
    module_ids: list[Reference] = Field(min_length=1)
    depends_on: list[Reference] = Field(default_factory=list)
    deliverables: list[Line] = Field(min_length=1, max_length=8)
    done_criteria: list[Line] = Field(min_length=1, max_length=16)
    estimate_hours: float = Field(gt=0, le=16, allow_inf_nan=False)
    priority: Literal["Must", "Should", "Could"]


class DevelopmentPlan(PlanningRecord):
    tasks: list[DevelopmentTask] = Field(min_length=1, max_length=48)
    rationale: Line


class PlanEdit(PlanningRecord):
    task_id: Reference
    depends_on: list[Reference]
    estimate_hours: float = Field(gt=0, le=16, allow_inf_nan=False)
    max_parallel: int = Field(strict=True, ge=1, le=4)


class PlanFinding(PlanningRecord):
    task_id: Reference | None = None
    severity: Literal["major", "minor"]
    issue: Line


class PlanReview(PlanningRecord):
    findings: list[PlanFinding] = Field(default_factory=list, max_length=16)


class TaskFile(PlanningRecord):
    path: Line
    content: Annotated[str, StringConstraints(min_length=1, max_length=40000)]


class TaskOutput(PlanningRecord):
    summary: Line
    files: list[TaskFile] = Field(default_factory=list, max_length=8)
    evidence: list[Line] = Field(min_length=1, max_length=8)


class TaskAssessment(PlanningRecord):
    findings: list[Line] = Field(default_factory=list, max_length=8)


class TaskProgress(PlanningRecord):
    status: TaskStatus = "pending"
    attempts: int = 0
    evidence: str = ""
    reason: str = ""
    output: TaskOutput | None = None
    review_findings: list[str] = Field(default_factory=list)


class TaskEvent(PlanningRecord):
    sequence: int
    task_id: str
    action: str
    from_status: str
    to_status: str
    actor: str
    reason: str
    timestamp: float
    plan_hash: str


class ScheduledTask(PlanningRecord):
    task_id: str
    batch: int
    earliest_start: float
    earliest_finish: float
    critical: bool
    planned_start: float | None = None
    planned_finish: float | None = None


class Engineering(PlanningRecord):
    source_text: str = ""
    requirements: DesignSpecification | None = None
    requirements_approval: str | None = None
    design: DesignSpecification | None = None
    design_approval: str | None = None
    plan: DevelopmentPlan | None = None
    plan_review: PlanReview | None = None
    plan_approval: str | None = None
    tasks: dict[str, TaskProgress] = Field(default_factory=dict)
    events: list[TaskEvent] = Field(default_factory=list)
    feedback: list[str] = Field(default_factory=list)
    max_parallel: int = Field(default=2, ge=1, le=4)
    plan_revisions: int = 0
    design_review_pending: bool = False
    schedule_rows: list[ScheduledTask] = Field(default_factory=list)


def fingerprint(value: BaseModel) -> str:
    return hashlib.sha256(
        json.dumps(value.model_dump(), ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def unique(items, label):
    ids = [item.id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label}编号重复。")
    return set(ids)


def validate_spec(spec: DesignSpecification) -> None:
    requirements = unique(spec.requirements, "需求")
    modules = unique(spec.modules, "模块")
    unique(spec.interfaces, "接口")
    covered = set()
    for module in spec.modules:
        if set(module.requirement_ids) - requirements:
            raise ValueError(f"{module.id} 引用了不存在的需求。")
        covered.update(module.requirement_ids)
    required = {item.id for item in spec.requirements if item.priority == "Must"}
    if required - covered:
        raise ValueError(f"必需需求缺少模块承接：{sorted(required - covered)}")
    for interface in spec.interfaces:
        if interface.source_module not in modules or (
            interface.target_module is not None and interface.target_module not in modules
        ):
            raise ValueError(f"{interface.id} 的接口端点模块不存在。")


def validate_plan(plan: DevelopmentPlan, spec: DesignSpecification) -> None:
    task_order(plan)
    requirements = {item.id for item in spec.requirements}
    modules = {item.id: item for item in spec.modules}
    covered = set()
    tested = set()
    for task in plan.tasks:
        if set(task.requirement_ids) - requirements or set(task.module_ids) - modules.keys():
            raise ValueError(f"{task.id} 的需求或模块引用不存在。")
        linked = {ref for key in task.module_ids for ref in modules[key].requirement_ids}
        if set(task.requirement_ids) - linked:
            missing = sorted(set(task.requirement_ids) - linked)
            candidates = [module.id for module in modules.values()
                          if set(module.requirement_ids).intersection(missing)]
            raise ValueError(
                f"{task.id} 的模块与需求追溯不一致。缺少承接 {'、'.join(missing)} 的模块，"
                f"可核对 {'、'.join(candidates)}，或移除任务未实际承接的需求。"
            )
        if task.kind != "test":
            covered.update(task.requirement_ids)
        if task.kind in {"test", "human"}:
            tested.update(task.requirement_ids)
    required = {item.id for item in spec.requirements if item.priority == "Must"}
    if required - covered:
        raise ValueError(f"必需需求缺少开发任务：{sorted(required - covered)}")
    if required - tested:
        raise ValueError(f"必需需求缺少测试或人工验收任务：{sorted(required - tested)}")

def task_order(plan: DevelopmentPlan) -> list[str]:
    ids = unique(plan.tasks, "任务")
    for task in plan.tasks:
        if set(task.depends_on) - ids or task.id in task.depends_on:
            raise ValueError(f"{task.id} 存在无效或自身依赖。")
        if len(set(task.depends_on)) != len(task.depends_on):
            raise ValueError(f"{task.id} 存在重复依赖。")
    try:
        return list(TopologicalSorter({t.id: t.depends_on for t in plan.tasks}).static_order())
    except CycleError as exc:
        path = " → ".join(exc.args[1])
        raise ValueError(f"任务依赖存在循环：{path}。请移除其中一项前置依赖。") from exc


def edited_plan(plan: DevelopmentPlan, edit: PlanEdit) -> DevelopmentPlan:
    if edit.task_id not in {task.id for task in plan.tasks}:
        raise ValueError("找不到要修改的任务，请刷新后重试。")
    updated = plan.model_copy(update={"tasks": [
        task.model_copy(update={
            "depends_on": edit.depends_on, "estimate_hours": edit.estimate_hours,
        }) if task.id == edit.task_id else task
        for task in plan.tasks
    ]})
    task_order(updated)
    return updated


def schedule(plan: DevelopmentPlan, max_parallel: int) -> list[dict]:
    """Compute dependency-only bounds and a timeline that respects the available slots."""
    if not 1 <= max_parallel <= 4:
        raise ValueError("最大并行任务数须为1到4。")
    by_id = {task.id: task for task in plan.tasks}
    order = task_order(plan)
    duration = {key: Decimal(str(task.estimate_hours)) for key, task in by_id.items()}
    zero = Decimal(0)
    finish = {}
    start = {}
    parent = {}
    for key in order:
        dependencies = by_id[key].depends_on
        previous = max(dependencies, key=lambda d: finish[d]) if dependencies else None
        parent[key] = previous
        start[key] = finish[previous] if previous else zero
        finish[key] = start[key] + duration[key]
    critical = set()
    cursor = max(order, key=lambda key: finish[key])
    while cursor:
        critical.add(cursor)
        cursor = parent[cursor]
    remaining, done, running = set(by_id), set(), {}
    planned_start, planned_finish, batches = {}, {}, {}
    rank = {"Must": 0, "Should": 1, "Could": 2}
    now, batch = zero, 0
    while remaining:
        ready = sorted(
            (key for key in remaining if set(by_id[key].depends_on) <= done),
            key=lambda key: (rank[by_id[key].priority], key),
        )[:max_parallel - len(running)]
        if ready:
            batch += 1
        for key in ready:
            batches[key] = batch
            planned_start[key] = now
            planned_finish[key] = running[key] = now + duration[key]
        remaining.difference_update(ready)
        now = min(running.values())
        completed = {key for key, end in running.items() if end == now}
        done.update(completed)
        for key in completed:
            del running[key]
    return [
        {"task_id": key, "batch": batches[key], "earliest_start": float(start[key]),
         "earliest_finish": float(finish[key]), "critical": key in critical,
         "planned_start": float(planned_start[key]), "planned_finish": float(planned_finish[key])}
        for key in order
    ]


def refresh_ready(engineering: Engineering) -> None:
    if not engineering.plan:
        return
    for task in engineering.plan.tasks:
        progress = engineering.tasks.setdefault(task.id, TaskProgress())
        if progress.status in {"pending", "ready"}:
            progress.status = (
                "ready" if all(engineering.tasks.get(d, TaskProgress()).status == "done"
                               for d in task.depends_on) else "pending"
            )


def transition(engineering: Engineering, task_id: str, action: str, evidence: str) -> None:
    if not engineering.plan or engineering.plan_approval != fingerprint(engineering.plan):
        raise ValueError("请先确认当前任务计划。")
    refresh_ready(engineering)
    task = next((t for t in engineering.plan.tasks if t.id == task_id), None)
    if task is None:
        raise ValueError("任务不存在。")
    progress = engineering.tasks[task_id]
    transitions = {
        "start": ({"ready"}, "running"),
        "submit": ({"running"}, "review"),
        "accept": ({"review"}, "done"),
        "reject": ({"review"}, "ready"),
        "block": ({"ready", "running", "review"}, "blocked"),
        "unblock": ({"blocked"}, "pending"),
        "retry": ({"failed"}, "pending"),
    }
    if action not in transitions or progress.status not in transitions[action][0]:
        raise ValueError("当前任务状态不支持这个操作。")
    if not evidence.strip():
        raise ValueError("请填写操作依据、交付证据或阻塞原因。")
    if action in {"start", "unblock", "retry"}:
        if not all(engineering.tasks[d].status == "done" for d in task.depends_on):
            raise ValueError("前置任务尚未通过验收。")
    if action == "start":
        if sum(p.status == "running" for p in engineering.tasks.values()) >= engineering.max_parallel:
            raise ValueError("已达到当前最大并行任务数。")
    if action == "accept" and progress.review_findings:
        raise ValueError("独立审查仍有未解决问题，请退回修复。")
    progress.status = transitions[action][1]
    progress.reason = evidence.strip()
    if action in {"submit", "accept"}:
        progress.evidence = evidence.strip()
    refresh_ready(engineering)


def diagram(engineering: Engineering) -> str:
    if not engineering.plan:
        return "digraph Tasks {}"
    rows = schedule(engineering.plan, engineering.max_parallel)
    critical = {row["task_id"] for row in rows if row["critical"]}
    nodes = []
    colours = {"done": "#d1f0d9", "blocked": "#ffd8d8", "failed": "#ffd8d8",
               "review": "#ffe7b3", "ready": "#dceeff", "running": "#d7e1ff"}
    for task in engineering.plan.tasks:
        state = engineering.tasks.get(task.id, TaskProgress()).status
        label = json.dumps(f"{task.id}  {task.title}\n{STATUS_NAMES[state]}", ensure_ascii=False)
        colour = colours.get(state, "#eeeeee")
        pen = 3 if task.id in critical else 1
        nodes.append(f'"{task.id}" [label={label}, fillcolor="{colour}", penwidth={pen}];')
        nodes.extend(f'"{dep}" -> "{task.id}";' for dep in task.depends_on)
    return ('digraph Tasks { rankdir=LR; node [shape=box, style="rounded,filled", '
            'fontname="Microsoft YaHei"]; ' + " ".join(nodes) + "}")


def timeline_svg(engineering: Engineering) -> str:
    """Render the computed planning data, never treating planned time as actual completion."""
    if not engineering.plan:
        return ""
    rows = schedule(engineering.plan, engineering.max_parallel)
    times = {row["task_id"]: row for row in rows}
    largest = max(row["planned_finish"] for row in rows)
    height = 58 + len(rows) * 36
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 {height}" '
             'width="100%" role="img" aria-label="任务排期甘特图" fill="currentColor">']
    for tick in range(6):
        x = 315 + tick * 112
        parts.extend([
            f'<text x="{x}" y="24" font-size="14">{largest * tick / 5:.3g} h</text>',
            f'<line x1="{x}" y1="33" x2="{x}" y2="{height - 12}" '
            'stroke="currentColor" opacity=".16"/>',
        ])
    tasks = sorted(engineering.plan.tasks, key=lambda t: (times[t.id]["planned_start"], t.id))
    for index, task in enumerate(tasks):
        row = times[task.id]
        x = 315 + row["planned_start"] / largest * 560
        width = task.estimate_hours / largest * 560
        y = 42 + index * 36
        label = html.escape(f"{task.id} {task.title[:15]}")
        colour = "#176b56" if row["critical"] else "#5586b5"
        detail = html.escape(
            f'{task.id} {task.title} · {row["planned_start"]:g}–{row["planned_finish"]:g} 小时'
        )
        parts.extend([
            f'<text x="8" y="{y + 18}" font-size="14">{label}</text>',
            f'<rect x="{x}" y="{y}" width="{width}" height="25" rx="4" fill="{colour}">'
            f'<title>{detail}</title></rect>',
        ])
    return "".join(parts) + "</svg>"


def export_plan(engineering: Engineering) -> str:
    if not engineering.plan:
        return ""
    lines = [f"# {engineering.design.title} 开发任务与执行编排",
             f"最大并行任务数 {engineering.max_parallel}",
             "| 任务 | 负责人 | 来源需求 | 模块 | 依赖 | 完成标准 |",
             "| --- | --- | --- | --- | --- | --- |"]
    def clean(value):
        return re.sub(r"[|\r\n]", " ", value)
    for task in engineering.plan.tasks:
        lines.append("| " + " | ".join(map(clean, [
            f"{task.id} {task.title}", task.owner, ", ".join(task.requirement_ids),
            ", ".join(task.module_ids), ", ".join(task.depends_on) or "无",
            " / ".join(task.done_criteria),
        ])) + " |")
    lines.extend(["", "## 预计执行排期", "| 任务 | 批次 | 开始（小时） | 完成（小时） |",
                  "| --- | --- | --- | --- |"])
    rows = schedule(engineering.plan, engineering.max_parallel)
    for row in sorted(rows, key=lambda r: (r["batch"], r["task_id"])):
        lines.append(f'| {row["task_id"]} | {row["batch"]} | '
                     f'{row["planned_start"]:g} | {row["planned_finish"]:g} |')
    lines.extend(["", "批次为建议计划。完成状态以独立验收和人工签收为准。"])
    return "\n".join(lines)
