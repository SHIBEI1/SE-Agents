from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime

from .chat import question_text
from .planning import STATUS_NAMES, fingerprint
from .presentation import ISSUE_STATUS, SIGNOFF, STAGES, STATUS
from .schemas import Blackboard


@dataclass(frozen=True)
class ProjectSummary:
    stage: str
    conclusion: str
    pending: tuple[str, ...]
    next_step: str
    reviews: str = ""
    context: str = ""
    changes: tuple[str, ...] = ()


def compact(text: str, limit: int = 220) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def revision_changes(board: Blackboard, previous: Blackboard | None) -> tuple[str, ...]:
    if previous is None:
        return ()
    changes: list[str] = []
    current, old = board.engineering, previous.engineering
    if current:
        for field, label in (("requirements", "需求规格"), ("design", "设计规格"), ("plan", "任务计划")):
            document = getattr(current, field)
            before = getattr(old, field) if old else None
            if document is None or document == before:
                continue
            items = document.tasks if field == "plan" else document.requirements
            if before is None:
                changes.append(f"生成{label}，包含 {len(items)} 项{'任务' if field == 'plan' else '需求'}。")
            else:
                earlier = before.tasks if field == "plan" else before.requirements
                by_id, earlier_by_id = {i.id: i for i in items}, {i.id: i for i in earlier}
                added = sorted(by_id.keys() - earlier_by_id.keys())
                removed = sorted(earlier_by_id.keys() - by_id.keys())
                edited = sorted(key for key in by_id.keys() & earlier_by_id.keys()
                                if by_id[key] != earlier_by_id[key])
                parts = [f"{name} {'、'.join(ids)}" for name, ids in
                         (("新增", added), ("修改", edited), ("移除", removed)) if ids]
                changes.append(f"{label}已更新：{'；'.join(parts) or '概述、模块、接口或说明发生变化'}。")
                if field != "plan":
                    for key in edited:
                        item = by_id[key]
                        changes.append(f"{key} 最新要求：{item.description}；验收：{item.acceptance}")
        for field, document, label in (
            ("requirements_approval", current.requirements, "需求基线"),
            ("design_approval", current.design, "设计规格"),
            ("plan_approval", current.plan, "任务计划"),
        ):
            value = getattr(current, field)
            if value != (getattr(old, field) if old else None):
                changes.append(f"{label}已确认。" if document and value == fingerprint(document)
                               else f"{label}确认已失效，需要重新审阅。")
        if current.feedback != (old.feedback if old else []):
            if current.feedback:
                changes.append(f"已记录修改意见：{current.feedback[-1]}")
        for task_id, progress in current.tasks.items():
            earlier = old.tasks.get(task_id) if old else None
            if earlier is None or progress != earlier:
                changes.append(f"{task_id}：{STATUS_NAMES[progress.status]}，已尝试 {progress.attempts} 次。")
    if board.proposal and board.proposal != previous.proposal:
        changes.append(f"推荐方案已更新为 {board.proposal.option_id}，版本 {board.proposal.revision}。")
    if board.technical_report and board.technical_report != previous.technical_report:
        changes.append("已保存新的技术报告，请查看最新章节。")
    return tuple(changes)


def summarize_project(
    board: Blackboard, run: dict, request: dict | None = None,
    *, previous: Blackboard | None = None,
) -> ProjectSummary:
    """只读取已有结果与状态；不调用模型、不修改任务或确认记录。"""
    stage = STAGES.get(run.get("stage", ""), "准备中")
    conclusion = "已记录你的想法，正在整理可供审阅的结果。"
    pending: list[str] = []
    next_step = "查看下方已有结果和活动记录。"
    review_counts = Counter(item.verdict for item in board.signoffs.values())
    reviews = " · ".join(
        f"{SIGNOFF[key]} {review_counts[key]} 位"
        for key in ("approve", "conditional", "reject") if review_counts[key]
    )

    if board.proposal:
        selected = next((o for o in board.options if o.id == board.proposal.option_id), None)
        if selected:
            conclusion = f"当前推荐：{selected.title}。{selected.approach}"
        for issue in board.objections:
            if issue.option_id == board.proposal.option_id and issue.status != "resolved":
                pending.append(f"{ISSUE_STATUS[issue.status]}：{issue.reason}")
        pending.extend(board.proposal_review_issues)
        next_step = "审阅下方推荐方案、遗留条件与复核意见，再决定是否采纳。"
    if board.technical_report:
        conclusion = "技术报告已生成，请审阅需求、设计和验收方法。"
        next_step = "展开下方完整技术报告，检查范围和验收条件。"

    engineering = board.engineering
    requirements_stage = run.get("stage") in {"requirements_spec", "requirements_confirm", "report_scope"}
    spec_stage = requirements_stage or run.get("stage") in {"engineering_design", "design_confirm"}
    if engineering:
        # 进入工程阶段后展示该阶段的待办，不把上游历史异议当作当前阻塞。
        pending = []
        spec = (engineering.requirements if requirements_stage else engineering.design) or engineering.requirements
        if spec:
            conclusion = f"{spec.title}：{spec.overview}"
        if engineering.plan and not spec_stage:
            plan = engineering.plan
            approved = engineering.plan_approval == fingerprint(plan)
            counts = Counter(
                engineering.tasks[task.id].status if task.id in engineering.tasks else "pending"
                for task in plan.tasks
            )
            conclusion = (
                f"已生成 {len(plan.tasks)} 项开发任务；"
                f"{'当前任务计划已确认' if approved else '当前任务计划尚未确认'}。"
                f"已验收 {counts['done']} 项，待验收 {counts['review']} 项，"
                f"执行中 {counts['running']} 项，失败 {counts['failed']} 项，阻塞 {counts['blocked']} 项。"
            )
            for task in plan.tasks:
                progress = engineering.tasks.get(task.id)
                if progress and progress.status in {"failed", "blocked", "review"}:
                    action = {
                        "failed": "执行失败", "blocked": "受阻", "review": "等待人工验收",
                    }[progress.status]
                    reason = f"：{progress.reason}" if progress.reason else ""
                    pending.append(f"{task.id} {task.title} · {action}{reason}")
            if engineering.plan_review:
                pending.extend(
                    f"计划评审待处理：{finding.issue}"
                    for finding in engineering.plan_review.findings if finding.severity == "major"
                )
            if not approved:
                next_step = "在下方审阅任务、依赖和评审意见，通过确认后再派发。"
            elif counts["done"] == len(plan.tasks):
                next_step = "所有任务均已签收；查看并导出交付物，按验收步骤检查整体项目。"
            elif counts["review"]:
                next_step = "打开下方“执行看板”，检查交付物、填写实际验收证据后签收。"
            elif counts["failed"] or counts["blocked"]:
                next_step = "打开下方“执行看板”，查看失败或阻塞原因并处理；独立就绪任务可继续。"
            else:
                next_step = "打开下方“执行看板”，选择可启动任务，再点击“执行此任务”或使用人工操作。"
        elif spec:
            next_step = "审阅下方规格及待确认问题，确认后继续生成设计或任务计划。"

    if board.project_build:
        pending = []
        conclusion = f"项目构建目录：{board.project_build.directory}。"
        if board.project_result:
            result = board.project_result
            conclusion += f"已生成 {len(result.files)} 个文件。"
            if result.acceptance:
                names = {"passed": "已通过本机验收", "failed": "本机验收失败", "not_run": "本机验收未运行"}
                conclusion += names[result.acceptance.status] + "。"
            pending.extend(result.review_notes)
        next_step = "在下方查看项目文件和验收结果，按生成项目的 README 启动检查。"

    context = conclusion
    if engineering and (engineering.plan is None or spec_stage):
        if engineering.design and not requirements_stage:
            spec = engineering.design
            confirmed = engineering.design_approval == fingerprint(spec)
            conclusion = (
                f"设计规格{'已确认' if confirmed else '已生成，等待审阅'}："
                f"{len(spec.requirements)} 条需求、{len(spec.modules)} 个模块、"
                f"{len(spec.interfaces)} 份接口定义。"
            )
        elif engineering.requirements:
            spec = engineering.requirements
            confirmed = engineering.requirements_approval == fingerprint(spec)
            conclusion = f"已整理 {len(spec.requirements)} 条需求；当前需求基线{'已确认' if confirmed else '尚未确认'}。"
            if not confirmed:
                conclusion += "请检查功能和验收条件，确认或提出修改。"

    if request:
        pending.insert(0, question_text(request))
        actions = {
            "proposal_confirm": "在下方审阅推荐方案，选择“采纳方案并整理需求”或填写修改意见退回。",
            "requirements_confirm": "在下方确认需求基线，或填写修改意见退回。",
            "design_confirm": "在下方选择“确认设计规格并拆任务”，或填写修改意见退回。",
            "plan_confirm": "在下方检查任务计划和排期，再使用任务计划确认入口。",
        }
        next_step = actions.get(request["kind"], "在下方问题区补充回答后继续。")
        if request["kind"] in actions:
            conclusion = f"{stage}：等待你的审阅决定。{conclusion}"
        if request.get("payload", {}).get("approval_blocked"):
            next_step = "当前确认被阻止；先处理下方列出的评审问题，再继续确认。"
    status = run.get("status", "")
    if status == "running":
        conclusion = f"正在{stage}。当前步骤仍在执行，新结果通过校验并保存后会更新这里。"
        next_step = "正在执行，请等待当前步骤结束；无需重复提交或再次调用。"
    elif status in {"failed", "interrupted", "cancelled"}:
        label = STATUS[status]
        detail = run.get("error") or board.stop_reason
        if detail:
            pending.insert(0, f"{label}：{detail}")
        conclusion = f"本次{stage}{label}。{detail or '本步骤没有正常完成。'}"
        conclusion += "已保存的结果仍保留，不能将其视为本次步骤成功生成的新结果。"
        next_step = "本次运行已停止推进；查看下方错误或活动记录，再决定是否继续任务。"
    elif status == "partial":
        conclusion = f"{stage}仅得到部分结果。{conclusion}"
        pending.insert(0, "当前只有部分结果，尚不能视为完整交付。")

    unique = tuple(dict.fromkeys(item.strip() for item in pending if item.strip()))
    return ProjectSummary(stage, conclusion, unique, next_step, reviews, context,
                          revision_changes(board, previous))


def render_project_summary(slot, board: Blackboard, run: dict, request: dict | None = None, *, store=None):
    import streamlit as st

    previous, saved_stage, activity = None, "", None
    revision = run.get("revision", 0)
    if store is not None:
        if revision > 0:
            previous = store.board(run["id"], revision - 1)
        saved_stage = next((item["stage"] for item in store.revisions(run["id"])
                            if item["revision"] == revision), "")
        events = store.activities(run["id"])
        activity = events[-1] if events else None
    summary = summarize_project(board, run, request, previous=previous)
    with slot.container():
        with st.expander("当前进展与待办", expanded=True):
            st.caption(f"当前阶段：{summary.stage} · {STATUS.get(run.get('status'), '状态待更新')}")
            preview = compact(summary.conclusion)
            st.write(preview)
            if activity:
                stamp = datetime.fromtimestamp(activity["created"]).strftime("%m-%d %H:%M:%S")
                st.caption(f"最近活动 · {stamp} · {activity['detail']}")
            if saved_stage:
                st.caption(f"上次保存：版本 {revision} · {STAGES.get(saved_stage, saved_stage)}")
            if summary.changes:
                st.markdown("**最近保存的变化**")
                for item in summary.changes[:3]:
                    st.write(f"• {compact(item, 300)}")
            if summary.reviews and board.engineering is None and not board.project_build:
                st.caption(f"各角色复核：{summary.reviews}")
            st.markdown("**待你处理或关注**")
            if summary.pending:
                for item in summary.pending[:3]:
                    st.write(f"• {compact(item, 300)}")
                if len(summary.pending) > 3:
                    st.caption(f"另有 {len(summary.pending) - 3} 项，可展开完整摘要查看。")
            else:
                st.write("当前记录未列出待处理事项；具体以当前阶段的审阅和验收结果为准。")
            st.markdown("**下一步**")
            st.write(summary.next_step)
            if (
                preview != summary.conclusion
                or summary.context != summary.conclusion
                or len(summary.changes) > 3
                or len(summary.pending) > 3
                or any(compact(item, 300) != item for item in summary.pending)
            ):
                with st.expander("展开完整摘要", expanded=False):
                    st.markdown("**完整概述**")
                    st.write(summary.conclusion)
                    if summary.context != summary.conclusion:
                        st.markdown("**已保存的项目内容（不代表当前步骤的新结果）**")
                        st.write(summary.context)
                    if summary.changes:
                        st.markdown("**最近保存的全部变化**")
                        for item in summary.changes:
                            st.write(f"• {item}")
                    if summary.pending:
                        st.markdown("**全部待处理事项**")
                        for index, item in enumerate(summary.pending, 1):
                            st.write(f"{index}. {item}")
            if run.get("status") != "running":
                st.markdown("[前往详细处理区 ↓](#project-actions)")
