from __future__ import annotations

import io
import zipfile

import streamlit as st

from .human import validate_answer
from .planning import (
    STATUS_NAMES,
    ScheduledTask,
    diagram,
    export_plan,
    fingerprint,
    schedule,
    timeline_svg,
)
from .reporting import export_design_spec, export_technical_report, technical_body
from .task_execution import update_task

ACTION_NAMES = {
    "start": "人工开始", "submit": "提交人工交付证据", "accept": "验收通过",
    "reject": "退回修复", "block": "报告阻塞", "unblock": "解除阻塞", "retry": "重新排队",
}
ACTIONS = {
    "pending": [], "ready": ["start", "block"], "running": ["submit", "block"],
    "review": ["accept", "reject", "block"], "blocked": ["unblock"],
    "failed": ["retry", "block"], "done": [],
}


def spec_view(spec):
    st.markdown(f"#### {spec.title}")
    st.write(spec.overview)
    st.dataframe(
        [{"需求": r.id, "优先级": r.priority, "描述": r.description, "验收标准": r.acceptance}
         for r in spec.requirements], hide_index=True, width="stretch",
    )
    with st.expander("模块、接口与约束"):
        st.dataframe(
            [{"模块": m.id, "名称": m.name, "职责": m.responsibility,
              "来源需求": "、".join(m.requirement_ids)} for m in spec.modules],
            hide_index=True, width="stretch",
        )
        if spec.interfaces:
            st.dataframe(
                [{"接口": i.id, "名称": i.name, "调用方": i.source_module,
                  "被调用方": i.target_module or "外部", "约定": i.contract}
                 for i in spec.interfaces], hide_index=True, width="stretch",
            )
        for item in spec.constraints:
            st.write(f"约束：{item}")
        for item in spec.assumptions:
            st.warning(f"待验证假设：{item}")


def edit_task_plan(service, run_id, engineering, request):
    tasks = {task.id: task for task in engineering.plan.tasks}

    def label(task_id):
        return f"{task_id} · {tasks[task_id].title}"

    with st.expander("编辑依赖与排期", expanded=True):
        st.caption("保存后更新排期草案。确认任务计划时，以最近保存的版本为准。")
        selected = st.selectbox("要编辑的任务", list(tasks), format_func=label,
                                key=f"edit-task-{run_id}")
        task = tasks[selected]
        with st.form(f"edit-plan-{request['id']}-{selected}"):
            dependencies = st.multiselect(
                "前置任务", [task_id for task_id in tasks if task_id != selected],
                default=task.depends_on, format_func=label,
                help="所选任务全部完成后，这项任务才能开始。留空表示没有前置任务。",
            )
            hours, parallel = st.columns(2)
            estimate = hours.number_input(
                "预计工时（小时）", min_value=0.0, max_value=16.0,
                value=task.estimate_hours, step=0.25,
            )
            max_parallel = parallel.number_input(
                "最大并行任务数", min_value=1, max_value=4, value=engineering.max_parallel,
            )
            submitted = st.form_submit_button("保存并重排")
        if not submitted:
            return
        if (dependencies == task.depends_on and estimate == task.estimate_hours
                and max_parallel == engineering.max_parallel):
            st.info("当前任务和并行数没有变化。")
            return
        answer = {
            "request_id": request["id"],
            "reason": f"已编辑 {selected} 的排期并重新计算甘特图。",
            "value": {
                "action": "edit_plan",
                "snapshot": request["payload"]["snapshot"],
                "edit": {"task_id": selected, "depends_on": dependencies,
                         "estimate_hours": estimate, "max_parallel": max_parallel},
            },
        }
        try:
            validate_answer(request, answer)
            result = service.execute(run_id, answer)
            if result["status"] != "waiting_input":
                raise ValueError("当前任务已离开计划确认阶段，请刷新后查看。")
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
            return
        st.session_state.notice = "排期已保存，甘特图已更新。"
        st.rerun()


def render_engineering(service, run_id, board, dispatch_task, *, live=False, request=None):
    engineering = board.engineering
    if engineering is None or board.project_build:
        return
    run = service.store.run(run_id)
    if engineering.requirements:
        with st.expander("需求规格与确认记录", expanded=engineering.design is None):
            spec_view(engineering.requirements)
            st.caption(
                "当前需求基线已确认" if engineering.requirements_approval
                == fingerprint(engineering.requirements) else "等待确认需求基线"
            )
    if engineering.design:
        spec = engineering.design
        confirmed = engineering.design_approval == fingerprint(spec)
        status = "已确认" if confirmed else "待确认"
        st.caption(
            f"设计规格{status} · {len(spec.requirements)} 条需求 · "
            f"{len(spec.modules)} 个模块 · {len(spec.interfaces)} 份接口定义"
        )
        if not live:
            st.download_button(
                "下载设计规格", export_design_spec(spec, confirmed=confirmed),
                file_name="design-spec.md", mime="text/markdown", key=f"design-spec-{run_id}",
                on_click="ignore",
            )
        with st.expander("设计规格", expanded=engineering.plan is None):
            spec_view(spec)
    if board.technical_report:
        if not live:
            st.download_button(
                "下载详细技术报告", export_technical_report(board, run["name"]),
                file_name="technical-report.md", mime="text/markdown",
                key=f"technical-report-{run_id}", on_click="ignore",
            )
        with st.expander("查看完整技术报告"):
            for title, body in board.technical_report.sections.items():
                st.markdown(f"### {title}\n\n{technical_body(title, body)}")
    if engineering.plan is None:
        return
    approved = engineering.plan_approval == fingerprint(engineering.plan)
    failed = run["status"] == "failed" and run["stage"] in {"task_decompose", "plan_review"}
    blocked = engineering.plan_review and any(
        finding.severity == "major" for finding in engineering.plan_review.findings
    )
    if approved:
        plan_status = "计划已确认，任务可按依赖启动。"
    elif failed:
        plan_status = "当前显示上一次保存的草案，继续任务后会重新生成或评审。"
    elif blocked:
        plan_status = "计划还有重大评审问题，修改并通过复核后才能启动任务。"
    elif engineering.plan_review is None and request and request["kind"] == "plan_confirm":
        plan_status = "已保存排期草案，确认时将先独立复核。"
    else:
        plan_status = "当前计划等待评审或确认。"
    st.subheader("开发任务与执行编排")
    st.caption(
        f"PM Agent 分解了 {len(engineering.plan.tasks)} 项任务，"
        f"最大并行任务数 {engineering.max_parallel}。{plan_status}"
    )
    approval_slot = None
    if not live and not approved and request and request["kind"] == "plan_confirm":
        edit_task_plan(service, run_id, engineering, request)
        approval_slot = st.container()
    st.dataframe(
        [{"任务": t.id, "工作项": t.title, "负责人": t.owner, "类型": t.kind,
          "来源需求": "、".join(t.requirement_ids), "模块": "、".join(t.module_ids),
          "前置依赖": "、".join(t.depends_on) or "无",
          "预计小时": t.estimate_hours, "完成标准": " / ".join(t.done_criteria)}
         for t in engineering.plan.tasks], hide_index=True, width="stretch",
    )
    if engineering.plan_review:
        for finding in engineering.plan_review.findings:
            display = st.error if finding.severity == "major" else st.info
            display(f"{finding.task_id or '整体计划'}：{finding.issue}")
    time_tab, graph_tab, board_tab, export_tab = st.tabs(
        ["执行批次与时间", "任务依赖图", "执行看板", "追溯与导出"]
    )
    rows = schedule(engineering.plan, engineering.max_parallel)
    with graph_tab:
        st.graphviz_chart(diagram(engineering), width="stretch")
        st.caption("粗框为仅按依赖和工时计算的关键路径任务。")
    with time_tab:
        total = max(row["planned_finish"] for row in rows)
        st.markdown(f"**预计总工期 {total:.4g} 小时**")
        st.html(timeline_svg(engineering))
        st.caption("排期考虑前置依赖和并行上限。深绿色标记仅按依赖和工时计算的关键路径任务。")
        st.dataframe(
            [{"任务": r["task_id"], "批次": r["batch"],
              "预计开始（小时）": r["planned_start"], "预计完成（小时）": r["planned_finish"]}
             for r in sorted(rows, key=lambda row: (row["batch"], row["task_id"]))],
            hide_index=True, width="stretch",
        )
    with board_tab:
        if not approved or not engineering.tasks:
            st.info(plan_status if failed or blocked else "确认任务计划后，编排器会解锁首批任务。")
        else:
            render_board(service, run_id, board, dispatch_task, live)
    with export_tab:
        for requirement in engineering.design.requirements:
            implementing = [t.id for t in engineering.plan.tasks
                            if requirement.id in t.requirement_ids]
            st.write(f"{requirement.id}　{requirement.description}　对应 {'、'.join(implementing)}")
        current = engineering.model_copy(
            update={"schedule_rows": [ScheduledTask.model_validate(row) for row in rows]}
        )
        st.download_button(
            "下载任务计划 JSON", current.model_dump_json(indent=2),
            file_name="development-plan.json", mime="application/json", key=f"plan-json-{run_id}",
        )
        st.download_button(
            "下载任务与编排说明", export_plan(engineering), file_name="development-plan.md",
            mime="text/markdown", key=f"plan-md-{run_id}",
        )
        if any(p.output for p in engineering.tasks.values()):
            archive = io.BytesIO()
            with zipfile.ZipFile(archive, "w") as z:
                for task_id, progress in engineering.tasks.items():
                    if progress.output:
                        for item in progress.output.files:
                            z.writestr(f"{task_id}/{item.path}", item.content)
            st.download_button(
                "下载分任务交付物", archive.getvalue(), file_name="task-deliverables.zip",
                mime="application/zip", key=f"deliverables-{run_id}",
            )
    return approval_slot


def render_board(service, run_id, board, dispatch_task, live):
    engineering = board.engineering
    st.caption("执行 Agent 生成交付物并经独立静态检查后进入待验收。人工运行或走查后提交证据并签收。")
    st.dataframe(
        [{"任务": t.id, "工作项": t.title, "状态": STATUS_NAMES[engineering.tasks[t.id].status],
          "尝试次数": engineering.tasks[t.id].attempts,
          "阻塞或操作依据": engineering.tasks[t.id].reason}
         for t in engineering.plan.tasks], hide_index=True, width="stretch",
    )
    if live:
        return
    run = service.store.run(run_id)
    cancelled = run["status"] == "cancelled"
    selected = st.selectbox("选择任务", [t.id for t in engineering.plan.tasks], key=f"task-{run_id}")
    task = next(t for t in engineering.plan.tasks if t.id == selected)
    progress = engineering.tasks[selected]
    st.write(f"**{task.title}**　负责人 {task.owner}")
    for criterion in task.done_criteria:
        st.write(f"- {criterion}")
    if progress.output:
        st.write(progress.output.summary)
        for item in progress.output.files:
            with st.expander(item.path):
                st.code(item.content)
        st.caption("Agent 提交的证据说明")
        for item in progress.output.evidence:
            st.write(item)
    for finding in progress.review_findings:
        st.error(f"独立审查意见：{finding}")
    if task.kind != "human" and progress.status == "ready":
        if progress.attempts >= 3:
            st.warning("Agent 已达到三次执行上限，请通过人工操作交付并验收。")
        if st.button("执行此任务", key=f"execute-{run_id}-{selected}",
                     disabled=cancelled or progress.attempts >= 3):
            dispatch_task(run_id, run["revision"], selected)
    choices = ACTIONS[progress.status]
    if progress.status == "failed":
        choices = ["retry"]
    if choices:
        with st.form(f"task-operation-{run_id}-{selected}-{run['revision']}"):
            action = st.selectbox("任务操作", choices, format_func=ACTION_NAMES.get)
            evidence = st.text_area("交付证据、验收结果或处理依据")
            checked = st.checkbox("验收通过时，我已逐条核对完成标准并确认所填证据")
            submitted = st.form_submit_button("提交操作", disabled=cancelled)
        if submitted:
            try:
                if action == "accept" and not checked:
                    raise ValueError("验收通过前请逐条核对完成标准。")
                update_task(service, run_id, run["revision"], selected, action, evidence)
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))
            else:
                st.rerun()
    if engineering.events:
        with st.expander("任务事件记录"):
            st.dataframe(
                [{"序号": e.sequence, "任务": e.task_id, "操作": e.action,
                  "原状态": e.from_status, "新状态": e.to_status, "执行者": e.actor,
                  "依据": e.reason} for e in engineering.events],
                hide_index=True, width="stretch",
            )
