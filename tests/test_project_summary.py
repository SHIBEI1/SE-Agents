from se_agents.planning import (
    DesignSpecification,
    DevelopmentPlan,
    Engineering,
    TaskProgress,
    fingerprint,
)
from se_agents.project_summary import summarize_project
from se_agents.schemas import Blackboard, Objection, Option, Proposal, Signoff


def engineering_board(approved=True):
    plan = DevelopmentPlan(tasks=[{
        "id": "T001", "title": "实现待办列表", "kind": "code", "owner": "开发 Agent",
        "requirement_ids": ["R01"], "module_ids": ["MOD01"], "depends_on": [],
        "deliverables": ["app.py"], "done_criteria": ["新增记录后可以查询"],
        "estimate_hours": 1, "priority": "Must",
    }], rationale="先完成最小可运行功能")
    return Blackboard(idea="待办网站", engineering=Engineering(
        plan=plan, plan_approval=fingerprint(plan) if approved else None,
        tasks={"T001": TaskProgress(status="review")},
    ))


def test_completed_planning_run_does_not_claim_tasks_are_accepted():
    board = engineering_board()
    summary = summarize_project(board, {"status": "completed", "stage": "orchestrate"})
    assert "已验收 0 项" in summary.conclusion
    assert "待验收 1 项" in summary.conclusion
    assert "验收证据" in summary.next_step
    assert "所有任务" not in summary.next_step


def test_modified_plan_requires_confirmation_again():
    board = engineering_board()
    board.engineering.plan.tasks[0].estimate_hours = 2
    summary = summarize_project(board, {"status": "completed", "stage": "orchestrate"})
    assert "尚未确认" in summary.conclusion
    assert "确认后再派发" in summary.next_step


def test_live_and_failed_states_do_not_offer_normal_confirmation():
    board = engineering_board()
    request = {"kind": "plan_confirm", "payload": {}}
    live = summarize_project(board, {"status": "running", "stage": "task_review"}, request)
    assert "无需重复" in live.next_step
    failed = summarize_project(board, {
        "status": "failed", "stage": "plan_review", "error": "返回内容校验失败",
    }, request)
    assert "错误" in failed.next_step
    assert any("校验失败" in item for item in failed.pending)


def test_all_accepted_tasks_show_export_action_without_mutating_state():
    board = engineering_board()
    board.engineering.tasks["T001"].status = "done"
    before = board.model_dump_json()
    summary = summarize_project(board, {"status": "completed", "stage": "task_update"})
    assert "所有任务均已签收" in summary.next_step
    assert before == board.model_dump_json()


def test_proposal_summary_filters_old_issues_and_preserves_conditional_verdicts():
    option = Option(id="OPT-001", title="本地待办", approach="标准库和 SQLite",
                    value="演示任务管理", features=["新增待办"], limitations="单人使用",
                    assumptions=[], claim_ids=[])
    def issue(key, option_id, status):
        return Objection(id=key, option_id=option_id, kind="factual", severity="major",
                         reason="需要核实运行环境", action="检查本机环境", author="pm",
                         status=status)
    board = Blackboard(
        idea="待办网站", options=[option],
        proposal=Proposal(option_id=option.id, sections={}, dependency_hash="snapshot"),
        objections=[issue("OBJ-001", option.id, "needs_evidence"),
                    issue("OBJ-002", option.id, "needs_evidence"),
                    issue("OBJ-003", "OPT-002", "open"),
                    issue("OBJ-004", option.id, "resolved")],
        signoffs={"pm": Signoff(verdict="conditional", reason="环境待核实")},
    )
    summary = summarize_project(board, {"status": "waiting_input", "stage": "proposal_confirm"})
    assert summary.conclusion.startswith("当前推荐：本地待办")
    assert len(summary.pending) == 1
    assert "有条件同意 1 位" == summary.reviews


def test_blocked_approval_and_partial_results_are_explicit():
    summary = summarize_project(engineering_board(False), {
        "status": "partial", "stage": "plan_confirm",
    }, {"kind": "plan_confirm", "payload": {"approval_blocked": True}})
    assert "确认被阻止" in summary.next_step
    assert any("部分结果" in item for item in summary.pending)


def long_spec_board():
    spec = DesignSpecification.model_validate({
        "title": "个人待办网站",
        "overview": "功能范围和环境说明。" * 45 + "末尾说明必须完整保留。",
        "requirements": [{"id": "R01", "description": "新增待办", "priority": "Must",
                          "acceptance": "新增后可查询"}],
        "modules": [{"id": "MOD01", "name": "待办", "responsibility": "管理待办",
                     "requirement_ids": ["R01"]}],
    })
    return Blackboard(idea="待办网站", engineering=Engineering(requirements=spec))


def test_summary_keeps_full_overview_and_long_failure_reason():
    board = long_spec_board()
    reason = "需要补充环境检查记录。" * 40 + "最后一项证据尚未提交。"
    summary = summarize_project(board, {
        "status": "failed", "stage": "requirements_confirm", "error": reason,
    })
    assert summary.context.endswith(board.engineering.requirements.overview)
    assert "本次确认需求基线执行失败" in summary.conclusion
    assert "成功生成的新结果" in summary.conclusion
    assert summary.pending[0].endswith(reason)


def test_long_summary_page_exposes_complete_text_without_model_calls():
    from streamlit.testing.v1 import AppTest

    board = long_spec_board()
    source = (
        "import streamlit as st\n"
        "from se_agents.schemas import Blackboard\n"
        "from se_agents.project_summary import render_project_summary\n"
        f"board = Blackboard.model_validate_json({board.model_dump_json()!r})\n"
        "render_project_summary(st.empty(), board, "
        "{'status': 'waiting_input', 'stage': 'requirements_confirm'})\n"
    )
    page = AppTest.from_string(source, default_timeout=30).run()
    assert not page.exception
    assert "展开完整摘要" in {item.label for item in page.expander}
    assert any(item.value.endswith("末尾说明必须完整保留。") for item in page.markdown)
    assert any("当前需求基线尚未确认" in item.value for item in page.markdown)


def test_summary_changes_with_current_step_even_when_saved_board_is_unchanged():
    board = long_spec_board()
    before = board.model_dump_json()
    waiting = summarize_project(board, {"status": "waiting_input", "stage": "requirements_confirm"})
    running = summarize_project(board, {"status": "running", "stage": "report_scope"})
    failed = summarize_project(board, {
        "status": "failed", "stage": "report_scope", "error": "需求编号不一致",
    })
    assert len({waiting.conclusion, running.conclusion, failed.conclusion}) == 3
    assert "正在编写详细需求" in running.conclusion
    assert "需求编号不一致" in failed.conclusion
    assert before == board.model_dump_json()
    assert failed == summarize_project(board, {
        "status": "failed", "stage": "report_scope", "error": "需求编号不一致",
    })


def test_summary_records_actual_requirement_edits_and_user_feedback():
    previous = long_spec_board()
    updated = previous.model_copy(deep=True)
    updated.engineering.requirements.requirements[0].description = "标题和截止日期均为必填"
    updated.engineering.feedback.append("请将截止日期设为必填")
    summary = summarize_project(updated, {
        "status": "waiting_input", "stage": "requirements_confirm",
    }, previous=previous)
    assert any("修改 R01" in item for item in summary.changes)
    assert any("截止日期设为必填" in item for item in summary.changes)
    updated.engineering.requirements_approval = fingerprint(updated.engineering.requirements)
    confirmed = summarize_project(updated, {"status": "running", "stage": "report_scope"},
                                  previous=previous)
    assert "需求基线已确认。" in confirmed.changes


def test_summary_does_not_show_old_plan_as_new_requirement_result():
    board = engineering_board()
    board.engineering.requirements = long_spec_board().engineering.requirements
    summary = summarize_project(board, {
        "status": "waiting_input", "stage": "requirements_confirm",
    })
    assert "已整理 1 条需求" in summary.conclusion
    assert "开发任务" not in summary.conclusion
