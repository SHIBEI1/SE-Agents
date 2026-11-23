import copy

import pytest

from se_agents.mock import mock_reply
from se_agents.planning import (
    DesignSpecification,
    DevelopmentPlan,
    fingerprint,
    schedule,
    validate_plan,
    validate_spec,
)
from se_agents.reporting import export_design_spec, export_technical_report
from se_agents.schemas import RunConfig
from se_agents.service import Service
from se_agents.task_execution import execute_task, update_task


def design(title="笔记社区"):
    return {
        "title": title, "overview": "基于已提供的设计实现首版核心操作。",
        "requirements": [
            {"id": "R01", "description": "用户可以登录", "priority": "Must",
             "acceptance": "正确凭据能登录，错误凭据返回401"},
            {"id": "R02", "description": "用户可以查询内容", "priority": "Must",
             "acceptance": "查询返回与保存内容一致的列表"},
        ],
        "modules": [
            {"id": "MOD01", "name": "账号", "responsibility": "验证身份",
             "requirement_ids": ["R01"]},
            {"id": "MOD02", "name": "内容", "responsibility": "提供内容列表",
             "requirement_ids": ["R02"]},
        ],
        "interfaces": [{"id": "IF01", "name": "查询", "source_module": "MOD01",
                        "target_module": "MOD02", "contract": "GET /items 返回JSON"}],
        "constraints": ["使用SQLite持久化"], "assumptions": [],
    }


def plan():
    def task(key, requirement, module, dependencies, kind="code"):
        return {
            "id": key, "title": f"实现 {requirement}", "kind": kind, "owner": "开发Agent",
            "requirement_ids": [requirement], "module_ids": [module],
            "depends_on": dependencies, "deliverables": [f"{key}.py"],
            "done_criteria": ["运行对应功能的验收用例，输出与接口约定一致"],
            "estimate_hours": 2, "priority": "Must",
        }
    tasks = [task("T001", "R01", "MOD01", []), task("T002", "R02", "MOD02", [])]
    testing = task("T003", "R01", "MOD01", ["T001", "T002"], "test")
    testing["requirement_ids"] = ["R01", "R02"]
    testing["module_ids"] = ["MOD01", "MOD02"]
    tasks.append(testing)
    return {"tasks": tasks, "rationale": "两个模块可独立实现，再合并验收。"}


@pytest.fixture
def engineering_replies(monkeypatch):
    calls = []

    def reply(phase, role, payload):
        calls.append((phase, role))
        if phase in {"engineering_design", "requirements_spec"}:
            title = "即时聊天" if "聊天" in str(payload) else "笔记社区"
            return design(title)
        if phase == "task_decompose":
            return plan()
        if phase in {"plan_review", "task_review", "report_review"}:
            return {"findings": []}
        if phase == "task_execute":
            return {
                "summary": "已提交源码，测试等待实际执行",
                "files": [{"path": p, "content": "print('generated')\n"}
                          for p in payload["任务"]["deliverables"]],
                "evidence": ["源码已提供，测试尚未运行"],
            }
        prose = "用户发起操作后检查身份并读取已保存内容，失败时返回明确错误。" * 10
        if phase == "report_scope":
            return {
                "sections": dict.fromkeys(payload["目录"], prose),
                "assumptions": [{"topic": "部署", "recommendation": "本地",
                                 "rationale": "先验证", "verify": "运行用例",
                                 "alternative": "部署服务器"}],
                "requirements": [{"id": r["id"], "priority": "必须", "actor": "用户",
                                  "behavior": r["description"], "acceptance": r["acceptance"]}
                                 for r in design()["requirements"]],
            }
        if phase == "report_architecture":
            return {"content": prose}
        if phase == "report_delivery":
            return {"sections": dict.fromkeys(payload["目录"], prose)}
        if phase == "report_tests":
            return {"tests": [{"requirements": ["R01", "R02"], "scenario": "核心操作",
                               "action": "登录后查询", "expected": "内容一致"}] * 5}
        return mock_reply(phase, role, payload)

    monkeypatch.setattr("se_agents.models.mock_reply", reply)
    return calls


def answer(service, run, **value):
    request = service.request(run)
    return service.execute(run, {
        "request_id": request["id"], "value": value, "reason": "依据当前文档确认",
    })


def approve(service, run):
    request = service.request(run)
    return answer(service, run, approve=True, snapshot=request["payload"]["snapshot"])


def planned(service):
    run = service.create("笔记社区设计规格", RunConfig(workflow_mode="planning"))
    assert service.execute(run)["status"] == "waiting_input"
    assert service.request(run)["kind"] == "design_confirm"
    assert approve(service, run)["status"] == "waiting_input"
    assert service.request(run)["kind"] == "plan_confirm"
    assert approve(service, run)["status"] == "completed"
    return run


def test_spec_to_planning_persists_and_uses_pm(tmp_path, engineering_replies):
    service = Service(tmp_path)
    run = planned(service)
    engineering = Service(tmp_path).store.board(run).engineering
    assert engineering.design_approval == fingerprint(engineering.design)
    assert engineering.plan_approval == fingerprint(engineering.plan)
    assert {k: p.status for k, p in engineering.tasks.items()} == {
        "T001": "ready", "T002": "ready", "T003": "pending",
    }
    assert ("task_decompose", "pm") in engineering_replies
    assert ("plan_review", "devil") in engineering_replies
    serial = schedule(engineering.plan, 1)
    parallel = schedule(engineering.plan, 2)
    assert len({r["batch"] for r in serial}) == 3
    assert len({r["batch"] for r in parallel}) == 2


def test_live_plan_edit_preserves_revision_and_recalculates_schedule(tmp_path, engineering_replies):
    service = Service(tmp_path)
    run = service.create("笔记社区设计规格", RunConfig(workflow_mode="planning"))
    service.execute(run)
    approve(service, run)
    before = service.store.run(run)["revision"]
    request = service.request(run)
    edit = {"task_id": "T001", "depends_on": ["T003"], "estimate_hours": 3, "max_parallel": 1}
    value = {"action": "edit_plan", "snapshot": request["payload"]["snapshot"], "edit": edit}
    with pytest.raises(ValueError, match="依赖存在循环"):
        answer(service, run, **value)
    assert service.store.run(run)["revision"] == before
    assert service.request(run)["id"] == request["id"]

    edit.update(task_id="T002", depends_on=["T001"])
    assert answer(service, run, **value)["status"] == "waiting_input"
    reopened = Service(tmp_path)
    updated = reopened.store.board(run).engineering
    assert updated.plan.tasks[1].depends_on == ["T001"]
    assert updated.max_parallel == 1
    assert updated.plan_approval is None and updated.plan_review is None
    assert {row.task_id: row.planned_start for row in updated.schedule_rows} == {
        "T001": 0, "T002": 2, "T003": 5,
    }
    assert reopened.store.board(run, before).engineering.plan.tasks[1].depends_on == []
    assert engineering_replies.count(("plan_review", "devil")) == 1
    with pytest.raises(ValueError, match="当前请求"):
        reopened.execute(run, {"request_id": request["id"], "value": value, "reason": "旧请求"})

    assert reopened.request(run)["payload"]["review_required"]
    assert approve(reopened, run)["status"] == "completed"
    ready = reopened.store.board(run).engineering
    assert ready.tasks["T001"].status == "ready"
    assert ready.tasks["T002"].status == "pending"
    assert engineering_replies.count(("plan_review", "devil")) == 2
    assert engineering_replies.count(("task_decompose", "pm")) == 1


def test_schedule_starts_waiting_tasks_when_a_slot_frees_up():
    raw = plan()
    raw["tasks"][0]["estimate_hours"] = 1
    raw["tasks"][1]["estimate_hours"] = 4
    raw["tasks"][2]["depends_on"] = ["T001"]
    row = next(r for r in schedule(DevelopmentPlan.model_validate(raw), 2) if r["task_id"] == "T003")
    assert row["planned_start"] == 1
    raw["tasks"][2]["depends_on"] = []
    row = next(r for r in schedule(DevelopmentPlan.model_validate(raw), 2) if r["task_id"] == "T003")
    assert row["earliest_start"] == 0
    assert row["planned_start"] == 1


def test_full_flow_reaches_requirements_design_and_plan(tmp_path, engineering_replies):
    service = Service(tmp_path)
    run = service.create(
        "校园工具\n目标用户：学生\n目标：便于查询\n限制：四周完成",
        RunConfig(workflow_mode="full", chat_mode=True),
    )
    service.execute(run)
    kinds = []
    for _ in range(12):
        request = service.request(run)
        if request is None:
            break
        kind = request["kind"]
        kinds.append(kind)
        approve(service, run)
    assert service.store.run(run)["status"] == "completed"
    assert kinds == ["proposal_confirm", "requirements_confirm", "design_confirm", "plan_confirm"]
    board = service.store.board(run)
    assert board.scores and board.objections and board.proposal
    assert any(item.kind == "resolve" and item.actor == "agent" for item in board.decisions)
    assert any(item.kind == "select" and item.actor == "orchestrator" for item in board.decisions)
    assert board.engineering.requirements_approval
    assert board.technical_report and board.engineering.plan_approval
    spec = export_design_spec(board.engineering.design, confirmed=True)
    assert all(value in spec for value in ("已确认", "R01", "MOD02", "IF01", "使用SQLite持久化"))
    report = export_technical_report(board, service.store.run(run)["name"])
    assert all(title in report for title in board.technical_report.sections)
    assert any(phase == "diverge" for phase, _ in engineering_replies)


def test_returned_proposal_reruns_with_feedback_and_fresh_confirmation(
    tmp_path, engineering_replies, monkeypatch,
):
    from se_agents import models

    previous = models.mock_reply
    feedback_seen = []

    def reply(phase, role, payload):
        if phase == "diverge" and payload.get("feedback"):
            feedback_seen.append((role, payload["feedback"]))
        return previous(phase, role, payload)

    monkeypatch.setattr(models, "mock_reply", reply)
    service = Service(tmp_path)
    run = service.create(
        "校园工具\n目标用户：学生\n目标：便于查询\n限制：四周完成",
        RunConfig(workflow_mode="full", chat_mode=True),
    )
    assert service.execute(run)["status"] == "waiting_input"
    first = service.request(run)
    assert first["kind"] == "proposal_confirm"
    assert answer(service, run, approve=False, snapshot=first["payload"]["snapshot"],
                  feedback="请优先减少数据维护工作。")["status"] == "waiting_input"
    second = service.request(run)
    assert second["kind"] == "proposal_confirm" and first["id"] != second["id"]
    assert {role for role, _ in feedback_seen} == {"pm", "user", "tech", "devil"}
    assert service.store.board(run).engineering.requirements is None
    assert approve(service, run)["status"] == "waiting_input"
    assert service.request(run)["kind"] == "requirements_confirm"


def test_all_blocked_recommendation_preserves_unverified_facts(
    tmp_path, engineering_replies, monkeypatch,
):
    from se_agents import models

    previous = models.mock_reply

    def reply(phase, role, payload):
        result = previous(phase, role, payload)
        if phase == "diverge":
            result = copy.deepcopy(result)
            for option in result["options"]:
                option["assumptions"] = [
                    {"statement": "尚未取得外部接口授权。", "critical": True,
                     "verification_plan": "向接口维护方核查。"}
                ]
                option["title"] += f" {role}"
        return result

    monkeypatch.setattr(models, "mock_reply", reply)
    service = Service(tmp_path)
    run = service.create(
        "校园工具\n目标用户：学生\n目标：便于查询\n限制：四周完成",
        RunConfig(workflow_mode="full", chat_mode=True),
    )
    assert service.execute(run)["status"] == "waiting_input"
    request = service.request(run)
    assert request["kind"] == "proposal_confirm"
    assert request["payload"]["needs_risk_ack"]
    board = service.store.board(run)
    assert len(board.shortlist) == 4
    assert {len(score.assessments) for score in board.scores.values()} == {4}
    assert board.proposal and board.proposal_review_issues
    assert not board.evidence
    assert all(a.status == "unverified" for option in board.options for a in option.assumptions)
    with pytest.raises(ValueError, match="遗留风险"):
        approve(service, run)
    assert answer(service, run, approve=True, snapshot=request["payload"]["snapshot"],
                  risk_ack=True)["status"] == "waiting_input"
    assert service.request(run)["kind"] == "requirements_confirm"


def test_stale_confirmation_cannot_advance(tmp_path, engineering_replies):
    service = Service(tmp_path)
    run = service.create("设计规格", RunConfig(workflow_mode="planning"))
    service.execute(run)
    before = service.store.run(run)["revision"]
    with pytest.raises(ValueError, match="版本"):
        answer(service, run, approve=True, snapshot="stale")
    assert service.store.run(run)["revision"] == before
    assert service.request(run)["kind"] == "design_confirm"


def test_agent_submission_requires_independent_acceptance(tmp_path, engineering_replies):
    service = Service(tmp_path)
    run = planned(service)
    revision = service.store.run(run)["revision"]
    execute_task(service, run, revision, "T001")
    engineering = service.store.board(run).engineering
    assert engineering.tasks["T001"].status == "review"
    assert engineering.tasks["T003"].status == "pending"
    with pytest.raises(ValueError, match="状态"):
        update_task(service, run, service.store.run(run)["revision"], "T003", "start", "提前执行")
    update_task(service, run, service.store.run(run)["revision"],
                "T001", "accept", "人工运行账号用例，正确和错误凭据均符合预期")
    assert service.store.board(run).engineering.tasks["T003"].status == "pending"
    execute_task(service, run, service.store.run(run)["revision"], "T002")
    update_task(service, run, service.store.run(run)["revision"],
                "T002", "accept", "人工核对列表用例输出")
    assert service.store.board(run).engineering.tasks["T003"].status == "ready"
    events = service.store.board(run).engineering.events
    assert any(e.task_id == "T002" and e.actor == "human" for e in events)
    assert events[-1].task_id == "T003" and events[-1].to_status == "ready"


def test_blocking_and_stale_board_actions(tmp_path, engineering_replies):
    service = Service(tmp_path)
    run = planned(service)
    revision = service.store.run(run)["revision"]
    update_task(service, run, revision, "T001", "block", "等待接口确认")
    assert service.store.board(run).engineering.tasks["T002"].status == "ready"
    assert service.store.board(run).engineering.tasks["T003"].status == "pending"
    with pytest.raises(ValueError, match="版本"):
        update_task(service, run, revision, "T001", "unblock", "已有接口")
    update_task(service, run, service.store.run(run)["revision"],
                "T001", "unblock", "已获得接口文档")
    assert service.store.board(run).engineering.tasks["T001"].status == "ready"


@pytest.mark.parametrize("defect", ["cycle", "missing_dependency", "missing_requirement", "no_test"])
def test_gate_rejects_invalid_task_graph(defect):
    spec = DesignSpecification.model_validate(design())
    value = plan()
    if defect == "cycle":
        value["tasks"][0]["depends_on"] = ["T003"]
    elif defect == "missing_dependency":
        value["tasks"][0]["depends_on"] = ["MISSING"]
    elif defect == "missing_requirement":
        value["tasks"][1]["requirement_ids"] = ["UNKNOWN"]
    else:
        value["tasks"][2]["kind"] = "doc"
    with pytest.raises(ValueError):
        validate_plan(DevelopmentPlan.model_validate(value), spec)


def test_gate_rejects_orphan_interface_and_uncovered_requirement():
    value = design()
    value["interfaces"][0]["target_module"] = "MISSING"
    with pytest.raises(ValueError, match="端点"):
        validate_spec(DesignSpecification.model_validate(value))
    value = copy.deepcopy(design())
    value["modules"] = value["modules"][:1]
    with pytest.raises(ValueError, match="承接"):
        validate_spec(DesignSpecification.model_validate(value))


def test_major_plan_findings_cannot_be_approved(tmp_path, engineering_replies, monkeypatch):
    from se_agents import models

    previous = models.mock_reply
    repair = False

    def reply(phase, role, payload):
        if phase == "plan_review" and not repair:
            return {"findings": [{"task_id": "T001", "severity": "major",
                                  "issue": "实现任务缺少必要业务步骤"}]}
        if phase == "task_decompose" and repair:
            assert payload["修改意见"] == ["在任务里补充账号校验步骤。"]
            assert payload["上一版评审意见"]["findings"][0]["severity"] == "major"
            result = previous(phase, role, payload)
            result["tasks"][0]["title"] += "并补充账号校验"
            return result
        return previous(phase, role, payload)

    monkeypatch.setattr(models, "mock_reply", reply)
    service = Service(tmp_path)
    run = service.create("设计规格", RunConfig(workflow_mode="planning"))
    service.execute(run)
    assert approve(service, run)["status"] == "waiting_input"
    board = service.store.board(run)
    assert board.engineering.plan_revisions == 2
    assert board.engineering.plan_approval is None
    request = service.request(run)
    assert request["kind"] == "plan_confirm" and request["payload"]["approval_blocked"]
    with pytest.raises(ValueError, match="重大评审"):
        approve(service, run)
    repair = True
    assert answer(service, run, approve=False, snapshot=request["payload"]["snapshot"],
                  feedback="在任务里补充账号校验步骤。")["status"] == "waiting_input"
    assert not service.request(run)["payload"]["approval_blocked"]
    assert approve(service, run)["status"] == "completed"


def test_failed_plan_rewrite_resumes_and_keeps_all_valid_criteria(
    tmp_path, engineering_replies, monkeypatch,
):
    from se_agents import models

    previous = models.mock_reply
    fail_rewrite = True

    def reply(phase, role, payload):
        result = previous(phase, role, payload)
        if phase == "plan_review" and len(payload["任务计划"]["tasks"][0]["done_criteria"]) == 1:
            return {"findings": [{"task_id": "T001", "severity": "major",
                                  "issue": "请覆盖额外的登录边界场景。"}]}
        if phase == "task_decompose" and payload["上一版任务索引"]:
            assert payload["需求到模块的映射"]["R01"] == ["MOD01"]
            result["tasks"][0]["done_criteria"] = [
                f"核对登录场景 {i} 的结果" for i in range(17 if fail_rewrite else 8)
            ]
        return result

    monkeypatch.setattr(models, "mock_reply", reply)
    service = Service(tmp_path)
    run = service.create("设计规格", RunConfig(workflow_mode="planning"))
    service.execute(run)
    failed = approve(service, run)
    assert failed["status"] == "failed" and failed["stage"] == "task_decompose"
    assert "done_criteria" in failed["error"] and "最多 16 条" in failed["error"]
    saved = service.store.board(run).engineering
    assert saved.design_approval and not saved.plan_approval
    assert len(saved.plan.tasks[0].done_criteria) == 1
    fail_rewrite = False
    assert service.execute(run)["status"] == "waiting_input"
    assert not service.request(run)["payload"]["approval_blocked"]
    assert len(service.store.board(run).engineering.plan.tasks[0].done_criteria) == 8


def test_static_findings_cannot_unlock_downstream(tmp_path, engineering_replies, monkeypatch):
    from se_agents import models

    previous = models.mock_reply

    def reply(phase, role, payload):
        if phase == "task_review":
            return {"findings": ["缺少接口实现"]}
        return previous(phase, role, payload)

    monkeypatch.setattr(models, "mock_reply", reply)
    service = Service(tmp_path)
    run = planned(service)
    execute_task(service, run, service.store.run(run)["revision"], "T001")
    with pytest.raises(ValueError, match="独立审查"):
        update_task(service, run, service.store.run(run)["revision"],
                    "T001", "accept", "仍尝试签收")
    assert service.store.board(run).engineering.tasks["T003"].status == "pending"


def test_human_task_never_dispatched(tmp_path, engineering_replies):
    service = Service(tmp_path)
    run = planned(service)
    board = service.store.board(run)
    board.engineering.plan.tasks[0].kind = "human"
    board.engineering.plan_approval = fingerprint(board.engineering.plan)
    service.store.publish(run, service.store.run(run)["revision"], "test-human", board)
    with pytest.raises(ValueError, match="人工"):
        execute_task(service, run, service.store.run(run)["revision"], "T001")


def test_different_design_input_is_preserved(tmp_path, engineering_replies):
    service = Service(tmp_path)
    run = service.create("即时聊天设计规格", RunConfig(workflow_mode="planning"))
    service.execute(run)
    assert service.store.board(run).engineering.design.title == "即时聊天"


def test_front_page_and_planning_dashboard(tmp_path, engineering_replies, monkeypatch):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    from se_agents import models

    monkeypatch.setenv("SE_AGENTS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(models, "configured_model", lambda: RunConfig())
    service = Service(tmp_path)
    run = planned(service)
    app_file = str(Path(__file__).resolve().parents[1] / "app.py")
    app = AppTest.from_file(app_file, default_timeout=30).run()
    assert not app.exception
    choice = next(item for item in app.selectbox if item.label == "选择起点")
    assert "课堂作业二：设计规格到任务编排" in choice.options
    choice.set_value("课堂作业二：设计规格到任务编排").run()
    assert not app.exception
    next(button for button in app.button if button.key == f"select-{run}").click().run()
    assert not app.exception
    assert "当前进展与待办" in {item.label for item in app.expander}
    assert {tab.label for tab in app.tabs} >= {
        "任务依赖图", "执行批次与时间", "执行看板", "追溯与导出",
    }
    assert any(button.label == "执行此任务" for button in app.button)
    assert len(app.dataframe) >= 4
    assert not any(expander.label == "候选与评分" for expander in app.expander)
    assert "下载设计规格" in {el.proto.label for el in app.get("download_button")}


def test_upload_design_and_edit_gantt_in_workbench(tmp_path, engineering_replies, monkeypatch):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    from se_agents import models

    monkeypatch.setenv("SE_AGENTS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(models, "configured_model", lambda: RunConfig(mode="deepseek"))
    real_init = models.ModelRunner.__init__

    def mock_init(self, store, run_id, config, **kwargs):
        real_init(self, store, run_id, config.model_copy(update={"mode": "mock"}), **kwargs)

    monkeypatch.setattr(models.ModelRunner, "__init__", mock_init)
    root = Path(__file__).resolve().parents[1]
    service = Service(tmp_path)
    app = AppTest.from_file(str(root / "app.py"), default_timeout=30).run()

    def click(label):
        next(button for button in app.button if button.label == label).click().run()
        if "work" in app.session_state:
            app.session_state["work"].result(timeout=15)
            app.run()
        assert not app.exception

    app.selectbox(key="workflow-choice").set_value("课堂作业二：设计规格到任务编排").run()
    spec = root / "examples" / "design-notes.md"
    app.file_uploader[0].set_value((spec.name, spec.read_bytes(), "text/markdown"))
    click("导入设计规格")
    run = service.store.list_runs()[0]["id"]
    assert "POST /sessions" in service.store.board(run).engineering.source_text
    click("确认设计规格并拆任务")
    assert any("预计总工期 4 小时" in field.value for field in app.markdown)
    app.selectbox(key=f"edit-task-{run}").set_value("T002").run()
    app.multiselect[0].set_value(["T001"])
    click("保存并重排")
    assert any("预计总工期 6 小时" in field.value for field in app.markdown)
    assert app.multiselect[0].value == ["T001"]
    assert any("甘特图已更新" in field.value for field in app.success)
    click("复核并确认任务计划")
    assert service.store.run(run)["status"] == "completed"
    assert not any(button.label == "保存并重排" for button in app.button)


def test_web_shows_single_recommendation_for_final_decision(
    tmp_path, engineering_replies, monkeypatch,
):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("SE_AGENTS_DATA_DIR", str(tmp_path))
    service = Service(tmp_path)
    run = service.create(
        "校园工具\n目标用户：学生\n目标：便于查询\n限制：四周完成",
        RunConfig(workflow_mode="full", chat_mode=True),
    )
    service.execute(run)
    original_config = Service.config

    def config(self, run_id):
        return original_config(self, run_id).model_copy(update={"mode": "deepseek"})

    monkeypatch.setattr(Service, "config", config)
    app_file = str(Path(__file__).resolve().parents[1] / "app.py")
    app = AppTest.from_file(app_file, default_timeout=30).run()
    next(button for button in app.button if button.key == f"select-{run}").click().run()
    assert not app.exception
    assert [heading.value for heading in app.subheader].count("推荐建设方案") == 1
    labels = {button.label for button in app.button}
    assert {"采纳方案并整理需求", "退回重新论证"} <= labels
    assert not app.chat_input


def test_twelve_candidates_score_in_batches_and_resume_after_invalid_reference(
    tmp_path, engineering_replies, monkeypatch,
):
    from se_agents import models

    previous = models.mock_reply
    fail_score = True
    batches = []

    def reply(phase, role, payload):
        result = previous(phase, role, payload)
        if phase == "diverge":
            source = result["options"][0]
            result["options"] = [
                dict(source, title=f"{role} 候选 {i}", assumptions=[]) for i in range(3)
            ]
        if phase == "score":
            ids = [item["id"] for item in payload["options"]]
            batches.append((role, ids))
            assert 1 <= len(ids) <= 3 and list(payload["allowed_conditions"]) == ids
            first = result["assessments"][0]
            first["scores"]["fit_note"] = "应该只保留数值"
            first["reasons_extra"] = "冗余说明"
            if fail_score and role == "pm" and ids[0] == "OPT-004":
                first["stance"] = "conditional"
                first["condition_ids"] = ["UNKNOWN"]
        return result

    monkeypatch.setattr(models, "mock_reply", reply)
    service = Service(tmp_path)
    run = service.create(
        "校园工具\n目标用户：学生\n目标：便于查询\n限制：四周完成",
        RunConfig(workflow_mode="full", chat_mode=True, max_tokens=800_000),
    )
    row = service.execute(run)
    assert row["status"] == "failed" and row["stage"] == "score"
    assert "OPT-004" in row["error"] and "condition_ids" in row["error"]
    board = service.store.board(run)
    assert len(board.shortlist) == 12 and not board.scores
    assert batches.count(("pm", ["OPT-001", "OPT-002", "OPT-003"])) == 1
    assert batches.count(("pm", ["OPT-004", "OPT-005", "OPT-006"])) == 3
    fail_score = False
    assert service.execute(run)["status"] == "waiting_input"
    assert service.request(run)["kind"] == "proposal_confirm"
    board = service.store.board(run)
    assert {len(card.assessments) for card in board.scores.values()} == {12}
    assert batches.count(("pm", ["OPT-001", "OPT-002", "OPT-003"])) == 1
    assert batches.count(("pm", ["OPT-004", "OPT-005", "OPT-006"])) == 4


def test_review_batches_resume_and_allow_omitted_empty_assumption_checks(
    tmp_path, engineering_replies, monkeypatch,
):
    from se_agents import models

    previous = models.mock_reply
    over_limit = True
    batches = []

    def reply(phase, role, payload):
        result = previous(phase, role, payload)
        if phase == "diverge":
            source = result["options"][0]
            result["options"] = [
                dict(source, title=f"{role} 候选 {i}", assumptions=[]) for i in range(3)
            ]
        if phase == "critique":
            ids = [option["id"] for option in payload["options"]]
            assert len(ids) <= 3
            batches.append((role, ids))
            if role != "devil":
                result.pop("checked_assumptions")
            if role == "user":
                count = 17 if over_limit and ids[0] == "OPT-004" else 9
                result["objections"] = [
                    {"option_id": ids[index % len(ids)], "kind": "factual", "severity": "minor",
                     "reason": f"场景 {index} 待验证。", "action": "保留验证项。"}
                    for index in range(count)
                ]
        return result

    monkeypatch.setattr(models, "mock_reply", reply)
    service = Service(tmp_path)
    run = service.create(
        "校园工具\n目标用户：学生\n目标：便于查询\n限制：四周完成",
        RunConfig(workflow_mode="full", chat_mode=True, max_tokens=800_000),
    )
    failed = service.execute(run)
    assert failed["status"] == "failed" and failed["stage"] == "critique"
    assert "objections" in failed["error"] and "最多 16 条" in failed["error"]
    over_limit = False
    assert service.execute(run)["status"] == "waiting_input"
    assert service.request(run)["kind"] == "proposal_confirm"
    initial = ["OPT-001", "OPT-002", "OPT-003"]
    assert batches.count(("user", initial)) == 1
    board = service.store.board(run)
    assert len(board.critiques["user"].objections) == 36
    assert board.critiques["user"].checked_assumptions == []


def test_critic_cannot_omit_a_required_assumption():
    from se_agents.deliberation import validate_critique
    from se_agents.schemas import CritiqueBatch

    reply = CritiqueBatch.model_validate({"reviews": [], "objections": []})
    payload = {"claims": [], "options": [], "critical_ids": ["ASM-001-1"]}
    with pytest.raises(ValueError, match="ASM-001-1"):
        validate_critique("devil", payload, reply)
    reply.checked_assumptions = ["ASM-002-1"]
    with pytest.raises(ValueError, match="ASM-002-1"):
        validate_critique("devil", payload, reply, allow_missing_checks=True)


def test_missing_critic_checks_are_reviewed_individually_and_resume(tmp_path, monkeypatch):
    from se_agents import models

    checked_ids = ["ASM-010-1", "ASM-011-1", "ASM-011-2", "ASM-012-1", "ASM-012-2"]
    fail_targeted_review = True
    targeted_calls = []
    batch_calls = []

    def reply(phase, role, payload):
        result = mock_reply(phase, role, payload)
        if phase == "diverge":
            source = result["options"][0]
            assumption = {
                "statement": "目标用户已有可导入的资料。",
                "critical": True,
                "verification_plan": "抽查五组用户资料，验证能否直接导入。",
            }
            result["options"] = [
                dict(source, title=f"{role} 候选 {index}",
                     assumptions=[assumption] * (1 if role == "devil" and index == 0 else 2))
                for index in range(3)
            ]
        if phase == "critique":
            first_id = payload["options"][0]["id"]
            batch_calls.append((role, first_id))
            if role == "devil" and first_id == "OPT-010":
                result.pop("checked_assumptions")
                result["objections"].append({
                    "option_id": "OPT-010", "kind": "tradeoff", "severity": "minor",
                    "reason": "首期支持的导入格式太多。", "action": "先支持一种格式。",
                })
        if phase == "critique_assumption":
            assumption_id = payload["assumption"]["id"]
            targeted_calls.append(assumption_id)
            if fail_targeted_review and assumption_id == "ASM-011-1":
                result.pop("action")
        return result

    monkeypatch.setattr(models, "mock_reply", reply)
    service = Service(tmp_path)
    run = service.create(
        "校园工具\n目标用户：学生\n目标：便于查询\n限制：四周完成",
        RunConfig(workflow_mode="full", chat_mode=True, max_calls=100, max_tokens=800_000),
    )
    first = service.execute(run)
    assert first["status"] == "failed" and "补查关键假设" in first["error"]
    fail_targeted_review = False
    assert service.execute(run)["status"] == "waiting_input"
    assert service.request(run)["kind"] == "proposal_confirm"
    assert batch_calls.count(("devil", "OPT-010")) == 1
    assert set(targeted_calls) == set(checked_ids)
    assert targeted_calls.count("ASM-010-1") == 1
    assert targeted_calls.count("ASM-011-1") == 4

    board = service.store.board(run)
    critic = board.critiques["devil"]
    assert set(checked_ids) <= set(critic.checked_assumptions)
    assert any(issue.kind == "tradeoff" for issue in critic.objections)
    for assumption_id in checked_ids:
        issue = next(item for item in board.objections if item.assumption_id == assumption_id)
        assert issue.author == "devil" and issue.status == "needs_evidence"
        assert "抽查五组" in issue.action
