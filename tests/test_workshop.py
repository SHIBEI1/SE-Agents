from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from se_agents.cli import demo
from se_agents.mock import mock_reply
from se_agents.reporting import export_markdown
from se_agents.schemas import ROLES, Frame, RunConfig
from se_agents.service import Service


def create_run(service, **config):
    return service.create(
        "校园自习空间助手",
        RunConfig(**config),
        Frame(audience="大学生", goal="减少查找时间", constraints="六周课程项目"),
    )


@pytest.mark.parametrize(("blocked", "status"), [(False, "completed"), (True, "partial")])
def test_demo_workflow(tmp_path, blocked, status):
    service = Service(tmp_path)
    run_id = demo(service, blocked)
    # A new application instance reads the persisted proposal and audit trail.
    service = Service(tmp_path)
    run = service.store.run(run_id)
    board = service.store.board(run_id)
    assert run["status"] == status
    assert board.proposal.option_id == ("OPT-002" if blocked else "OPT-001")
    assert len(board.claims) == 4
    assert len(board.options) == 3
    assert set(board.signoffs) == set(ROLES)
    assert board.review_snapshot_hash
    assert {decision.actor for decision in board.decisions} == {"demo"}
    calls = service.store.call_log(run_id)
    assert len(calls) == 16
    for phase in ("diverge", "score", "signoff"):
        branch = [call for call in calls if call["task"].startswith(phase + ":")]
        assert len({call["input_hash"] for call in branch}) == 1
        assert len({call["prompt_hash"] for call in branch}) == 4
    assert any(issue.status == "needs_evidence" for issue in board.objections)
    report = export_markdown(board, service.config(run_id), run, calls)
    assert "保留分歧与证据" in report
    assert ("部分结果" if blocked else "可供用户审阅") in report


def test_invalid_answer_does_not_advance_checkpoint(tmp_path):
    service = Service(tmp_path)
    run_id = service.create("有待澄清的软件 Idea", RunConfig())
    assert service.execute(run_id)["status"] == "waiting_input"
    request = service.request(run_id)
    assert request["kind"] == "frame"
    with pytest.raises(ValueError, match="不匹配"):
        service.execute(run_id, {"request_id": "stale", "value": {}, "reason": "旧答案"})
    assert not service.store.call_log(run_id)
    assert service.request(run_id) == request
    response = {"request_id": request["id"], "value": {"stop": True}, "reason": "先保留问题"}
    assert service.execute(run_id, response)["status"] == "partial"


def test_parallel_budget_does_not_reveal_partial_branches(tmp_path):
    service = Service(tmp_path)
    run_id = create_run(service, max_calls=1)
    assert service.execute(run_id)["status"] == "partial"
    assert len(service.store.call_log(run_id)) == 1
    assert service.store.board(run_id).claims == []


def test_draft_revision_requires_fresh_signatures(tmp_path, monkeypatch):
    def one_rejection(phase, role, payload):
        if phase == "signoff" and role == "devil" and payload["proposal"]["revision"] == 0:
            return {"verdict": "reject", "reason": "缺少具体的原型验证步骤。", "condition_ids": []}
        return mock_reply(phase, role, payload)

    monkeypatch.setattr("se_agents.models.mock_reply", one_rejection)
    service = Service(tmp_path)
    run_id = demo(service, blocked=False)
    run = service.store.run(run_id)
    assert run["status"] == "completed"
    board = service.store.board(run_id)
    assert board.proposal.revision == 1
    signoff_revs = [
        r["revision"] for r in service.store.revisions(run_id) if r["stage"] == "signoff"
    ]
    assert len(signoff_revs) == 2
    first = service.store.board(run_id, signoff_revs[0])
    assert first.review_snapshot_hash != board.review_snapshot_hash
    assert first.signoffs["devil"].verdict == "reject"
    assert board.signoffs["devil"].verdict == "approve"


def test_late_model_response_cannot_publish_after_cancel(tmp_path, monkeypatch):
    called, release = Event(), Event()

    def hold_divergence(phase, role, payload):
        if phase == "diverge":
            called.set()
            if not release.wait(timeout=10):
                raise RuntimeError("测试未释放并发分支")
        return mock_reply(phase, role, payload)

    monkeypatch.setattr("se_agents.models.mock_reply", hold_divergence)
    service = Service(tmp_path)
    run_id = create_run(service)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(service.execute, run_id)
        try:
            assert called.wait(timeout=10)
            service.store.request_cancel(run_id)
        finally:
            release.set()
        assert future.result(timeout=10)["status"] == "cancelled"
    assert service.store.board(run_id).claims == []
