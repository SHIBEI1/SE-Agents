import sqlite3

import pytest

from se_agents.cli import demo
from se_agents.schemas import RunConfig
from se_agents.service import Busy, Service


def test_legacy_names_and_deleting_a_task(tmp_path):
    service = Service(tmp_path)
    existing = demo(service, blocked=False)
    idea = service.store.board(existing).idea
    waiting = service.create(idea, RunConfig())
    service.execute(waiting)
    original_request = service.request(waiting)
    with service.store.connect(write=True) as db:
        db.execute("DROP INDEX runs_name")
        db.execute("ALTER TABLE runs DROP COLUMN name")

    service = Service(tmp_path)
    tasks = {run["id"]: run for run in service.store.list_runs()}
    assert len({run["name"] for run in tasks.values()}) == 2
    assert service.request(waiting) == original_request
    assert service.store.board(existing).proposal

    service.store.rename(existing, "校园自习助手（已完成）")
    assert service.store.run(existing)["name"] == "校园自习助手（已完成）"
    with pytest.raises(ValueError, match="同名"):
        service.store.rename(waiting, "校园自习助手（已完成）")
    with service.worker_lock(), pytest.raises(Busy):
        service.delete_run(existing)

    service.delete_run(existing)
    assert [run["id"] for run in service.store.list_runs()] == [waiting]
    assert service.request(waiting) == original_request
    assert service.store.call_log(existing) == []
    assert service.store.revisions(existing) == []
    with sqlite3.connect(service.checkpoints) as db:
        assert (
            db.execute("SELECT 1 FROM checkpoints WHERE thread_id=?", (existing,)).fetchone()
            is None
        )
    assert service.store.pending_checkpoint_deletions() == []


def test_new_task_names_are_unique(tmp_path):
    service = Service(tmp_path)
    first = service.create("# 校园自习助手\n第一版", RunConfig())
    second = service.create("另一项想法", RunConfig(), name="校园自习助手")
    assert service.store.run(first)["name"] == "校园自习助手"
    assert service.store.run(second)["name"] == "校园自习助手（2）"
