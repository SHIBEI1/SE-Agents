from se_agents import deliberation as rules
from se_agents.mock import mock_reply
from se_agents.schemas import RunConfig
from se_agents.service import Service
from se_agents.workflow import saved_critiques


def test_chat_persists_clarifications_and_resumes(tmp_path):
    service = Service(tmp_path)
    run_id = service.create("我想做校园物资借用助手。", RunConfig(chat_mode=True))
    assert service.execute(run_id)["status"] == "waiting_input"
    frame = service.request(run_id)
    service.execute(run_id, chat=(frame["id"], "给学生用"))
    assert service.request(run_id)["id"] == frame["id"]
    assert service.store.chat_messages(run_id)[-1]["kind"] == "assistant"

    service = Service(tmp_path)
    service.execute(run_id, chat=(frame["id"], "希望减少重复预约，四周内完成"))
    assert service.store.board(run_id).frame.audience == "学生"
    assert service.store.board(run_id).frame.goal == "减少重复预约"
    value = service.request(run_id)
    service.execute(run_id, chat=(value["id"], "先验证查询，接受人工维护。"))
    fact = service.request(run_id)
    assert fact["id"] != value["id"]
    service.execute(run_id, chat=(fact["id"], "暂无材料"))
    selection = service.request(run_id)
    service.execute(run_id, chat=(selection["id"], "选 1，2"))
    assert service.request(run_id)["id"] == selection["id"]
    service.execute(run_id, chat=(selection["id"], "选第一个，因为开发周期较短。"))
    assert service.store.run(run_id)["status"] == "completed"
    rows = service.store.chat_messages(run_id)
    assert [row["request_id"] for row in rows if row["kind"] == "question"] == [
        frame["id"],
        value["id"],
        fact["id"],
        selection["id"],
    ]
    assert rows[-1]["content"] == "选第一个，因为开发周期较短。"
    service.delete_run(run_id)
    assert service.store.chat_messages(run_id) == []


def test_chat_skips_questions_already_answered_in_idea(tmp_path):
    service = Service(tmp_path)
    run_id = service.create(
        "校园自习助手\n目标用户：在校学生\n目标：缩短找座位的时间\n限制：四周完成",
        RunConfig(chat_mode=True),
    )
    service.execute(run_id)
    assert service.store.board(run_id).frame.audience == "在校学生"
    assert service.request(run_id)["kind"] == "resolve"


def test_review_recovers_two_candidates_from_same_author(tmp_path, monkeypatch):
    fail_pm = True
    seen = []

    def reply(phase, role, payload):
        if phase == "diverge":
            if role != "pm":
                return {"options": [], "note": "暂无额外方向。"}
            first = mock_reply(phase, "pm", payload)["options"][0]
            second = mock_reply(phase, "tech", payload)["options"][0]
            return {"options": [first, second], "note": "比较独立实现与外部集成。"}
        if phase != "critique":
            return mock_reply(phase, role, payload)
        seen.append(role)
        targets = payload["review_target_ids"]
        assert len(targets) == (0 if role == "pm" else 1)
        if role == "pm" and fail_pm:
            return {"objections": []}
        result = mock_reply(phase, role, payload)
        assert [item["claim_id"] for item in result["reviews"]] == targets
        return result

    monkeypatch.setattr("se_agents.models.mock_reply", reply)
    service = Service(tmp_path)
    run_id = service.create(
        "点餐系统\n目标用户：店员\n目标：提高点餐效率\n限制：四周完成",
        RunConfig(chat_mode=True),
    )
    assert service.execute(run_id)["status"] == "failed"
    board = service.store.board(run_id)
    assert board.shortlist == ["OPT-001", "OPT-002"]
    saved = saved_critiques(service.store, run_id, rules.critique_input(board))
    assert set(saved) == {"user", "tech", "devil"}
    assert saved["user"].objections
    assert any(
        "reviews" in (row["diagnostic"] or "") for row in service.store.call_log(run_id)
    )

    fail_pm = False
    seen.clear()
    assert service.execute(run_id)["status"] == "waiting_input"
    assert seen == ["pm"]
    assert service.store.board(run_id).objections
