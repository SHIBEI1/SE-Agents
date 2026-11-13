import pytest

from se_agents.models import ModelError, ModelRunner, configured_model
from se_agents.reporting import export_markdown
from se_agents.schemas import RunConfig
from se_agents.service import Service
from se_agents.technical import Chapter


def test_configuration_requires_model_api(monkeypatch):
    for name in ("MODEL_API_KEY", "MODEL_BASE_URL", "MODEL_NAME", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="深度求索接口密钥"):
        configured_model()
    monkeypatch.setenv("MODEL_API_KEY", "test-token")
    monkeypatch.setenv("MODEL_BASE_URL", "https://example.invalid/v1/")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    config = configured_model()
    assert config.mode == "api"
    assert config.technical_mode
    assert config.base_url == "https://example.invalid/v1"
    assert "test-token" not in config.model_dump_json()


def test_technical_report_can_proceed_with_delegated_decisions(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_BASE_URL", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-token")
    requirements = [
        {
            "id": f"R{i:02}",
            "priority": "必须",
            "actor": "前台员工",
            "behavior": f"执行点餐流程中的第 {i} 项业务操作。",
            "acceptance": "操作完成后订单状态和账单保持一致。",
        }
        for i in range(1, 6)
    ]
    phases = []

    def reply(self, phase, role, payload, schema, validate=None):
        phases.append(phase)
        prose = (
            "前台员工提交餐品，服务端校验库存并保存订单。"
            "结账前需要检查金额和状态，失败时应允许重试并避免重复写入。"
        ) * 7
        if phase == "design_probe":
            value = {
                "questions": [
                    {
                        "question": "是否接入线上支付？",
                        "impact": "影响外部接口和结账边界。",
                        "default": "首版先人工确认收款。",
                    }
                ]
            }
        elif phase == "report_scope":
            assert payload["输入"]["用户实际回复"] == ["你决定"]
            value = {
                "sections": {**dict.fromkeys(payload["目录"], prose), "R01": "", "额外章节": {}},
                "assumptions": [
                    {
                        "topic": "支付",
                        "recommendation": "人工确认收款",
                        "rationale": "降低外部依赖",
                        "verify": "确认收银流程",
                        "alternative": "二期接入聚合支付",
                    }
                ],
                "requirements": requirements,
            }
        elif phase == "report_architecture":
            value = {"content": prose * 20}
        elif phase == "report_delivery":
            value = {"sections": {title: prose for title in payload["目录"]}}
        elif phase == "report_tests":
            value = {
                "tests": [
                    {
                        "requirements": [row["id"]],
                        "scenario": "验证点餐状态流转",
                        "action": "提交一笔订单并结账",
                        "expected": "订单进入已结账",
                    }
                    for row in [requirements[0]] * 19 + requirements[1:]
                ]
            }
        else:
            value = {"findings": []}
        result = schema.model_validate(value)
        if validate:
            validate(result)
        return result

    monkeypatch.setattr(ModelRunner, "run", reply)
    service = Service(tmp_path)
    config = configured_model()
    run_id = service.create("我想做一个餐厅前台点餐系统。", config)
    assert service.execute(run_id)["status"] == "waiting_input"
    pending = service.request(run_id)
    assert pending["kind"] == "design_clarify"
    run = service.execute(run_id, chat=(pending["id"], "你决定"))
    assert run["status"] == "completed"
    board = service.store.board(run_id)
    document = export_markdown(board, config, run, [])
    assert "餐厅前台点餐系统" in document
    assert "人工确认收款" in document and "需求与测试追踪" in document
    assert phases.count("report_architecture") == 4
    assert "diverge" not in phases
    assert "额外章节" not in board.technical_report.sections
    assert len(board.technical_report.sections["数据模型与状态流转"]) > 4800
    tracking = board.technical_report.sections["需求与测试追踪"]
    assert tracking.count("**测试 ") == 18
    assert all(row["id"] in tracking for row in requirements)


def test_report_resume_retries_exhausted_section_and_reuses_saved_output(tmp_path, monkeypatch):
    service = Service(tmp_path)
    config = RunConfig()
    run_id = service.create("校园物资借用助手", config)
    service.store.begin(run_id)
    payload = {"当前章节": "数据模型与状态流转"}
    valid = {"content": "按物资编号关联借用记录和归还记录。" * 20}
    monkeypatch.setattr("se_agents.models.mock_reply", lambda *args: {"content": "太短"})

    runner = ModelRunner(service.store, run_id, config)
    with pytest.raises(ModelError, match="文本为空"):
        runner.run("report_architecture", "tech", payload, Chapter)
    assert len(service.store.call_log(run_id)) == 3

    monkeypatch.setattr("se_agents.models.mock_reply", lambda *args: valid)
    resumed = ModelRunner(service.store, run_id, config, resuming=True)
    assert resumed.run("report_architecture", "tech", payload, Chapter).content == valid["content"]
    assert resumed.run("report_architecture", "tech", payload, Chapter).content == valid["content"]
    assert len(service.store.call_log(run_id)) == 4
