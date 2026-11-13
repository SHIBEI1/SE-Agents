import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from se_agents import models
from se_agents.acceptance import RuntimeAcceptance, validate_runtime_plan, verify_project
from se_agents.models import ModelRunner
from se_agents.project import Plan, recover_project, validate_plan, write_project
from se_agents.reporting import export_markdown
from se_agents.schemas import Blackboard, ProjectBuild, RunConfig, TechnicalReport
from se_agents.service import Service


@pytest.mark.parametrize(
    "test_body,expected",
    [
        ("def test_total(self): self.assertEqual(24 + 24, 48)", "passed"),
        ("def setUp(self): pass", "failed"),
    ],
)
def test_python_acceptance_uses_current_interpreter(tmp_path, monkeypatch, test_body, expected):
    service = Service(tmp_path)
    config = RunConfig()
    run = service.create("测试收银", config)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "must-not-reach-project")
    files = {
        "app/settings.py": "EXPECTED_TOTAL = 48\n",
        "app/server.py": (
            "import os, sqlite3\n"
            "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
            "from app.settings import EXPECTED_TOTAL\n"
            "assert 'DEEPSEEK_API_KEY' not in os.environ\n"
            "db = sqlite3.connect(os.environ['DB_PATH'])\n"
            "db.execute('CREATE TABLE orders (amount INTEGER)')\n"
            "db.executemany('INSERT INTO orders VALUES (?)', [(24,), (24,)])\n"
            "class Handler(BaseHTTPRequestHandler):\n"
            "    def do_GET(self):\n"
            "        self.send_response(200)\n"
            "        self.send_header('Content-Type', 'application/json')\n"
            "        self.end_headers()\n"
            "        total = db.execute('SELECT SUM(amount) FROM orders').fetchone()[0]\n"
            "        assert total == EXPECTED_TOTAL\n"
            "        self.wfile.write(('{\"total\": %d}' % total).encode())\n"
            "HTTPServer((os.environ['HOST'], int(os.environ['PORT'])), Handler).serve_forever()\n"
        ),
        "tests/test_total.py": f"import unittest\nclass Orders(unittest.TestCase):\n    {test_body}\n",
    }
    plan = RuntimeAcceptance.model_validate(
        {
            "runtime": "python",
            "surface": "api",
            "entry": "app/server.py",
            "tests": ["tests/test_total.py"],
            "checks": [
                {"label": "账单", "path": "/total", "fields": {"total": 48}},
                {"label": "服务可访问", "path": "/health"},
            ],
        }
    )
    result = verify_project(ModelRunner(service.store, run, config), files, plan, enabled=True)
    assert result.status == expected, result.output
    if expected == "passed":
        assert "通过：账单" in result.output
        assert "按 README 启动：python3 -m app.server" in result.output
    else:
        assert "没有实际运行的测试用例" in result.output


@pytest.mark.parametrize("entry", ["./src/server.js", "node src/server.js", r"src\server.js"])
def test_project_plan_normalizes_node_entry_and_ignores_notes(entry):
    plan = Plan.model_validate(
        {
            "contracts": "查询账单合计",
            "note": "模型补充说明",
            "files": [
                {"path": path, "purpose": path, "dependencies": []}
                for path in ("package.json", "src/server.js", "tests/order.test.js", "README.md")
            ],
            "acceptance": {
                "surface": "api",
                "entry": entry,
                "tests": ["./tests/order.test.js"],
                "checks": [
                    {"label": "账单", "path": "/total", "fields": {"total": 48}},
                    {"label": "健康检查", "path": "/health"},
                ],
            },
        }
    )
    validate_plan(plan)
    assert plan.acceptance.entry == "src/server.js"
    assert plan.acceptance.tests == ["tests/order.test.js"]
    invalid = RuntimeAcceptance.model_validate(
        {**plan.acceptance.model_dump(), "entry": "npm start"}
    )
    with pytest.raises(ValueError, match="acceptance.entry='npm start'"):
        validate_runtime_plan(invalid, {item.path for item in plan.files})


def plan_files():
    return [
        {"path": "app.py", "purpose": "计算订单总价", "dependencies": []},
        {"path": "tests/test_app.py", "purpose": "覆盖点餐计价", "dependencies": ["app.py"]},
        {"path": "README.md", "purpose": "说明启动和测试", "dependencies": ["app.py"]},
    ]


@pytest.mark.parametrize("existing_directory", [False, True])
def test_confirmed_report_builds_project(tmp_path, monkeypatch, existing_directory):
    for name in ("MODEL_API_KEY", "MODEL_BASE_URL", "MODEL_NAME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-token")
    service = Service(tmp_path / "workbench")
    original = Blackboard(
        idea="一个餐厅前台点餐程序",
        technical_report=TechnicalReport(
            model="设计模型", sections={"计价规则": "金额使用整数分。"}
        ),
        outcome="completed",
    )
    source_id = service.store.create(original, RunConfig(), "点餐程序")
    service.store.finish(source_id, "completed")
    target = tmp_path / "项目目录" / "点餐程序"
    if existing_directory:
        (target / "docs").mkdir(parents=True)
        (target / "docs" / "旧笔记.md").write_text("已有文件", encoding="utf-8")
    with pytest.raises(ValueError, match="报告版本"):
        service.create_build(source_id, 1, str(target))
    occupied = tmp_path / "已有文件"
    occupied.touch()
    with pytest.raises(ValueError, match="必须是可用的目录"):
        service.create_build(source_id, 0, str(occupied))
    run_id = service.create_build(source_id, 0, str(target))
    assert not (target / "app.py").exists()
    source = service.store.run(source_id)
    report = export_markdown(original, service.config(source_id), source, [])
    service.delete_run(source_id)

    review_rounds = []
    attempts_by_path = {}
    first_question = {
        "question": "点完菜后，是否需要前台确认再送到后厨？",
        "impact": "影响前台的提交步骤。",
        "default": "直接送到后厨",
        "options": ["直接送到后厨", "由前台确认后发送"],
    }
    second_question = {
        "question": "后厨是否优先看到最早的订单？",
        "impact": "影响排队显示。",
        "default": "最早的订单排在前面",
        "options": ["最早的订单排在前面", "最新的订单排在前面"],
        "paths": ["app.py"],
    }

    def reply(*, messages, **kwargs):
        system, user = (message["content"] for message in messages)
        payload = json.loads(user.split("\n上次校验问题：", 1)[0])
        if "澄清问题与建议默认值" in payload:
            phase = "project_probe"
        elif "当前文件" in payload:
            phase = "project_file"
        elif "审查轮次" in payload:
            phase = "project_review"
        else:
            phase = "project_plan"
        if phase == "project_probe":
            assert payload["已确认报告"] == original.technical_report.sections
        else:
            assert payload["已确认报告"] == report
            if existing_directory:
                assert "由前台确认后发送" in payload["用户确认的构建取舍"][0]
        if phase == "project_probe":
            value = {"questions": [first_question] if existing_directory else []}
        elif phase == "project_plan":
            files = plan_files()
            files[1]["dependencies_note"] = "测试需要引用 app.total。"
            value = {
                "contracts": {"函数": ["app.total(prices)"], "金额单位": "分"},
                "files": files,
                "acceptance": {
                    "runtime": "python",
                    "surface": "api",
                    "entry": "app.py",
                    "tests": ["tests/test_app.py"],
                    "checks": [
                        {"label": "账单", "path": "/total", "fields": {"total": 950}},
                        {"label": "服务可访问", "path": "/health"},
                    ],
                },
            }
        elif phase == "project_file":
            assert payload["共享契约"]["金额单位"] == "分"
            path = payload["当前文件"]["path"]
            attempts_by_path[path] = attempts_by_path.get(path, 0) + 1
            if path == "app.py":
                if payload["修复意见"]:
                    assert payload["原文件"] == "def total(prices):\n    return prices[0]\n"
                    if existing_directory:
                        assert "最早的订单排在前面" in payload["修复意见"][1]
                    body = "def total(prices):\n    return sum(prices)\n"
                else:
                    body = "def total(prices):\n    return prices[0]\n"
            elif path == "README.md":
                if "输出要求" not in payload:
                    return SimpleNamespace(
                        usage=None,
                        choices=[SimpleNamespace(
                            finish_reason="length", message=SimpleNamespace(content="")
                        )],
                    )
                assert payload["README 启动命令"] == "python3 ./app.py"
                body = "# 点餐程序\n\n<!-- project-startup -->\n"
            else:
                assert "app.py" in payload["依赖源码"]
                if not existing_directory and attempts_by_path[path] <= 3:
                    return SimpleNamespace(
                        usage=None,
                        choices=[SimpleNamespace(
                            finish_reason="stop", message=SimpleNamespace(
                                content=json.dumps({"note": "代码说明，没有源码"})
                            )
                        )],
                    )
                body = (
                    "from unittest import TestCase\nfrom app import total\n\n"
                    "class PriceTest(TestCase):\n"
                    "    def test_total(self):\n"
                    "        self.assertEqual(total([300, 650]), 950)\n"
                )
            value = {"content": body, "note": "可忽略的附加说明"}
        else:
            assert phase == "project_review" and "你是质疑者" in system
            review_rounds.append(payload["审查轮次"])
            first = payload["审查轮次"] == 1
            assert ("sum(prices)" in payload["源码"]["app.py"]) != first
            value = {
                "findings": [{"path": "app.py", "issue": "应汇总所有菜品价格。"}] if first else []
            }
            if first and existing_directory:
                value["choice"] = second_question
        return SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(
                finish_reason="stop", message=SimpleNamespace(content=json.dumps(value))
            )],
        )

    monkeypatch.setattr(models, "OpenAI", lambda **kwargs: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=reply)), close=lambda: None
    ))
    run = service.execute(run_id)
    if existing_directory:
        assert run["status"] == "waiting_input"
        request = service.request(run_id)
        assert request["kind"] == "project_clarify"
        assert not (target / "app.py").exists()
        invalid = service.execute(run_id, chat=(request["id"], "选 3"))
        assert invalid["status"] == "waiting_input"
        run = service.execute(run_id, chat=(request["id"], "选 2"))
        assert run["status"] == "waiting_input"
        request = service.request(run_id)
        assert request["kind"] == "project_review_clarify"
        assert not (target / "app.py").exists()
        run = service.execute(run_id, chat=(request["id"], "你决定"))
    else:
        assert run["status"] == "failed"
        assert "生成 tests/test_app.py 失败" in run["error"]
        assert "content：缺少必填字段" in run["error"]
        assert not target.exists()
        assert attempts_by_path == {"app.py": 1, "tests/test_app.py": 3}
        run = service.execute(run_id)
        assert attempts_by_path["tests/test_app.py"] == 4
        assert attempts_by_path["app.py"] == 2  # Initial generation and review repair only.
    board = service.store.board(run_id)
    assert run["status"] == "partial"
    assert board.project_result.acceptance.status == "not_run"
    assert review_rounds == [1, 2]
    if existing_directory:
        assert (target / "docs" / "旧笔记.md").read_text(encoding="utf-8") == "已有文件"
        assert [decision.kind for decision in board.decisions] == [
            "project_clarify", "project_review_clarify",
        ]
        contract = json.loads((target / "docs/project-contract.json").read_text(encoding="utf-8"))
        assert "由前台确认后发送" in contract["user_decisions"][0]
        assert "最早的订单排在前面" in contract["user_decisions"][1]
    assert (target / "docs/approved-design.md").read_text(encoding="utf-8") == report
    readme = (target / "README.md").read_text(encoding="utf-8")
    assert "```bash\npython3 ./app.py\n```" in readme
    assert "<!-- project-startup -->" not in readme
    assert board.project_result == recover_project(board.project_build, run_id)
    assert set(board.project_result.files) == {
        "app.py",
        "tests/test_app.py",
        "README.md",
        "docs/approved-design.md",
        "docs/build-checks.md",
        "docs/project-contract.json",
    }
    subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
        cwd=target,
        check=True,
        timeout=10,
    )
    (target / "app.py").write_text("# 用户后续修改\n", encoding="utf-8")
    with pytest.raises(ValueError, match="无法确认属于本次构建"):
        recover_project(board.project_build, run_id)


@pytest.mark.parametrize(
    "path", ["../app.py", "README.md/sub.md", ".env", "docs", "CON.txt", "src/AUX", "name."]
)
def test_plan_rejects_unsafe_or_reserved_paths(path):
    files = plan_files()
    files.append({"path": path, "purpose": "额外文件", "dependencies": []})
    with pytest.raises(ValueError):
        validate_plan(Plan(contracts="共享约定", files=files))


@pytest.mark.parametrize("symlink", [False, True])
def test_existing_project_conflict_preserves_directory(tmp_path, symlink):
    target = tmp_path / "project"
    target.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    if symlink:
        try:
            (target / "docs").symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            if getattr(exc, "winerror", None) == 1314:
                pytest.skip("当前 Windows 用户没有创建符号链接的权限。")
            raise
        conflict = "docs"
    else:
        (target / "README.md").write_text("已有说明", encoding="utf-8")
        conflict = "README.md"
    build = ProjectBuild(
        source_run_id="source",
        source_revision=0,
        report_hash="hash",
        source_name="测试报告",
        source_report_partial=False,
        directory=str(target),
    )
    with pytest.raises(ValueError, match=conflict):
        write_project(build, "run", {"src/app.py": "pass", "README.md": "# 新说明"}, [], "# 方案")
    assert {item.name for item in target.iterdir()} == {conflict}
    assert not list(outside.iterdir())
    if not symlink:
        assert (target / conflict).read_text(encoding="utf-8") == "已有说明"
