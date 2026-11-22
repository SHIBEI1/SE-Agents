from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from pydantic import Field, JsonValue, field_validator, model_validator

from .models import ModelRunner
from .schemas import ProjectAcceptance, Record, Text

MAX_LOG = 200_000
STARTUP_MARKER = "<!-- project-startup -->"


class HttpCheck(Record):
    label: Text
    method: Literal["GET", "POST", "PATCH", "PUT", "DELETE"] = "GET"
    path: Text
    status: int = Field(default=200, ge=100, le=599)
    body: JsonValue = None
    headers: dict[str, str] = Field(default_factory=dict)
    fields: dict[str, JsonValue] = Field(default_factory=dict)
    capture: dict[str, str] = Field(default_factory=dict)
    html: str | None = None


class RuntimeAcceptance(Record):
    runtime: Literal["python", "node"] = "node"
    surface: Literal["web", "api"]
    entry: Text
    tests: list[str] = Field(min_length=1, max_length=10)
    checks: list[HttpCheck] = Field(min_length=2, max_length=16)

    def startup_args(self) -> list[str]:
        path = PurePosixPath(self.entry)
        module = path.with_suffix("").parts
        if (
            self.runtime == "python"
            and len(module) > 1
            and all(part.isidentifier() for part in module)
        ):
            return ["-m", ".".join(module)]
        return [f"./{self.entry}"]

    def startup_command(self) -> str:
        executable = "python3" if self.runtime == "python" else "node"
        return shlex.join([executable, *self.startup_args()])

    @model_validator(mode="before")
    @classmethod
    def infer_runtime(cls, value):
        if isinstance(value, dict) and "runtime" not in value:
            entry = value.get("entry")
            if isinstance(entry, str) and entry.strip().endswith(".py"):
                return {**value, "runtime": "python"}
        return value

    @field_validator("entry")
    @classmethod
    def normalize_entry(cls, entry: str) -> str:
        # Accept an unambiguous invocation, but keep only its file path.
        command = re.fullmatch(r"(?:node|python3?)\s+([A-Za-z0-9_./\\-]+)", entry)
        if command:
            entry = command[1]
        return entry.replace("\\", "/").removeprefix("./")

    @field_validator("tests")
    @classmethod
    def normalize_tests(cls, tests: list[str]) -> list[str]:
        return [path.strip().replace("\\", "/").removeprefix("./") for path in tests]


def validate_runtime_plan(plan: RuntimeAcceptance, paths: set[str]) -> None:
    python = plan.runtime == "python"
    suffixes = (".py",) if python else (".js", ".cjs", ".mjs")
    language = "Python" if python else "JavaScript"
    if plan.entry not in paths or not plan.entry.endswith(suffixes):
        candidates = "、".join(sorted(paths - set(plan.tests)))
        raise ValueError(
            f"acceptance.entry={plan.entry!r} 必须是 files 中的 {language} 启动文件路径。"
            f"请核对文件清单（{candidates}），填写相对路径。"
        )
    pattern = r"(?:^|/)test_[^/]+\.py$" if python else r"\.test\.(?:js|cjs|mjs)$"
    if any(path not in paths or not re.search(pattern, path) for path in plan.tests):
        naming = "test_*.py" if python else "*.test.js、*.test.cjs 或 *.test.mjs"
        raise ValueError(f"测试须引用文件清单中的 {naming}，且每个文件只列一次。")
    if len(set(plan.tests)) != len(plan.tests):
        raise ValueError("测试文件路径不得重复。")
    if any(
        not step.path.startswith("/") or step.path.startswith(("//", "/\\")) for step in plan.checks
    ):
        raise ValueError("验收请求必须使用以 / 开头的本机应用路径。")
    if not any(step.fields and step.status < 300 for step in plan.checks):
        raise ValueError("至少需要一条成功的 HTTP 验收请求断言业务数据。")
    if plan.surface == "web" and not any(
        step.html and step.method == "GET" and step.status == 200 for step in plan.checks
    ):
        raise ValueError("网页项目须包含页面访问和正文内容检查。")


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


PYTHON_TESTS = """
import pathlib
import sys
import unittest

root = pathlib.Path.cwd()
sys.path.insert(0, str(root))
suite = unittest.TestSuite()
for name in sys.argv[1:]:
    path = root / name
    suite.addTests(unittest.TestLoader().discover(str(path.parent), pattern=path.name))
result = unittest.TextTestRunner(verbosity=2).run(suite)
if result.testsRun <= len(result.skipped):
    sys.exit("没有实际运行的测试用例")
sys.exit(0 if result.wasSuccessful() else 1)
"""


def verify_project(
    model: ModelRunner, files: dict[str, str], plan: RuntimeAcceptance | None, enabled: bool
) -> ProjectAcceptance:
    if not enabled:
        return ProjectAcceptance(status="not_run", summary="本次构建未启用本机运行验收。")
    if plan is None:
        return ProjectAcceptance(status="not_run", summary="该项目类型尚无自动运行验收。")
    python = plan.runtime == "python"
    if not python:
        node = shutil.which("node")
        if not node:
            return ProjectAcceptance(status="not_run", summary="未找到 Node.js 22.18 或更高版本。")
        try:
            version = subprocess.check_output([node, "--version"], timeout=5).decode().strip()
            major, minor, *_ = map(int, version.removeprefix("v").split("."))
        except (OSError, ValueError, subprocess.SubprocessError):
            return ProjectAcceptance(status="not_run", summary="无法识别本机 Node.js 版本。")
        if (major, minor) < (22, 18):
            return ProjectAcceptance(status="not_run", summary="需要 Node.js 22.18 或更高版本。")

    started = time.monotonic()
    logs = []
    status: Literal["passed", "failed"] = "failed"
    summary = "项目运行验收未通过。"
    model.store.set_stage(model.run_id, "project_verify")
    try:
        validate_runtime_plan(plan, set(files))
        if not python:
            package = json.loads(files["package.json"])
            if not isinstance(package, dict) or any(
                package.get(key)
                for key in (
                    "dependencies",
                    "devDependencies",
                    "optionalDependencies",
                    "peerDependencies",
                )
            ):
                return ProjectAcceptance(
                    status="not_run", summary="本机自动验收支持仅使用 Node.js 内置模块的项目。"
                )
        with tempfile.TemporaryDirectory(prefix="se-agents-check-") as directory:
            root = Path(directory).resolve()
            for name, content in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content.encode("utf-8"))
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            # Use a fresh environment so application code does not inherit model credentials.
            env = {k: os.environ[k] for k in ("SYSTEMROOT", "WINDIR") if k in os.environ}
            env.update(
                {
                    "HOME": str(root),
                    "USERPROFILE": str(root),
                    "TMP": str(root),
                    "TEMP": str(root),
                    "TMPDIR": str(root),
                    "NO_COLOR": "1",
                    "JWT_SECRET": "local-acceptance-secret",
                    "ADMIN_PASSWORD": "local-acceptance-password",
                    "NODE_ENV": "test",
                    "PORT": str(port),
                    "HOST": "127.0.0.1",
                    "DB_PATH": str(root / "acceptance.sqlite"),
                }
            )
            args = (
                [sys.executable, "-X", "utf8", "-I", "-S", "-u"]
                if python
                else [
                    node,
                    "--permission",
                    f"--allow-fs-read={root}",
                    f"--allow-fs-write={root}",
                ]
            )

            def launch(command, output, *, prefix=args):
                model.store.check_cancel(model.run_id)
                return subprocess.Popen(
                    [*prefix, *command],
                    cwd=root,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                )

            def tail(output):
                size = output.seek(0, os.SEEK_END)
                output.seek(max(0, size - 5000))
                return output.read().decode("utf-8", errors="replace").strip()

            test_command = (
                ["-c", PYTHON_TESTS, *plan.tests]
                if python
                else [
                    "--test",
                    "--experimental-test-isolation=none",
                    "--test-reporter=tap",
                    *[f"./{path}" for path in plan.tests],
                ]
            )
            entry_command = plan.startup_args()
            entry_prefix = [sys.executable, "-X", "utf8"] if python else args
            tick = time.monotonic()
            model.store.activity(model.run_id, "运行项目自动化测试")
            with (
                tempfile.TemporaryFile() as output,
                launch(test_command, output) as proc,
            ):
                try:
                    while proc.poll() is None:
                        model.store.check_cancel(model.run_id)
                        if time.monotonic() - tick > 60 or output.seek(0, os.SEEK_END) > MAX_LOG:
                            raise TimeoutError("自动化测试超过时间或日志大小限制。")
                        time.sleep(0.2)
                finally:
                    if proc.poll() is None:
                        proc.kill()
                        proc.wait()
                    result = tail(output)
                    logs.append(result)
                if proc.returncode or (
                    not python and not re.search(r"^# pass [1-9][0-9]*$", result, re.M)
                ):
                    raise ValueError("项目测试未通过，或实际通过的测试数为零。")
            model.store.activity(model.run_id, "项目自动化测试通过", time.monotonic() - tick)

            startup = f"按 README 启动：{plan.startup_command()}"
            model.store.activity(model.run_id, startup)
            logs.append(startup)
            with (
                tempfile.TemporaryFile() as output,
                launch(entry_command, output, prefix=entry_prefix) as proc,
            ):
                try:
                    check_http(model, plan, port, proc, logs, output)
                finally:
                    if proc.poll() is None:
                        proc.kill()
                        proc.wait()
                    server_log = tail(output)
                    if server_log:
                        logs.append(server_log)
        status = "passed"
        summary = f"{'Python' if python else 'Node.js'} 项目测试和 {len(plan.checks)} 项 HTTP 验收均通过。"
    except (KeyError, OSError, ValueError) as exc:
        logs.append(str(exc))
    return ProjectAcceptance(
        status=status,
        summary=summary,
        output="\n\n".join(logs)[-12000:],
        elapsed=time.monotonic() - started,
    )


def check_http(model, plan, port, proc, logs, output):
    start = time.monotonic()
    while True:
        model.store.check_cancel(model.run_id)
        if proc.poll() is not None:
            raise ValueError("服务启动后退出。")
        if time.monotonic() - start > 15 or output.seek(0, os.SEEK_END) > MAX_LOG:
            raise TimeoutError("服务未能按时启动。")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.3):
                break
        except OSError:
            time.sleep(0.2)

    opener = build_opener(ProxyHandler({}), NoRedirect())
    saved = {}

    def fill(value):
        if isinstance(value, str):
            if match := re.fullmatch(r"\{\{([A-Za-z0-9_]+)\}\}", value):
                return saved[match[1]]
            return re.sub(r"\{\{([A-Za-z0-9_]+)\}\}", lambda m: str(saved[m[1]]), value)
        if isinstance(value, list):
            return [fill(item) for item in value]
        if isinstance(value, dict):
            return {key: fill(item) for key, item in value.items()}
        return value

    def field(body, path):
        for key in path.split("."):
            body = body[int(key)] if isinstance(body, list) else body[key]
        return body

    for step in plan.checks:
        model.store.check_cancel(model.run_id)
        if output.seek(0, os.SEEK_END) > MAX_LOG:
            raise TimeoutError("服务日志超过大小限制。")
        tick = time.monotonic()
        try:
            headers = {key: str(value) for key, value in fill(step.headers).items()}
            body = json.dumps(fill(step.body)).encode() if step.body is not None else None
            if body is not None:
                headers.setdefault("Content-Type", "application/json")
            path = fill(step.path)
            if (
                not isinstance(path, str)
                or not path.startswith("/")
                or any(char in path for char in ("\r", "\n", "\\", "#"))
            ):
                raise ValueError("验收路径无效")
            request = Request(
                f"http://127.0.0.1:{port}" + path,
                data=body,
                headers=headers,
                method=step.method,
            )
            try:
                response = opener.open(request, timeout=3)
            except HTTPError as exc:
                response = exc
            with response:
                if response.status != step.status:
                    raise ValueError(f"期望 HTTP {step.status}，实际 {response.status}")
                raw = bytearray()
                while chunk := response.read1(8192):
                    raw.extend(chunk)
                    if len(raw) > MAX_LOG:
                        raise ValueError("响应超过验收大小限制")
                    if time.monotonic() - tick > 5:
                        raise TimeoutError("响应超过验收时间限制")
                    model.store.check_cancel(model.run_id)
                if step.html and (
                    response.headers.get_content_type() != "text/html"
                    or step.html not in raw.decode("utf-8")
                ):
                    raise ValueError("HTML 页面缺少预期内容")
                if step.fields or step.capture:
                    data = json.loads(raw)
                    for key, expected in step.fields.items():
                        if field(data, key) != fill(expected):
                            raise ValueError(f"响应字段 {key} 与预期不一致")
                    saved.update({name: field(data, key) for name, key in step.capture.items()})
        except (ValueError, KeyError, TypeError, IndexError, OSError) as exc:
            model.store.activity(model.run_id, f"{step.label} · 失败", time.monotonic() - tick)
            raise ValueError(f"{step.label}：{exc}") from exc
        model.store.activity(model.run_id, f"{step.label} · 通过", time.monotonic() - tick)
        logs.append(f"通过：{step.label}")
