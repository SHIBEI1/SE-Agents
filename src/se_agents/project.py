from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
import time
import tomllib
from collections.abc import Callable, Sequence
from contextlib import ExitStack, suppress
from pathlib import Path, PurePosixPath, PureWindowsPath

from pydantic import ConfigDict, Field, JsonValue, field_validator

from .acceptance import STARTUP_MARKER, RuntimeAcceptance, validate_runtime_plan, verify_project
from .models import ModelError, ModelOutputTruncated, ModelRunner
from .schemas import (
    DesignQuestion,
    ProjectAcceptance,
    ProjectBuild,
    ProjectBuildResult,
    Record,
    Text,
    canonical,
)
from .storage import BudgetLimit
from .windows_io import is_reparse_point, publish_windows_directory

MARKER = ".se-agents-build.json"
DESIGN = "docs/approved-design.md"
CHECKS = "docs/build-checks.md"
CONTRACTS = "docs/project-contract.json"
RESERVED = {MARKER, DESIGN, CHECKS, CONTRACTS}
FILE_LIMIT = 30_000
PROJECT_LIMIT = 360_000
MARKER_LIMIT = 120_000
COMPONENT = re.compile(r"[A-Za-z0-9_.@+()\[\]-]+")


class PlannedFile(Record):
    model_config = ConfigDict(extra="ignore")

    path: Text
    purpose: Text
    dependencies: list[str] = Field(max_length=8)


class Plan(Record):
    model_config = ConfigDict(extra="ignore")

    contracts: str | list[JsonValue] | dict[str, JsonValue]
    files: list[PlannedFile] = Field(min_length=3, max_length=32)
    acceptance: RuntimeAcceptance | None = None

    @field_validator("contracts")
    @classmethod
    def validate_contracts(cls, value: str | list | dict) -> str | list | dict:
        if isinstance(value, str):
            value = value.strip()
        if not value:
            raise ValueError("共享契约不能为空。")
        if len(canonical(value)) > 12_000:
            raise ValueError("共享契约过长，请精简到 12,000 字符以内。")
        return value


class SourceFile(Record):
    model_config = ConfigDict(extra="ignore")

    content: str = Field(min_length=1, max_length=FILE_LIMIT)


class Finding(Record):
    path: Text
    issue: Text


class Review(Record):
    findings: list[Finding] = Field(max_length=5)


class ReviewChoice(DesignQuestion):
    paths: list[Text] = Field(min_length=1, max_length=8)


class InitialReview(Review):
    choice: ReviewChoice | None = None


def project_destination(raw: str) -> Path:
    path = Path(raw.strip()).expanduser()
    if not path.is_absolute() or not path.name or path.name == "..":
        raise ValueError("请填写项目的完整目录路径。")
    path = path.parent.resolve() / path.name
    if path.is_symlink() or is_reparse_point(path) or (path.exists() and not path.is_dir()):
        raise ValueError("目标路径必须是可用的目录。")
    for parent in path.parents:
        if parent.exists():
            if not parent.is_dir():
                raise ValueError(f"项目路径经过了非目录文件：{parent}")
            break
    return path


def check_path(raw: str) -> None:
    parts = raw.split("/")
    if len(raw) > 220 or any(
        not COMPONENT.fullmatch(part)
        or PureWindowsPath(part).is_reserved()
        or part.endswith(".")
        or part in {".", "..", ".git", ".venv", "node_modules", "__pycache__"}
        or (part.startswith(".env") and part != ".env.example")
        for part in parts
    ):
        raise ValueError(f"不允许的项目文件路径：{raw}")


def validate_plan(plan: Plan) -> None:
    seen: set[str] = set()
    previous_paths: set[str] = set()
    for entry in plan.files:
        path = entry.path
        check_path(path)
        normalized = path.casefold()
        if (
            normalized in RESERVED
            or normalized in seen
            or any(
                previous.startswith(normalized + "/") or normalized.startswith(previous + "/")
                for previous in seen | RESERVED
            )
        ):
            raise ValueError(f"文件路径重复、冲突或为保留路径：{path}")
        if not set(entry.dependencies).issubset(previous_paths):
            raise ValueError(f"{path} 的依赖须引用已经列出的文件。")
        seen.add(normalized)
        previous_paths.add(path)
    if "README.md" not in previous_paths:
        raise ValueError("文件计划缺少 README.md。")
    if plan.acceptance:
        if plan.acceptance.runtime == "node" and "package.json" not in previous_paths:
            raise ValueError("Node.js 验收需要 package.json。")
        validate_runtime_plan(plan.acceptance, previous_paths)
    if not any(
        (
            path.startswith(("tests/", "test/"))
            or any(part in {"tests", "test", "__tests__", "spec"} for part in path.split("/")[:-1])
            or re.search(r"(^|[/_.-])tests?([/_.-]|$)|tests?\.[a-z]+$", path)
        )
        and PurePosixPath(path).suffix not in {"", ".md", ".txt", ".json", ".toml", ".yaml", ".yml"}
        for path in seen
    ):
        raise ValueError("文件计划须包含可执行的自动化测试源文件。")


def validate_source(path: str, source: SourceFile) -> None:
    content = source.content
    if not content.strip() or "\x00" in content or len(content.encode("utf-8")) > FILE_LIMIT:
        raise ValueError(f"{path} 为空或超过单文件大小限制，请拆分实现。")
    extension = PurePosixPath(path).suffix.lower()
    try:
        if extension == ".py":
            compile(content, path, "exec", dont_inherit=True)
        elif extension == ".json":
            json.loads(content)
        elif extension == ".toml":
            tomllib.loads(content)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"{path} 语法错误：{exc}") from exc


def generate_files(
    model: ModelRunner,
    idea: str,
    report: str,
    *,
    verify_locally: bool,
    decisions: Sequence[str] = (),
    decide: Callable[[DesignQuestion], str] | None = None,
) -> tuple[dict[str, str], list[str], str, ProjectAcceptance]:
    def stage(phase: str) -> None:
        model.store.set_stage(model.run_id, phase)

    choices = list(decisions)
    stage("project_plan")
    plan = model.run(
        "project_plan",
        "tech",
        {
            "原始想法": idea,
            "已确认报告": report,
            "用户确认的构建取舍": choices,
            "已启用本机验收": verify_locally,
        },
        Plan,
        validate_plan,
    )
    model.store.activity(
        model.run_id,
        f"已确定 {len(plan.files)} 个交付文件",
    )
    if plan.acceptance:
        surface = "网页" if plan.acceptance.surface == "web" else "API"
        model.store.activity(
            model.run_id, f"已规划{surface}入口和 {len(plan.acceptance.checks)} 项 HTTP 验收"
        )
    entries = {entry.path: entry for entry in plan.files}
    file_map: dict[str, str] = {}

    def generate(entry: PlannedFile, problems: list[str]) -> None:
        stage("project_file")
        started = time.monotonic()
        action = "修复" if problems else "生成"
        model.store.activity(model.run_id, f"{action} {entry.path}")
        dependency_code = {path: file_map[path] for path in entry.dependencies}
        existing = file_map.get(entry.path)
        payload = {
            "用户想法": idea,
            "已确认报告": report,
            "用户确认的构建取舍": choices,
            "共享契约": plan.contracts,
            "独立验收约定": plan.acceptance.model_dump() if plan.acceptance else None,
            "README 启动命令": plan.acceptance.startup_command() if plan.acceptance else None,
            "文件清单": {path: item.purpose for path, item in entries.items()},
            "当前文件": entry.model_dump(),
            "依赖源码": dependency_code,
            "原文件": existing,
            "修复意见": problems,
        }

        def check(output: SourceFile) -> None:
            validate_source(entry.path, output)
            if entry.path == "README.md" and plan.acceptance:
                if output.content.splitlines().count(STARTUP_MARKER) != 1:
                    raise ValueError(f"README 启动处必须有一行 {STARTUP_MARKER}。")

        try:
            try:
                result = model.run("project_file", "tech", payload, SourceFile, check)
            except ModelOutputTruncated:
                payload["输出要求"] = (
                    "上次响应超出了输出长度。用紧凑实现完整交付当前文件，目标 4500 字符以内。"
                    "合并重复逻辑，精简注释、示例和空行。保留约定接口、核心功能和异常处理。"
                    "返回完整 JSON 和完整文件。"
                )
                result = model.run("project_file", "tech", payload, SourceFile, check)
        except ModelError as exc:
            raise ModelError(f"{action} {entry.path} 失败。{exc}") from exc
        content = result.content
        if entry.path == "README.md" and plan.acceptance:
            command = plan.acceptance.startup_command()
            windows = (
                "\n\nWindows 下可将 `python3` 换成 `python`，请使用 Python 3.11 或更高版本。"
                if plan.acceptance.runtime == "python"
                else ""
            )
            content = re.sub(
                rf"(?m)^{re.escape(STARTUP_MARKER)}(?=\r?$)",
                lambda _: f"在项目根目录执行以下命令。\n\n```bash\n{command}\n```{windows}",
                content,
                count=1,
            )
        other_size = sum(
            len(body.encode("utf-8")) for path, body in file_map.items() if path != entry.path
        )
        if other_size + len(content.encode("utf-8")) > PROJECT_LIMIT:
            raise ValueError("项目超过本轮构建的源码大小上限。")
        file_map[entry.path] = content
        model.store.activity(model.run_id, f"已{action} {entry.path}", time.monotonic() - started)

    def validate_review(result: Review) -> None:
        if any(item.path not in file_map for item in result.findings):
            raise ValueError("审查意见必须引用项目中的文件。")
        if isinstance(result, InitialReview) and result.choice:
            if any(path not in file_map for path in result.choice.paths):
                raise ValueError("交互取舍必须引用项目中受影响的文件。")

    def review(round_number: int) -> Review:
        stage("project_review")
        can_ask = round_number == 1 and decide is not None
        return model.run(
            "project_review",
            "devil",
            {
                "用户想法": idea,
                "已确认报告": report,
                "用户确认的构建取舍": choices,
                "允许提出用户取舍": can_ask,
                "共享契约": plan.contracts,
                "源码": file_map,
                "审查轮次": round_number,
            },
            InitialReview if can_ask else Review,
            validate_review,
        )

    for entry in plan.files:
        generate(entry, [])
    notes = []
    try:
        initial = review(1)
        model.store.activity(model.run_id, f"源码审查发现 {len(initial.findings)} 项问题")
        by_path: dict[str, list[str]] = {}
        for issue in initial.findings:
            by_path.setdefault(issue.path, []).append(issue.issue)
        if decide and isinstance(initial, InitialReview) and (choice := initial.choice):
            question = DesignQuestion.model_validate(choice.model_dump(exclude={"paths"}))
            reply = decide(question)
            choices.append(reply)
            for path in choice.paths:
                by_path.setdefault(path, []).append(f"按本次用户确认调整。{reply}")
        if by_path:
            notes = [
                f"{path}（待复查）：{issue}"
                for path, issues in by_path.items() for issue in issues
            ]
            for entry in plan.files:
                if entry.path in by_path:
                    generate(entry, by_path[entry.path])
            notes = [f"{item.path}：{item.issue}" for item in review(2).findings]
    except (ModelError, BudgetLimit, ValueError) as exc:
        notes.insert(0, f"审查或修复未完成。{exc}")

    acceptance = verify_project(model, file_map, plan.acceptance, verify_locally)
    if acceptance.status == "failed":
        try:
            stage("project_runtime_repair")
            repair = model.run(
                "project_runtime_repair",
                "tech",
                {
                    "共享契约": plan.contracts,
                    "用户确认的构建取舍": choices,
                    "验收约定": plan.acceptance.model_dump(),
                    "源码": file_map,
                    "验收日志": acceptance.output,
                },
                Review,
                validate_review,
            )
            by_path = {}
            for issue in repair.findings:
                by_path.setdefault(issue.path, []).append(issue.issue)
            for entry in plan.files:
                if entry.path in by_path:
                    generate(entry, by_path[entry.path])
            if by_path:
                acceptance = verify_project(model, file_map, plan.acceptance, verify_locally)
                notes = [f"{item.path}：{item.issue}" for item in review(3).findings]
        except (ModelError, BudgetLimit, ValueError) as exc:
            notes.append(f"运行验收后的修复或复查未完成。{exc}")
    model.store.activity(model.run_id, acceptance.summary, acceptance.elapsed)
    contract = json.dumps(
        {
            "contracts": plan.contracts,
            "acceptance": plan.acceptance.model_dump() if plan.acceptance else None,
            "startup_command": plan.acceptance.startup_command() if plan.acceptance else None,
            "user_decisions": choices,
        },
        ensure_ascii=False,
        indent=2,
    )
    return file_map, notes, contract, acceptance


def rename_exclusive(source: Path, destination: Path) -> None:
    # Publish the whole directory in one operation and refuse any existing destination.
    if sys.platform == "win32":
        os.rename(source, destination)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    src, dst = os.fsencode(source), os.fsencode(destination)
    if sys.platform == "darwin":
        status = libc.renamex_np(ctypes.c_char_p(src), ctypes.c_char_p(dst), ctypes.c_uint(4))
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        status = libc.renameat2(
            -100, ctypes.c_char_p(src), -100, ctypes.c_char_p(dst), ctypes.c_uint(1)
        )
    else:
        raise OSError("此系统不支持安全发布新项目目录。")
    if status != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), destination)


def project_checks(notes: list[str], acceptance: ProjectAcceptance | None = None) -> str:
    issues = "\n".join(f"- {note}" for note in notes) or "本轮源码审查未留下待修复项。"
    report = (
        "# 构建检查\n\n已检查文件清单和路径，并校验 Python、JSON 和 TOML 文件的语法。"
        "源码的模型审查结果见下方。\n\n"
        "运行验收请按 README 安装依赖、启动服务并运行项目自带测试。\n\n"
        f"## 审查结果\n\n{issues}\n"
    )
    if acceptance:
        report += f"\n## 运行验收\n\n{acceptance.summary} 用时 {acceptance.elapsed:.1f} 秒。\n"
        if acceptance.output:
            report += (
                "\n" + "\n".join(f"    {line}" for line in acceptance.output.splitlines()) + "\n"
            )
    return report


def publish_into_directory(staged: Path, destination: Path, paths: list[str]) -> None:
    if sys.platform == "win32":
        publish_windows_directory(staged, destination, paths)
        return
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    created_dirs: list[tuple[int, str]] = []
    published: list[tuple[int, str, Path]] = []
    with ExitStack() as handles:
        root = os.open(destination, flags)
        handles.callback(os.close, root)
        directories = {(): root}
        targets = []
        try:
            for path in paths:
                parts = PurePosixPath(path).parts
                for index, part in enumerate(parts[:-1]):
                    prefix = parts[: index + 1]
                    if prefix in directories:
                        continue
                    parent = directories[prefix[:-1]]
                    try:
                        os.mkdir(part, dir_fd=parent)
                    except FileExistsError:
                        pass
                    else:
                        created_dirs.append((parent, part))
                    try:
                        folder = os.open(part, flags, dir_fd=parent)
                    except OSError as exc:
                        if exc.errno not in {errno.ENOTDIR, errno.ENOENT, errno.ELOOP}:
                            raise
                        raise ValueError(f"项目路径冲突：{'/'.join(prefix)}") from exc
                    handles.callback(os.close, folder)
                    directories[prefix] = folder
                folder = directories[parts[:-1]]
                name = parts[-1]
                try:
                    os.stat(name, dir_fd=folder, follow_symlinks=False)
                except FileNotFoundError:
                    targets.append((folder, name, path))
                else:
                    raise ValueError(f"项目文件已存在：{path}")

            # Link only staged, complete files. link() fails if another writer wins the name.
            for folder, name, path in targets:
                source = staged / path
                try:
                    os.link(source, name, dst_dir_fd=folder, follow_symlinks=False)
                except FileExistsError as exc:
                    raise ValueError(f"项目文件已存在：{path}") from exc
                published.append((folder, name, source))
        except BaseException:
            for folder, name, source in reversed(published):
                with suppress(FileNotFoundError):
                    current = os.stat(name, dir_fd=folder, follow_symlinks=False)
                    if os.path.samestat(source.stat(), current):
                        os.unlink(name, dir_fd=folder)
            for folder, name in reversed(created_dirs):
                with suppress(OSError):
                    os.rmdir(name, dir_fd=folder)
            raise


def write_project(
    build: ProjectBuild,
    run_id: str,
    files: dict[str, str],
    notes: list[str],
    report: str,
    *,
    contract: str | None = None,
    acceptance: ProjectAcceptance | None = None,
) -> ProjectBuildResult:
    destination = project_destination(build.directory)
    payload = {**files, DESIGN: report, CHECKS: project_checks(notes, acceptance)}
    if contract is not None:
        payload[CONTRACTS] = contract
    result = ProjectBuildResult(
        files={
            path: hashlib.sha256(body.encode("utf-8")).hexdigest() for path, body in payload.items()
        },
        review_notes=notes,
        acceptance=acceptance,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = destination if destination.is_dir() else destination.parent
    with tempfile.TemporaryDirectory(prefix=".se-agents-", dir=staging_parent) as folder:
        staged = Path(folder) / "project"
        staged.mkdir()
        for path, body in payload.items():
            check_path(path)
            target = staged / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body.encode("utf-8"))
        (staged / MARKER).write_text(
            canonical(
                {"run_id": run_id, "report_hash": build.report_hash, "result": result.model_dump()}
            ),
            encoding="utf-8",
        )
        if staging_parent == destination:
            publish_into_directory(staged, destination, [*payload, MARKER])
        else:
            try:
                rename_exclusive(staged, destination)
            except FileExistsError:
                publish_into_directory(staged, destination, [*payload, MARKER])
    return result


def recover_project(build: ProjectBuild, run_id: str) -> ProjectBuildResult | None:
    destination = project_destination(build.directory)
    marker = destination / MARKER
    try:
        marker_info = marker.lstat()
    except FileNotFoundError:
        return None
    try:
        if (
            destination.is_symlink()
            or is_reparse_point(destination)
            or not stat.S_ISREG(marker_info.st_mode)
            or marker_info.st_size > MARKER_LIMIT
        ):
            raise ValueError
        data = json.loads(marker.read_text(encoding="utf-8"))
        if data["run_id"] != run_id or data["report_hash"] != build.report_hash:
            raise ValueError
        result = ProjectBuildResult.model_validate(data["result"])
        if len(result.files) > 35 or not {DESIGN, CHECKS, "README.md"} <= result.files.keys():
            raise ValueError
        for path, checksum in result.files.items():
            check_path(path)
            item = destination / path
            info = item.lstat()
            if (
                item.resolve(strict=True) != item
                or not stat.S_ISREG(info.st_mode)
                or info.st_size > PROJECT_LIMIT
            ):
                raise ValueError
            if hashlib.sha256(item.read_bytes()).hexdigest() != checksum:
                raise ValueError
        return result
    except (OSError, ValueError, KeyError, TypeError):
        raise ValueError(
            f"目录中的 {MARKER} 无法确认属于本次构建，或之前生成的文件已被修改。"
        ) from None
