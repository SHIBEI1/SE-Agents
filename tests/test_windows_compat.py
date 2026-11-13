import errno
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from se_agents import project, service, windows_io
from se_agents.schemas import ProjectBuild


def test_worker_lock_blocks_another_process_and_releases(tmp_path):
    workspace = service.Service(tmp_path)
    script = """
from pathlib import Path
import sys
from se_agents.service import Busy, Service
workspace = Service(Path(sys.argv[1]))
try:
    with workspace.worker_lock():
        pass
except Busy:
    sys.exit(7)
"""
    with workspace.worker_lock():
        busy = subprocess.run([sys.executable, "-c", script, str(tmp_path)], timeout=15)
    free = subprocess.run([sys.executable, "-c", script, str(tmp_path)], timeout=15)
    assert busy.returncode == 7
    assert free.returncode == 0


def test_windows_lock_releases_after_exception(tmp_path, monkeypatch):
    workspace = service.Service(tmp_path)
    held = set()
    positions = []

    def locking(descriptor, mode, length):
        positions.append(os.lseek(descriptor, 0, os.SEEK_CUR))
        assert length == 1
        if mode == 1:
            if held:
                raise PermissionError(errno.EACCES, "locked")
            held.add("lock")
        else:
            held.remove("lock")

    monkeypatch.setattr(service, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(
        service, "file_locks", SimpleNamespace(locking=locking, LK_NBLCK=1, LK_UNLCK=2)
    )
    with pytest.raises(RuntimeError, match="工作失败"), workspace.worker_lock():
        with pytest.raises(service.Busy), workspace.worker_lock():
            pass
        raise RuntimeError("工作失败")
    with workspace.worker_lock():
        assert held
    assert not held
    assert positions == [0] * 5


@pytest.mark.parametrize("reparse", [False, True])
def test_windows_directory_handle_closes_on_rejection(tmp_path, monkeypatch, reparse):
    closed = []

    def create(path, access, share, security, disposition, flags, template):
        assert path == str(tmp_path)
        assert access == 0x80 and share == 1 and disposition == 3
        assert flags == 0x02200000
        return 123

    def close(handle):
        closed.append(handle)
        return True

    monkeypatch.setattr(
        windows_io.ctypes,
        "WinDLL",
        lambda *args, **kwargs: SimpleNamespace(CreateFileW=create, CloseHandle=close),
        raising=False,
    )
    monkeypatch.setattr(windows_io, "is_reparse_point", lambda path: reparse)
    if reparse:
        with pytest.raises(ValueError, match="项目路径冲突"), windows_io.directory_handle(tmp_path):
            pytest.fail("目录连接不应进入写入阶段。")
    else:
        with pytest.raises(RuntimeError, match="工作失败"), windows_io.directory_handle(tmp_path):
            raise RuntimeError("工作失败")
    assert closed == [123]


@pytest.fixture
def windows_backend(monkeypatch):
    # Exercise the Windows publication algorithm on any host. A Windows test run
    # uses the actual native directory handles and exclusive rename implementation.
    if sys.platform != "win32":

        @contextmanager
        def directory_handle(path):
            if not path.is_dir() or path.is_symlink() or windows_io.is_reparse_point(path):
                raise ValueError(f"项目路径冲突：{path}")
            yield

        original_rename = os.rename

        def rename(source, destination):
            if os.path.lexists(destination):
                raise FileExistsError(errno.EEXIST, "exists", destination)
            original_rename(source, destination)

        simulated_os = SimpleNamespace(**{**vars(os), "rename": rename})
        monkeypatch.setattr(windows_io, "directory_handle", directory_handle)
        monkeypatch.setattr(windows_io, "os", simulated_os)
        monkeypatch.setattr(project, "os", simulated_os)
        monkeypatch.setattr(project, "sys", SimpleNamespace(platform="win32"))


@pytest.mark.parametrize("existing", [False, True])
def test_windows_build_publication(tmp_path, windows_backend, existing):
    target = tmp_path / "中文目录" / "项目"
    if existing:
        (target / "src").mkdir(parents=True)
        (target / "src" / "notes.txt").write_text("原始文件", encoding="utf-8")
    build = ProjectBuild(
        source_run_id="source",
        source_revision=0,
        report_hash="hash",
        source_name="设计",
        source_report_partial=False,
        directory=str(target),
    )
    result = project.write_project(
        build, "run", {"src/app.py": "print('你好')\n", "README.md": "# 项目\n"}, [], "# 方案"
    )
    assert project.recover_project(build, "run") == result
    assert (target / "src" / "app.py").read_text(encoding="utf-8") == "print('你好')\n"
    if existing:
        assert (target / "src" / "notes.txt").read_text(encoding="utf-8") == "原始文件"


@pytest.mark.parametrize("failure", ["conflict", "write_error", "reparse"])
def test_windows_publication_rolls_back(tmp_path, windows_backend, monkeypatch, failure):
    staged = tmp_path / "staged"
    destination = tmp_path / "destination"
    (staged / "src").mkdir(parents=True)
    destination.mkdir()
    (staged / "src" / "app.py").write_text("pass", encoding="utf-8")
    (staged / "README.md").write_text("# 项目", encoding="utf-8")
    if failure == "conflict":
        (destination / "README.md").write_text("原说明", encoding="utf-8")
    elif failure == "reparse":
        (destination / "src").mkdir()
        original = windows_io.is_reparse_point
        monkeypatch.setattr(
            windows_io,
            "is_reparse_point",
            lambda path: path == destination / "src" or original(path),
        )
    else:
        original_rename = windows_io.os.rename

        def fail_second_move(source, target):
            if Path(source) == staged / "README.md":
                raise PermissionError("写入失败")
            original_rename(source, target)

        monkeypatch.setattr(windows_io.os, "rename", fail_second_move)

    with pytest.raises((ValueError, PermissionError)):
        project.publish_into_directory(staged, destination, ["src/app.py", "README.md"])
    assert (staged / "src" / "app.py").read_text(encoding="utf-8") == "pass"
    assert not (destination / "src" / "app.py").exists()
    if failure == "conflict":
        assert (destination / "README.md").read_text(encoding="utf-8") == "原说明"
    elif failure == "write_error":
        assert list(destination.iterdir()) == []
