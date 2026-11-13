from __future__ import annotations

import ctypes
import os
import stat
from contextlib import ExitStack, contextmanager, suppress
from ctypes import wintypes
from pathlib import Path, PurePosixPath


def is_reparse_point(path: Path) -> bool:
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except FileNotFoundError:
        return False
    return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


@contextmanager
def directory_handle(path: Path):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create.restype = wintypes.HANDLE
    close = kernel.CloseHandle
    close.argtypes = [wintypes.HANDLE]
    close.restype = wintypes.BOOL

    # Pin the directory while publishing. Excluding WRITE and DELETE sharing prevents
    # another writer from replacing it with a junction or renaming it under us.
    file_read_attributes = 0x80
    file_share_read = 0x1
    open_existing = 3
    backup_semantics = 0x02000000
    open_reparse_point = 0x00200000
    handle = create(
        str(path),
        file_read_attributes,
        file_share_read,
        None,
        open_existing,
        backup_semantics | open_reparse_point,
        None,
    )
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not path.is_dir() or path.is_symlink() or is_reparse_point(path):
            raise ValueError(f"项目路径冲突：{path}")
        yield
    finally:
        close(handle)


def publish_windows_directory(staged: Path, destination: Path, paths: list[str]) -> None:
    created: list[Path] = []
    published: list[tuple[Path, Path, os.stat_result]] = []
    with ExitStack() as handles:
        handles.enter_context(directory_handle(destination))
        opened = {destination}
        targets = []
        try:
            for path in paths:
                target = destination.joinpath(*PurePosixPath(path).parts)
                parents = list(target.relative_to(destination).parents)[:-1]
                for relative in reversed(parents):
                    folder = destination / relative
                    if folder in opened:
                        continue
                    try:
                        folder.mkdir()
                    except FileExistsError:
                        pass
                    else:
                        created.append(folder)
                    handles.enter_context(directory_handle(folder))
                    opened.add(folder)
                if os.path.lexists(target):
                    raise ValueError(f"项目文件已存在：{path}")
                targets.append((staged / path, target))

            # Windows rename fails on existing names and works without hard-link support.
            for source, target in targets:
                identity = source.stat()
                try:
                    os.rename(source, target)
                except FileExistsError as exc:
                    raise ValueError(f"项目文件已存在：{target.relative_to(destination)}") from exc
                published.append((source, target, identity))
        except BaseException:
            for source, target, identity in reversed(published):
                with suppress(FileNotFoundError):
                    if os.path.samestat(identity, target.lstat()):
                        os.rename(target, source)
            # Windows requires directory handles to close before directories can be removed.
            handles.close()
            for folder in reversed(created):
                with suppress(OSError):
                    folder.rmdir()
            raise
