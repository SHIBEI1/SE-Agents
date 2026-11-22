from __future__ import annotations

import errno
import logging
import sqlite3
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from .chat import chat_answer, question_text, structured_reply
from .human import validate_answer
from .models import ModelError, ModelRunner, configured_model
from .planning import Engineering
from .project import project_destination
from .schemas import Blackboard, Frame, FrameNotes, ProjectBuild, RunConfig, digest
from .storage import BudgetLimit, Cancelled, Store
from .workflow import Workshop

if sys.platform == "win32":
    import msvcrt as file_locks
else:
    import fcntl as file_locks

logger = logging.getLogger(__name__)
PROJECT_TOKEN_LIMIT = 1_500_000


class Busy(RuntimeError):
    pass


class Service:
    def __init__(self, data_dir: Path):
        self.store = Store(data_dir / "workshop.sqlite")
        self.checkpoints = self.store.path.with_name("checkpoints.sqlite")
        self.lockfile = self.store.path.with_suffix(".worker.lock")
        try:
            with self.worker_lock():
                self.store.recover_abandoned()
                self.clean_deleted_checkpoints()
        except Busy:
            pass

    @contextmanager
    def worker_lock(self):
        with self.lockfile.open("a+b") as lock:
            try:
                if sys.platform == "win32":
                    lock.seek(0)
                    file_locks.locking(lock.fileno(), file_locks.LK_NBLCK, 1)
                else:
                    file_locks.flock(lock.fileno(), file_locks.LOCK_EX | file_locks.LOCK_NB)
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise
                raise Busy("已有任务在执行，请稍后重试。") from exc
            try:
                yield
            finally:
                if sys.platform == "win32":
                    lock.seek(0)
                    file_locks.locking(lock.fileno(), file_locks.LK_UNLCK, 1)

    def create(
        self,
        idea: str,
        config: RunConfig,
        frame: Frame | None = None,
        *,
        name: str = "",
        frame_hints: FrameNotes | None = None,
    ) -> str:
        idea = idea.strip()
        limit = 60_000 if config.workflow_mode == "planning" else 10_000
        if not 1 <= len(idea) <= limit:
            raise ValueError(f"输入须为 1 到 {limit:,} 个字符。")
        if config.mode != "mock":
            runner = ModelRunner(self.store, "preflight", config)
            runner.close()
        engineering = None
        if config.workflow_mode != "legacy":
            if config.workflow_mode == "full":
                config = config.model_copy(update={"internal_deliberation": True})
            engineering = Engineering(
                source_text=idea if config.workflow_mode == "planning" else "",
                max_parallel=config.max_parallel_tasks,
            )
        return self.store.create(
            Blackboard(idea=idea, frame=frame, frame_hints=frame_hints, engineering=engineering),
            config, name,
        )

    def create_build(
        self, source_id: str, revision: int, directory: str, *, verify_locally: bool = False
    ) -> str:
        source = self.store.run(source_id)
        if source["revision"] != revision:
            raise ValueError("报告版本已更新，请刷新后重新确认。")
        board = self.store.board(source_id, revision)
        if (
            source["status"] not in {"completed", "partial"}
            or not board.technical_report
            or board.project_build
        ):
            raise ValueError("请先生成并审阅技术报告。")
        directory = str(project_destination(directory))
        config = configured_model().model_copy(
            update={
                "max_calls": 80,
                "max_tokens": PROJECT_TOKEN_LIMIT,
                "max_seconds": 3600,
                "max_output_tokens": 8192,
            }
        )
        board.project_build = ProjectBuild(
            source_run_id=source_id,
            source_revision=revision,
            report_hash=digest(board.technical_report.model_dump()),
            source_name=source["name"],
            source_report_partial=source["status"] == "partial",
            directory=directory,
            verify_locally=verify_locally,
        )
        board.outcome = None
        board.stop_reason = None
        return self.store.create(board, config, f"{source['name'][:50]}－项目构建")

    def create_planning(self, source_id: str, revision: int) -> str:
        source = self.store.run(source_id)
        if source["revision"] != revision:
            raise ValueError("报告版本已更新，请刷新后重新选择。")
        board = self.store.board(source_id, revision)
        if not board.technical_report:
            raise ValueError("请先生成技术报告。")
        text = "\n\n".join(
            f"## {title}\n{body}" for title, body in board.technical_report.sections.items()
        )
        config = configured_model().model_copy(update={
            "workflow_mode": "planning", "max_calls": 100, "max_tokens": 800_000,
            "max_output_tokens": 8192,
        })
        return self.create(text, config, name=f"{source['name'][:45]}－任务编排")

    def delete_run(self, run_id: str) -> None:
        with self.worker_lock():
            self.store.recover_abandoned()
            self.store.delete(run_id)
            self.clean_deleted_checkpoints()

    def clean_deleted_checkpoints(self) -> None:
        pending = self.store.pending_checkpoint_deletions()
        if not pending:
            return
        try:
            with SqliteSaver.from_conn_string(str(self.checkpoints)) as checkpointer:
                for run_id in pending:
                    checkpointer.delete_thread(run_id)
                    self.store.checkpoint_deleted(run_id)
        except sqlite3.Error:
            logger.warning("任务已移除，检查点将在下次启动任务时继续清理。", exc_info=True)

    def config(self, run_id: str) -> RunConfig:
        return RunConfig.model_validate_json(self.store.run(run_id)["config"])

    @staticmethod
    def graph_config(run_id: str, config: RunConfig) -> dict:
        return {
            "configurable": {"thread_id": run_id},
            "max_concurrency": config.concurrency,
            "recursion_limit": 80,
        }

    @staticmethod
    def pending(snapshot) -> dict | None:
        return snapshot.interrupts[0].value if snapshot.interrupts else None

    def request(self, run_id: str) -> dict | None:
        if self.store.run(run_id)["status"] != "waiting_input":
            return None
        with SqliteSaver.from_conn_string(str(self.checkpoints)) as checkpointer:
            graph = Workshop(self.store, None).build(checkpointer)
            return self.pending(graph.get_state(self.graph_config(run_id, self.config(run_id))))

    def execute(
        self,
        run_id: str,
        answer: dict | None = None,
        *,
        demo: bool = False,
        chat: tuple[str, str] | None = None,
    ) -> dict:
        if answer is not None and chat is not None:
            raise ValueError("每次只提交一条回复。")
        if chat is not None and not 1 <= len(chat[1].strip()) <= 2400:
            raise ValueError("回复须为 1 到 2400 个字符。")
        with self.worker_lock():
            self.store.recover_abandoned()
            self.clean_deleted_checkpoints()
            row = self.store.run(run_id)
            if row["status"] not in {"queued", "waiting_input", "failed", "interrupted"}:
                raise ValueError("该运行已结束。")
            config = self.config(run_id)
            if (
                config.max_tokens == 650_000
                and self.store.board(run_id, row["revision"]).project_build
            ):
                # Older build tasks keep their recorded usage and can finish with the new limit.
                config = config.model_copy(update={"max_tokens": PROJECT_TOKEN_LIMIT})
            if demo and config.mode != "mock":
                raise ValueError("演示决策仅用于模拟模式。")
            runner = ModelRunner(
                self.store, run_id, config, resuming=row["status"] in {"failed", "interrupted"}
            )
            try:
                with SqliteSaver.from_conn_string(str(self.checkpoints)) as checkpointer:
                    graph = Workshop(self.store, runner).build(checkpointer)
                    graph_config = self.graph_config(run_id, config)
                    snapshot = graph.get_state(graph_config)
                    request = self.pending(snapshot)
                    if request:
                        self.store.add_chat_message(
                            run_id, request["id"], "question", question_text(request)
                        )
                        if answer is None and chat is None:
                            self.store.finish(run_id, "waiting_input")
                            return self.store.run(run_id)
                        if chat is not None and chat[0] != request["id"]:
                            raise ValueError("问题已更新，请刷新后回答当前问题。")
                        if answer is not None:
                            validate_answer(request, answer)
                    elif answer is not None or chat is not None:
                        raise ValueError("当前阶段没有等待中的人工请求。")
                    elif not snapshot.values:
                        event = {"run_id": run_id, "revision": 0}
                    elif snapshot.next:
                        event = None
                    else:
                        board = self.store.board(run_id)
                        if board.outcome is None:
                            raise RuntimeError("找不到可继续的图检查点。")
                        self.store.finish(run_id, board.outcome)
                        return self.store.run(run_id)

                    self.store.begin(run_id)
                    try:
                        if request:
                            if chat is not None:
                                request_id, message = chat
                                message = message.strip()
                                self.store.add_chat_message(run_id, request_id, "user", message)
                                replies = [
                                    row["content"]
                                    for row in self.store.chat_messages(run_id)
                                    if row["request_id"] == request_id and row["kind"] == "user"
                                ]
                                try:
                                    answer = chat_answer(request, message, replies, runner)
                                except ValueError as exc:
                                    self.store.add_chat_message(
                                        run_id, request_id, "assistant", str(exc)
                                    )
                                    self.store.finish(run_id, "waiting_input")
                                    return self.store.run(run_id)
                            else:
                                self.store.add_chat_message(
                                    run_id, request["id"], "user", structured_reply(answer)
                                )
                            event = Command(
                                resume={
                                    "request_id": answer["request_id"],
                                    "value": answer["value"],
                                    "reason": answer["reason"].strip(),
                                    "actor": "demo" if demo else "human",
                                    "answered_at": time.time(),
                                }
                            )
                        graph.invoke(event, graph_config, durability="sync")
                        board = self.store.board(run_id)
                        if board.outcome:
                            self.store.finish(run_id, board.outcome)
                        elif pending := self.pending(graph.get_state(graph_config)):
                            self.store.add_chat_message(
                                run_id, pending["id"], "question", question_text(pending)
                            )
                            self.store.finish(run_id, "waiting_input")
                        else:
                            raise RuntimeError("流程结束但未保存结果。")
                    except BudgetLimit as exc:
                        current = self.store.run(run_id)["revision"]
                        board = self.store.board(run_id, current)
                        board.outcome, board.stop_reason = "partial", str(exc)
                        try:
                            self.store.publish(run_id, current, "budget", board)
                        except Cancelled:
                            self.store.finish(run_id, "cancelled")
                        else:
                            self.store.finish(run_id, "partial")
                    except Cancelled:
                        self.store.finish(run_id, "cancelled")
                    except ModelError as exc:
                        self.store.finish(run_id, "failed", str(exc))
                    except Exception:
                        self.store.finish(run_id, "failed", "流程执行异常，已保留已提交版本。")
                        raise
            finally:
                runner.close()
            return self.store.run(run_id)
