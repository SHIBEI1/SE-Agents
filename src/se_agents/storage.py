from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from itertools import count
from pathlib import Path
from uuid import uuid4

from .presentation import STAGES
from .schemas import Blackboard, RunConfig, canonical


class BudgetLimit(RuntimeError):
    pass


class Cancelled(RuntimeError):
    pass


def suggested_name(idea: str) -> str:
    line = next((line.strip(" #>*-\t") for line in idea.splitlines() if line.strip(" #>*-\t")), "")
    title = " ".join(line.split()).strip("。！？.!?")
    return title[:28] + ("…" if len(title) > 28 else "") if title else "未命名任务"


class Store:
    def __init__(self, path: Path):
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, created REAL NOT NULL, config TEXT NOT NULL,
                    status TEXT NOT NULL, stage TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 0,
                    cancel INTEGER NOT NULL DEFAULT 0, elapsed REAL NOT NULL DEFAULT 0,
                    started REAL, error TEXT, name TEXT
                );
                CREATE TABLE IF NOT EXISTS revisions (
                    run_id TEXT REFERENCES runs(id), revision INTEGER, base INTEGER,
                    stage TEXT NOT NULL, board TEXT NOT NULL,
                    PRIMARY KEY (run_id, revision), UNIQUE (run_id, base, stage)
                );
                CREATE TABLE IF NOT EXISTS calls (
                    id INTEGER PRIMARY KEY, run_id TEXT REFERENCES runs(id),
                    task TEXT NOT NULL, role TEXT NOT NULL, input_hash TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL, reserved INTEGER NOT NULL,
                    tokens INTEGER, usage TEXT, status TEXT NOT NULL, elapsed REAL
                );
                CREATE INDEX IF NOT EXISTS calls_run ON calls(run_id);
                CREATE TABLE IF NOT EXISTS task_results (
                    run_id TEXT REFERENCES runs(id), task TEXT, payload TEXT NOT NULL,
                    PRIMARY KEY (run_id, task)
                );
                CREATE TABLE IF NOT EXISTS deleted_checkpoints (run_id TEXT PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS activities (
                    id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id),
                    created REAL NOT NULL, detail TEXT NOT NULL, elapsed REAL
                );
                CREATE INDEX IF NOT EXISTS activities_run ON activities(run_id, id);
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id),
                    kind TEXT NOT NULL, request_id TEXT NOT NULL, content TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS chat_messages_run ON chat_messages(run_id, id);
                CREATE UNIQUE INDEX IF NOT EXISTS chat_question_once
                ON chat_messages(run_id, request_id) WHERE kind='question';
            """)
        with self.connect(write=True) as db:
            columns = {row["name"] for row in db.execute("PRAGMA table_info(runs)")}
            if "name" not in columns:
                db.execute("ALTER TABLE runs ADD COLUMN name TEXT")
            call_columns = {row["name"] for row in db.execute("PRAGMA table_info(calls)")}
            if "diagnostic" not in call_columns:
                db.execute("ALTER TABLE calls ADD COLUMN diagnostic TEXT")
            unnamed = db.execute(
                """SELECT runs.id, revisions.board FROM runs JOIN revisions
                   ON revisions.run_id=runs.id AND revisions.revision=0
                   WHERE runs.name IS NULL ORDER BY runs.created, runs.id"""
            ).fetchall()
            for row in unnamed:
                idea = Blackboard.model_validate_json(row["board"]).idea
                name = self.unique_name(db, suggested_name(idea))
                db.execute("UPDATE runs SET name=? WHERE id=?", (name, row["id"]))
            db.execute("CREATE UNIQUE INDEX IF NOT EXISTS runs_name ON runs(name)")

    @contextmanager
    def connect(self, *, write: bool = False):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def unique_name(db: sqlite3.Connection, name: str) -> str:
        candidate = name
        for number in count(2):
            if not db.execute("SELECT 1 FROM runs WHERE name=?", (candidate,)).fetchone():
                return candidate
            suffix = f"（{number}）"
            candidate = name[: 60 - len(suffix)] + suffix

    def create(self, board: Blackboard, config: RunConfig, name: str = "") -> str:
        run_id = uuid4().hex[:12]
        title = name.strip() or suggested_name(board.idea)
        if len(title) > 60:
            raise ValueError("任务名称最多 60 个字符。")
        with self.connect(write=True) as db:
            db.execute(
                """INSERT INTO runs(id, created, config, status, stage, name)
                   VALUES(?, ?, ?, 'queued', 'frame', ?)""",
                (run_id, time.time(), config.model_dump_json(), self.unique_name(db, title)),
            )
            db.execute(
                "INSERT INTO revisions VALUES(?, 0, -1, 'input', ?)",
                (run_id, board.model_dump_json()),
            )
        return run_id

    def run(self, run_id: str) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise ValueError("找不到该任务。")
        return dict(row)

    def list_runs(self) -> list[dict]:
        with self.connect() as db:
            return [
                dict(row) for row in db.execute("SELECT * FROM runs ORDER BY created DESC, id DESC")
            ]

    def rename(self, run_id: str, name: str) -> None:
        name = name.strip()
        if not 1 <= len(name) <= 60:
            raise ValueError("任务名称须为 1 到 60 个字符。")
        with self.connect(write=True) as db:
            if db.execute("SELECT 1 FROM runs WHERE name=? AND id<>?", (name, run_id)).fetchone():
                raise ValueError("已有同名任务，请换一个名称。")
            if not db.execute("UPDATE runs SET name=? WHERE id=?", (name, run_id)).rowcount:
                raise ValueError("找不到该任务。")

    def delete(self, run_id: str) -> None:
        with self.connect(write=True) as db:
            row = db.execute("SELECT status FROM runs WHERE id=?", (run_id,)).fetchone()
            if row is None:
                raise ValueError("找不到该任务。")
            if row["status"] == "running":
                raise ValueError("请先停止任务，再删除。")
            db.execute("DELETE FROM chat_messages WHERE run_id=?", (run_id,))
            db.execute("DELETE FROM activities WHERE run_id=?", (run_id,))
            db.execute("DELETE FROM task_results WHERE run_id=?", (run_id,))
            db.execute("DELETE FROM calls WHERE run_id=?", (run_id,))
            db.execute("DELETE FROM revisions WHERE run_id=?", (run_id,))
            db.execute("DELETE FROM runs WHERE id=?", (run_id,))
            db.execute("INSERT OR IGNORE INTO deleted_checkpoints VALUES(?)", (run_id,))

    def add_chat_message(self, run_id: str, request_id: str, kind: str, content: str) -> None:
        with self.connect(write=True) as db:
            db.execute(
                """INSERT INTO chat_messages(run_id, request_id, kind, content)
                   VALUES(?, ?, ?, ?) ON CONFLICT DO NOTHING""",
                (run_id, request_id, kind, content),
            )

    def chat_messages(self, run_id: str) -> list[dict]:
        with self.connect() as db:
            return [
                dict(row)
                for row in db.execute(
                    """SELECT request_id, kind, content FROM chat_messages
                   WHERE run_id=? ORDER BY id""",
                    (run_id,),
                )
            ]

    def pending_checkpoint_deletions(self) -> list[str]:
        with self.connect() as db:
            return [row["run_id"] for row in db.execute("SELECT run_id FROM deleted_checkpoints")]

    def checkpoint_deleted(self, run_id: str) -> None:
        with self.connect(write=True) as db:
            db.execute("DELETE FROM deleted_checkpoints WHERE run_id=?", (run_id,))

    def board(self, run_id: str, revision: int | None = None) -> Blackboard:
        with self.connect() as db:
            row = db.execute(
                """SELECT board FROM revisions WHERE run_id=?
                   AND revision=COALESCE(?, (SELECT revision FROM runs WHERE id=?))""",
                (run_id, revision, run_id),
            ).fetchone()
        if row is None:
            raise ValueError("找不到该黑板版本。")
        return Blackboard.model_validate_json(row["board"])

    def publish(self, run_id: str, base: int, stage: str, board: Blackboard) -> int:
        with self.connect(write=True) as db:
            old = db.execute(
                "SELECT revision FROM revisions WHERE run_id=? AND base=? AND stage=?",
                (run_id, base, stage),
            ).fetchone()
            if old:
                return old["revision"]
            row = db.execute("SELECT revision, cancel FROM runs WHERE id=?", (run_id,)).fetchone()
            if row["cancel"]:
                raise Cancelled()
            if row["revision"] != base:
                raise RuntimeError("黑板版本冲突。")
            revision = base + 1
            db.execute(
                "INSERT INTO revisions VALUES(?, ?, ?, ?, ?)",
                (run_id, revision, base, stage, board.model_dump_json()),
            )
            db.execute("UPDATE runs SET revision=? WHERE id=?", (revision, run_id))
            return revision

    def set_stage(self, run_id: str, stage: str) -> None:
        with self.connect(write=True) as db:
            changed = db.execute(
                "UPDATE runs SET stage=? WHERE id=? AND stage<>?", (stage, run_id, stage)
            ).rowcount
            if changed:
                db.execute(
                    "INSERT INTO activities(run_id, created, detail) VALUES(?, ?, ?)",
                    (run_id, time.time(), STAGES.get(stage, stage)),
                )

    def activity(self, run_id: str, detail: str, elapsed: float | None = None) -> None:
        with self.connect(write=True) as db:
            db.execute(
                "INSERT INTO activities(run_id, created, detail, elapsed) VALUES(?, ?, ?, ?)",
                (run_id, time.time(), detail, elapsed),
            )

    def activities(self, run_id: str) -> list[dict]:
        with self.connect() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT created, detail, elapsed FROM activities WHERE run_id=? ORDER BY id",
                    (run_id,),
                )
            ]

    def request_cancel(self, run_id: str) -> None:
        with self.connect(write=True) as db:
            db.execute(
                """UPDATE runs SET cancel=1,
                   status=CASE WHEN status='running' THEN status ELSE 'cancelled' END
                   WHERE id=? AND status IN ('queued', 'running', 'waiting_input', 'interrupted')""",
                (run_id,),
            )

    def begin(self, run_id: str) -> None:
        with self.connect(write=True) as db:
            row = db.execute("SELECT cancel FROM runs WHERE id=?", (run_id,)).fetchone()
            if row["cancel"]:
                raise Cancelled()
            db.execute(
                "UPDATE runs SET status='running', started=?, error=NULL WHERE id=?",
                (time.time(), run_id),
            )

    def recover_abandoned(self) -> None:
        # The caller holds the exclusive worker lock.
        with self.connect(write=True) as db:
            db.execute("UPDATE calls SET status='unknown' WHERE status='pending'")
            db.execute(
                """UPDATE runs SET status=CASE WHEN cancel=1 THEN 'cancelled' ELSE 'interrupted' END,
                   elapsed=elapsed+MAX(0, ?-COALESCE(started, ?)),
                   started=NULL WHERE status='running'""",
                (time.time(), time.time()),
            )

    def finish(self, run_id: str, status: str, error: str | None = None) -> None:
        with self.connect(write=True) as db:
            db.execute(
                """UPDATE runs SET status=CASE WHEN cancel=1 THEN 'cancelled' ELSE ? END,
                   elapsed=elapsed+MAX(0, ?-COALESCE(started, ?)), started=NULL, error=? WHERE id=?""",
                (status, time.time(), time.time(), error, run_id),
            )

    def check_cancel(self, run_id: str) -> None:
        if self.run(run_id)["cancel"]:
            raise Cancelled()

    def reserve(
        self,
        run_id: str,
        task: str,
        role: str,
        input_hash: str,
        prompt_hash: str,
        estimate: int,
        config: RunConfig,
    ) -> tuple[int, float]:
        with self.connect(write=True) as db:
            row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if row["cancel"]:
                raise Cancelled()
            remaining = config.max_seconds - row["elapsed"] - (time.time() - row["started"])
            if remaining <= 0:
                raise BudgetLimit("达到运行时间上限。")
            usage = db.execute(
                """SELECT COUNT(*) n, COALESCE(SUM(COALESCE(tokens, reserved)), 0) tokens
                   FROM calls WHERE run_id=?""",
                (run_id,),
            ).fetchone()
            if usage["n"] >= config.max_calls:
                raise BudgetLimit("达到模型请求次数上限。")
            if usage["tokens"] + estimate > config.max_tokens:
                raise BudgetLimit("剩余 Token 预算不足以启动下一次请求。")
            cursor = db.execute(
                """INSERT INTO calls(run_id, task, role, input_hash, prompt_hash, reserved, status)
                   VALUES(?, ?, ?, ?, ?, ?, 'pending')""",
                (run_id, task, role, input_hash, prompt_hash, estimate),
            )
            return int(cursor.lastrowid), min(120, remaining)

    def end_call(
        self,
        call_id: int,
        status: str,
        usage: dict | None,
        seconds: float,
        *,
        diagnostic: str | None = None,
    ) -> None:
        tokens = usage.get("total_tokens") if usage is not None else None
        with self.connect(write=True) as db:
            db.execute(
                "UPDATE calls SET status=?, tokens=?, usage=?, elapsed=?, diagnostic=? WHERE id=?",
                (
                    status,
                    tokens,
                    canonical(usage) if usage is not None else None,
                    seconds,
                    diagnostic,
                    call_id,
                ),
            )

    def cached(self, run_id: str, task: str) -> str | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT payload FROM task_results WHERE run_id=? AND task=?",
                (run_id, task),
            ).fetchone()
        return row["payload"] if row else None

    def cache(self, run_id: str, task: str, payload: str) -> None:
        with self.connect(write=True) as db:
            db.execute(
                "INSERT OR IGNORE INTO task_results VALUES(?, ?, ?)", (run_id, task, payload)
            )

    def cached_phase(self, run_id: str, phase: str) -> list[tuple[str, str]]:
        with self.connect() as db:
            return [
                (row["task"], row["payload"])
                for row in db.execute(
                    "SELECT task, payload FROM task_results WHERE run_id=? AND task GLOB ? ORDER BY rowid DESC",
                    (run_id, f"{phase}:*"),
                )
            ]

    def call_log(self, run_id: str) -> list[dict]:
        with self.connect() as db:
            return [
                dict(row)
                for row in db.execute(
                    """SELECT task, role, input_hash, prompt_hash, tokens, status, elapsed, diagnostic
                   FROM calls WHERE run_id=? ORDER BY id""",
                    (run_id,),
                )
            ]

    def revisions(self, run_id: str) -> list[dict]:
        with self.connect() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT revision, stage FROM revisions WHERE run_id=? ORDER BY revision",
                    (run_id,),
                )
            ]
