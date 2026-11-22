from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from .models import configured_model
from .reporting import export_json, export_markdown
from .schemas import Frame, RunConfig
from .service import Busy, Service


def read_idea(path: Path) -> str:
    if path.suffix.lower() not in {".txt", ".md"}:
        raise ValueError("请选择 UTF-8 .txt 或 .md 文件。")
    with path.open("rb") as stream:
        data = stream.read(1_048_577)
    if len(data) > 1_048_576:
        raise ValueError("文件超过 1 MiB。")
    return data.decode("utf-8-sig")


def describe(service: Service, run_id: str) -> dict:
    row = service.store.run(run_id)
    board = service.store.board(run_id, row["revision"])
    return {
        "id": run_id,
        "status": row["status"],
        "stage": row["stage"],
        "revision": row["revision"],
        "reason": board.stop_reason or row["error"],
        "request": service.request(run_id),
    }


def demo(service: Service, blocked: bool) -> str:
    run_id = service.create(
        "开发一个帮助大学生查找和预约校园自习空间的软件。",
        RunConfig(),
        Frame(
            audience="在校大学生",
            goal="减少查找开放自习空间的时间",
            constraints="六周课程项目，先用公开或授权数据验证原型。",
        ),
    )
    service.execute(run_id)
    while request := service.request(run_id):
        if request["kind"] == "resolve" and request["payload"]["issue"]["kind"] == "value":
            value = {"choice": "优先快速验证查询价值，接受人工更新的维护成本。"}
        elif request["kind"] == "resolve":
            value = {"verdict": "defer"}
        elif request["kind"] == "select":
            value = {"option_id": "OPT-002" if blocked else "OPT-001"}
        else:
            raise RuntimeError("模拟样例遇到未知人工步骤。")
        service.execute(
            run_id,
            {
                "request_id": request["id"],
                "value": value,
                "reason": "内置模拟场景的演示决策。",
            },
            demo=True,
        )
    return run_id


def main() -> None:
    load_dotenv(Path.cwd() / ".env", encoding="utf-8-sig")
    parser = argparse.ArgumentParser(prog="se-agents", description="黑板驱动的多专家软件创意论证")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start", help="读取软件想法并使用真实模型生成技术报告")
    start.add_argument("file", type=Path)
    start.add_argument("--workflow", choices=["full", "planning", "legacy"], default="full")
    start.add_argument("--max-parallel", type=int, choices=range(1, 5), default=2)
    for name in ("show", "resume", "cancel"):
        commands.add_parser(name).add_argument("run_id")
    answer = commands.add_parser("answer")
    answer.add_argument("run_id")
    answer.add_argument("file", type=Path, help="包含 request_id、value、reason 的 JSON")
    reply = commands.add_parser("reply", help="用自然语言回答当前问题")
    reply.add_argument("run_id")
    reply.add_argument("text")
    export = commands.add_parser("export")
    export.add_argument("run_id")
    export.add_argument("--format", choices=["markdown", "json"], default="markdown")
    args = parser.parse_args()
    service = Service(args.data_dir)
    try:
        if args.command == "start":
            config = configured_model().model_copy(update={
                "workflow_mode": args.workflow, "max_parallel_tasks": args.max_parallel,
                "chat_mode": args.workflow == "full", "max_calls": 100, "max_tokens": 800_000,
                "max_output_tokens": 8192,
            })
            run_id = service.create(read_idea(args.file), config)
            service.execute(run_id)
        else:
            run_id = args.run_id
            if (
                args.command in {"resume", "answer", "reply"}
                and service.config(run_id).mode == "mock"
            ):
                raise ValueError("历史模拟任务只供查看。请使用真实模型新建技术报告任务。")
            if args.command == "resume":
                service.execute(run_id)
            elif args.command == "answer":
                service.execute(run_id, json.loads(args.file.read_text(encoding="utf-8")))
            elif args.command == "reply":
                pending = service.request(run_id)
                if pending is None:
                    raise ValueError("当前任务没有等待回答的问题。")
                service.execute(run_id, chat=(pending["id"], args.text))
            elif args.command == "cancel":
                service.store.request_cancel(run_id)
            elif args.command == "export":
                run = service.store.run(run_id)
                board = service.store.board(run_id, run["revision"])
                render = export_json if args.format == "json" else export_markdown
                print(render(board, service.config(run_id), run, service.store.call_log(run_id)))
                return
        print(json.dumps(describe(service, run_id), ensure_ascii=False, indent=2))
    except (ValueError, OSError, Busy) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
