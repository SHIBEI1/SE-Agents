from __future__ import annotations

import time

from .models import ModelError, ModelRunner
from .planning import (
    TaskAssessment,
    TaskEvent,
    TaskOutput,
    fingerprint,
    transition,
)
from .project import check_path
from .storage import BudgetLimit, Cancelled


def publish_event(service, run_id, base, board, task_id, before, action, actor, reason):
    engineering = board.engineering
    old = service.store.board(run_id, base).engineering
    engineering.events.append(TaskEvent(
        sequence=len(engineering.events) + 1, task_id=task_id, action=action,
        from_status=before, to_status=engineering.tasks[task_id].status,
        actor=actor, reason=reason, timestamp=time.time(), plan_hash=fingerprint(engineering.plan),
    ))
    for key, progress in engineering.tasks.items():
        if key != task_id and old.tasks[key].status != progress.status:
            engineering.events.append(TaskEvent(
                sequence=len(engineering.events) + 1, task_id=key, action="dependencies_updated",
                from_status=old.tasks[key].status, to_status=progress.status,
                actor="orchestrator", reason=f"前置任务 {task_id} 状态变化后重新检查依赖",
                timestamp=time.time(), plan_hash=fingerprint(engineering.plan),
            ))
    revision = service.store.publish(run_id, base, "task_update", board)
    service.store.activity(run_id, f"{task_id} {action}：{reason[:180]}")
    return revision


def update_task(service, run_id, revision, task_id, action, evidence):
    with service.worker_lock():
        row = service.store.run(run_id)
        if row["revision"] != revision:
            raise ValueError("看板版本已更新，请刷新后重试。")
        board = service.store.board(run_id)
        engineering = board.engineering
        if engineering is None or task_id not in engineering.tasks:
            raise ValueError("没有已确认的任务看板。")
        before = engineering.tasks[task_id].status
        transition(engineering, task_id, action, evidence)
        publish_event(service, run_id, revision, board, task_id, before, action, "human", evidence)
    return service.store.run(run_id)


def execute_task(service, run_id, revision, task_id):
    with service.worker_lock():
        row = service.store.run(run_id)
        if row["revision"] != revision:
            raise ValueError("看板版本已更新，请刷新后重试。")
        board = service.store.board(run_id)
        engineering = board.engineering
        if engineering is None or engineering.plan is None:
            raise ValueError("请先生成任务编排。")
        task = next((t for t in engineering.plan.tasks if t.id == task_id), None)
        if task is None or task.kind == "human":
            raise ValueError("此任务应由人工执行。")
        progress = engineering.tasks[task_id]
        if progress.attempts >= 3:
            raise ValueError("该任务已达到三次执行上限，请人工处理或调整计划。")
        before = progress.status
        transition(engineering, task_id, "start", "派发给执行 Agent")
        progress.attempts += 1
        current = publish_event(
            service, run_id, revision, board, task_id, before, "dispatch", "orchestrator",
            f"第 {progress.attempts} 次派发",
        )
        service.store.begin(run_id)
        runner = None
        try:
            runner = ModelRunner(service.store, run_id, service.config(run_id), resuming=True)
            previous = {
                key: p.output.model_dump() for key, p in engineering.tasks.items()
                if p.output is not None and p.status == "done"
            }

            def check(output):
                allowed = set(task.deliverables)
                paths = [item.path for item in output.files]
                if len(set(paths)) != len(paths):
                    raise ValueError("交付文件路径重复。")
                for path in paths:
                    check_path(path)
                    if path not in allowed:
                        raise ValueError(f"文件 {path} 不在本任务交付清单中。")
                if task.kind in {"code", "test", "doc"} and set(paths) != allowed:
                    raise ValueError("尚未提供任务卡指定的全部文件。")

            service.store.set_stage(run_id, "task_execute")
            output = runner.run(
                "task_execute", "tech",
                {"任务": task.model_dump(), "设计规格": engineering.design.model_dump(),
                 "已验收前置产物": previous, "上轮产物": (
                     progress.output.model_dump() if progress.output else None),
                 "修复意见": progress.review_findings, "尝试次数": progress.attempts},
                TaskOutput, check,
            )
            progress.output = output
            service.store.set_stage(run_id, "task_review")
            review = runner.run(
                "task_review", "devil",
                {"任务": task.model_dump(), "设计规格": engineering.design.model_dump(),
                 "交付物": output.model_dump(), "前置产物": previous},
                TaskAssessment,
            )
            progress.evidence = "\n".join(output.evidence)
            progress.review_findings = review.findings
            progress.status = "review"
            progress.reason = "已生成交付物并完成独立静态审查，等待实际验收。"
            publish_event(service, run_id, current, board, task_id, "running",
                          "submit", "executor", progress.reason)
            service.store.finish(run_id, "completed")
        except Cancelled:
            service.store.finish(run_id, "cancelled")
        except (ModelError, BudgetLimit, ValueError, OSError) as exc:
            progress.status = "failed"
            progress.reason = str(exc)
            publish_event(service, run_id, current, board, task_id, "running",
                          "failed", "executor", str(exc))
            service.store.finish(run_id, "completed", str(exc))
        except Exception:
            progress.status = "failed"
            progress.reason = "执行服务异常，请检查日志后重试或人工处理。"
            publish_event(service, run_id, current, board, task_id, "running",
                          "failed", "executor", progress.reason)
            service.store.finish(run_id, "completed", progress.reason)
            raise
        finally:
            if runner:
                runner.close()
    return service.store.run(run_id)
