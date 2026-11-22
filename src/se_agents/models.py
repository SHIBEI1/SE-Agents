from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import TypeVar
from urllib.parse import urlsplit

from openai import APIConnectionError, APIStatusError, OpenAI
from pydantic import BaseModel, ValidationError

from . import prompts
from .mock import mock_reply
from .presentation import STAGES
from .schemas import ROLE_NAMES, RunConfig, canonical, digest
from .storage import Store

T = TypeVar("T", bound=BaseModel)


class ModelError(RuntimeError):
    pass


class ModelOutputTruncated(ModelError):
    pass


def configured_model() -> RunConfig:
    endpoint = os.environ.get("MODEL_BASE_URL", "").strip().rstrip("/")
    key = os.environ.get("MODEL_API_KEY", "").strip()
    name = os.environ.get("MODEL_NAME", "").strip()
    if endpoint or key or name:
        parsed = urlsplit(endpoint)
        if (
            not key
            or not name
            or parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("请完整配置通用模型的接口地址、模型名和密钥。")
        return RunConfig(mode="api", base_url=endpoint, model=name, technical_mode=True)
    if not os.environ.get("DEEPSEEK_API_KEY", "").strip():
        raise ValueError("请先在项目的配置文件中填写深度求索接口密钥。")
    return RunConfig(
        mode="deepseek",
        model=os.environ.get("DEEPSEEK_MODEL", "").strip() or "deepseek-flash",
        technical_mode=True,
    )


def validation_detail(exc: ValueError) -> str:
    if not isinstance(exc, ValidationError):
        return str(exc)
    descriptions = {
        "missing": "缺少必填字段",
        "literal_error": "取值不在允许范围内",
        "extra_forbidden": "包含未定义字段",
        "json_invalid": "不是有效的结构化数据",
        "too_long": "条目数量超限",
        "string_too_short": "文本为空",
        "string_too_long": "文本长度超限",
        "string_type": "应为文本",
    }
    errors = exc.errors(include_input=False, include_url=False)
    return "；".join(
        f"{'.'.join(map(str, error['loc'])) or '整体输出'}："
        f"{descriptions.get(error['type'], '格式或类型不符合要求')}"
        + (
            f"（最多 {error['ctx']['max_length']} 字符）"
            if error["type"] == "string_too_long"
            else f"（最多 {error['ctx']['max_length']} 条）"
            if error["type"] == "too_long" and "max_length" in error.get("ctx", {})
            else ""
        )
        for error in errors[:3]
    )


class ModelRunner:
    def __init__(self, store: Store, run_id: str, config: RunConfig, *, resuming: bool = False):
        self.store, self.run_id, self.config = store, run_id, config
        self.resuming = resuming
        key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        base_url = "https://api.deepseek.com"
        if config.mode == "api":
            current = configured_model()
            if current.mode != "api" or current.base_url != config.base_url:
                raise ValueError("该任务使用的接口地址与当前配置不同，请恢复原配置或新建任务。")
            base_url = current.base_url
            key = os.environ.get("MODEL_API_KEY", "").strip()
        if config.mode == "deepseek" and not key:
            raise ValueError("请先在项目配置文件中填写深度求索的接口密钥。")
        self.client = (
            OpenAI(api_key=key, base_url=base_url, max_retries=0) if config.mode != "mock" else None
        )

    def close(self) -> None:
        if self.client:
            self.client.close()

    def run(
        self,
        phase: str,
        role: str,
        payload: dict,
        schema: type[T],
        validate: Callable[[T], None] | None = None,
    ) -> T:
        task_label = f"{ROLE_NAMES[role]}在{STAGES[phase]}阶段"
        input_hash = digest(payload)
        file_stage = phase == "project_file"
        base = prompts.COMMON.rstrip("，") + "。" if file_stage else prompts.BASE
        system = f"{base}\n{prompts.ROLES[role]}\n{prompts.TASKS[phase]}"
        system += "\nJSON Schema: " + canonical(schema.model_json_schema())
        prompt_hash = digest(system)
        task_prefix = f"{phase}:{role}:{input_hash[:20]}:"
        task = task_prefix + prompt_hash[:12]
        report_stage = phase.startswith("report_")
        repairable_stage = report_stage or file_stage or phase in {
            "project_plan", "project_probe", "requirements_spec", "engineering_design",
            "task_decompose", "plan_review", "task_execute", "task_review", "internal_resolution",
            "critique", "critique_assumption", "score",
        }
        cached = self.store.cached(self.run_id, task)
        if cached is not None:
            return schema.model_validate_json(cached)
        if file_stage or (self.resuming and (report_stage or phase == "critique")):
            for saved_task, saved in self.store.cached_phase(self.run_id, phase):
                if saved_task.startswith(task_prefix):
                    try:
                        result = schema.model_validate_json(saved)
                        if validate:
                            validate(result)
                    except ValueError:
                        continue
                    return result

        previous = [row for row in self.store.call_log(self.run_id) if row["task"] == task]
        if (
            file_stage
            and len(previous) >= 2
            and all(
                row["diagnostic"] in {"模型输出被截断", "模型返回空响应或内容被截断"}
                for row in previous[-2:]
            )
        ):
            raise ModelOutputTruncated(f"{task_label}的输出被截断。")
        first_attempt = 0 if repairable_stage and self.resuming else len(previous)
        last_diagnostic = previous[-1]["diagnostic"] if previous and repairable_stage else None
        repair = (
            f"\n上次校验问题：{last_diagnostic}。请按要求重新返回完整的 JSON。"
            if last_diagnostic
            else ""
        )
        for attempt in range(first_attempt, 3):
            content = canonical(payload) + repair
            request_estimate = (
                len(system.encode()) + len(content.encode()) + 128 + self.config.max_output_tokens
            )
            call_id, timeout = self.store.reserve(
                self.run_id,
                task,
                role,
                input_hash,
                digest(system + repair),
                request_estimate,
                self.config,
            )
            started = time.monotonic()
            usage = None
            truncated = False
            try:
                if self.client is None:
                    raw = canonical(mock_reply(phase, role, payload))
                    usage = {
                        "total_tokens": max(1, (len(system) + len(content) + len(raw)) // 3),
                        "simulated": True,
                    }
                else:
                    completion = self.client.chat.completions.create(
                        model=self.config.model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": content},
                        ],
                        response_format={"type": "json_object"},
                        max_tokens=self.config.max_output_tokens,
                        **(
                            {"extra_body": {"thinking": {"type": "disabled"}}}
                            if self.config.mode == "deepseek"
                            else {}
                        ),
                        timeout=timeout,
                    )
                    usage = completion.usage.model_dump(mode="json") if completion.usage else None
                    raw = completion.choices[0].message.content if completion.choices else None
                    truncated = bool(
                        completion.choices and completion.choices[0].finish_reason == "length"
                    )
                    if truncated:
                        raise ValueError("模型输出被截断")
                    if not raw:
                        raise ValueError("模型返回空响应")
                try:
                    result = schema.model_validate_json(raw)
                except ValidationError as exc:
                    # A stray annotation is harmless if every required scoring field is valid.
                    if phase != "score" or any(
                        error["type"] != "extra_forbidden"
                        for error in exc.errors(include_input=False, include_url=False)
                    ):
                        raise
                    result = schema.model_validate_json(raw, extra="ignore")
                if validate:
                    validate(result)
            except (APIConnectionError, APIStatusError) as exc:
                self.store.end_call(call_id, "request_error", usage, time.monotonic() - started)
                status = getattr(exc, "status_code", None)
                if attempt < 2 and (status is None or status in {408, 429} or status >= 500):
                    time.sleep(0.5 * (attempt + 1))
                    continue
                label = f"状态码 {status}" if status else "连接错误"
                raise ModelError(f"{task_label}调用模型失败（{label}）。") from None
            except ValueError as exc:
                detail = validation_detail(exc)
                self.store.end_call(
                    call_id,
                    "invalid_output",
                    usage,
                    time.monotonic() - started,
                    diagnostic=detail,
                )
                if (repairable_stage or not repair) and attempt < 2:
                    repair = f"\n上次校验问题：{detail}。请按要求重新返回完整的 JSON。"
                    if file_stage:
                        repair += ' 将完整文件文本放在 {"content":"..."} 中，勿用说明代替源码。'
                    continue
                if truncated:
                    raise ModelOutputTruncated(f"{task_label}的输出被截断。") from None
                raise ModelError(f"{task_label}的输出未通过校验：{detail}") from None
            except BaseException:
                self.store.end_call(call_id, "unknown", usage, time.monotonic() - started)
                raise
            self.store.end_call(call_id, "ok", usage, time.monotonic() - started)
            self.store.check_cancel(self.run_id)
            self.store.cache(self.run_id, task, result.model_dump_json())
            return result
        raise ModelError(f"{task_label}已达到请求次数上限。")
