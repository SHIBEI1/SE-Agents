from __future__ import annotations

from .schemas import Frame, canonical

DEVELOPMENT_QUESTIONS = {"design_clarify", "project_clarify", "project_review_clarify"}


def validate_answer(request: dict, answer: dict) -> None:
    if not isinstance(answer, dict) or answer.get("request_id") != request["id"]:
        raise ValueError("该回答不匹配当前请求。请刷新后重试。")
    reason = answer.get("reason")
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 2400:
        raise ValueError("请填写有效的理由（最多 2400 字符）。")
    value = answer.get("value")
    if not isinstance(value, dict) or len(canonical(value)) > 12_000:
        raise ValueError("请提交有效的回答。")
    if value == {"stop": True}:
        return
    kind = request["kind"]
    payload = request["payload"]
    if kind in DEVELOPMENT_QUESTIONS:
        reply = value.get("reply")
        if not isinstance(reply, str) or not 1 <= len(reply.strip()) <= 2400:
            raise ValueError("请用自然语言回复，也可以说「你决定」。")
    elif kind == "frame":
        Frame.model_validate(value)
    elif kind == "shortlist":
        options = value.get("option_ids")
        available = {item["id"] for item in payload["options"]}
        if (
            not isinstance(options, list)
            or not 1 <= len(options) <= 3
            or any(not isinstance(item, str) for item in options)
            or len(set(options)) != len(options)
            or not set(options) <= available
        ):
            raise ValueError("请选择 1–3 个不同的有效候选。")
    elif kind == "select":
        if value.get("option_id") not in [item["option_id"] for item in payload["rankings"]]:
            raise ValueError("请选择评分阶段中的有效候选。")
    elif kind == "resolve":
        if payload["issue"]["kind"] == "value":
            choice = value.get("choice")
            if not isinstance(choice, str) or not choice.strip() or len(choice) > 2400:
                raise ValueError("请填写实际偏好或裁决。")
        else:
            verdict = value.get("verdict")
            if not isinstance(verdict, str) or verdict not in {
                "supported",
                "refuted",
                "inconclusive",
                "defer",
            }:
                raise ValueError("请选择事实核查结论。")
            if verdict != "defer" and any(
                not isinstance(value.get(field), str)
                or not value[field].strip()
                or len(value[field]) > 2400
                for field in ("source", "excerpt")
            ):
                raise ValueError("请填写可核对的来源与内容摘要。")
    else:
        raise ValueError("不支持的人工请求。")
