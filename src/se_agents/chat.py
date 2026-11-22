from __future__ import annotations

import re

from .human import APPROVAL_QUESTIONS, DEVELOPMENT_QUESTIONS, validate_answer
from .models import ModelError, ModelRunner
from .presentation import EVIDENCE, zh
from .schemas import FrameNotes

LABELS = {"audience": "目标用户", "goal": "目标", "constraints": "限制"}
UNDECIDED = {
    "audience": "目标用户待确定",
    "goal": "改善目标待确定",
    "constraints": "范围与限制待确定",
}
DEFER_FRAME = {"还没定", "还没定，先分析", "先分析", "暂不确定"}
ALIASES = {
    "目标用户": "audience",
    "用户": "audience",
    "使用者": "audience",
    "目标": "goal",
    "希望改善的结果": "goal",
    "要解决的问题": "goal",
    "范围与限制": "constraints",
    "限制": "constraints",
    "约束": "constraints",
    "核查结论": "verdict",
    "结论": "verdict",
    "材料来源": "source",
    "来源": "source",
    "材料摘要": "excerpt",
    "摘要": "excerpt",
}
LABELED = re.compile(r"(?<![\w])(" + "|".join(ALIASES) + r")\s*[：:]\s*")
CHINESE_DIGITS = str.maketrans("一二三四五六七八九", "123456789")


def labeled_fields(text: str) -> dict[str, str]:
    matches = list(LABELED.finditer(text))
    fields = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        value = text[match.end() : end].strip(" \t\n，；。")
        if value:
            fields[ALIASES[match[1]]] = value
    return fields


def frame_notes(text: str, model: ModelRunner) -> dict[str, str]:
    fields = {key: value for key, value in labeled_fields(text).items() if key in LABELS}
    phrases = {
        "audience": r"^(?:给(.+?)用|(?:面向|用户是|目标用户是)(.+))$",
        "goal": r"^(?:希望|想要|想|主要解决)(.+)$",
        "constraints": r"^((?:没有|暂无)(?:额外)?限制|(?:[0-9一二三四五六七八九十]+)(?:周|天|个月)内?.+)$",
    }
    for clause in re.split(r"[，。；,;\n]", text):
        for key, pattern in phrases.items():
            if key not in fields and (match := re.fullmatch(pattern, clause.strip())):
                fields[key] = next(value for value in match.groups() if value).strip()
    if len(fields) == len(LABELS) or model.config.mode == "mock":
        return fields

    def check(notes: FrameNotes) -> None:
        if any(value and value not in text for value in notes.model_dump().values()):
            raise ValueError("摘录必须来自用户原话。")

    try:
        extracted = model.run("chat_frame", "pm", {"用户原话": text}, FrameNotes, check)
    except ModelError:
        return fields
    return {**extracted.model_dump(exclude_none=True), **fields}


def question_text(request: dict) -> str:
    kind, payload = request["kind"], request["payload"]
    if kind == "proposal_confirm":
        return "Agent 已汇总一个推荐方案。请审阅后确认，或者填写意见退回重新论证。"
    if kind == "plan_confirm" and payload.get("approval_blocked"):
        return "计划还有重大评审问题。请查看评审意见，调整排期或填写意见退回 PM 重新拆分。"
    if kind == "plan_confirm" and payload.get("review_required"):
        return "已保存排期草案。确认时，质疑者会先复核这版计划。"
    if kind in APPROVAL_QUESTIONS:
        return "请审阅下方当前版本，确认后进入下一阶段，也可以填写修改意见退回。"
    if kind in DEVELOPMENT_QUESTIONS:
        intro = {
            "design_clarify": "",
            "project_clarify": "构建前，有一个实现方向想请你确认。\n\n",
            "project_review_clarify": "初版源码已生成。审查时发现一个使用上的取舍。\n\n",
        }[kind]
        blocks = []
        for item in payload["questions"]:
            options = "\n".join(
                f"  - {index}. {option}" for index, option in enumerate(item.get("options", []), 1)
            )
            block = f"- **{item['question']}**\n  - {item['impact']}\n  - 默认建议：{item['default']}"
            blocks.append(f"{block}\n{options}" if options else block)
        questions = "\n\n".join(blocks)
        return f"{intro}{questions}\n\n可以直接回答，也可以选「你决定」采用默认建议。"
    if kind == "frame":
        known = payload.get("known", {})
        missing = [label for key, label in LABELS.items() if key not in known]
        labels = "、".join(missing)
        examples = {
            "目标用户": "目标用户：在校学生",
            "目标": "目标：减少重复预约",
            "限制": "限制：四周内完成课程演示",
        }
        sample = "；".join(examples[label] for label in missing)
        return (
            f"还需要确认**{labels}**。例如「{sample}」。\n\n"
            "这些内容也可以在比较方向时再定。回复「先分析」即可继续。"
        )
    if kind in {"shortlist", "select"}:
        options = payload.get("options", []) if kind == "shortlist" else payload["rankings"]
        if kind == "shortlist":
            intro = "有多个候选方向。请回复要比较的 1 到 3 个编号，例如「选 1 和 2」。"
        else:
            intro = "候选方向已比较完毕。你想按哪一个起草方案？可以回复「选 1」并附上理由。"
        lines = [intro]
        for index, item in enumerate(options, 1):
            label = zh(item["title"])
            if kind == "select":
                score = item["score"]
                label += f" · {score:.2f} 分" if score is not None else " · 资料不足"
                if payload["blocking"].get(item["option_id"]):
                    label += " · 有待核查事项"
            lines.append(f"{index}. {label}")
        return "\n\n".join(lines)
    issue = payload["issue"]
    if issue["kind"] == "value":
        return f"**需要你决定一项取舍**\n\n{zh(issue['reason'])}\n\n{zh(issue['action'])}"
    return (
        f"**有一项事实需要核查**\n\n{zh(issue['reason'])}\n\n{zh(issue['action'])}\n\n"
        "可以回复「暂无材料」。如果有材料，请写上「结论：支持／反驳／不确定」「来源」和「摘要」。"
    )


def option_choices(text: str, items: list[dict], *, multiple: bool) -> list[str]:
    # Numbering is the order shown in the chat. IDs are also accepted explicitly.
    head = text
    if not multiple:
        head = re.split(r"[，。；,;\n]\s*(?=(?:因为|理由|考虑到|先|这样))", text, maxsplit=1)[0]
    selected = set()
    for item in items:
        identifier = item.get("id", item.get("option_id"))
        identifier_pattern = re.escape(identifier)
        if (multiple and re.search(rf"(?<![A-Z0-9]){identifier_pattern}(?![0-9])", head, re.I)) or (
            not multiple
            and re.fullmatch(
                rf"(?:我)?(?:选择|选|用)?\s*{identifier_pattern}[。！!]?",
                head,
                re.I,
            )
        ):
            selected.add(identifier)
        title = item["title"]
        if head.strip() == title or re.fullmatch(
            r"(?:我)?(?:选择|选|用)\s*" + re.escape(title), head
        ):
            selected.add(identifier)
    normalized = head.translate(CHINESE_DIGITS).strip()
    numbers = re.fullmatch(
        r"(?:我)?(?:选择|选|用)?\s*(?:方案)?\s*(?:第)?\s*(\d+)(?:\s*(?:个|号|项|种|方案))?"
        r"((?:\s*[、,，和及]\s*\d+)*)[。！!]?",
        normalized,
    )
    if numbers:
        for number in re.findall(r"\d+", numbers[1] + numbers[2]):
            index = int(number)
            if 1 <= index <= len(items):
                item = items[index - 1]
                selected.add(item.get("id", item.get("option_id")))
            else:
                raise ValueError("请回复列表里显示的方案编号。")
    return [
        item.get("id", item.get("option_id"))
        for item in items
        if item.get("id", item.get("option_id")) in selected
    ]


def chat_answer(request: dict, text: str, replies: list[str], model: ModelRunner) -> dict:
    text = text.strip()
    kind, payload = request["kind"], request["payload"]
    if text.rstrip("。！!") in {"结束并保留结果", "结束任务"}:
        value = {"stop": True}
    elif kind in APPROVAL_QUESTIONS:
        if text.rstrip("。！!") not in {"确认", "同意", "通过"}:
            raise ValueError("请使用规格或计划下方的确认表单，也可以回复「结束并保留结果」。")
        value = {"approve": True, "snapshot": payload["snapshot"]}
    elif kind in DEVELOPMENT_QUESTIONS:
        questions = payload["questions"]
        options = questions[0].get("options", []) if len(questions) == 1 else []
        number = re.fullmatch(r"(?:我?选(?:择)?\s*)?(\d+)[。！!]?", text) if options else None
        if number:
            index = int(number[1])
            if not 1 <= index <= len(options):
                raise ValueError("请选择对话中列出的编号，或直接描述你的要求。")
            reply = options[index - 1]
            if kind == "design_clarify":
                reply = f"关于「{questions[0]['question']}」，我选择：{reply}"
        elif text.rstrip("。！!") == "你决定" and (
            kind != "design_clarify" or any(item.get("options") for item in questions)
        ):
            if kind == "design_clarify":
                reply = "我同意采用默认建议。\n" + "\n".join(
                    f"- {item['question']} {item['default'][:950]}" for item in questions
                )
            else:
                reply = f"采用默认建议。{questions[0]['default']}"
        else:
            reply = text
        value = {"reply": reply[:2400]}
    elif kind == "frame":
        defer = text.rstrip("。！!") in DEFER_FRAME
        source = "\n".join(reply for reply in replies if reply.rstrip("。！!") not in DEFER_FRAME)
        fields = {**payload.get("known", {}), **(frame_notes(source, model) if source else {})}
        missing = [label for key, label in LABELS.items() if key not in fields]
        if missing and not defer:
            raise ValueError(f"可以继续补充{'、'.join(missing)}，也可以回复「先分析」。")
        value = {**UNDECIDED, **fields}
    elif kind in {"shortlist", "select"}:
        multiple = kind == "shortlist"
        items = payload["options"] if multiple else payload["rankings"]
        choices = option_choices(text, items, multiple=multiple)
        decision = re.split(r"[，。；,;\n]", text, maxsplit=1)[0]
        if re.search(r"不选|不要|不考虑|还是|或者|[？?]", decision) or not 1 <= len(choices) <= (
            3 if multiple else 1
        ):
            hint = "选 1 和 2" if multiple else "选 1"
            raise ValueError(f"我还不能确定你选的是哪个方向，请回复明确的编号，例如「{hint}」。")
        value = {"option_ids": choices} if multiple else {"option_id": choices[0]}
    elif payload["issue"]["kind"] == "value":
        value = {"choice": text}
    else:
        fields = labeled_fields(text)
        verdicts = {"支持": "supported", "反驳": "refuted", "不确定": "inconclusive"}
        verdict = verdicts.get(fields.get("verdict", ""))
        if verdict and fields.get("source") and fields.get("excerpt"):
            value = {"verdict": verdict, "source": fields["source"], "excerpt": fields["excerpt"]}
        elif any(
            phrase in text for phrase in ("暂无材料", "没有材料", "待核查", "不知道", "不清楚")
        ):
            value = {"verdict": "defer"}
        else:
            raise ValueError(
                "如果暂时没有证据，请回复「暂无材料」。有材料时请一并提供结论、来源和摘要。"
            )
    answer = {"request_id": request["id"], "value": value, "reason": text[:2400]}
    validate_answer(request, answer)
    return answer


def structured_reply(answer: dict) -> str:
    value = answer["value"]
    if value == {"stop": True}:
        return "结束并保留结果。"
    if value.get("action") == "edit_plan":
        return answer["reason"]
    if "reply" in value:
        return value["reply"]
    if "choice" in value:
        return value["choice"]
    if "option_id" in value:
        return f"选择{zh(value['option_id'])}。{answer['reason']}"
    if "option_ids" in value:
        return "选择" + "、".join(map(zh, value["option_ids"]))
    if "verdict" in value:
        return " · ".join(
            filter(
                None,
                (
                    EVIDENCE[value["verdict"]],
                    value.get("source"),
                    value.get("excerpt"),
                ),
            )
        )
    return "\n".join(f"{label}：{value[key]}" for key, label in LABELS.items() if key in value)
