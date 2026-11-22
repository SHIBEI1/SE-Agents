from __future__ import annotations

import re

from .schemas import Decision

STATUS = {
    "queued": "待开始",
    "running": "论证中",
    "waiting_input": "等你决定",
    "completed": "可供审阅",
    "partial": "部分结果",
    "failed": "执行失败",
    "cancelled": "已停止",
    "interrupted": "已中断",
}
STAGES = {
    "internal_resolution": "Agent 处理方案取舍",
    "proposal_confirm": "确认推荐方案",
    "reopen_discussion": "按用户意见重新论证",
    "requirements_spec": "整理需求规格",
    "requirements_confirm": "确认需求基线",
    "engineering_design": "整理设计规格",
    "design_confirm": "确认设计规格",
    "task_decompose": "PM 分解开发任务",
    "plan_review": "独立评审任务计划",
    "plan_confirm": "确认任务与编排",
    "orchestrate": "生成执行编排",
    "task_execute": "执行开发任务",
    "task_review": "独立检查交付物",
    "task_update": "更新任务看板",
    "project_plan": "规划软件项目",
    "project_probe": "核对实现方向",
    "project_clarify": "确认实现方向",
    "project_review_clarify": "确认构建中的取舍",
    "project_file": "生成项目文件",
    "project_review": "审查项目实现",
    "project_verify": "运行项目验收",
    "project_runtime_repair": "定位运行失败并规划修复",
    "project_build": "写入项目目录",
    "design_probe": "规划技术设计",
    "design_clarify": "澄清关键问题",
    "report_scope": "编写详细需求",
    "report_architecture": "设计架构与接口",
    "report_delivery": "编写实施与验收计划",
    "report_tests": "设计验收测试",
    "report_review": "复核技术设计",
    "technical_report": "汇总技术报告",
    "chat_frame": "理解输入",
    "input": "提交想法",
    "frame": "明确目标",
    "diverge": "独立构思",
    "cluster": "合并方案",
    "shortlist": "筛选方案",
    "critique": "审查方案",
    "critique_assumption": "补查关键假设",
    "resolve": "处理分歧",
    "score": "独立评分",
    "select": "选择方案",
    "draft": "起草方案",
    "signoff": "签署方案",
    "revise": "修订方案",
    "budget": "达到预算上限",
    "stop": "保留部分结果",
}
SIGNOFF = {"approve": "同意", "conditional": "有条件同意", "reject": "不同意"}
STANCE = {
    "support": "支持",
    "conditional": "有条件支持",
    "oppose": "反对",
    "abstain": "保留意见",
    "agree": "同意",
}
ISSUE_KIND = {"factual": "事实待核查", "value": "目标取舍", "tradeoff": "方案权衡"}
ISSUE_STATUS = {
    "open": "待处理",
    "needs_evidence": "待补材料",
    "needs_human": "待你决定",
    "resolved": "已处理",
    "retained": "保留分歧",
}
EVIDENCE = {
    "unverified": "待核查",
    "supported": "材料支持",
    "refuted": "材料反驳",
    "inconclusive": "材料不足",
    "defer": "保留待核查",
}
CALL_STATUS = {
    "ok": "成功",
    "pending": "请求中",
    "unknown": "结果未知",
    "request_error": "请求失败",
    "invalid_output": "输出未通过校验",
}
REFS = {"OPT": "方案", "ASM": "假设", "OBJ": "异议", "CLM": "观点", "EVD": "证据", "DEC": "决定"}
_LABEL = re.compile(r"\b(OPT|ASM|OBJ|CLM|EVD|DEC)-([0-9]+(?:-[0-9]+)?)")
_WORDS = {
    "Proposal": "论证方案",
    "Idea": "想法",
    "Token": "词元",
    "DeepSeek": "深度求索",
    "needs_evidence": "待补材料",
    "needs_human": "待你决定",
    "unverified": "待核查",
    "inconclusive": "材料不足",
    "supported": "材料支持",
    "refuted": "材料反驳",
    "resolved": "已处理",
    "retained": "保留分歧",
    "open": "待处理",
}
_WORD = re.compile(r"(?<![A-Za-z_])(" + "|".join(_WORDS) + r")(?![A-Za-z_])")


def zh(text: str) -> str:
    return _WORD.sub(
        lambda m: _WORDS[m[0]],
        _LABEL.sub(
            lambda m: f"{REFS[m[1]]} {m[2]}",
            text,
        ),
    )


def decision_text(decision: Decision) -> str:
    value = decision.value
    if "reply" in value:
        question = value.get("question")
        return zh(f"{question} → {value['reply']}" if question else value["reply"])
    if "option_id" in value:
        return zh(value["option_id"])
    if "option_ids" in value:
        return zh("、".join(value["option_ids"]))
    if "choice" in value:
        return zh(value["choice"])
    if "verdict" in value:
        return EVIDENCE.get(value["verdict"], "已记录核查意见")
    return zh(decision.reason)


def ranking_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "候选方案": f"{zh(row['option_id'])} · {zh(row['title'])}",
            "综合评分": "资料不足" if row["score"] is None else f"{row['score']:.2f}",
            "支持比例": f"{row['consensus']:.0%}",
        }
        for row in rows
    ]
