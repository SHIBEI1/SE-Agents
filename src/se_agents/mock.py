"""Deterministic examples for exercising the workflow without network calls."""

from .schemas import WEIGHTS

OPTIONS = {
    "directory": {
        "title": "轻量信息目录",
        "approach": "以人工维护的信息目录作为可独立验证的课程原型。",
        "value": "先验证目标用户是否能更快找到需要的信息。",
        "features": ["浏览与筛选", "查看来源和更新时间", "提交纠错反馈"],
        "limitations": "仅模拟已授权或公开数据，不承诺实时同步。",
        "assumptions": [
            {
                "statement": "目标用户有重复查找信息的场景。",
                "critical": False,
                "verification_plan": "请三位目标用户完成同一信息查找任务，记录耗时与反馈。",
            }
        ],
    },
    "integration": {
        "title": "接入外部平台",
        "approach": "使用现有平台的数据接口，将查询和操作集中到一个入口。",
        "value": "可减少重复录入，但效果依赖数据接口。",
        "features": ["同步公开状态", "统一查询与操作入口"],
        "limitations": "需要取得真实接口文档与使用授权。",
        "assumptions": [
            {
                "statement": "外部平台提供可用且已授权的接口。",
                "critical": True,
                "verification_plan": "取得平台方提供的接口文档和授权说明。",
            }
        ],
    },
    "manual": {
        "title": "人工服务验证",
        "approach": "以表单和人工记录模拟服务，先检验需求再决定是否开发。",
        "value": "低成本发现高频问题和业务流程。",
        "features": ["收集请求", "人工响应", "记录需求与响应时间"],
        "limitations": "适合小样本，人工维护成本随规模增长。",
        "assumptions": [],
    },
}


def mock_reply(task: str, role: str, payload: dict) -> dict:
    if task == "diverge":
        key = {"pm": "directory", "user": "directory", "tech": "integration", "devil": "manual"}[
            role
        ]
        note = {
            "pm": "先用最小功能验证价值，避免提前承担集成成本。",
            "user": "先确认信息查找的痛点与频率，需要实际用户任务验证。",
            "tech": "接口集成收益较大，需把依赖和验证条件写清。",
            "devil": "服务需求可能不值得开发，建议先用人工方式检验。",
        }[role]
        return {"options": [OPTIONS[key]], "note": note}

    if task == "critique":
        seen = set()
        reviews = []
        for claim in payload["claims"]:
            if claim["author"] != role and claim["author"] not in seen:
                reviews.append(
                    {
                        "claim_id": claim["id"],
                        "stance": "conditional",
                        "reason": "方向具有验证价值，应先核查其依赖与维护成本。",
                    }
                )
                seen.add(claim["author"])
        objections = []
        if role == "user":
            option = payload["options"][0]
            objections.append(
                {
                    "option_id": option["id"],
                    "kind": "value",
                    "severity": "minor",
                    "reason": "便利性与数据维护成本之间需要明确优先级。",
                    "action": "由用户明确本轮是优先快速验证，还是优先功能覆盖。",
                }
            )
        return {
            "reviews": reviews,
            "objections": objections,
            "checked_assumptions": payload["critical_ids"] if role == "devil" else [],
        }

    if task == "critique_assumption":
        return {
            "reason": f"尚缺少验证「{payload['assumption']['statement']}」的实测或用户反馈。",
            "action": payload["assumption"]["verification_plan"],
        }

    if task == "internal_resolution":
        return {"decisions": [
            {"issue_id": issue["id"], "choice": "优先验证核心场景",
             "rationale": "根据当前原型目标控制实施范围。"}
            for issue in payload["issues"]
        ]}

    if task == "score":
        assessments = []
        for option in payload["options"]:
            blocked = [
                item["id"]
                for item in option["assumptions"]
                if item["critical"] and item["status"] != "supported"
            ]
            grade = 4 if option["title"] == OPTIONS["directory"]["title"] else 3
            scores = dict.fromkeys(WEIGHTS, grade)
            if blocked:
                scores["feasibility"] = None
            assessments.append(
                {
                    "option_id": option["id"],
                    "scores": scores,
                    "reasons": {
                        name: "依据原型范围与当前依赖判断，实际效果待验证。" for name in WEIGHTS
                    },
                    "stance": "conditional" if blocked else "support",
                    "condition_ids": blocked,
                    "rationale": "按现有材料评价，可行性取决于已列出的前提。",
                }
            )
        return {"assessments": assessments}

    if task == "signoff":
        blocking = payload["blocking_ids"]
        return {
            "verdict": "conditional" if blocking else "approve",
            "condition_ids": blocking,
            "reason": "关键前提尚待核实。" if blocking else "范围、验证计划及未决事项已经列明。",
        }

    if task == "revise":
        return {
            "changes": ["明确小样本验证的负责人、时间及成功条件。"],
            "rationale": "将验证计划写成可检查的交付项。",
        }

    raise ValueError(f"Unknown mock task: {task}")
