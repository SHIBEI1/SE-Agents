from __future__ import annotations

from .schemas import (
    DIMENSIONS,
    ROLES,
    WEIGHTS,
    Assessment,
    Assumption,
    Blackboard,
    Claim,
    Critique,
    Divergence,
    Objection,
    Option,
    Proposal,
    Role,
    Scorecard,
    Signoff,
    digest,
)


def cluster(replies: dict[Role, Divergence], board: Blackboard) -> None:
    groups: dict[str, Option] = {}
    for role in ROLES:
        reply = replies[role]
        board.divergence_notes[role] = reply.note
        for seed in reply.options:
            key = digest(seed.model_dump())
            if key not in groups:
                option_id = f"OPT-{len(groups) + 1:03}"
                assumptions = [
                    Assumption(
                        **item.model_dump(), id=f"ASM-{option_id[4:]}-{index}", option_id=option_id
                    )
                    for index, item in enumerate(seed.assumptions, 1)
                ]
                groups[key] = Option(
                    **seed.model_dump(exclude={"assumptions"}),
                    id=option_id,
                    claim_ids=[],
                    assumptions=assumptions,
                )
            option = groups[key]
            claim = Claim(
                id=f"CLM-{len(board.claims) + 1:03}",
                author=role,
                option_id=option.id,
                rationale=reply.note,
            )
            board.claims.append(claim)
            option.claim_ids.append(claim.id)
    board.options = list(groups.values())


def shortlisted(board: Blackboard) -> list[Option]:
    return [item for item in board.options if item.id in board.shortlist]


def critique_input(board: Blackboard) -> dict:
    options = shortlisted(board)
    ids = {option.id for option in options}
    return {
        "idea": board.idea,
        "frame": board.frame.model_dump(),
        "options": [item.model_dump() for item in options],
        "claims": [claim.model_dump() for claim in board.claims if claim.option_id in ids],
        "critical_ids": [a.id for option in options for a in option.assumptions if a.critical],
    }


def critique_targets(payload: dict, role: Role) -> dict:
    by_author = {}
    for claim in payload["claims"]:
        if claim["author"] != role:
            by_author.setdefault(claim["author"], claim["id"])
    return {
        **payload,
        "review_target_ids": list(by_author.values())[:2],
        "required_checked_assumption_ids": payload["critical_ids"] if role == "devil" else [],
    }


def validate_critique(role: Role, payload: dict, reply: Critique) -> None:
    authors = {item["id"]: item["author"] for item in payload["claims"] if item["author"] != role}
    reviewed_ids = [item.claim_id for item in reply.reviews]
    if len(set(reviewed_ids)) != len(reviewed_ids) or any(
        ref not in authors for ref in reviewed_ids
    ):
        raise ValueError("无效、重复或属于自己的审查对象。")
    if len({authors[ref] for ref in reviewed_ids}) < min(2, len(set(authors.values()))):
        raise ValueError("审查必须覆盖至少两个其他作者。")
    options = {item["id"]: {a["id"] for a in item["assumptions"]} for item in payload["options"]}
    for item in reply.objections:
        if item.option_id not in options or (
            item.assumption_id and item.assumption_id not in options[item.option_id]
        ):
            raise ValueError("异议引用无效。")
    critical = set(payload["critical_ids"])
    if not set(reply.checked_assumptions) <= critical or (
        role == "devil" and not critical <= set(reply.checked_assumptions)
    ):
        raise ValueError("关键假设检查不完整。")


def record_objections(board: Blackboard) -> None:
    for role in ROLES:
        for issue in board.critiques[role].objections:
            board.objections.append(
                Objection(
                    **issue.model_dump(),
                    id=f"OBJ-{len(board.objections) + 1:03}",
                    author=role,
                    status={
                        "value": "needs_human",
                        "factual": "needs_evidence",
                        "tradeoff": "open",
                    }[issue.kind],
                )
            )
    for option in shortlisted(board):
        for assumption in option.assumptions:
            if not assumption.critical or any(
                item.kind == "factual"
                and item.severity == "major"
                and item.assumption_id == assumption.id
                for item in board.objections
            ):
                continue
            board.objections.append(
                Objection(
                    id=f"OBJ-{len(board.objections) + 1:03}",
                    author="program",
                    option_id=option.id,
                    assumption_id=assumption.id,
                    kind="factual",
                    severity="major",
                    reason=assumption.statement,
                    action=assumption.verification_plan,
                    status="needs_evidence",
                )
            )


def condition_status(board: Blackboard, option_id: str) -> dict[str, bool]:
    option = next(item for item in board.options if item.id == option_id)
    result = {item.id: item.status == "supported" for item in option.assumptions}
    result.update(
        {
            item.id: item.status == "resolved"
            for item in board.objections
            if item.option_id == option_id
        }
    )
    return result


def valid_conditions(refs: list[str], status: dict[str, bool], required: bool) -> bool:
    return (
        (not required or bool(refs)) and len(set(refs)) == len(refs) and set(refs) <= status.keys()
    )


def validate_score(board: Blackboard, score: Scorecard) -> None:
    expected = set(board.shortlist)
    actual = [item.option_id for item in score.assessments]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("必须对每个候选评分一次。")
    for item in score.assessments:
        status = condition_status(board, item.option_id)
        if item.reasons.keys() != WEIGHTS.keys() or not valid_conditions(
            item.condition_ids,
            status,
            item.stance == "conditional",
        ):
            raise ValueError("评分依据或条件引用不完整。")


def option_assessments(board: Blackboard, option_id: str) -> list[Assessment]:
    return [
        item
        for role in ROLES
        if role in board.scores
        for item in board.scores[role].assessments
        if item.option_id == option_id
    ]


def consensus(board: Blackboard, option_id: str) -> float:
    status = condition_status(board, option_id)
    return sum(
        item.stance in {"support", "conditional"}
        and (item.stance == "support" or bool(item.condition_ids))
        and all(status.get(ref, False) for ref in item.condition_ids)
        for item in option_assessments(board, option_id)
    ) / len(ROLES)


def rankings(board: Blackboard, weights: dict[str, float]) -> list[dict]:
    results = []
    for option in shortlisted(board):
        assessments = option_assessments(board, option.id)
        complete = len(assessments) == len(ROLES) and all(
            None not in item.scores.model_dump().values() for item in assessments
        )
        values = []
        if complete:
            for item in assessments:
                scores = item.scores.model_dump()
                values.append(sum(weights[key] * scores[key] for key in weights))
        results.append(
            {
                "option_id": option.id,
                "title": option.title,
                "score": round(sum(values) / len(ROLES), 3) if values else None,
                "range": [round(min(values), 3), round(max(values), 3)] if values else None,
                "consensus": consensus(board, option.id),
            }
        )
    return sorted(
        results,
        key=lambda row: (-row["score"] if row["score"] is not None else 1, row["option_id"]),
    )


def sensitivity(board: Blackboard, weights: dict[str, float]) -> bool | None:
    ranked = [row for row in rankings(board, weights) if row["score"] is not None]
    if len(ranked) < 2:
        return None
    baseline = ranked[0]["option_id"]
    for key in weights:
        for delta in (-0.05, 0.05):
            perturbed = weights | {key: max(0, weights[key] + delta)}
            total = sum(perturbed.values())
            ranked = rankings(board, {name: weight / total for name, weight in perturbed.items()})
            if ranked[0]["option_id"] != baseline:
                return True
    return False


def blocking_ids(board: Blackboard, option_id: str) -> list[str]:
    option = next(item for item in board.options if item.id == option_id)
    return [
        item.id for item in option.assumptions if item.critical and item.status != "supported"
    ] + [
        issue.id
        for issue in board.objections
        if issue.option_id == option_id and issue.severity == "major" and issue.status != "resolved"
    ]


def compose(board: Blackboard, option_id: str, weights: dict[str, float]) -> Proposal:
    option = next(item for item in board.options if item.id == option_id)
    decision = next(item for item in reversed(board.decisions) if item.kind == "select")
    related_issues = [item for item in board.objections if item.option_id == option_id]
    dependencies = {
        "option": option.model_dump(),
        "frame": board.frame.model_dump(),
        "objections": [item.model_dump() for item in related_issues],
        "decisions": [item.model_dump() for item in board.decisions],
        "rankings": rankings(board, weights),
        "weights": weights,
    }
    sections = {
        "背景与问题": board.idea,
        "目标用户": board.frame.audience,
        "价值主张": option.value,
        "范围与非目标": f"选定方向：{option.title}\n\n约束：{board.frame.constraints}\n\n边界：{option.limitations}",
        "核心功能": "\n".join(f"- {feature}" for feature in option.features),
        "技术方案概要": option.approach,
        "商业模式或资源方式": "课程原型。记录数据、开发和维护所需资源，商业假设待单独验证。",
        "成功指标": f"目标：{board.frame.goal}\n\n先采集当前做法的基线，再与原型任务结果对比。",
        "风险与关键假设": "\n".join(
            f"- {item.id} [{item.status}] {item.statement}。验证：{item.verification_plan}"
            for item in option.assumptions
        )
        or "当前候选未登记假设，仍需通过实际原型验证。",
        "里程碑": "1. 确认范围与验证标准\n2. 构建最小原型\n3. 验证主要假设并复核方向",
        "未决问题与分歧": "\n".join(
            f"- {item.id} [{item.status}] {item.reason}。下一步：{item.action}"
            for item in related_issues
            if item.status != "resolved"
        )
        or "当前选定方向暂无已登记的开放异议。",
        "决策依据": f"{decision.id} 选择 {option.id}。{decision.reason}\n\n"
        f"关联论点：{', '.join(option.claim_ids)}",
    }
    return Proposal(option_id=option_id, sections=sections, dependency_hash=digest(dependencies))


def validate_signoff(board: Blackboard, reply: Signoff) -> None:
    status = condition_status(board, board.proposal.option_id)
    if not valid_conditions(reply.condition_ids, status, reply.verdict == "conditional"):
        raise ValueError("签署条件必须引用当前方案的假设或异议。")


def signoffs_pass(board: Blackboard) -> bool:
    status = condition_status(board, board.proposal.option_id)
    return set(board.signoffs) == set(ROLES) and all(
        item.verdict in {"approve", "conditional"}
        and (item.verdict == "approve" or bool(item.condition_ids))
        and all(status.get(ref, False) for ref in item.condition_ids)
        for item in board.signoffs.values()
    )


def scoring_rubric(weights: dict[str, float]) -> dict:
    return {name: {"label": DIMENSIONS[name], "weight": weight} for name, weight in weights.items()}
