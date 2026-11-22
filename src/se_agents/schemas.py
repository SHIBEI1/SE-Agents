from __future__ import annotations

import hashlib
import json
import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from .planning import Engineering

Role = Literal["pm", "user", "tech", "devil"]
ROLES: tuple[Role, ...] = ("pm", "user", "tech", "devil")
ROLE_NAMES = {"pm": "产品经理", "user": "用户研究员", "tech": "技术架构师", "devil": "质疑者"}
WEIGHTS = {"value": 0.30, "feasibility": 0.25, "cost": 0.15, "fit": 0.15, "risk": 0.15}
DIMENSIONS = {
    "value": "用户价值",
    "feasibility": "技术可行性",
    "cost": "成本可承受性",
    "fit": "差异化与场景适配",
    "risk": "风险可控性",
}
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2400)]
ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]
Grade = Annotated[int, Field(strict=True, ge=1, le=5)]


def canonical(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)


class RunConfig(Record):
    mode: Literal["mock", "deepseek", "api"] = "mock"
    chat_mode: bool = False
    technical_mode: bool = False
    workflow_mode: Literal["legacy", "full", "planning"] = "legacy"
    internal_deliberation: bool = False
    max_parallel_tasks: int = Field(default=2, ge=1, le=4)
    model: ShortText = "deepseek-flash"
    base_url: str | None = None
    max_calls: int = Field(default=48, ge=1, le=100)
    max_tokens: int = Field(default=160_000, ge=1)
    max_seconds: int = Field(default=1800, ge=1)
    max_output_tokens: int = Field(default=4096, ge=256, le=8192)
    concurrency: int = Field(default=2, ge=1, le=4)
    max_draft_revisions: int = Field(default=1, ge=0, le=1)
    weights: dict[str, float] = Field(default_factory=lambda: WEIGHTS.copy())

    @model_validator(mode="after")
    def validate_weights(self) -> RunConfig:
        if (
            self.weights.keys() != WEIGHTS.keys()
            or any(not math.isfinite(weight) or weight < 0 for weight in self.weights.values())
            or not math.isclose(sum(self.weights.values()), 1, abs_tol=1e-9)
        ):
            raise ValueError("评分权重必须包含五个维度，非负且总和为 1")
        return self


class Frame(Record):
    audience: Text
    goal: Text
    constraints: Text


class FrameNotes(Record):
    audience: Text | None = None
    goal: Text | None = None
    constraints: Text | None = None


class AssumptionSeed(Record):
    statement: Text
    critical: bool
    verification_plan: Text


class OptionSeed(Record):
    title: ShortText
    approach: Text
    value: Text
    features: list[Text] = Field(min_length=1, max_length=5)
    limitations: Text
    assumptions: list[AssumptionSeed] = Field(max_length=3)


class Divergence(Record):
    options: list[OptionSeed] = Field(max_length=3)
    note: Text


class Assumption(AssumptionSeed):
    id: str
    option_id: str
    status: Literal["unverified", "supported", "refuted", "inconclusive"] = "unverified"
    evidence_ids: list[str] = Field(default_factory=list)


class Claim(Record):
    id: str
    author: Role
    option_id: str
    rationale: Text


class Option(OptionSeed):
    id: str
    claim_ids: list[str]
    assumptions: list[Assumption]


class Review(Record):
    claim_id: str
    stance: Literal["agree", "oppose", "conditional"]
    reason: Text


class ObjectionSeed(Record):
    option_id: str
    assumption_id: str | None = None
    kind: Literal["factual", "value", "tradeoff"]
    severity: Literal["major", "minor"]
    reason: Text
    action: Text


class Objection(ObjectionSeed):
    id: str
    author: Role | Literal["program"]
    status: Literal["open", "needs_evidence", "needs_human", "resolved", "retained"] = "open"
    decision_id: str | None = None
    evidence_id: str | None = None


class Critique(Record):
    reviews: list[Review] = Field(max_length=64)
    objections: list[ObjectionSeed] = Field(max_length=64)
    checked_assumptions: list[str] = Field(default_factory=list, max_length=36)


class CritiqueBatch(Critique):
    reviews: list[Review] = Field(max_length=16)
    objections: list[ObjectionSeed] = Field(max_length=16)


class AssumptionReview(Record):
    reason: Text
    action: Text


class Scores(Record):
    value: Grade | None
    feasibility: Grade | None
    cost: Grade | None
    fit: Grade | None
    risk: Grade | None


class ScoreReasons(Record):
    value: Text
    feasibility: Text
    cost: Text
    fit: Text
    risk: Text


class Assessment(Record):
    option_id: str
    scores: Scores
    reasons: ScoreReasons
    stance: Literal["support", "conditional", "oppose", "abstain"]
    rationale: Text
    condition_ids: list[str] = Field(default_factory=list)


class Scorecard(Record):
    assessments: list[Assessment] = Field(min_length=1, max_length=12)


class Tradeoff(Record):
    issue_id: ShortText
    choice: Text
    rationale: Text


class TradeoffDecisions(Record):
    decisions: list[Tradeoff] = Field(max_length=32)


class Signoff(Record):
    verdict: Literal["approve", "conditional", "reject"]
    reason: Text
    condition_ids: list[str] = Field(default_factory=list)


class DraftRevision(Record):
    changes: list[Text] = Field(min_length=1, max_length=5)
    rationale: Text


class Decision(Record):
    id: str
    kind: Literal[
        "frame", "shortlist", "resolve", "select", "stop", "design_clarify",
        "project_clarify", "project_review_clarify",
        "proposal_confirm", "requirements_confirm", "design_confirm", "plan_confirm",
    ]
    value: dict
    reason: Text
    actor: Literal["human", "demo", "agent", "orchestrator"]
    answered_at: float


class Evidence(Record):
    id: str
    source: Text
    excerpt: Text
    judgment: Literal["supported", "refuted", "inconclusive"]


class Proposal(Record):
    option_id: str
    sections: dict[str, str]
    dependency_hash: str
    revision: int = 0


class DesignQuestion(Record):
    question: Text
    impact: Text
    default: Text
    options: list[ShortText] = Field(default_factory=list, max_length=3)


class DesignProbe(Record):
    questions: list[DesignQuestion] = Field(max_length=2)


class TechnicalReport(Record):
    model: ShortText
    sections: dict[str, str]


class ProjectBuild(Record):
    source_run_id: str
    source_revision: int
    report_hash: str
    source_name: str
    source_report_partial: bool
    directory: str
    verify_locally: bool = False


class ProjectAcceptance(Record):
    status: Literal["passed", "failed", "not_run"]
    summary: str
    output: str = ""
    elapsed: float = 0


class ProjectBuildResult(Record):
    files: dict[str, str]
    review_notes: list[str]
    acceptance: ProjectAcceptance | None = None


class Blackboard(Record):
    idea: str
    frame: Frame | None = None
    frame_hints: FrameNotes | None = None
    divergence_notes: dict[Role, str] = Field(default_factory=dict)
    claims: list[Claim] = Field(default_factory=list)
    options: list[Option] = Field(default_factory=list)
    shortlist: list[str] = Field(default_factory=list)
    critiques: dict[Role, Critique] = Field(default_factory=dict)
    objections: list[Objection] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    scores: dict[Role, Scorecard] = Field(default_factory=dict)
    decisions: list[Decision] = Field(default_factory=list)
    proposal: Proposal | None = None
    proposal_review_issues: list[str] = Field(default_factory=list)
    design_questions: list[DesignQuestion] = Field(default_factory=list)
    technical_report: TechnicalReport | None = None
    project_build: ProjectBuild | None = None
    project_questions: list[DesignQuestion] = Field(default_factory=list)
    project_result: ProjectBuildResult | None = None
    signoffs: dict[Role, Signoff] = Field(default_factory=dict)
    review_snapshot_hash: str | None = None
    outcome: Literal["completed", "partial"] | None = None
    stop_reason: str | None = None
    engineering: Engineering | None = None
