from __future__ import annotations

import operator
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, TypedDict, TypeVar

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from . import deliberation as rules
from .chat import frame_notes
from .human import validate_answer
from .models import ModelError, ModelRunner
from .project import generate_files, recover_project, write_project
from .reporting import export_markdown
from .schemas import (
    ROLES,
    Blackboard,
    Critique,
    Decision,
    DesignProbe,
    DesignQuestion,
    Divergence,
    DraftRevision,
    Evidence,
    Frame,
    Role,
    Scorecard,
    Signoff,
    digest,
)
from .storage import Store
from .technical import build_report, context

T = TypeVar("T")


class GraphState(TypedDict, total=False):
    run_id: str
    revision: int
    branches: Annotated[list[str], operator.add]
    halt: bool
    revise: bool


class ProjectDecisionStopped(Exception):
    def __init__(self, answer: dict):
        self.answer = answer


def saved_critiques(store: Store, run_id: str, payload: dict) -> dict[Role, Critique]:
    legacy_hash = digest(payload)[:20]
    allowed_hashes = {
        role: {legacy_hash, digest(rules.critique_targets(payload, role))[:20]} for role in ROLES
    }
    saved = {}
    for task, raw in store.cached_phase(run_id, "critique"):
        _, role, input_hash, _ = task.split(":", 3)
        if role in saved or input_hash not in allowed_hashes.get(role, ()):
            continue
        try:
            review = Critique.model_validate_json(raw)
            rules.validate_critique(role, payload, review)
        except ValueError:
            continue
        saved[role] = review
    return saved


class Workshop:
    def __init__(self, store: Store, model: ModelRunner):
        self.store = store
        self.model = model

    def build(self, checkpointer):
        graph = StateGraph(GraphState)
        stages = {
            "design_probe": self.design_probe,
            "design_clarify": self.design_clarify,
            "technical_report": self.technical_report,
            "project_probe": self.project_probe,
            "project_clarify": self.project_clarify,
            "project_build": self.project_build,
            "frame": self.frame,
            "cluster": self.cluster,
            "shortlist": self.shortlist,
            "critique": self.critique,
            "resolve": self.resolve,
            "score": self.score,
            "select": self.select,
            "draft": self.draft,
            "signoff": self.signoff,
            "revise": self.revise,
        }
        for name, fn in stages.items():
            graph.add_node(name, fn)
        branches = [f"diverge_{role}" for role in ROLES]
        for role, branch in zip(ROLES, branches, strict=True):
            graph.add_node(branch, self.divergence_node(role))
        graph.add_conditional_edges(
            START,
            self.initial_stage,
            ["design_probe", "frame", "project_probe"],
        )
        graph.add_edge("design_probe", "design_clarify")
        graph.add_conditional_edges(
            "design_clarify",
            lambda state: END if state.get("halt") else "technical_report",
            [END, "technical_report"],
        )
        graph.add_edge("technical_report", END)
        graph.add_edge("project_probe", "project_clarify")
        graph.add_conditional_edges(
            "project_clarify",
            lambda state: END if state.get("halt") else "project_build",
            [END, "project_build"],
        )
        graph.add_edge("project_build", END)
        graph.add_conditional_edges(
            "frame",
            lambda state: END if state.get("halt") else branches,
            [END, *branches],
        )
        graph.add_edge(branches, "cluster")
        for start, destination in (
            ("cluster", "shortlist"),
            ("shortlist", "critique"),
            ("critique", "resolve"),
            ("resolve", "score"),
            ("score", "select"),
            ("select", "draft"),
            ("draft", "signoff"),
        ):
            graph.add_conditional_edges(
                start,
                lambda state, next_node=destination: END if state.get("halt") else next_node,
                [END, destination],
            )
        graph.add_conditional_edges(
            "signoff",
            lambda state: "revise" if state.get("revise") else END,
            ["revise", END],
        )
        graph.add_edge("revise", "signoff")
        return graph.compile(checkpointer=checkpointer)

    def initial_stage(self, state: GraphState) -> str:
        if self.store.board(state["run_id"], state["revision"]).project_build:
            return "project_probe"
        return "design_probe" if self.model.config.technical_mode else "frame"

    def load(self, state: GraphState, phase: str) -> Blackboard:
        self.store.check_cancel(state["run_id"])
        self.store.set_stage(state["run_id"], phase)
        return self.store.board(state["run_id"], state["revision"])

    def commit(self, state: GraphState, phase: str, board: Blackboard, **updates) -> GraphState:
        revision = self.store.publish(state["run_id"], state["revision"], phase, board)
        return {"revision": revision, **updates}

    def parallel(self, task: Callable[[Role], T]) -> dict[Role, T]:
        with ThreadPoolExecutor(max_workers=self.model.config.concurrency) as pool:
            futures = {role: pool.submit(task, role) for role in ROLES}
            return {role: future.result() for role, future in futures.items()}

    @staticmethod
    def ask(state: GraphState, kind: str, key: str, payload: dict) -> dict:
        request = {
            "id": f"{state['run_id']}:{state['revision']}:{kind}:{key}",
            "kind": kind,
            "payload": payload,
        }
        answer = interrupt(request)
        validate_answer(request, answer)
        return answer

    @staticmethod
    def log_decision(board: Blackboard, kind: str, answer: dict) -> Decision:
        decision = Decision(
            id=f"DEC-{len(board.decisions) + 1:03}",
            kind=kind,
            value=answer["value"],
            reason=answer["reason"],
            actor=answer["actor"],
            answered_at=answer["answered_at"],
        )
        board.decisions.append(decision)
        return decision

    def user_stop(
        self, state: GraphState, phase: str, board: Blackboard, answer: dict
    ) -> GraphState:
        self.log_decision(board, "stop", answer)
        board.outcome, board.stop_reason = "partial", "用户选择结束当前讨论并保留结果。"
        return self.commit(state, phase, board, halt=True)

    def frame(self, state: GraphState) -> GraphState:
        board = self.load(state, "frame")
        if board.frame is None:
            known = board.frame_hints.model_dump(exclude_none=True) if board.frame_hints else {}
            if self.model.config.chat_mode and len(known) < len(Frame.model_fields):
                known = {**frame_notes(board.idea, self.model), **known}
            if len(known) == len(Frame.model_fields):
                board.frame = Frame.model_validate(known)
            else:
                payload = {"idea": board.idea, "known": known}
                answer = self.ask(state, "frame", "initial", payload)
                if answer["value"] == {"stop": True}:
                    return self.user_stop(state, "frame", board, answer)
                board.frame = Frame.model_validate(answer["value"])
                self.log_decision(board, "frame", answer)
        return self.commit(state, "frame", board)

    def design_probe(self, state: GraphState) -> GraphState:
        board = self.load(state, "design_probe")
        probe = self.model.run("design_probe", "pm", context(board), DesignProbe)
        board.design_questions = probe.questions
        return self.commit(state, "design_probe", board)

    def design_clarify(self, state: GraphState) -> GraphState:
        board = self.load(state, "design_clarify")
        if board.design_questions:
            answer = self.ask(
                state,
                "design_clarify",
                "scope",
                {"questions": [item.model_dump(exclude_defaults=True) for item in board.design_questions]},
            )
            if answer["value"] == {"stop": True}:
                return self.user_stop(state, "design_clarify", board, answer)
            self.log_decision(board, "design_clarify", answer)
        return self.commit(state, "design_clarify", board)

    def technical_report(self, state: GraphState) -> GraphState:
        board = self.load(state, "technical_report")
        board.technical_report, needs_review = build_report(self.model, board)
        board.outcome = "partial" if needs_review else "completed"
        board.stop_reason = (
            "已生成完整设计草案，审查中还有事项需要复核。"
            if needs_review
            else "技术报告已生成。默认设计和待验证事项已标注。"
        )
        return self.commit(state, "technical_report", board)

    def project_probe(self, state: GraphState) -> GraphState:
        board = self.load(state, "project_probe")
        if not board.technical_report:
            raise ModelError("构建任务缺少已确认的技术报告。")
        probe = self.model.run(
            "project_probe",
            "tech",
            {**context(board), "已确认报告": board.technical_report.sections},
            DesignProbe,
        )
        board.project_questions = probe.questions
        return self.commit(state, "project_probe", board)

    def project_clarify(self, state: GraphState) -> GraphState:
        phase = "project_clarify"
        board = self.load(state, phase)
        for index, question in enumerate(board.project_questions):
            answer = self.ask(state, phase, str(index), {"questions": [question.model_dump()]})
            if answer["value"] == {"stop": True}:
                return self.user_stop(state, phase, board, answer)
            self.log_decision(board, phase, answer).value["question"] = question.question
        return self.commit(state, phase, board)

    def project_build(self, state: GraphState) -> GraphState:
        board = self.load(state, "project_build")
        build = board.project_build
        if not build or not board.technical_report:
            raise ModelError("构建任务缺少已确认的技术报告。")
        if build.report_hash != digest(board.technical_report.model_dump()):
            raise ModelError("技术报告与确认的版本不一致，请重新确认。")
        run_id = state["run_id"]

        def build_decision(value: dict) -> str:
            question = value.get("question")
            return f"{question}\n用户回复：{value['reply']}" if question else value["reply"]

        decisions = [
            build_decision(item.value) for item in board.decisions if item.kind == "project_clarify"
        ]

        def decide(question: DesignQuestion) -> str:
            kind = "project_review_clarify"
            self.store.set_stage(run_id, kind)
            answer = self.ask(state, kind, "review", {"questions": [question.model_dump()]})
            if answer["value"] == {"stop": True}:
                raise ProjectDecisionStopped(answer)
            decision = self.log_decision(board, kind, answer)
            decision.value["question"] = question.question
            return build_decision(decision.value)

        try:
            result = recover_project(build, run_id)
            if result is None:
                approved = board.model_copy(
                    update={"outcome": "partial" if build.source_report_partial else "completed"}
                )
                report = export_markdown(
                    approved, self.model.config, {"name": build.source_name}, []
                )
                files, notes, contract, acceptance = generate_files(
                    self.model, board.idea, report, verify_locally=build.verify_locally,
                    decisions=decisions, decide=decide,
                )
                self.store.set_stage(run_id, "project_build")
                self.store.check_cancel(run_id)
                result = write_project(
                    build, run_id, files, notes, report, contract=contract, acceptance=acceptance
                )
                self.store.activity(run_id, f"已写入 {len(result.files)} 个交付文件")
        except ProjectDecisionStopped as exc:
            return self.user_stop(state, "project_build", board, exc.answer)
        except (ValueError, OSError) as exc:
            raise ModelError(f"构建未能完成。{exc}") from exc
        board.project_result = result
        passed = result.acceptance is not None and result.acceptance.status == "passed"
        board.outcome = "completed" if passed and not result.review_notes else "partial"
        summary = result.acceptance.summary if result.acceptance else "该版本未记录运行验收。"
        board.stop_reason = f"项目已写入。{summary}"
        if result.review_notes:
            board.stop_reason += "源码审查仍有待处理事项。"
        return self.commit(state, "project_build", board)

    @staticmethod
    def divergence_input(board: Blackboard) -> dict:
        return {"idea": board.idea, "frame": board.frame.model_dump()}

    def divergence_node(self, role: Role):
        def run(state: GraphState) -> GraphState:
            board = self.load(state, "diverge")
            self.model.run("diverge", role, self.divergence_input(board), Divergence)
            return {"branches": [role]}

        return run

    def cluster(self, state: GraphState) -> GraphState:
        board = self.load(state, "cluster")
        if set(state["branches"]) != set(ROLES):
            raise RuntimeError("独立发散屏障未收到四个角色。")
        payload = self.divergence_input(board)
        replies = {role: self.model.run("diverge", role, payload, Divergence) for role in ROLES}
        rules.cluster(replies, board)
        if not board.options:
            board.outcome, board.stop_reason = "partial", "专家未形成可讨论候选。"
        return self.commit(state, "cluster", board, halt=board.outcome is not None)

    def shortlist(self, state: GraphState) -> GraphState:
        board = self.load(state, "shortlist")
        if len(board.options) <= 3:
            board.shortlist = [item.id for item in board.options]
        else:
            answer = self.ask(
                state,
                "shortlist",
                "options",
                {"options": [item.model_dump() for item in board.options]},
            )
            if answer["value"] == {"stop": True}:
                return self.user_stop(state, "shortlist", board, answer)
            board.shortlist = answer["value"]["option_ids"]
            self.log_decision(board, "shortlist", answer)
        return self.commit(state, "shortlist", board)

    def critique(self, state: GraphState) -> GraphState:
        board = self.load(state, "critique")
        payload = rules.critique_input(board)
        saved = saved_critiques(self.store, state["run_id"], payload)

        def review(role: Role) -> Critique:
            if role in saved:
                return saved[role]
            return self.model.run(
                "critique",
                role,
                rules.critique_targets(payload, role),
                Critique,
                lambda reply: rules.validate_critique(role, payload, reply),
            )

        board.critiques = self.parallel(review)
        rules.record_objections(board)
        return self.commit(state, "critique", board)

    def resolve(self, state: GraphState) -> GraphState:
        board = self.load(state, "resolve")
        for issue in board.objections:
            needs_answer = issue.status == "needs_human" or (
                issue.status == "needs_evidence"
                and issue.severity == "major"
                and not issue.decision_id
            )
            if not needs_answer:
                continue
            answer = self.ask(state, "resolve", issue.id, {"issue": issue.model_dump()})
            if answer["value"] == {"stop": True}:
                return self.user_stop(state, "resolve", board, answer)
            decision = self.log_decision(
                board,
                "resolve",
                {
                    **answer,
                    "value": {**answer["value"], "issue_id": issue.id},
                },
            )
            issue.decision_id = decision.id
            if issue.kind == "value":
                issue.status = "resolved"
                continue
            value = answer["value"]
            if value["verdict"] == "defer":
                continue
            evidence = Evidence(
                id=f"EVD-{len(board.evidence) + 1:03}",
                source=value["source"],
                excerpt=value["excerpt"],
                judgment=value["verdict"],
            )
            board.evidence.append(evidence)
            issue.evidence_id = evidence.id
            if evidence.judgment == "supported":
                issue.status = "resolved"
            for option in board.options:
                for assumption in option.assumptions:
                    if assumption.id == issue.assumption_id:
                        assumption.status = evidence.judgment
                        assumption.evidence_ids.append(evidence.id)
        return self.commit(state, "resolve", board)

    def score(self, state: GraphState) -> GraphState:
        board = self.load(state, "score")
        payload = {
            "idea": board.idea,
            "frame": board.frame.model_dump(),
            "options": [item.model_dump() for item in rules.shortlisted(board)],
            "objections": [item.model_dump() for item in board.objections],
            "decisions": [item.model_dump() for item in board.decisions],
            "evidence": [item.model_dump() for item in board.evidence],
            "rubric": rules.scoring_rubric(self.model.config.weights),
        }
        board.scores = self.parallel(
            lambda role: self.model.run(
                "score",
                role,
                payload,
                Scorecard,
                lambda reply: rules.validate_score(board, reply),
            )
        )
        return self.commit(state, "score", board)

    def select(self, state: GraphState) -> GraphState:
        board = self.load(state, "select")
        answer = self.ask(
            state,
            "select",
            "ranked",
            {
                "rankings": rules.rankings(board, self.model.config.weights),
                "sensitivity_changed": rules.sensitivity(board, self.model.config.weights),
                "blocking": {
                    item.id: rules.blocking_ids(board, item.id) for item in rules.shortlisted(board)
                },
            },
        )
        if answer["value"] == {"stop": True}:
            return self.user_stop(state, "select", board, answer)
        self.log_decision(board, "select", answer)
        return self.commit(state, "select", board)

    def draft(self, state: GraphState) -> GraphState:
        board = self.load(state, "draft")
        selection = next(item for item in reversed(board.decisions) if item.kind == "select")
        board.proposal = rules.compose(
            board, selection.value["option_id"], self.model.config.weights
        )
        return self.commit(state, "draft", board)

    def signoff(self, state: GraphState) -> GraphState:
        board = self.load(state, "signoff")
        proposal = board.proposal
        option_id = proposal.option_id
        blocking = rules.blocking_ids(board, option_id)
        snapshot = digest(proposal.model_dump())
        payload = {
            "proposal": proposal.model_dump(),
            "review_snapshot": snapshot,
            "blocking_ids": blocking,
            "conditions": rules.condition_status(board, option_id),
            "objections": [
                item.model_dump() for item in board.objections if item.option_id == option_id
            ],
        }
        board.signoffs = self.parallel(
            lambda role: self.model.run(
                "signoff",
                role,
                payload,
                Signoff,
                lambda reply: rules.validate_signoff(board, reply),
            )
        )
        board.review_snapshot_hash = snapshot
        selected = next(
            item
            for item in rules.rankings(board, self.model.config.weights)
            if item["option_id"] == option_id
        )
        reasons = []
        if blocking:
            reasons.append(f"重大异议或关键假设待处理：{', '.join(blocking)}")
        if selected["score"] is None:
            reasons.append("选定方向评分不完整")
        if selected["consensus"] < 0.75:
            reasons.append("有效支持未达到 75%")
        if not reasons and not rules.signoffs_pass(board):
            if proposal.revision < self.model.config.max_draft_revisions:
                return self.commit(state, "signoff", board, revise=True)
            reasons.append("未获得四位专家对当前版本的有效签署")
        board.outcome = "partial" if reasons else "completed"
        board.stop_reason = "；".join(reasons) if reasons else "选定方向通过共识与版本签署门槛。"
        return self.commit(state, "signoff", board, revise=False)

    def revise(self, state: GraphState) -> GraphState:
        board = self.load(state, "revise")
        revision = self.model.run(
            "revise",
            "pm",
            {
                "proposal": board.proposal.model_dump(),
                "signoffs": {
                    role: signoff.model_dump() for role, signoff in board.signoffs.items()
                },
            },
            DraftRevision,
        )
        board.proposal.revision += 1
        board.proposal.sections["修订与验证补充"] = (
            revision.rationale + "\n\n" + "\n".join(f"- {line}" for line in revision.changes)
        )
        board.signoffs = {}
        board.review_snapshot_hash = None
        return self.commit(state, "revise", board, revise=False)
