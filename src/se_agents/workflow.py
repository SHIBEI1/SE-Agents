from __future__ import annotations

import operator
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, TypedDict, TypeVar

from langgraph.graph import END, START, StateGraph
from langgraph.types import Overwrite, interrupt

from . import deliberation as rules
from .chat import frame_notes
from .human import validate_answer
from .models import ModelError, ModelRunner
from .planning import (
    DesignSpecification,
    DevelopmentPlan,
    PlanEdit,
    PlanReview,
    ScheduledTask,
    TaskProgress,
    edited_plan,
    fingerprint,
    refresh_ready,
    schedule,
    validate_plan,
    validate_spec,
)
from .project import generate_files, recover_project, write_project
from .reporting import export_markdown
from .schemas import (
    ROLES,
    AssumptionReview,
    Blackboard,
    Critique,
    CritiqueBatch,
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
    TradeoffDecisions,
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
    plan_action: str


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
            "requirements_spec": self.requirements_spec,
            "requirements_confirm": self.requirements_confirm,
            "engineering_design": self.engineering_design,
            "design_confirm": self.design_confirm,
            "task_decompose": self.task_decompose,
            "plan_review": self.plan_review,
            "plan_confirm": self.plan_confirm,
            "orchestrate": self.orchestrate,
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
            "proposal_confirm": self.proposal_confirm,
            "reopen_discussion": self.reopen_discussion,
        }
        for name, fn in stages.items():
            graph.add_node(name, fn)
        branches = [f"diverge_{role}" for role in ROLES]
        for role, branch in zip(ROLES, branches, strict=True):
            graph.add_node(branch, self.divergence_node(role))
        graph.add_conditional_edges(
            START,
            self.initial_stage,
            ["design_probe", "frame", "project_probe", "engineering_design"],
        )
        graph.add_edge("design_probe", "design_clarify")
        graph.add_conditional_edges(
            "design_clarify",
            lambda state: END if state.get("halt") else "technical_report",
            [END, "technical_report"],
        )
        graph.add_conditional_edges(
            "technical_report",
            lambda state: "engineering_design" if self.flow_mode(state) == "full" else END,
            ["engineering_design", END],
        )
        graph.add_edge("requirements_spec", "requirements_confirm")
        graph.add_conditional_edges(
            "requirements_confirm",
            lambda state: END if state.get("halt") else (
                "requirements_spec" if state.get("revise") else "technical_report"
            ),
            [END, "requirements_spec", "technical_report"],
        )
        graph.add_edge("engineering_design", "design_confirm")
        graph.add_conditional_edges(
            "design_confirm",
            lambda state: END if state.get("halt") else (
                "engineering_design" if state.get("revise") else "task_decompose"
            ),
            [END, "engineering_design", "task_decompose"],
        )
        graph.add_edge("task_decompose", "plan_review")
        graph.add_conditional_edges(
            "plan_review",
            lambda state: END if state.get("halt") else (
                "task_decompose" if state.get("revise") else
                "orchestrate" if state.get("plan_action") == "approved" else "plan_confirm"
            ),
            [END, "task_decompose", "plan_confirm", "orchestrate"],
        )
        graph.add_conditional_edges(
            "plan_confirm",
            lambda state: END if state.get("halt") else (
                "task_decompose" if state.get("revise") else {
                    "saved": "plan_confirm", "review": "plan_review",
                }.get(state.get("plan_action"), "orchestrate")
            ),
            [END, "task_decompose", "orchestrate", "plan_confirm", "plan_review"],
        )
        graph.add_edge("orchestrate", END)
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
            self.after_signoff,
            ["revise", "proposal_confirm", "requirements_spec", END],
        )
        graph.add_edge("revise", "signoff")
        graph.add_conditional_edges(
            "proposal_confirm",
            lambda state: END if state.get("halt") else (
                "reopen_discussion" if state.get("revise") else "requirements_spec"
            ),
            [END, "reopen_discussion", "requirements_spec"],
        )
        graph.add_edge("reopen_discussion", "frame")
        return graph.compile(checkpointer=checkpointer)

    def initial_stage(self, state: GraphState) -> str:
        if self.store.board(state["run_id"], state["revision"]).project_build:
            return "project_probe"
        if self.flow_mode(state) == "planning":
            return "engineering_design"
        if self.flow_mode(state) == "full":
            return "frame"
        return "design_probe" if self.model.config.technical_mode else "frame"

    def flow_mode(self, state: GraphState) -> str:
        engineering = self.store.board(state["run_id"], state["revision"]).engineering
        if engineering is None:
            return "legacy"
        return "planning" if engineering.source_text else "full"

    def after_signoff(self, state: GraphState) -> str:
        if state.get("revise"):
            return "revise"
        if self.model.config.internal_deliberation:
            return "proposal_confirm"
        board = self.store.board(state["run_id"], state["revision"])
        if self.flow_mode(state) == "full" and board.outcome == "completed":
            return "requirements_spec"
        return END

    def proposal_confirm(self, state: GraphState) -> GraphState:
        board = self.load(state, "proposal_confirm")
        snapshot = digest(board.proposal.model_dump())
        answer = self.ask(state, "proposal_confirm", snapshot[:12], {
            "snapshot": snapshot,
            "needs_risk_ack": bool(board.proposal_review_issues),
        })
        if answer["value"] == {"stop": True}:
            return self.user_stop(state, "proposal_confirm", board, answer)
        self.log_decision(board, "proposal_confirm", answer)
        return self.commit(state, "proposal_confirm", board, revise=not answer["value"]["approve"])

    def reopen_discussion(self, state: GraphState) -> GraphState:
        board = self.load(state, "reopen_discussion")
        decision = board.decisions[-1]
        board.engineering.feedback.append(decision.value["feedback"])
        board.divergence_notes.clear()
        board.claims.clear()
        board.options.clear()
        board.shortlist.clear()
        board.critiques.clear()
        board.objections.clear()
        board.evidence.clear()
        board.scores.clear()
        board.signoffs.clear()
        board.proposal = None
        board.review_snapshot_hash = None
        board.proposal_review_issues.clear()
        board.outcome = board.stop_reason = None
        return self.commit(state, "reopen_discussion", board, branches=Overwrite([]), revise=False)

    def requirements_spec(self, state: GraphState) -> GraphState:
        board = self.load(state, "requirements_spec")
        board.outcome = None
        engineering = board.engineering
        engineering.requirements = self.model.run(
            "requirements_spec", "user",
            {"想法": board.idea, "选定方案": board.proposal.model_dump(),
             "尚未解决的评审条件": board.proposal_review_issues,
             "修改意见": engineering.feedback},
            DesignSpecification, validate_spec,
        )
        engineering.requirements_approval = None
        return self.commit(state, "requirements_spec", board, halt=False, revise=False)

    def confirm_engineering(self, state: GraphState, kind: str, document) -> GraphState:
        board = self.load(state, kind)
        engineering = board.engineering
        expected = fingerprint(document)
        risk = kind == "design_confirm" and engineering.design_review_pending
        answer = self.ask(state, kind, expected[:12], {
            "snapshot": expected, "document": document.model_dump(), "needs_risk_ack": risk,
            "approval_blocked": False,
        })
        if answer["value"] == {"stop": True}:
            return self.user_stop(state, kind, board, answer)
        self.log_decision(board, kind, answer)
        approved = answer["value"]["approve"]
        field = {"requirements_confirm": "requirements_approval",
                 "design_confirm": "design_approval"}[kind]
        setattr(engineering, field, expected if approved else None)
        if not approved:
            engineering.feedback.append(answer["value"]["feedback"])
        return self.commit(state, kind, board, halt=False, revise=not approved)

    def requirements_confirm(self, state: GraphState) -> GraphState:
        board = self.store.board(state["run_id"], state["revision"])
        return self.confirm_engineering(state, "requirements_confirm", board.engineering.requirements)

    def engineering_design(self, state: GraphState) -> GraphState:
        board = self.load(state, "engineering_design")
        engineering = board.engineering
        baseline = engineering.requirements

        def check(spec):
            validate_spec(spec)
            if baseline and spec.requirements != baseline.requirements:
                raise ValueError("设计规格必须保留已确认需求，不能改写需求基线。")

        engineering.design = self.model.run(
            "engineering_design", "tech",
            {"输入设计规格": engineering.source_text or board.technical_report.sections,
             "已确认需求": baseline.model_dump() if baseline else None,
             "修改意见": engineering.feedback},
            DesignSpecification, check,
        )
        engineering.design_approval = None
        return self.commit(state, "engineering_design", board, halt=False, revise=False)

    def design_confirm(self, state: GraphState) -> GraphState:
        board = self.store.board(state["run_id"], state["revision"])
        return self.confirm_engineering(state, "design_confirm", board.engineering.design)

    def task_decompose(self, state: GraphState) -> GraphState:
        board = self.load(state, "task_decompose")
        engineering = board.engineering
        if engineering.design_approval != fingerprint(engineering.design):
            raise ValueError("当前设计规格尚未确认。")
        engineering.plan = self.model.run(
            "task_decompose", "pm",
            {"设计规格": engineering.design.model_dump(),
             "需求到模块的映射": {
                 req.id: [module.id for module in engineering.design.modules
                          if req.id in module.requirement_ids]
                 for req in engineering.design.requirements
             },
             "上一版任务索引": [
                 {"id": task.id, "title": task.title, "depends_on": task.depends_on}
                 for task in engineering.plan.tasks
             ] if engineering.plan else [],
             "上一版评审意见": engineering.plan_review.model_dump() if engineering.plan_review else {},
             "最大并行任务数": engineering.max_parallel,
             "修改意见": [
                 item.value["feedback"] for item in board.decisions
                 if item.kind == "plan_confirm" and item.value.get("feedback")
             ]},
            DevelopmentPlan, lambda plan: validate_plan(plan, engineering.design),
        )
        engineering.plan_approval = None
        engineering.plan_review = None
        engineering.schedule_rows = []
        board.stop_reason = None
        return self.commit(state, "task_decompose", board, halt=False, revise=False, plan_action="")

    def plan_review(self, state: GraphState) -> GraphState:
        board = self.load(state, "plan_review")
        engineering = board.engineering
        ids = {t.id for t in engineering.plan.tasks}

        def check(review):
            if any(f.task_id is not None and f.task_id not in ids for f in review.findings):
                raise ValueError("评审必须引用实际任务编号。")

        engineering.plan_review = self.model.run(
            "plan_review", "devil",
            {"设计规格": engineering.design.model_dump(), "任务计划": engineering.plan.model_dump()},
            PlanReview, check,
        )
        major = any(f.severity == "major" for f in engineering.plan_review.findings)
        manual = state.get("plan_action") == "review"
        revise = major and not manual and engineering.plan_revisions < 2
        if revise:
            engineering.plan_revisions += 1
        approved = manual and not major
        if approved:
            engineering.plan_approval = fingerprint(engineering.plan)
        return self.commit(state, "plan_review", board, halt=False, revise=revise,
                           plan_action="approved" if approved else "")

    def plan_confirm(self, state: GraphState) -> GraphState:
        board = self.load(state, "plan_confirm")
        engineering = board.engineering
        plan = engineering.plan
        expected = fingerprint(plan)
        review = engineering.plan_review
        answer = self.ask(state, "plan_confirm", expected[:12], {
            "snapshot": expected, "document": plan.model_dump(), "needs_risk_ack": False,
            "approval_blocked": bool(review and any(f.severity == "major" for f in review.findings)),
            "review_required": review is None,
        })
        value = answer["value"]
        if value == {"stop": True}:
            return self.user_stop(state, "plan_confirm", board, answer)
        self.log_decision(board, "plan_confirm", answer)
        if value.get("action") == "edit_plan":
            edit = PlanEdit.model_validate(value["edit"])
            engineering.plan = edited_plan(plan, edit)
            engineering.max_parallel = edit.max_parallel
            engineering.plan_approval = None
            if engineering.plan != plan:
                engineering.plan_review = None
            engineering.schedule_rows = [
                ScheduledTask.model_validate(row)
                for row in schedule(engineering.plan, engineering.max_parallel)
            ]
            return self.commit(state, "plan_confirm", board, halt=False, revise=False,
                               plan_action="saved")
        if not value["approve"]:
            engineering.plan_approval = None
            engineering.feedback.append(value["feedback"])
            engineering.plan_revisions = 0
            return self.commit(state, "plan_confirm", board, halt=False, revise=True, plan_action="")
        if review is not None:
            engineering.plan_approval = expected
        return self.commit(state, "plan_confirm", board, halt=False, revise=False,
                           plan_action="review" if review is None else "approved")

    def orchestrate(self, state: GraphState) -> GraphState:
        board = self.load(state, "orchestrate")
        engineering = board.engineering
        if engineering.plan_approval != fingerprint(engineering.plan):
            raise ValueError("当前任务计划尚未确认。")
        engineering.tasks = {task.id: TaskProgress() for task in engineering.plan.tasks}
        engineering.schedule_rows = [
            ScheduledTask.model_validate(row)
            for row in schedule(engineering.plan, engineering.max_parallel)
        ]
        refresh_ready(engineering)
        board.outcome, board.stop_reason = (
            "completed", "任务分解与执行编排已生成。请在任务看板启动工作并逐项验收。",
        )
        return self.commit(state, "orchestrate", board, halt=False, revise=False)

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
        if board.engineering:
            board.engineering.design_review_pending = needs_review
            board.outcome = None
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
        payload = {"idea": board.idea, "frame": board.frame.model_dump()}
        if board.engineering and board.engineering.feedback:
            payload["feedback"] = board.engineering.feedback
        return payload

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
        if self.model.config.internal_deliberation or len(board.options) <= 3:
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
        options = rules.shortlisted(board)
        saved = saved_critiques(self.store, state["run_id"], rules.critique_input(board))
        batch_size = 3
        payloads = [
            rules.critique_input(board, options[offset:offset + batch_size])
            for offset in range(0, len(options), batch_size)
        ]

        def review(role: Role) -> Critique:
            if role in saved:
                return saved[role]
            batches = [
                self.model.run(
                    "critique", role, rules.critique_targets(payload, role), CritiqueBatch,
                    lambda reply, batch=payload: rules.validate_critique(
                        role, batch, reply, allow_missing_checks=True
                    ),
                )
                for payload in payloads
            ]
            return Critique(
                reviews=[item for batch in batches for item in batch.reviews],
                objections=[item for batch in batches for item in batch.objections],
                checked_assumptions=[item for batch in batches for item in batch.checked_assumptions],
            )

        board.critiques = self.parallel(review)
        critic = board.critiques["devil"]
        checked = set(critic.checked_assumptions)
        checked.update(o.assumption_id for o in critic.objections if o.kind == "factual")
        assumption_reviews = {}
        for option in options:
            for assumption in option.assumptions:
                if assumption.critical and assumption.id not in checked:
                    assumption_reviews[assumption.id] = self.model.run(
                        "critique_assumption", "devil",
                        {"goal": board.frame.goal, "constraints": board.frame.constraints,
                         "option": {"title": option.title, "approach": option.approach},
                         "assumption": assumption.model_dump(include={
                             "id", "statement", "verification_plan",
                         })},
                        AssumptionReview,
                    )
        critic.checked_assumptions = [
            a.id for option in options for a in option.assumptions if a.critical
        ]
        rules.record_objections(board, assumption_reviews)
        return self.commit(state, "critique", board)

    def resolve(self, state: GraphState) -> GraphState:
        board = self.load(state, "resolve")
        if self.model.config.internal_deliberation:
            self.resolve_internally(board)
            return self.commit(state, "resolve", board)
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

    def resolve_internally(self, board: Blackboard) -> None:
        issues = {issue.id: issue for issue in board.objections if issue.kind != "factual"}
        if not issues:
            return

        def check(result: TradeoffDecisions) -> None:
            ids = [item.issue_id for item in result.decisions]
            if len(ids) != len(set(ids)) or set(ids) - issues.keys():
                raise ValueError("取舍决定必须引用一条有效且不重复的非事实异议。")

        result = self.model.run(
            "internal_resolution", "pm",
            {**rules.critique_input(board),
             "cross_reviews": {role: item.model_dump() for role, item in board.critiques.items()},
             "issues": [item.model_dump() for item in issues.values()]},
            TradeoffDecisions, check,
        )
        for item in result.decisions:
            decision = self.log_decision(board, "resolve", {
                "value": {"issue_id": item.issue_id, "choice": item.choice},
                "reason": item.rationale, "actor": "agent", "answered_at": time.time(),
            })
            issue = issues[item.issue_id]
            issue.status, issue.decision_id = "resolved", decision.id
        for issue in issues.values():
            if not issue.decision_id:
                issue.status = "retained"

    def score(self, state: GraphState) -> GraphState:
        board = self.load(state, "score")
        options = rules.shortlisted(board)
        batch_size = 3

        def assess(role: Role) -> Scorecard:
            assessments = []
            for offset in range(0, len(options), batch_size):
                batch = options[offset:offset + batch_size]
                option_ids = [item.id for item in batch]
                objections = [o for o in board.objections if o.option_id in option_ids]
                decision_ids = {o.decision_id for o in objections}
                # Legacy resolve decisions record evidence indirectly through objections.
                evidence_ids = {o.evidence_id for o in objections} | {
                    ref for option in batch for a in option.assumptions for ref in a.evidence_ids
                }
                payload = {
                    "idea": board.idea,
                    "frame": board.frame.model_dump(),
                    "options": [item.model_dump() for item in batch],
                    "allowed_conditions": {
                        ref: rules.condition_status(board, ref) for ref in option_ids
                    },
                    "objections": [item.model_dump() for item in objections],
                    "decisions": [
                        item.model_dump() for item in board.decisions if item.id in decision_ids
                    ],
                    "evidence": [
                        item.model_dump() for item in board.evidence if item.id in evidence_ids
                    ],
                    "rubric": rules.scoring_rubric(self.model.config.weights),
                }
                result = self.model.run(
                    "score", role, payload, Scorecard,
                    lambda reply, ids=option_ids: rules.validate_score(board, reply, ids),
                )
                assessments.extend(result.assessments)
            return Scorecard(assessments=assessments)

        board.scores = self.parallel(assess)
        return self.commit(state, "score", board)

    def select(self, state: GraphState) -> GraphState:
        board = self.load(state, "select")
        if self.model.config.internal_deliberation:
            ranked = rules.rankings(board, self.model.config.weights)
            selected = min(ranked, key=lambda row: (
                bool(rules.blocking_ids(board, row["option_id"])),
                row["score"] is None or row["consensus"] < 0.75,
                -(row["score"] or 0), -row["consensus"], row["option_id"],
            ))
            score = selected["score"] if selected["score"] is not None else "资料不足"
            self.log_decision(board, "select", {
                "value": {"option_id": selected["option_id"]},
                "reason": (f"优先推荐无重大阻塞且达到评分和支持门槛的方案，"
                           f"综合评分 {score}，有效支持 {selected['consensus']:.0%}。"),
                "actor": "orchestrator", "answered_at": time.time(),
            })
            return self.commit(state, "select", board)
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
        if self.model.config.internal_deliberation:
            payload["tradeoff_decisions"] = [
                item.model_dump() for item in board.decisions
                if item.id in {issue.decision_id for issue in board.objections
                               if issue.option_id == option_id}
            ]
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
        if self.model.config.internal_deliberation:
            board.proposal_review_issues = reasons
            board.outcome = board.stop_reason = None
            return self.commit(state, "signoff", board, revise=False)
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
