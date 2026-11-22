from __future__ import annotations

import hashlib
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from se_agents import deliberation as rules
from se_agents.chat import question_text
from se_agents.engineering_ui import render_engineering
from se_agents.human import APPROVAL_QUESTIONS, DEVELOPMENT_QUESTIONS
from se_agents.models import configured_model
from se_agents.presentation import (
    CALL_STATUS,
    EVIDENCE,
    ISSUE_KIND,
    ISSUE_STATUS,
    SIGNOFF,
    STAGES,
    STANCE,
    STATUS,
    decision_text,
    ranking_rows,
    zh,
)
from se_agents.reporting import export_json, export_markdown, technical_body
from se_agents.schemas import DIMENSIONS, ROLE_NAMES
from se_agents.service import Busy, Service
from se_agents.task_execution import execute_task
from se_agents.workflow import saved_critiques

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", encoding="utf-8-sig")
st.set_page_config(page_title="软件开发工作台", page_icon="⌘", layout="wide")
st.html("""
<style>
[data-testid="stAppDeployButton"], [data-testid="stMainMenu"] {display: none;}
[data-testid="stHeader"] {background: transparent;}
[data-testid="stMainBlockContainer"] {max-width: 1040px; padding: 3.5rem 2.5rem 5rem;}
[data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.16);}
[data-testid="stSidebar"] [data-testid="stHeading"] h2 {font-size: 1rem; letter-spacing: -.025em;}
[data-testid="stSidebar"] button[kind="tertiary"] {justify-content: flex-start; text-align: left;}
[data-testid="stSidebar"] button[kind="tertiary"] p {
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
[data-testid="stHeading"] h1 {font-size: clamp(1.8rem, 4vw, 2.65rem); letter-spacing: -.035em;}
[data-testid="stChatMessage"] {border-radius: 16px; padding: .7rem .85rem;}
[data-testid="stChatMessage"][aria-label*="你"] {background: rgba(128,128,128,.08);}
[data-testid="stChatInput"] {border-radius: 18px; box-shadow: 0 2px 8px rgba(0,0,0,.06);}
[data-testid="stExpander"] details {border-radius: 12px;}
[data-testid="stTabs"] [role="tab"] {padding-inline: .9rem;}
@media (max-width: 680px) {
    [data-testid="stMainBlockContainer"] {padding: 3rem 1rem 4rem;}
}
</style>
""")


@st.cache_resource
def get_workspace(data_dir: str):
    service = Service(Path(data_dir))
    return service, ThreadPoolExecutor(max_workers=1)


service, pool = get_workspace(os.environ.get("SE_AGENTS_DATA_DIR", str(ROOT / "data")))


def schedule(run_id: str, chat: tuple[str, str] | None = None) -> None:
    st.session_state.next_task = run_id
    st.session_state.active_task = run_id
    st.session_state.pending_chat = chat
    st.session_state.work = pool.submit(service.execute, run_id, chat=chat)
    st.rerun()


def schedule_task(run_id: str, revision: int, task_id: str) -> None:
    st.session_state.next_task = run_id
    st.session_state.active_task = run_id
    st.session_state.pending_chat = None
    st.session_state.work = pool.submit(execute_task, service, run_id, revision, task_id)
    st.rerun()


def schedule_approval(run_id: str, request: dict, value: dict, reason: str) -> None:
    st.session_state.next_task = run_id
    st.session_state.active_task = run_id
    st.session_state.pending_chat = None
    answer = {"request_id": request["id"], "value": value, "reason": reason}
    st.session_state.work = pool.submit(service.execute, run_id, answer)
    st.rerun()


def activity_panel(run: dict, *, live: bool = False) -> None:
    elapsed = (
        run["elapsed"] + max(0, time.time() - run["started"]) if run["started"] else run["elapsed"]
    )
    minutes, seconds = divmod(int(elapsed), 60)
    duration = f"{minutes} 分 {seconds} 秒" if minutes else f"{seconds} 秒"
    events = service.store.activities(run["id"])
    calls = service.store.call_log(run["id"])
    if not events and not calls and not live:
        return
    stage = f" · {STAGES.get(run['stage'], '准备中')}" if live else ""
    st.caption(f"{'已运行' if live else '用时'} {duration}{stage}")
    key = f"activity-{run['id']}"
    saved = f"{key}-expanded"
    if key not in st.session_state and saved in st.session_state:
        st.session_state[key] = st.session_state[saved]

    def remember_activity_state() -> None:
        st.session_state[saved] = st.session_state[key]

    panel = st.expander("活动记录", key=key, on_change=remember_activity_state)
    if not panel.open:
        return
    with panel:
        for item in reversed(events[-20:] if live else events):
            stamp = time.strftime("%H:%M:%S", time.localtime(item["created"]))
            spent = f" · {item['elapsed']:.1f} 秒" if item["elapsed"] is not None else ""
            st.text(f"{stamp}  {item['detail']}{spent}")
        if calls:
            st.caption(f"模型请求 {len(calls)} 次")
            for call in reversed(calls[-3:] if live else calls):
                stage = STAGES.get(call["task"].split(":")[0], "模型请求")
                status = CALL_STATUS.get(call["status"], "已结束")
                spent = f" · {call['elapsed']:.1f} 秒" if call["elapsed"] is not None else ""
                st.text(f"{stage} · {status}{spent}")


@st.fragment(run_every="1s")
def running(run_id: str) -> None:
    run = service.store.run(run_id)
    future = st.session_state.get("work")
    if (future is not None and future.done()) or (future is None and run["status"] != "running"):
        st.rerun()
    activity_panel(run, live=True)
    render_engineering(service, run_id, service.store.board(run_id), schedule_task, live=True)
    if st.button("停止任务", key=f"cancel-{run_id}"):
        service.store.request_cancel(run_id)
        st.rerun()


def confirm_delete(run_id: str, name: str) -> None:
    st.warning(f"删除「{name}」及其论证记录？删除后无法恢复。")
    delete, cancel = st.columns(2)
    if delete.button("确认删除", key=f"confirm-delete-{run_id}", type="primary"):
        try:
            service.delete_run(run_id)
        except (ValueError, Busy) as exc:
            st.error(str(exc))
        else:
            st.session_state.pop("confirm_task", None)
            st.session_state.next_task = None
            st.session_state.notice = "任务已删除。"
            st.rerun()
    if cancel.button("返回", key=f"keep-{run_id}"):
        st.session_state.pop("confirm_task", None)
        st.rerun()


def task_controls(run_id: str, run: dict) -> None:
    if run["status"] == "running":
        return
    with st.expander("管理任务"):
        with st.form(f"rename-{run_id}"):
            name = st.text_input("任务名称", value=run["name"], max_chars=60)
            submitted = st.form_submit_button("保存名称")
        if submitted:
            try:
                service.store.rename(run_id, name)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state.notice = "任务名称已更新。"
                st.rerun()
        if st.button("删除任务", key=f"delete-{run_id}"):
            st.session_state.confirm_task = run_id
        if st.session_state.get("confirm_task") == run_id:
            confirm_delete(run_id, run["name"])


def message(kind: str, text: str) -> None:
    user = kind == "user"
    with st.chat_message(
        "你" if user else "开发助手", avatar=":material/person:" if user else ":material/terminal:"
    ):
        st.markdown(text)


def read_idea(idea, upload=None) -> str:
    if upload is None:
        return idea
    if upload.size > 1_048_576:
        raise ValueError("文件超过 1 兆字节。")
    try:
        content = upload.getvalue().decode("utf-8-sig")
    except UnicodeError as exc:
        raise ValueError("无法读取文件，请将内容粘贴到输入框。") from exc
    return "\n\n".join(part for part in (idea.strip(), content.strip()) if part)


def new_chat(name: str) -> None:
    st.title("软件工程工作流")
    choices = {"完整流程：从想法到任务编排": "full",
               "课堂作业二：设计规格到任务编排": "planning",
               "技术报告与项目构建": "legacy"}
    mode = choices[st.selectbox("选择起点", list(choices), key="workflow-choice")]
    parallel = st.number_input("最大并行任务数", min_value=1, max_value=4, value=2, step=1)
    if mode == "planning":
        st.write("上传已有设计规格，确认模块与需求后，由 PM Agent 拆分任务。可编辑依赖并重排甘特图。")
    elif mode == "full":
        st.write("Agent 独立构思、比较并复核候选，汇总一个推荐方案供你确认。")
    else:
        st.write("输入想法生成技术报告，经确认后可构建项目。")
    config = None
    try:
        config = configured_model()
        config = config.model_copy(update={
            "workflow_mode": mode, "max_parallel_tasks": int(parallel),
            "chat_mode": mode == "full", "max_calls": 100,
            "max_tokens": 800_000, "max_output_tokens": 8192,
        })
    except ValueError as exc:
        st.warning(str(exc))
    if config:
        st.caption(f"当前模型为 {config.model}。发送后将调用模型。")
    if mode == "planning":
        with st.form("import-design-spec"):
            upload = st.file_uploader("选择设计规格文件", type=["md", "txt"], max_upload_size=1)
            idea = st.text_area("或粘贴设计规格", max_chars=60_000, height=150)
            submitted = st.form_submit_button("导入设计规格", type="primary", disabled=config is None)
        if not submitted:
            return
    else:
        submission = st.chat_input(
            "输入软件想法，按回车发送", key="new_idea", max_chars=10000,
            accept_file=True, file_type=["txt", "md"], max_upload_size=1,
            disabled=config is None,
        )
        if not submission:
            return
        if isinstance(submission, str):
            idea, upload = submission, None
        else:
            idea = submission.text or ""
            upload = submission.files[0] if submission.files else None
    try:
        schedule(service.create(read_idea(idea, upload), config, name=name))
    except ValueError as exc:
        st.error(zh(str(exc)))


def details(run_id: str, board, config, calls: list[dict]) -> None:
    with st.expander("候选与评分"):
        if board.scores:
            st.table(ranking_rows(rules.rankings(board, config.weights)))
        for option in board.options:
            st.markdown(f"**{zh(option.title)}**")
            st.write(zh(option.approach))
            for claim in board.claims:
                if claim.option_id == option.id:
                    st.caption(f"{ROLE_NAMES[claim.author]} · {zh(claim.rationale)}")
            for role, card in board.scores.items():
                for row in card.assessments:
                    if row.option_id == option.id:
                        grades = " · ".join(
                            f"{DIMENSIONS[key]} {score if score is not None else '待核查'}"
                            for key, score in row.scores.model_dump().items()
                        )
                        st.caption(f"{ROLE_NAMES[role]} · {STANCE[row.stance]} · {grades}")
                        st.caption(zh(row.rationale))
    with st.expander("分歧与证据"):
        if not board.critiques:
            saved = (
                saved_critiques(service.store, run_id, rules.critique_input(board))
                if board.shortlist and board.frame
                else {}
            )
            if saved:
                st.caption(f"{len(saved)} 位专家已通过校验，以下意见待汇总。")
                for role, review in saved.items():
                    for issue in review.objections:
                        st.write(
                            f"{ROLE_NAMES[role]} · {zh(issue.option_id)} · {ISSUE_KIND[issue.kind]}"
                        )
                        st.caption(zh(f"{issue.reason} · {issue.action}"))
                if not any(review.objections for review in saved.values()):
                    st.caption("已通过的审查暂未提出异议。")
            else:
                st.caption("专家审查尚未完成，暂时没有可显示的异议。")
        elif not board.objections:
            st.caption("专家审查已完成，未提出异议。")
        for issue in board.objections:
            st.write(
                f"{zh(issue.option_id)} · {ISSUE_KIND[issue.kind]} · {ISSUE_STATUS[issue.status]}"
            )
            st.caption(zh(f"{issue.reason} · {issue.action}"))
        for evidence in board.evidence:
            st.write(f"{EVIDENCE[evidence.judgment]} · {evidence.source}")
            st.caption(zh(evidence.excerpt))
        if not board.evidence:
            st.caption("暂无用户补充的核查材料。")
    with st.expander("过程记录"):
        stages = [STAGES.get(row["stage"], "任务更新") for row in service.store.revisions(run_id)]
        st.caption(" → ".join(stages))
        for decision in board.decisions:
            st.write(f"{STAGES[decision.kind]} · {decision_text(decision)}")
        st.caption(f"模型请求 {len(calls)} 次")
        if calls:
            st.table(
                [
                    {
                        "阶段": STAGES.get(row["task"].split(":")[0], "模型分析"),
                        "角色": ROLE_NAMES.get(row["role"], "专家"),
                        "结果": CALL_STATUS.get(row["status"], "状态未知"),
                        "校验详情": row["diagnostic"] or "—",
                    }
                    for row in calls
                ]
            )


future = st.session_state.get("work")
if future is not None and future.done():
    st.session_state.pop("work")
    st.session_state.pop("pending_chat", None)
    try:
        future.result()
    except (ValueError, Busy) as exc:
        st.error(zh(str(exc)))
    except Exception:
        st.error("任务未能继续，请稍后恢复。")
active = st.session_state.get("work") is not None

with st.sidebar:
    st.header("⌘ 软件开发工作台")
    history = {row["id"]: row for row in service.store.list_runs()}
    if st.button("新建任务", icon=":material/edit_square:", width="stretch", disabled=active):
        st.session_state.next_task = None
        st.rerun()
    if "next_task" in st.session_state:
        st.session_state.task_selection = st.session_state.pop("next_task")
    if st.session_state.get("task_selection") not in history:
        st.session_state.task_selection = None
    selected = st.session_state.task_selection
    st.caption("任务")
    search = st.text_input("搜索任务", placeholder="搜索任务", label_visibility="collapsed")
    for task_id, task in history.items():
        if search.strip().casefold() not in task["name"].casefold():
            continue
        if st.button(
            task["name"],
            key=f"select-{task_id}",
            width="stretch",
            disabled=active,
            type="secondary" if task_id == selected else "tertiary",
            icon=":material/folder_open:"
            if task_id == selected
            else ":material/chat_bubble_outline:",
        ):
            st.session_state.next_task = task_id
            st.rerun()
    if selected is None:
        with st.expander("更多设置"):
            name = st.text_input("任务名称（选填）", max_chars=60)

if notice := st.session_state.pop("notice", None):
    st.success(notice)
if st.session_state.get("last_opened_task") != selected:
    st.session_state.pop("confirm_task", None)
    st.session_state.last_opened_task = selected

if active:
    selected = st.session_state.active_task
if selected is None:
    new_chat(name)
    st.stop()

run_id = selected
run = service.store.run(run_id)
board = service.store.board(run_id, run["revision"])
config = service.config(run_id)
st.title(run["name"])
status = "构建中" if board.project_build and run["status"] == "running" else STATUS[run["status"]]
st.caption(f"{status} · {'历史模拟记录' if config.mode == 'mock' else config.model}")
if not active:
    task_controls(run_id, run)
transcript = service.store.chat_messages(run_id)
thread = st.container(border=False, key=f"thread-{run_id}")
with thread:
    message("user", board.idea)
    if board.decisions and (not transcript or board.project_build):
        with st.expander("此前的决定"):
            for decision in board.decisions:
                st.write(f"{STAGES[decision.kind]} · {decision_text(decision)}")
    for entry in transcript:
        message(entry["kind"], entry["content"])

if active or run["status"] == "running":
    with thread:
        if pending_chat := st.session_state.get("pending_chat"):
            request_id, content = pending_chat
            if not any(
                entry["request_id"] == request_id
                and entry["kind"] == "user"
                and entry["content"] == content.strip()
                for entry in transcript
            ):
                message("user", content)
        with st.chat_message("开发助手", avatar=":material/terminal:"):
            running(run_id)
    st.stop()

if board.engineering is None and not config.technical_mode and st.button(
    "用原始想法生成技术报告", type="primary"
):
    try:
        schedule(service.create(board.idea, configured_model(), name=run["name"]))
    except ValueError as exc:
        st.error(zh(str(exc)))

plan_failed = bool(
    board.engineering and run["status"] == "failed"
    and run["stage"] in {"task_decompose", "plan_review"}
)
if plan_failed:
    st.error(zh(run["error"] or "任务计划生成或评审失败。"))
    if st.button("继续任务", key=f"resume-plan-{run_id}"):
        schedule(run_id)

request = service.request(run_id) if config.mode != "mock" else None
approval_slot = render_engineering(service, run_id, board, schedule_task, request=request)

if request:
    if not any(
        row["kind"] == "question" and row["request_id"] == request["id"] for row in transcript
    ):
        with thread:
            message("assistant", question_text(request))
    if request["kind"] in APPROVAL_QUESTIONS:
        proposal_question = request["kind"] == "proposal_confirm"
        if proposal_question:
            option = next(item for item in board.options if item.id == board.proposal.option_id)
            st.subheader("推荐建设方案")
            st.markdown(f"**{option.title}**")
            if board.proposal_review_issues:
                st.warning("这份推荐方案还有待处理的评审条件。")
                for issue in board.proposal_review_issues:
                    st.write(zh(issue))
            for title, content in board.proposal.sections.items():
                with st.expander(title, expanded=title in {
                    "范围与非目标", "核心功能", "决策依据", "未决问题与分歧",
                }):
                    st.markdown(zh(content))
            with st.expander("各角色的复核结论"):
                for role, signoff in board.signoffs.items():
                    st.write(f"{ROLE_NAMES[role]} · {SIGNOFF[signoff.verdict]} · {zh(signoff.reason)}")
        confirmation_labels = {
            "proposal_confirm": "采纳方案并整理需求",
            "design_confirm": "确认设计规格并拆任务",
            "plan_confirm": (
                "复核并确认任务计划" if request["payload"].get("review_required") else "确认任务计划"
            ),
        }
        with (approval_slot or st.container()), st.form(f"approval-{request['id']}"):
            feedback = st.text_area("修改意见，退回时必填", max_chars=2400)
            risk_ack = st.checkbox("我已查看遗留问题与风险") if (
                request["payload"].get("needs_risk_ack")
            ) else False
            confirm, reject = st.columns(2)
            approved = confirm.form_submit_button(
                confirmation_labels.get(request["kind"], "确认当前版本并继续"),
                type="primary",
                disabled=bool(request["payload"].get("approval_blocked")),
            )
            rejected = reject.form_submit_button("退回重新论证" if proposal_question else "退回修改")
        if approved or rejected:
            if rejected and not feedback.strip():
                st.error("请填写修改意见。")
            elif approved and request["payload"].get("needs_risk_ack") and not risk_ack:
                st.error("请先查看并确认遗留风险。")
            else:
                schedule_approval(run_id, request, {
                    "approve": approved, "snapshot": request["payload"]["snapshot"],
                    "feedback": feedback, "risk_ack": risk_ack,
                }, feedback.strip() or "已审阅并确认当前版本")
        activity_panel(run)
        if board.options:
            details(run_id, board, config, service.store.call_log(run_id))
        st.stop()
    if request["kind"] in DEVELOPMENT_QUESTIONS:
        questions = request["payload"]["questions"]
        with thread:
            if len(questions) == 1:
                for index, option in enumerate(questions[0].get("options", []), 1):
                    if st.button(option, key=f"choice-{request['id']}-{index}"):
                        schedule(run_id, (request["id"], f"选 {index}"))
            if st.button("你决定", key=f"delegate-{request['id']}"):
                schedule(run_id, (request["id"], "你决定"))
    if reply := st.chat_input("回复上面的问题", key=f"reply-{run_id}", max_chars=2400):
        schedule(run_id, (request["id"], reply))
    activity_panel(run)
    if board.options:
        details(run_id, board, config, service.store.call_log(run_id))
    st.stop()

if not plan_failed and (board.stop_reason or run["error"]):
    with thread:
        message("assistant", zh(run["error"] or board.stop_reason))
if (
    not plan_failed and config.mode != "mock"
    and run["status"] in {"queued", "interrupted", "failed"}
    and st.button("继续任务")
):
    schedule(run_id)

calls = service.store.call_log(run_id)
activity_panel(run)
if build := board.project_build:
    st.subheader("软件项目构建")
    st.caption(f"按「{build.source_name}」的技术报告第 {build.source_revision} 版构建。")
    st.code(build.directory, language=None)
    if result := board.project_result:
        files_tab, checks_tab = st.tabs(["项目文件", "验收结果"])
        with checks_tab:
            st.caption("打开项目里的 README.md，按说明配置、启动和运行测试。")
            if check := result.acceptance:
                badge = {"passed": "验收通过", "failed": "验收失败", "not_run": "验收未运行"}
                st.write(f"**{badge[check.status]}** · {check.summary}")
                if check.output:
                    with st.expander("运行验收日志"):
                        st.code(check.output, language=None)
            if result.review_notes:
                st.warning("源码审查仍有待处理事项，请在验收前查看。")
                for note in result.review_notes:
                    st.text(note)
        with files_tab:
            name = st.selectbox(
                "项目文件",
                sorted(result.files),
                key=f"build-file-{run_id}",
                index=sorted(result.files).index("README.md") if "README.md" in result.files else 0,
            )
            if name:
                folder = Path(build.directory).resolve()
                path = folder / name
                try:
                    if path.resolve(strict=True) != path or path.stat().st_size > 360_000:
                        raise ValueError
                    content = path.read_bytes()
                    if hashlib.sha256(content).hexdigest() != result.files[name]:
                        raise ValueError
                except (OSError, ValueError, RuntimeError):
                    st.caption("文件已移动或修改，可在目标目录查看当前版本。")
                else:
                    syntax = {
                        ".py": "python",
                        ".js": "javascript",
                        ".json": "json",
                        ".html": "html",
                        ".css": "css",
                        ".md": "markdown",
                        ".sql": "sql",
                    }
                    st.code(content.decode("utf-8"), language=syntax.get(path.suffix))
                    st.download_button(
                        "下载此文件", content, file_name=path.name, mime="text/plain"
                    )
elif board.technical_report and board.engineering is None:
    with thread, st.chat_message("开发助手", avatar=":material/terminal:"):
        st.subheader("软件开发技术报告")
        title = "目标、角色与版本范围"
        st.write(technical_body(title, board.technical_report.sections.get(title, "")))
    with st.expander("查看完整技术报告"):
        for title, content in board.technical_report.sections.items():
            st.markdown(f"### {title}\n\n{technical_body(title, content)}")
    if board.engineering is None and st.button("按当前报告生成任务编排", type="primary"):
        try:
            schedule(service.create_planning(run_id, run["revision"]))
        except (ValueError, Busy) as exc:
            st.error(str(exc))
    if board.engineering is None and run["status"] in {"completed", "partial"}:
        with st.expander("确认方案并构建项目"):
            if run["status"] == "partial":
                st.warning("这版报告标为待复核草案。请先查看其中的待处理事项。")
            with st.form(f"build-{run_id}"):
                st.caption("工作台将在指定目录生成源码、测试和启动说明。目录不存在时会自动创建。")
                directory = st.text_input("项目目录", placeholder="~/Downloads/我的软件项目")
                verify = st.checkbox(
                    "运行本机验收",
                    help="Python 项目复用工作台的 Python。会在临时目录执行生成的代码。"
                    "只在信任生成内容时启用，本机运行仍可能访问网络和本机文件。",
                )
                approved = st.checkbox("我已审阅并确认这版技术方案")
                if st.form_submit_button("开始构建项目", type="primary"):
                    if not approved:
                        st.error("请先确认这版技术方案。")
                    else:
                        try:
                            build_id = service.create_build(
                                run_id, run["revision"], directory, verify_locally=verify
                            )
                        except (ValueError, OSError) as exc:
                            st.error(zh(str(exc)))
                        else:
                            schedule(build_id)
elif board.proposal:
    with thread, st.chat_message("开发助手", avatar=":material/terminal:"):
        st.subheader("论证方案")
        option = next(item for item in board.options if item.id == board.proposal.option_id)
        st.write(zh(option.title))
        summary = {"目标用户", "价值主张", "核心功能"}
        for title, content in board.proposal.sections.items():
            if title in summary:
                st.markdown(f"**{zh(title)}**\n\n{zh(content)}")
        with st.expander("完整方案"):
            for title, content in board.proposal.sections.items():
                st.markdown(f"**{zh(title)}**\n\n{zh(content)}")
        if board.signoffs:
            with st.expander("专家签署"):
                for role, signoff in board.signoffs.items():
                    st.write(
                        f"{ROLE_NAMES[role]} · {SIGNOFF[signoff.verdict]} · {zh(signoff.reason)}"
                    )

if not board.project_build and (
    board.technical_report or board.proposal or board.options or board.outcome
):
    filename = re.sub(r'[\\/:*?"<>|]+', "－", run["name"])
    report, data = st.columns(2)
    report_title = (
        "任务编排" if board.engineering and board.engineering.plan
        else "技术报告" if board.technical_report
        else "设计规格" if board.engineering and board.engineering.design else "论证摘要"
    )
    report.download_button(
        f"下载{report_title}",
        export_markdown(board, config, run, calls),
        file_name=f"{filename}－{report_title}.md",
        mime="text/markdown",
        type="primary",
    )
    data.download_button(
        "导出数据",
        export_json(board, config, run, calls),
        file_name=f"{filename}－分析数据.json",
        mime="application/json",
    )
if not board.project_build and (
    board.options or (board.engineering is None and (board.decisions or board.technical_report))
):
    details(run_id, board, config, calls)
