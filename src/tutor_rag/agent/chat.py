"""终端对话入口：支持一次性提问与交互式 REPL。"""

from __future__ import annotations

from ..config import get_settings
from ..trace import setup_logging
from .factory import build_tutor_agent
from .graph import DEFAULT_THREAD_ID, TutorAgent
from .llm import build_chat_model, model_label

HELP_TEXT = (
    "命令：/help 帮助  /clear 清空当前会话  "
    "/model 查看当前模型  /model <provider:model> 切换模型  /exit 退出"
)


def _run_once(agent: TutorAgent, text: str, thread_id: str, stream: bool) -> None:
    if stream:
        for piece in agent.stream(text, thread_id):
            print(piece, end="", flush=True)
        print()
    else:
        print(agent.invoke(text, thread_id).content)


def _handle_model_command(agent: TutorAgent, argument: str) -> None:
    if not argument:
        print(f"当前模型: {model_label(agent.model)}")
        return
    try:
        agent.switch_model(build_chat_model(model_name=argument))
        print(f"已切换模型: {argument}（历史对话保留）")
    except Exception as exc:
        print(f"切换失败: {exc}")


def run_chat(
    model_name: str | None = None,
    thread_id: str = DEFAULT_THREAD_ID,
    once: str | None = None,
    stream: bool = True,
    retrieval: bool = True,
) -> int:
    settings = get_settings()
    setup_logging(settings.paths.logs_dir / f"chat-{thread_id}.log")

    try:
        agent = build_tutor_agent(settings=settings, retrieval=retrieval)
        if model_name:
            agent.switch_model(build_chat_model(model_name=model_name))
    except Exception as exc:
        print(f"模型初始化失败: {exc}")
        print(
            "请检查 .env：DEEPSEEK_API_KEY 是否已配置、"
            "TUTOR_RAG_LLM_MODEL 是否正确（如 deepseek:deepseek-flash）"
        )
        return 1

    if once is not None:
        try:
            _run_once(agent, once, thread_id, stream)
        except Exception as exc:
            print(f"调用失败: {exc}")
            return 1
        return 0

    mode_text = "已启用教材检索" if agent.retrieval_service else "未启用教材检索"
    print(f"初中家教对话（{mode_text}，输入 /help 查看命令）")
    while True:
        try:
            text = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text in {"/exit", "/quit"}:
            break
        if text == "/help":
            print(HELP_TEXT)
            continue
        if text == "/clear":
            agent.reset(thread_id)
            print("已清空当前会话")
            continue
        if text == "/model" or text.startswith("/model "):
            _handle_model_command(agent, text[len("/model"):].strip())
            continue
        try:
            _run_once(agent, text, thread_id, stream)
        except Exception as exc:
            print(f"调用失败: {exc}")
    return 0
