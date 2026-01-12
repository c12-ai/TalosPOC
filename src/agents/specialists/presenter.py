"""
Presenter utilities for generating user-facing messages.

Keep this module thin: it accepts `list[AnyMessage]` and uses a prompt to generate
the final or HITL-review copy. Upstream nodes should avoid directly appending
internal logs/drafts into `messages`.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from copilotkit.langgraph import copilotkit_customize_config
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig

from src.utils.models import PLANNER_MODEL

ReviewKind = Literal["plan_review", "tlc_confirm"]


FINAL_SYSTEM_PROMPT = """
你是 Talos (实验室智能助手) 的 Presenter，总结最近一步的操作，并简短的描述给用户50字以内的答复。

要求:
- 你只输出给用户看的最终答复, 不要输出内部日志、工具调用细节、调试信息。
- 你会收到一段对话消息 (Human/AI/System). 其中可能包含一条以 "CONTEXT_JSON:" 开头的 SystemMessage, 里面是结构化上下文。
- 如果信息不足, 明确列出需要用户确认/补充的点。
- 输出简洁、结构清晰, 优先使用要点列表与步骤。
""".strip()

REVIEW_SYSTEM_PROMPT = """
你是 Talos 的 HITL Presenter。

要求:
- 你只输出一个"需要用户确认/审阅"的短提示, 不要泄露内部日志或推理。
- 你会收到:
  1) 对话消息 (Human/AI/System)
  2) 一段 review JSON (SystemMessage, 以 "REVIEW_JSON:" 开头) 包含 kind 与 args
- 输出应告诉用户: 你在让他确认什么 + 下一步如何操作 (确认/拒绝/修改)。
""".strip()


def _invoke(
    messages: list[AnyMessage],
    *,
    system_prompt: str,
    config: RunnableConfig | None = None,
) -> str:
    resp = PLANNER_MODEL.invoke(
        [SystemMessage(content=system_prompt), *messages],
        config=config,
    )
    content = getattr(resp, "content", "")
    return str(content or "").strip()


def present_final(
    messages: list[AnyMessage], config: RunnableConfig | None = None
) -> str:
    """Generate the final user-visible answer from messages."""
    invoke_config = (
        copilotkit_customize_config(config, emit_messages=True)
        if config
        else copilotkit_customize_config(emit_messages=True)
    )
    return (
        _invoke(messages, system_prompt=FINAL_SYSTEM_PROMPT, config=invoke_config)
        or "已完成。"
    )


def present_review(
    messages: list[AnyMessage],
    *,
    kind: ReviewKind,
    args: dict[str, Any],
    config: RunnableConfig | None = None,
) -> str:
    """Generate review prompt shown before interrupt."""
    invoke_config = (
        copilotkit_customize_config(config, emit_messages=False)
        if config
        else copilotkit_customize_config(emit_messages=False)
    )
    review_json = json.dumps({"kind": kind, "args": args}, ensure_ascii=False)
    review_msg = SystemMessage(content=f"REVIEW_JSON:\n{review_json}")
    return (
        _invoke(
            [review_msg, *messages],
            system_prompt=REVIEW_SYSTEM_PROMPT,
            config=invoke_config,
        )
        or "请确认是否继续。"
    )
