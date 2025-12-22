from __future__ import annotations

import uuid
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.types import Command

from src.main import talos_agent
from src.utils.logging_config import logger
from src.utils.tools import _pretty


def _prompt_bool(prompt: str) -> bool:
    while True:
        raw = input(prompt).strip().lower()
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please enter y/n")


def _assistant_reply_from_state(state: dict[str, Any]) -> str:
    def _get_output(obj: Any) -> Any:
        if obj is None:
            return None
        output = getattr(obj, "output", None)
        if output is not None:
            return output
        if isinstance(obj, dict):
            return obj.get("output")
        return None

    # Prefer returning the final planning output when present.
    plan = state.get("plan")
    plan_out = _get_output(plan)
    plan_steps = getattr(plan_out, "plan_steps", None)
    if isinstance(plan_steps, list) and plan_steps:
        lines: list[str] = ["Plan:"]
        for idx, s in enumerate(plan_steps, 1):
            title = getattr(s, "title", None)
            executor = getattr(s, "executor", None)
            status = getattr(s, "status", None)
            status_value = getattr(status, "value", status)
            out = getattr(s, "output", None)
            if out is not None:
                lines.append(f"{idx}. {title} ({executor}) [{status_value}] -> {out}")
            else:
                lines.append(f"{idx}. {title} ({executor}) [{status_value}]")
        return "\n".join(lines)

    # Fallback to intermediate states.
    admittance_state = state.get("admittance_state")
    value = getattr(admittance_state, "value", str(admittance_state)).lower()

    if value == "no":
        admittance = state.get("admittance")
        feedback = getattr(_get_output(admittance), "feedback", None)
        return f"Admittance: NO\n{feedback}" if feedback else "Admittance: NO"

    if value == "yes":
        intention = state.get("intention")
        output = _get_output(intention)
        return f"Admittance: YES\nIntention: {output}" if output is not None else "Admittance: YES"

    return "Done."


def _run_once(conversation: list[AnyMessage], thread_id: str) -> dict[str, Any]:
    config: Any = {"configurable": {"thread_id": thread_id}}

    next_input: Any = {
        "user_input": list(conversation),
        "messages": list(conversation),
    }

    while True:
        interrupt_payload: Any | None = None
        last_state: dict[str, Any] | None = None

        for state in talos_agent.stream(next_input, config=config, stream_mode="values"):
            last_state = state
            if "__interrupt__" in state:
                interrupt_payload = state["__interrupt__"]
                break

        if interrupt_payload is None:
            return last_state or {}

        print("\n--- HUMAN REVIEW REQUIRED ---")
        print(_pretty(interrupt_payload))

        approve = _prompt_bool("Approve? (y/n): ")
        comment = ""
        if approve:
            comment = input("Optional comment (enter to skip): ").strip()
        else:
            comment = input("Edit the user input (required to revise): ").strip()

        next_input = Command(resume={"approval": approve, "comment": comment})


def main() -> None:
    """Run the Talos LangGraph in an interactive terminal session."""
    thread_id = f"console-{uuid.uuid4()}"
    print(f"Thread: {thread_id}")
    logger.info(f"==== Session Started with Thread: {thread_id} ====")

    conversation: list[AnyMessage] = []

    while True:
        user_text = input("\nUser input (blank to quit): ").strip()
        if not user_text:
            return

        conversation.append(HumanMessage(content=user_text))

        print(f"Converstaion so far {conversation}")

        final_state = _run_once(conversation=conversation, thread_id=thread_id)

        updated_conversation: Any | None = final_state.get("messages") or final_state.get("user_input")
        if isinstance(updated_conversation, list) and updated_conversation:
            conversation = list(updated_conversation)

        conversation.append(AIMessage(content=_assistant_reply_from_state(final_state)))

        print("\n--- FINAL STATE ---")
        print(_pretty(final_state))
        logger.info(f"==== Session Ended with Thread: {thread_id} ==== \n")


if __name__ == "__main__":
    main()
