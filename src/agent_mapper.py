"""
# Wrap functions in Agent, turn into Langgraph nodes.

Also including other Langgraph specific function e.g., interrupt().
"""

from typing import Any

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.types import interrupt

from src.agents.tlc_agent import TLCAgent
from src.classes.agent_flow_state import TLCState
from src.classes.operation import OperationResponse, OperationResume, OperationRouting
from src.classes.system_enum import AdmittanceState, ExecutionStatusEnum
from src.classes.system_state import TODO, HumanApproval, IntentionDetectionFin, PlanningAgentOutput, UserAdmittance
from src.functions.admittance import WatchDogAgent
from src.functions.human_interaction import HumanInLoop
from src.functions.intention_detection import IntentionDetectionAgent
from src.functions.planner import Planner
from src.utils.logging_config import logger

watch_dog = WatchDogAgent()
human_interact_agent = HumanInLoop()
intention_detect_agent = IntentionDetectionAgent()
planner_agent = Planner()
tlc_agent = TLCAgent()


# region <utils>


def _ensure_messages(state: TLCState) -> list[AnyMessage]:
    return list(state.messages or state.user_input or [])


def _dump_messages_for_human_review(messages: list[AnyMessage]) -> list[dict[str, Any]]:
    return [{"type": m.__class__.__name__, "content": getattr(m, "content", None)} for m in messages]


def _apply_human_revision(messages: list[AnyMessage], revised_text: str) -> list[AnyMessage]:
    revised_text = revised_text.strip()
    if not revised_text:
        return list(messages)

    updated = list(messages)
    for idx in range(len(updated) - 1, -1, -1):
        if isinstance(updated[idx], HumanMessage):
            updated[idx] = HumanMessage(content=revised_text)
            return updated

    updated.append(HumanMessage(content=revised_text))
    return updated


# endregion


# region <dependent functions>


def request_user_confirm(state: TLCState) -> dict[str, OperationResponse[str, HumanApproval]]:
    """Request user confirmation on the intention detection result."""
    logger.info("Requesting user confirmation for intention detection result.")

    intention = state.intention
    if intention is None:
        raise ValueError("Missing 'intention' in state")

    print("intention", intention)

    reviewed = intention.output
    logger.info("Intention detection result (reviewed): {}", reviewed)

    messages = _ensure_messages(state)

    interrupt_payload = {
        "question": "Do you approve this intention? If not, edit the user input in 'comment'.",
        "intention": reviewed.model_dump(mode="json"),
        "current_user_input": _dump_messages_for_human_review(messages),
    }
    payload = interrupt(interrupt_payload)
    resume = OperationResume(**payload)

    if resume.approval:
        logger.info("User approved via interrupt. payload={}", payload)
    else:
        logger.warning("User rejected via interrupt. payload={}", payload)

    confirmation = human_interact_agent.post_human_confirmation(
        reviewed=reviewed,
        approval=resume.approval,
        comment=resume.comment,
    )

    updates: dict[str, Any] = {"human_confirmation": confirmation}

    edited_text = (resume.comment or "").strip()
    if not resume.approval and edited_text:
        revised_messages = _apply_human_revision(messages, edited_text)
        updates["messages"] = revised_messages
        updates["user_input"] = revised_messages

    return updates


def user_admittance_node(
    state: TLCState,
) -> dict[str, OperationResponse[list[AnyMessage], UserAdmittance] | AdmittanceState]:
    """If user input is within domain and capacity of this Agent, return YES, otherwise NO."""
    messages = _ensure_messages(state)
    res = watch_dog.run(user_input=messages)

    logger.debug(
        "Admittance decision: within_domain={}, within_capacity={}",
        res.output.within_domain,
        res.output.within_capacity,
    )

    return {
        "admittance": res,
        "admittance_state": AdmittanceState.YES
        if res.output.within_domain and res.output.within_capacity
        else AdmittanceState.NO,
    }


def intention_detection_node(state: TLCState) -> dict[str, OperationResponse[list[AnyMessage], IntentionDetectionFin]]:
    """Run intention detection on user input."""
    messages = _ensure_messages(state)
    logger.info("Running intention_detection_node with {} messages", len(messages))
    res = intention_detect_agent.run(user_input=messages)
    logger.debug(f"Intention detection output: {res}")

    return {
        "intention": res,
    }


def planner_node(state: TLCState) -> dict[str, Any]:
    """Generate a TODO plan based on the latest user messages (post-confirmation)."""
    messages = _ensure_messages(state)
    logger.info("Running planner_node with {} messages", len(messages))
    res = planner_agent.run(user_input=messages)
    logger.debug("Planner output: {}", res)

    return {"plan": res, "plan_cursor": 0}


def _get_plan_output(state: TLCState) -> PlanningAgentOutput:
    plan = state.plan
    if plan is None:
        raise ValueError("Missing 'plan' in state")
    return plan.output


def _get_current_todo(state: TLCState) -> tuple[int, TODO]:
    cursor = int(state.plan_cursor)
    return cursor, _get_plan_output(state).todo_list[cursor]


def dispatch_todo_node(_state: TLCState) -> dict[str, Any]:
    """No-op node used as a stable point for conditional routing to per-todo executors."""
    return {}


def route_next_todo(state: TLCState) -> str:
    """Route to the next todo executor based on the current todo name; END when done."""
    plan_out = _get_plan_output(state)
    cursor = int(state.plan_cursor)
    if cursor >= len(plan_out.todo_list):
        logger.info("All todos executed. cursor={} total={}", cursor, len(plan_out.todo_list))
        return "done"

    todo = plan_out.todo_list[cursor]
    name = (todo.name or "").strip().lower()
    logger.info("Routing todo cursor={} name={}", cursor, name)

    if "tlc" in name:
        return "execute_tlc"

    return "execute_unsupported"


def execute_tlc_node(state: TLCState) -> dict[str, Any]:
    """Execute TLC-related todo using TLCAgent (compound extraction + placeholder MCP lookup)."""
    cursor, todo = _get_current_todo(state)

    todo.status = ExecutionStatusEnum.IN_PROGRESS
    messages = _ensure_messages(state)
    res = tlc_agent.run(user_input=messages)

    todo.output = {
        "agent": "tlc_agent",
        "result": res.output.model_dump(mode="json"),
    }
    todo.status = ExecutionStatusEnum.COMPLETED

    logger.info("Executed TLC todo cursor={} id={} name={}", cursor, todo.id, todo.name)
    return {"plan": state.plan}


def execute_unsupported_node(state: TLCState) -> dict[str, Any]:
    """Mark todo as on-hold when no executor exists yet."""
    cursor, todo = _get_current_todo(state)
    todo.output = {
        "agent": "executor",
        "note": f"No executor implemented for task name '{todo.name}'.",
    }
    todo.status = ExecutionStatusEnum.ON_HOLD
    logger.warning("No executor for todo cursor={} id={} name={}", cursor, todo.id, todo.name)
    return {"plan": state.plan}


def advance_todo_cursor_node(state: TLCState) -> dict[str, Any]:
    """Advance plan_cursor to the next todo."""
    cursor = int(state.plan_cursor)
    return {"plan_cursor": cursor + 1}


# endregion


# region <router>


def route_admittance(state: TLCState) -> str:
    """Select the next node based on the admittance decision."""
    decision = state.admittance_state or AdmittanceState.NO
    logger.info("Routing based on admittance_state={}", decision.value)
    return decision.value


def route_human_confirm_intention(state: TLCState) -> str:
    """
    Route Human confirmation response after intention detection.

    Proceed to next node if confirmed, otherwise go back to intention detection (Revise).
    """
    human_confirmation = state.human_confirmation

    if human_confirmation and human_confirmation.output.approval:
        logger.info("Human confirmation approved, proceeding.")
        return OperationRouting.PROCEED.value

    logger.info("Human confirmation rejected or missing, revising.")
    return OperationRouting.REVISE.value


# endregion
