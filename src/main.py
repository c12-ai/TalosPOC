from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src import node_mapper
from src.classes.agent_flow_state import TLCState
from src.classes.system_enum import AdmittanceState
from src.functions.planner import planner_subgraph


def create_talos_workflow() -> StateGraph:
    """Create the Talos workflow definition (uncompiled)."""
    workflow = StateGraph(TLCState)

    # region <router function placeholder>
    workflow.add_node("user_admittance", node_mapper.user_admittance_node)
    workflow.add_node("intention_detection", node_mapper.intention_detection_node)
    workflow.add_node("dispatcher", node_mapper.dispatcher_node)
    workflow.add_node("bottom_line_handler", node_mapper.bottom_line_handler_node)
    workflow.add_node("request_user_confirm", node_mapper.request_user_confirm)

    workflow.add_node("dispatch_execute", node_mapper.dispatcher_execute)
    workflow.add_node("consulting_handler", node_mapper.consulting_handler_node)
    workflow.add_node("query_handler", node_mapper.query_handler_node)

    workflow.add_node("dispatch_todo", node_mapper.dispatch_todo_node)
    workflow.add_node("planner", planner_subgraph.compiled)
    workflow.add_node("prepare_tlc_step", node_mapper.prepare_tlc_step_node)
    workflow.add_node("finalize_tlc_step", node_mapper.finalize_tlc_step_node)
    workflow.add_node("execute_unsupported", node_mapper.execute_unsupported_node)
    workflow.add_node("advance_todo_cursor", node_mapper.route_advance_todo_cursor_node)
    # workflow.add_node("checkpoint", node_mapper.survey_inspect)

    # region <draw>

    workflow.add_edge(START, "user_admittance")
    workflow.add_conditional_edges(
        "user_admittance",
        node_mapper.route_admittance,
        {
            AdmittanceState.YES.value: "intention_detection",
            AdmittanceState.NO.value: "bottom_line_handler",
        },
    )

    # Go to bottom line handler if not admitted and END flow
    workflow.add_edge("bottom_line_handler", END)

    workflow.add_conditional_edges(
        "intention_detection",
        node_mapper.dispatcher_node,
        {
            "executor": "dispatch_execute",
            "query": "query_handler",  # Placeholder, not implemented yet
            "consulting": "consulting_handler",  # Placeholder, not implemented yet
        },
    )

    workflow.add_edge("consulting_handler", END)
    workflow.add_edge("query_handler", END)

    # Executing Workflow
    workflow.add_edge("planner", "dispatch_todo")

    workflow.add_conditional_edges(
        "dispatch_todo",
        node_mapper.route_next_todo,
        {
            "prepare_tlc_step": "prepare_tlc_step",
            "execute_unsupported": "execute_unsupported",
            "done": END,
        },
    )

    workflow.add_edge("prepare_tlc_step", "finalize_tlc_step")
    workflow.add_edge("finalize_tlc_step", "advance_todo_cursor")
    workflow.add_edge("execute_unsupported", "advance_todo_cursor")
    workflow.add_edge("advance_todo_cursor", "dispatch_todo")

    # endregion
    return workflow


def create_talos_agent(*, checkpointer: MemorySaver | None = None) -> Any:
    """Compile the Talos workflow into a runnable graph."""
    workflow = create_talos_workflow()
    if checkpointer is None:
        checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


talos_agent = create_talos_agent()


def _export_workflow_png() -> None:
    output_path = Path(__file__).resolve().parents[1] / "static" / "workflow.png"
    talos_agent.get_graph().draw_png(output_file_path=str(output_path))


if __name__ == "__main__":
    _export_workflow_png()
