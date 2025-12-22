from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src import agent_mapper
from src.classes.agent_flow_state import TLCState
from src.classes.operation import OperationRouting
from src.classes.system_enum import AdmittanceState

checkpointer = MemorySaver()
talos_workflow = StateGraph(TLCState)

# region <router function placeholder>


talos_workflow.add_node("user_admittance", agent_mapper.user_admittance_node)
talos_workflow.add_node("intention_detection", agent_mapper.intention_detection_node)
talos_workflow.add_node("request_user_confirm", agent_mapper.request_user_confirm)
talos_workflow.add_node("planner", agent_mapper.planner_node)
talos_workflow.add_node("dispatch_todo", agent_mapper.dispatch_todo_node)
talos_workflow.add_node("execute_tlc", agent_mapper.execute_tlc_node)
talos_workflow.add_node("execute_unsupported", agent_mapper.execute_unsupported_node)
talos_workflow.add_node("advance_todo_cursor", agent_mapper.advance_todo_cursor_node)
# talos_workflow.add_node("checkpoint", agent_mapper.survey_inspect)

talos_workflow.add_edge(START, "user_admittance")
talos_workflow.add_conditional_edges(
    "user_admittance",
    agent_mapper.route_admittance,
    {
        AdmittanceState.YES.value: "intention_detection",
        AdmittanceState.NO.value: END,
    },
)
talos_workflow.add_edge("intention_detection", "request_user_confirm")
talos_workflow.add_conditional_edges(
    "request_user_confirm",
    agent_mapper.route_human_confirm_intention,
    {
        OperationRouting.PROCEED.value: "planner",
        OperationRouting.REVISE.value: "intention_detection",
    },
)

talos_workflow.add_edge("planner", "dispatch_todo")
talos_workflow.add_conditional_edges(
    "dispatch_todo",
    agent_mapper.route_next_todo,
    {
        "execute_tlc": "execute_tlc",
        "execute_unsupported": "execute_unsupported",
        "done": END,
    },
)
talos_workflow.add_edge("execute_tlc", "advance_todo_cursor")
talos_workflow.add_edge("execute_unsupported", "advance_todo_cursor")
talos_workflow.add_edge("advance_todo_cursor", "dispatch_todo")


# endregion

talos_agent = talos_workflow.compile(checkpointer=checkpointer)
# talos_agent = talos_workflow.compile()


def _export_workflow_png() -> None:
    output_path = Path(__file__).resolve().parents[1] / "static" / "workflow.png"
    talos_agent.get_graph().draw_png(output_file_path=str(output_path))


if __name__ == "__main__":
    _export_workflow_png()
