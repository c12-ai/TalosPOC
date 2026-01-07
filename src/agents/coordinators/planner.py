"""
Planner Agent: Generate a plan based on user input. The output should be a sequence of TODOs: a set of restricted commands.

1. We need to have a registry of executors, which is a dictionary of ExecutorKey and a callable function.
2. Planner only output executor that exist in the registry. we may need to verify that.

Guardrails:
1. Plan freeze, once approved, we will have a none changeable plan hash, if user want to change / edit it, we need to re-enter the audit workflow.
2. 幂等/续跑: 基于 step.id + status, 已完成的不重复跑; 失败标记 on_hold 并记录错误

TODO: Clean Code
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AnyMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from src.models.core import PlanningAgentOutput, PlanStep
from src.models.enums import ExecutionStatusEnum, ExecutorKey
from src.models.operation import OperationInterruptPayload, OperationResponse
from src.models.planner import PlannerGraphState
from src.presenter import present_review
from src.utils.logging_config import logger
from src.utils.messages import MsgUtils
from src.utils.models import PLANNER_MODEL
from src.utils.PROMPT import PLANNER_SYSTEM_PROMPT
from src.utils.tools import coerce_operation_resume


class PlannerSubgraph:
    """Planning subgraph: generate_plan -> plan_review (approve/revise loop)."""

    def __init__(self, *, with_checkpointer: bool = False) -> None:
        """
        Build and compile the internal LangGraph subgraph.

        Args:
            with_checkpointer: If True, compiles with MemorySaver for standalone usage with HITL.
                               If False (default), parent graph's checkpointer will be used.

        """
        logger.info("PlannerSubgraph initialized with model={}", PLANNER_MODEL)

        subgraph = StateGraph(PlannerGraphState)
        subgraph.add_node("generate_plan", self._generate_plan)
        subgraph.add_node("interrupt_plan_review", self._interrupt_plan_review)

        subgraph.add_edge(START, "generate_plan")
        subgraph.add_edge("generate_plan", "interrupt_plan_review")
        subgraph.add_conditional_edges("interrupt_plan_review", self._route_plan_review, {"revise": "generate_plan", "approved": END})

        checkpointer = MemorySaver() if with_checkpointer else None
        self.compiled = subgraph.compile(checkpointer=checkpointer)

        # Keep the LLM agent initialization here (after compile) to match the TLCAgent pattern and avoid unnecessary init cost when only graph
        # structure is inspected.
        self._agent = create_agent(
            model=PLANNER_MODEL,
            response_format=ToolStrategy[PlanningAgentOutput](PlanningAgentOutput),
            system_prompt=PLANNER_SYSTEM_PROMPT,
        )

    def _plan(self, *, user_input: list[AnyMessage]) -> OperationResponse[list[AnyMessage], PlanningAgentOutput]:
        """
        Generate a plan based on user messages.

        This is a small seam used by the planning node. Tests can monkeypatch this method to make the workflow deterministic without replacing the
        whole subgraph.
        """
        start_time = datetime.now()
        logger.info("PlannerSubgraph._plan triggered with {} messages", len(user_input))

        # NOTE: Placeholder planning logic.
        plan_steps = [
            PlanStep(
                id="1",
                title="TLC IPC monitoring (extract compounds + lookup Rf)",
                executor=ExecutorKey.TLC_AGENT,
                args={},
                # TLC execution includes its own form confirmation stage, so avoid double HITL.
                requires_human_approval=False,
                status=ExecutionStatusEnum.NOT_STARTED,
                output=None,
            ),
        ]
        plan = PlanningAgentOutput(plan_steps=plan_steps, plan_hash="MOCK HASH")

        end_time = datetime.now()
        return OperationResponse[list[AnyMessage], PlanningAgentOutput](
            operation_id=f"planner_{uuid4()}",
            input=user_input,
            output=plan,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
        )

    @staticmethod
    def _ensure_messages(state: PlannerGraphState) -> list[AnyMessage]:
        return MsgUtils.ensure_messages(state)  # type: ignore[arg-type]

    @staticmethod
    def _ensure_work_messages(state: PlannerGraphState) -> list[AnyMessage]:
        # Use the latest human-only messages as execution context.
        # (For now, revisions are appended by presenter nodes into `messages`.)
        return MsgUtils.only_human_messages(MsgUtils.ensure_messages(state))  # type: ignore[arg-type]

    def _generate_plan(self, state: PlannerGraphState) -> dict[str, Any]:
        work_messages = self._ensure_work_messages(state)
        logger.info("PlannerSubgraph.generate_plan with {} messages", len(work_messages))

        res = self._plan(user_input=work_messages)
        plan_out = res.output

        return {
            "plan": res,
            "plan_cursor": 0,
            "plan_approved": False,
            "messages": MsgUtils.append_thinking(
                MsgUtils.ensure_messages(state),  # type: ignore[arg-type]
                f"[planner] plan_created plan_hash={plan_out.plan_hash} steps={len(plan_out.plan_steps)}",
            ),
        }

    def _interrupt_plan_review(self, state: PlannerGraphState) -> dict[str, Any]:
        """
        Interrupt + apply resume for plan review.

        This mirrors the TLCAgent HITL pattern: build the review payload, interrupt, then either approve (END) or append a revision message and
        loop back to `generate_plan`.
        """
        if state.plan is None:
            raise ValueError("Missing 'plan' before interrupt_plan_review")

        plan_out = state.plan.output
        steps_preview = [
            {
                "id": s.id,
                "title": s.title,
                "executor": str(s.executor),
                "requires_human_approval": s.requires_human_approval,
                "status": s.status.value,
                "args": s.args,
            }
            for s in plan_out.plan_steps
        ]
        args = {"plan_hash": plan_out.plan_hash, "plan_steps": steps_preview}
        payload = OperationInterruptPayload(
            message=present_review(
                MsgUtils.only_human_messages(MsgUtils.ensure_messages(state)),  # type: ignore[arg-type]
                kind="plan_review",
                args=args,
            ),
            args=args,
        )

        raw = interrupt(payload.model_dump(mode="json"))
        resume = coerce_operation_resume(raw)

        if resume.approval:
            return {"plan_approved": True}

        # On reject: append the revision instruction (if provided) and clear old plan so parent graph won't treat it as executable.
        messages = MsgUtils.ensure_messages(state)  # type: ignore[arg-type]
        edited_text = (resume.comment or "").strip()
        if edited_text:
            messages = MsgUtils.append_user_message(messages, edited_text)

        return {"messages": messages, "plan": None, "plan_cursor": 0, "plan_approved": False}

    @staticmethod
    def _route_plan_review(state: PlannerGraphState) -> str:
        return "approved" if state.plan_approved else "revise"


if __name__ == "__main__":
    from pathlib import Path

    planner = PlannerSubgraph()

    output_path = Path(__file__).resolve().parents[3] / "assets" / "planner_graph.png"
    planner.compiled.get_graph(xray=True).draw_mermaid_png(output_file_path=str(output_path))
    logger.success(f"Workflow exported to {output_path}")
