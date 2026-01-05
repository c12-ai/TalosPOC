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
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage, AnyMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from src.agents.coordinators.human_interaction import HumanInLoop
from src.models.core import PlanningAgentOutput, PlanStep
from src.models.enums import ExecutionStatusEnum, ExecutorKey
from src.models.operation import OperationResponse
from src.utils.logging_config import logger
from src.utils.messages import only_human_messages
from src.utils.models import PLANNER_MODEL
from src.utils.PROMPT import PLANNER_SYSTEM_PROMPT

if TYPE_CHECKING:
    from collections.abc import Callable

# Registry of executors
executor_registry: dict[ExecutorKey, Callable[..., Any]] = {}


class Planner:
    def __init__(self) -> None:
        """Initialize the Planner Agent."""
        self.planner = create_agent(
            model=PLANNER_MODEL,
            response_format=ToolStrategy[PlanningAgentOutput](PlanningAgentOutput),
            system_prompt=PLANNER_SYSTEM_PROMPT,
        )
        logger.info("Planner initialized")

    def run(self, user_input: list[AnyMessage]) -> OperationResponse[list[AnyMessage], PlanningAgentOutput]:
        """
        Generate a plan based on user input. The output should be a sequence of TODOs: a set of restricted commands.

        Args:
            user_input (list[AnyMessage]): The user input messages ordered chronologically.

        Returns:
            OperationResponse[list[AnyMessage], PlanningAgentOutput]: The operation response containing the plan.

        """
        start_time = datetime.now()
        logger.info("Planner.run triggered with {} messages", len(user_input))

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


class PlannerGraphState(BaseModel):
    """LangGraph subgraph state schema for planning + HITL review."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    messages: list[AnyMessage] = Field(default_factory=list)
    user_input: list[AnyMessage] = Field(default_factory=list)

    plan: OperationResponse[list[AnyMessage], PlanningAgentOutput] | None = None
    plan_cursor: int = 0
    plan_approved: bool = False


class PlannerSubgraph:
    """Planning subgraph: generate_plan -> plan_review (approve/revise loop)."""

    def __init__(self, *, with_checkpointer: bool = False) -> None:
        """Build and compile the internal planning subgraph."""
        self._planner = Planner()
        self._human = HumanInLoop()

        subgraph = StateGraph(PlannerGraphState)
        subgraph.add_node("generate_plan", self._generate_plan)
        subgraph.add_node("plan_review", self._plan_review)

        subgraph.add_edge(START, "generate_plan")
        subgraph.add_edge("generate_plan", "plan_review")
        subgraph.add_conditional_edges(
            "plan_review",
            self._route_plan_review,
            {
                "revise": "generate_plan",
                "approved": END,
            },
        )

        checkpointer = MemorySaver() if with_checkpointer else None
        self.compiled = subgraph.compile(checkpointer=checkpointer)

    @staticmethod
    def _ensure_messages(state: PlannerGraphState) -> list[AnyMessage]:
        return list(state.messages or state.user_input)

    def _generate_plan(self, state: PlannerGraphState) -> dict[str, Any]:
        messages = self._ensure_messages(state)
        logger.info("PlannerSubgraph.generate_plan with {} messages", len(messages))

        res = self._planner.run(user_input=messages)
        plan_out = res.output

        steps_preview = "\n".join(
            [
                f"- {idx + 1}. {s.title} (executor={s.executor}, requires_human_approval={s.requires_human_approval})"
                for idx, s in enumerate(plan_out.plan_steps)
            ],
        )

        updated_messages = list(messages)
        updated_messages.append(
            AIMessage(
                content="\n".join(
                    [
                        "[planner] plan_created",
                        f"plan_hash={plan_out.plan_hash}",
                        "steps:",
                        steps_preview or "- (no steps)",
                    ],
                ),
            ),
        )

        return {
            "plan": res,
            "plan_cursor": 0,
            "plan_approved": False,
            "messages": updated_messages,
            "user_input": only_human_messages(updated_messages),
        }

    def _plan_review(self, state: PlannerGraphState) -> dict[str, Any]:
        if state.plan is None:
            raise ValueError("Missing 'plan' before plan_review")
        messages = self._ensure_messages(state)

        updates = self._human.review_plan(plan_out=state.plan.output, messages=messages)
        if not updates.get("plan_approved"):
            # Explicitly clear old plan on reject, so parent graph won't treat it as executable.
            updates["plan"] = None
            updates["plan_cursor"] = 0
        return updates

    @staticmethod
    def _route_plan_review(state: PlannerGraphState) -> str:
        return "approved" if state.plan_approved else "revise"


planner_subgraph = PlannerSubgraph()


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    planner = Planner()
    result = planner.run(user_input=[HumanMessage(content="帮我查一下阿司匹林的属性,然后设计一个TLC条件")])
    print(result.model_dump_json(indent=2))
