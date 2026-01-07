from langchain_core.messages import AnyMessage
from pydantic import BaseModel, ConfigDict, Field

from src.models.core import PlanningAgentOutput
from src.models.operation import OperationResponse


class PlannerGraphState(BaseModel):
    """LangGraph subgraph state schema for planning + HITL review."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    messages: list[AnyMessage] = Field(default_factory=list)
    user_input: list[AnyMessage] = Field(default_factory=list)
    thinking: list[AnyMessage] = Field(default_factory=list)

    plan: OperationResponse[list[AnyMessage], PlanningAgentOutput] | None = None
    plan_cursor: int = 0
    plan_approved: bool = False
