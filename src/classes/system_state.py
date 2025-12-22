from enum import StrEnum
from typing import Any
from uuid import uuid4

from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field, field_validator

from src.classes.system_enum import ExecutionStatusEnum, GoalTypeEnum


class IDAIDetermine(BaseModel):
    matched_goal_type: GoalTypeEnum
    reason: str

    def __str__(self) -> str:
        return f"Matched goal type: {self.matched_goal_type}, reason: {self.reason}"


class IntentionDetectionFin(IDAIDetermine):
    winner_id: str  # matched goal_id
    evidences: list[IDAIDetermine]


class WatchDogAIDetermined(BaseModel):
    within_domain: bool = Field(..., description="表示用户输入是否在系统领域范围内")
    within_capacity: bool = Field(..., description="表示用户输入是否在系统可执行范围内")
    feedback: str = Field(..., description="对用户输入的判别依据和进一步输入建议")

    @field_validator("within_capacity", "within_domain")
    @classmethod
    def check_boolean(cls, v: bool) -> bool:
        """Validate that the field is a boolean."""
        if not isinstance(v, bool):
            raise TypeError("within_prefixed attr must be a boolean")
        return v


class UserAdmittance(WatchDogAIDetermined):
    id: str
    user_input: list[AnyMessage]


class HumanApproval(BaseModel):
    """Represents the outcome of a human approval step."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for this approval decision.")
    approval: bool = Field(..., description="Whether the human reviewer approved proceeding.")
    reviewed: IntentionDetectionFin = Field(..., description="The operation response that was reviewed.")
    comment: str | None = Field(None, description="Revised user input provided when the decision is 'edit'.")


class Compound(BaseModel):
    compound_name: str = Field(..., description="Compound name, IUPAC standard name, English")
    smiles: str = Field(..., description="SMILES expression of the compound")


class TLCAIOutput(BaseModel):
    compounds: list[Compound] = Field(..., description="List of compounds extracted from the text")


class TLCAgentOutput(TLCAIOutput):
    pass


class ExecutorKey(StrEnum):
    TLC_AGENT = "tlc_agent.run"
    # COLUMN_AGENT = "column.recommend"
    # ROBOT_TLC = "robot.tlc_spot"
    # PROPERTY_LOOKUP = "property.lookup"


class PlanStep(BaseModel):
    id: str
    title: str
    executor: ExecutorKey
    args: dict[str, Any] = Field(default_factory=dict)  # Different for each executor
    requires_human_approval: bool = True
    status: ExecutionStatusEnum = ExecutionStatusEnum.NOT_STARTED
    output: Any | None = None


class PlanningAgentOutput(BaseModel):
    plan_steps: list[PlanStep] = Field(..., description="List of steps to be executed")
    plan_hash: str = Field(..., description="Hash of the plan")
