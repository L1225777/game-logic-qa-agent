from collections.abc import Collection
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from .models import NextActionSpec, NextActionType


InvestigationStepOutcome = Literal["processed", "rejected", "terminal"]
InvestigationFinalStatus = Literal[
    "finished",
    "clarification_required",
    "human_review_required",
    "max_steps_exceeded",
]


class InvestigationActionTrace(BaseModel):
    action_type: NextActionType
    tool_name_supplied: bool
    trusted_tool_name: str | None = None
    had_tool_args: bool
    requested_expansion_count: int


def summarize_investigation_action(
    action: NextActionSpec,
    trusted_tool_names: Collection[str],
) -> InvestigationActionTrace:
    trusted_tool_name = (
        action.tool_name if action.tool_name in trusted_tool_names else None
    )
    return InvestigationActionTrace(
        action_type=action.action_type,
        tool_name_supplied=action.tool_name is not None,
        trusted_tool_name=trusted_tool_name,
        had_tool_args=bool(action.tool_args),
        requested_expansion_count=len(action.expand_task_ids),
    )


class InvestigationStepTrace(BaseModel):
    step_number: int
    action: InvestigationActionTrace
    scope_version_before: int
    scope_version_after: int
    scope_task_ids_before: list[str] = Field(default_factory=list)
    scope_task_ids_after: list[str] = Field(default_factory=list)
    investigation_status_before: str
    investigation_status_after: str
    new_issue_types: list[str] = Field(default_factory=list)
    new_decision_error_count: int
    outcome: InvestigationStepOutcome


class InvestigationTraceRecorder(Protocol):
    def record_step(self, step: InvestigationStepTrace) -> None: ...

    def record_final_status(self, status: InvestigationFinalStatus) -> None: ...


class InMemoryInvestigationTraceRecorder:
    def __init__(self) -> None:
        self.steps: list[InvestigationStepTrace] = []
        self.final_status: InvestigationFinalStatus | None = None

    def record_step(self, step: InvestigationStepTrace) -> None:
        self.steps.append(step)

    def record_final_status(self, status: InvestigationFinalStatus) -> None:
        self.final_status = status
