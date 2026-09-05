from typing import Any, Literal

from pydantic import BaseModel, Field


TaskStatus = Literal["inactive", "active", "paused", "completed"]
NextActionType = Literal[
    "call_tool", "expand_scope", "clarify", "human_review", "finish"
]
DecisionRejectionCode = Literal[
    "missing_tool_name", "unexpected_action_fields", "tool_already_called",
    "unknown_tool", "no_expandable_tasks", "empty_scope_expansion",
    "unauthorized_scope_expansion",
]


class NPCRequirement(BaseModel):
    npc_id: str
    location: str
    state: str


class Task(BaseModel):
    id: str
    dependencies: list[str] = Field(default_factory=list)
    npc_requirements: list[NPCRequirement] = Field(default_factory=list)
    exclusive_group: str | None = None


class NPCRuntimeState(BaseModel):
    location: str
    state: str
    occupied_by_task_id: str | None = None


class GameRuntimeState(BaseModel):
    task_statuses: dict[str, TaskStatus] = Field(default_factory=dict)
    npc_states: dict[str, NPCRuntimeState] = Field(default_factory=dict)


class ValidationIssue(BaseModel):
    issue_type: str
    message: str
    task_ids: list[str] = Field(default_factory=list)
    npc_ids: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    checker_name: str


class ImpactAnalysisResult(BaseModel):
    changed_task_ids: list[str] = Field(default_factory=list)
    affected_task_ids: list[str] = Field(default_factory=list)
    affected_npc_ids: list[str] = Field(default_factory=list)


class NextActionSpec(BaseModel):
    action_type: NextActionType
    tool_name: str | None = None
    tool_args: dict[str, Any] = Field(default_factory=dict)
    expand_task_ids: list[str] = Field(default_factory=list)
    reason: str


class AgentInvestigationState(BaseModel):
    investigation_goal: str
    impact_analysis: ImpactAnalysisResult
    scope_task_ids: list[str] = Field(default_factory=list)
    expandable_task_ids: list[str] = Field(default_factory=list)
    scope_version: int = 1
    called_tool_names: list[str] = Field(default_factory=list)
    called_tool_scope_versions: dict[str, list[int]] = Field(default_factory=dict)
    issues: list[ValidationIssue] = Field(default_factory=list)
    decision_errors: list[str] = Field(default_factory=list)
    last_decision_rejection: DecisionRejectionCode | None = None
    investigation_status: str = "start"
