"""Bounded decision facts projected by the controller before each Agent round."""

from collections import Counter
from typing import Literal, cast, get_args

from pydantic import BaseModel, ConfigDict, Field

from .models import AgentInvestigationState, DecisionRejectionCode
from .tools import ToolRegistry, tool_call_is_blocked


_MAX_GOAL_CHARS = 2000
_MAX_TASK_IDS = 128
_MAX_TASK_ID_CHARS = 128

ContextStatus = Literal[
    "start", "running", "finished", "clarification_required",
    "human_review_required", "max_steps_exceeded", "unknown",
]
ContextFindingType = Literal[
    "dependency_cycle", "missing_dependency", "potential_npc_conflict",
    "missing_npc_runtime_state", "npc_location_mismatch", "npc_state_mismatch",
    "npc_occupied_by_other_task",
]


class ProviderToolCapability(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    blocked_by_call_history: bool
    last_called_scope_version: int | None


class ProviderFindingCount(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    issue_type: ContextFindingType
    count: int


class ProviderDecisionContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    investigation_goal: str = Field(max_length=_MAX_GOAL_CHARS)
    goal_truncated: bool
    scope_task_ids: tuple[str, ...] = Field(max_length=_MAX_TASK_IDS)
    omitted_scope_task_count: int
    expandable_task_ids: tuple[str, ...] = Field(max_length=_MAX_TASK_IDS)
    omitted_expandable_task_count: int
    scope_version: int
    active_tools: tuple[ProviderToolCapability, ...]
    finding_counts: tuple[ProviderFindingCount, ...]
    unrecognized_finding_count: int
    decision_error_count: int
    last_decision_rejection: DecisionRejectionCode | None
    investigation_status: ContextStatus


def _bounded_task_ids(task_ids: list[str]) -> tuple[tuple[str, ...], int]:
    unique_ids = sorted(set(task_ids))
    visible = tuple(
        task_id for task_id in unique_ids if len(task_id) <= _MAX_TASK_ID_CHARS
    )[:_MAX_TASK_IDS]
    # Omit long IDs rather than shortening them into different identifiers.
    return visible, len(unique_ids) - len(visible)


def build_provider_decision_context(
    state: AgentInvestigationState, tool_registry: ToolRegistry,
) -> ProviderDecisionContext:
    """Copy trusted facts only; no serialization of state or execution of Tools.

    Finding labels are a fixed vocabulary, not a claim of checker coverage or
    independent validation. Call-history summaries only concern active Tools.
    """
    scope_ids, omitted_scope = _bounded_task_ids(state.scope_task_ids)
    expandable_ids, omitted_expandable = _bounded_task_ids(state.expandable_task_ids)
    capabilities = tuple(
        ProviderToolCapability(
            name=name,
            blocked_by_call_history=tool_call_is_blocked(name, state),
            last_called_scope_version=max((
                version for version in state.called_tool_scope_versions.get(name, [])
                if 0 < version <= state.scope_version
            ), default=None),
        )
        for name in tool_registry.active_tool_names()
    )
    counts = Counter(
        issue.issue_type for issue in state.issues
        if issue.issue_type in get_args(ContextFindingType)
    )
    status = state.investigation_status
    if status not in get_args(ContextStatus):
        status = "unknown"
    return ProviderDecisionContext(
        investigation_goal=state.investigation_goal[:_MAX_GOAL_CHARS],
        goal_truncated=len(state.investigation_goal) > _MAX_GOAL_CHARS,
        scope_task_ids=scope_ids,
        omitted_scope_task_count=omitted_scope,
        expandable_task_ids=expandable_ids,
        omitted_expandable_task_count=omitted_expandable,
        scope_version=state.scope_version,
        active_tools=capabilities,
        finding_counts=tuple(
            ProviderFindingCount(issue_type=cast(ContextFindingType, kind), count=count)
            for kind, count in sorted(counts.items())
        ),
        unrecognized_finding_count=len(state.issues) - sum(counts.values()),
        decision_error_count=len(state.decision_errors),
        last_decision_rejection=state.last_decision_rejection,
        investigation_status=cast(ContextStatus, status),
    )
