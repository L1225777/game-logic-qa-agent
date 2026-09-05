"""Maintenance guards; runtime projections remain explicit allowlists.

Every upstream field is exposed, summarized/derived, or internal-only at each
sensitive boundary. A model field addition must receive a deliberate review here.
An entirely internal nested object does not need its own child-field projection.
"""

import pytest

from game_qa_agent.models import AgentInvestigationState, NextActionSpec, ValidationIssue
from game_qa_agent.trace import InvestigationStepTrace


@pytest.mark.parametrize(("model", "exposed", "summarized", "internal"), [
    pytest.param(
        AgentInvestigationState,
        {"scope_version", "last_decision_rejection"},
        {"investigation_goal", "scope_task_ids", "expandable_task_ids",
         "called_tool_names", "called_tool_scope_versions", "issues",
         "decision_errors", "investigation_status"},
        {"impact_analysis", "tool_executions"},
        id="state-to-provider-context",
    ),
    pytest.param(
        ValidationIssue,
        set(), {"issue_type"},
        {"message", "task_ids", "npc_ids", "evidence", "checker_name"},
        id="issue-to-provider-context",
    ),
    pytest.param(
        NextActionSpec,
        {"action_type"}, {"tool_name", "tool_args", "expand_task_ids"}, {"reason"},
        id="action-to-trace",
    ),
    pytest.param(
        AgentInvestigationState,
        {"scope_version", "scope_task_ids", "investigation_status"},
        {"issues", "decision_errors"},
        {"investigation_goal", "impact_analysis", "expandable_task_ids",
         "called_tool_names", "called_tool_scope_versions", "last_decision_rejection",
         "tool_executions"},
        id="state-to-trace",
    ),
    pytest.param(
        ValidationIssue,
        {"issue_type"}, set(),
        {"message", "task_ids", "npc_ids", "evidence", "checker_name"},
        id="issue-to-trace",
    ),
    pytest.param(
        AgentInvestigationState,
        {"scope_version"},
        {"scope_task_ids", "issues", "decision_errors", "investigation_status"},
        {"investigation_goal", "impact_analysis", "expandable_task_ids",
         "called_tool_names", "called_tool_scope_versions", "last_decision_rejection",
         "tool_executions"},
        id="state-to-report",
    ),
    pytest.param(
        ValidationIssue,
        set(), {"issue_type", "checker_name", "task_ids", "npc_ids"},
        {"message", "evidence"},
        id="issue-to-report",
    ),
    pytest.param(
        InvestigationStepTrace,
        {"step_number", "scope_version_before", "scope_version_after"},
        {"scope_task_ids_before", "scope_task_ids_after", "outcome"},
        {"action", "investigation_status_before", "investigation_status_after",
         "new_issue_types", "new_decision_error_count"},
        id="trace-step-to-report",
    ),
])
def test_sensitive_projection_classifies_every_upstream_field(
    model, exposed, summarized, internal,
) -> None:
    assert not (exposed & summarized or exposed & internal or summarized & internal)
    assert exposed | summarized | internal == set(model.model_fields)
