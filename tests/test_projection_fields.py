"""Maintenance guards; runtime projections remain explicit allowlists.

Every upstream field is exposed, summarized/derived, or internal-only at each
sensitive boundary. A model field addition must receive a deliberate review here.
An entirely internal nested object does not need its own child-field projection.
"""

import pytest

from game_qa_agent.eval import (
    AgentEvaluationCase, AgentEvaluationCaseResult, AgentEvaluationEvidence,
    AgentEvaluationExpectationResult, AgentEvaluationHarnessError,
)
from game_qa_agent.models import (
    AgentInvestigationState, GameRuntimeState, NextActionSpec, NPCRequirement,
    NPCRuntimeState, Task, ToolExecutionRecord, ValidationIssue,
)
from game_qa_agent.report import QAInvestigationReport, QAReportFinding
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
    pytest.param(
        AgentEvaluationCase,
        {"case_id", "max_steps", "trace_enabled"},
        {"initial_state", "tasks", "runtime_state", "scripted_actions",
         "expected_final_status", "expected_issue_types", "expected_trusted_scope_task_ids",
         "expected_decision_error_count", "expected_trace_step_count", "expected_trace_final_status"},
        set(), id="case-to-evidence-plan",
    ),
    pytest.param(
        AgentInvestigationState,
        {"scope_version"},
        {"scope_task_ids", "expandable_task_ids", "called_tool_names",
         "called_tool_scope_versions", "tool_executions", "issues", "decision_errors",
         "investigation_status"},
        {"investigation_goal", "impact_analysis", "last_decision_rejection"},
        id="state-to-evidence-package",
    ),
    # Fixture fields are used only in a digest of trusted, non-sensitive local
    # definitions. Hashing is not redaction of secrets or Tool observations.
    pytest.param(
        Task, set(), {"id", "dependencies", "npc_requirements", "exclusive_group"},
        set(), id="task-to-evidence-fixture-identity",
    ),
    pytest.param(
        NPCRequirement, set(), {"npc_id", "location", "state"}, set(),
        id="requirement-to-evidence-fixture-identity",
    ),
    pytest.param(
        GameRuntimeState, set(), {"task_statuses", "npc_states"}, set(),
        id="runtime-fixture-to-evidence-identity",
    ),
    pytest.param(
        NPCRuntimeState, set(), {"location", "state", "occupied_by_task_id"}, set(),
        id="npc-fixture-to-evidence-identity",
    ),
    pytest.param(
        NextActionSpec, {"action_type"}, {"tool_name", "tool_args", "expand_task_ids"},
        {"reason"}, id="scripted-action-to-evidence-plan",
    ),
    pytest.param(
        ValidationIssue, set(), {"issue_type", "checker_name", "task_ids", "npc_ids"},
        {"message", "evidence"}, id="issue-to-canonical-evidence",
    ),
    pytest.param(
        ToolExecutionRecord,
        {"execution_number", "tool_name", "scope_version", "status", "issue_indices"},
        set(), set(), id="execution-to-canonical-evidence",
    ),
    pytest.param(
        AgentEvaluationCaseResult, set(),
        {"outcome", "evidence", "expectation_results", "harness_error"},
        {"case_id", "passed_expectations", "failed_expectations"},
        id="eval-result-to-evidence-package",
    ),
    pytest.param(
        AgentEvaluationEvidence, {"trace_step_count", "trace_final_status"}, set(),
        {"actual_final_status", "actual_issue_types", "actual_trusted_scope_task_ids",
         "actual_decision_error_count"},
        id="eval-observation-to-evidence-package",
    ),
    pytest.param(
        AgentEvaluationExpectationResult, {"expectation", "passed"}, set(),
        {"expected", "actual"}, id="eval-check-to-evidence-package",
    ),
    pytest.param(
        AgentEvaluationHarnessError, {"stage"}, set(), {"exception_type"},
        id="harness-error-to-evidence-package",
    ),
    pytest.param(
        QAInvestigationReport,
        {"status", "scope_task_ids", "scope_version", "decision_error_count"}, {"findings"},
        {"issue_count", "issue_type_counts", "unrecognized_issue_count", "trace", "limitations"},
        id="report-to-canonical-evidence",
    ),
    pytest.param(
        QAReportFinding,
        {"issue_type", "checker_name", "scoped_task_ids", "unscoped_task_count", "npc_count"},
        set(), set(), id="nested-safe-finding-to-canonical-evidence",
    ),
])
def test_sensitive_projection_classifies_every_upstream_field(
    model, exposed, summarized, internal,
) -> None:
    assert not (exposed & summarized or exposed & internal or summarized & internal)
    assert exposed | summarized | internal == set(model.model_fields)
