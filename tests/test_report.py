from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from game_qa_agent import (
    QAInvestigationReport,
    build_qa_report,
    render_qa_report_markdown,
)
from game_qa_agent import checkers, orchestration
from game_qa_agent.analysis import build_task_index
from game_qa_agent.eval import (
    AgentEvaluationCase,
    build_deterministic_evaluation_cases,
    run_evaluation_suite,
)
from game_qa_agent.models import (
    GameRuntimeState,
    NPCRequirement,
    NPCRuntimeState,
    Task,
    ValidationIssue,
)
from game_qa_agent.providers import DeepSeekProvider
from game_qa_agent.tools import ToolRegistry, build_default_tool_registry
from game_qa_agent.trace import InMemoryInvestigationTraceRecorder

from conftest import ScriptedProvider, action


def case_by_id(case_id: str) -> AgentEvaluationCase:
    return next(
        case for case in build_deterministic_evaluation_cases()
        if case.case_id == case_id
    )


def investigate(case: AgentEvaluationCase):
    trace = InMemoryInvestigationTraceRecorder()
    state = orchestration.run_agent_investigation(
        case.initial_state.model_copy(deep=True),
        build_default_tool_registry(),
        build_task_index(case.tasks),
        case.runtime_state,
        ScriptedProvider(case.scripted_actions),
        max_steps=case.max_steps,
        trace_recorder=trace,
    )
    return state, trace


@pytest.mark.parametrize("case_id", [
    "normal_success",
    "authorized_dynamic_scope_expansion",
    "rejected_unauthorized_scope_expansion",
    "max_step_exhaustion",
])
def test_real_investigation_report_structure_and_trace(case_id: str) -> None:
    case = case_by_id(case_id)
    state, trace = investigate(case)

    report = build_qa_report(state, trace)

    assert isinstance(report, QAInvestigationReport)
    assert report.status == case.expected_final_status
    assert report.scope_task_ids == tuple(case.expected_trusted_scope_task_ids)
    assert report.issue_count == len(case.expected_issue_types)
    assert report.issue_type_counts == dict(Counter(case.expected_issue_types))
    assert report.unrecognized_issue_count == 0
    assert report.decision_error_count == case.expected_decision_error_count
    assert report.trace is not None
    assert report.trace.step_count == case.expected_trace_step_count
    assert report.trace.final_status == case.expected_trace_final_status
    assert report.trace.rejected_step_count == case.expected_decision_error_count
    assert QAInvestigationReport.model_validate_json(report.model_dump_json()) == report
    if case_id == "authorized_dynamic_scope_expansion":
        assert report.scope_version == 2
        assert len(report.trace.scope_changes) == 1
        assert report.trace.scope_changes[0].model_dump() == {
            "step_number": 2,
            "scope_version_before": 1,
            "scope_version_after": 2,
            "added_task_ids": ("task_event_7",),
        }
    else:
        assert report.scope_version == 1
        assert report.trace.scope_changes == ()
    if case_id == "rejected_unauthorized_scope_expansion":
        assert "task_c" in state.decision_errors[0]
        assert '"task_c"' not in report.model_dump_json()
        assert "`task_c`" not in render_qa_report_markdown(report)
        assert "task_event_7" not in report.model_dump_json()
        occupation = next(
            finding for finding in report.findings
            if finding.issue_type == "npc_occupied_by_other_task"
        )
        assert occupation.scoped_task_ids == ("task_b",)
        assert occupation.unscoped_task_count == 1


def all_finding_case() -> AgentEvaluationCase:
    case = case_by_id("normal_success")
    case.tasks = [
        Task(id="task_a", dependencies=["task_b", "private-missing-id"], npc_requirements=[
            NPCRequirement(npc_id="npc_a", location="village", state="available"),
            NPCRequirement(npc_id="private-npc-id", location="forest", state="ready"),
        ]),
        Task(id="task_b", dependencies=["task_a"], npc_requirements=[
            NPCRequirement(npc_id="npc_a", location="town", state="available"),
        ]),
    ]
    case.initial_state.scope_task_ids = ["task_b", "task_a"]
    case.runtime_state = GameRuntimeState(
        task_statuses={"task_a": "active", "task_b": "inactive"},
        npc_states={"npc_a": NPCRuntimeState(
            location="private-runtime-location", state="private-runtime-state",
            occupied_by_task_id="private-occupant-id",
        )},
    )
    case.scripted_actions = [
        action("call_tool", tool_name="dependency_cycle_checker"),
        action("call_tool", tool_name="dependency_reference_checker"),
        action("call_tool", tool_name="npc_static_conflict_checker"),
        action("call_tool", tool_name="npc_runtime_checker"),
        action("finish"),
    ]
    return case


def test_all_current_checker_types_have_safe_structured_details() -> None:
    state, trace = investigate(all_finding_case())

    report = build_qa_report(state, trace)

    assert report.issue_count == 7
    assert report.issue_type_counts == {
        "dependency_cycle": 1,
        "missing_dependency": 1,
        "potential_npc_conflict": 1,
        "missing_npc_runtime_state": 1,
        "npc_location_mismatch": 1,
        "npc_state_mismatch": 1,
        "npc_occupied_by_other_task": 1,
    }
    findings = {finding.issue_type: finding for finding in report.findings}
    assert findings["dependency_cycle"].scoped_task_ids == ("task_a", "task_b")
    assert findings["dependency_cycle"].npc_count == 0
    assert findings["potential_npc_conflict"].checker_name == "npc_static_conflict_checker"
    assert findings["potential_npc_conflict"].npc_count == 1
    assert findings["npc_location_mismatch"].scoped_task_ids == ("task_a",)
    assert findings["npc_occupied_by_other_task"].unscoped_task_count == 1
    assert "static_risk_not_runtime_proof" in report.limitations
    output = report.model_dump_json() + render_qa_report_markdown(report)
    for private_text in [
        "private-missing-id", "private-npc-id", "private-runtime-location",
        "private-runtime-state", "private-occupant-id", "required_location",
        "actual_location", "missing_dependency_id", "village", "npc_a",
    ]:
        assert private_text not in output


def test_distinct_recorded_findings_with_equal_projections_remain_counted() -> None:
    state, _ = investigate(all_finding_case())
    original = next(issue for issue in state.issues if issue.issue_type == "npc_location_mismatch")
    distinct = original.model_copy(deep=True)
    distinct.evidence["actual_location"] = "different-private-location"
    distinct.message = "a different recorded finding"
    state.issues.append(distinct)

    report = build_qa_report(state)

    assert report.issue_count == 8
    assert report.issue_type_counts["npc_location_mismatch"] == 2
    projected = [finding for finding in report.findings if finding.issue_type == "npc_location_mismatch"]
    assert len(projected) == 2
    assert projected[0] == projected[1]


@pytest.mark.parametrize(("action_type", "status", "limitation"), [
    ("clarify", "clarification_required", "Clarification is required"),
    ("human_review", "human_review_required", "Human review is required"),
    ("call_tool", "max_steps_exceeded", "The step limit was reached"),
])
def test_abnormal_completion_has_explicit_limitations(action_type, status, limitation) -> None:
    case = case_by_id("normal_success")
    case.max_steps = 1
    case.scripted_actions = [action(
        action_type,
        tool_name="dependency_reference_checker" if action_type == "call_tool" else None,
    )]
    state, trace = investigate(case)

    report = build_qa_report(state, trace)

    assert report.status == status
    assert status in report.limitations
    assert limitation in render_qa_report_markdown(report)
    assert report.trace.final_status == status


@pytest.mark.parametrize(("status", "limitation"), [
    ("start", "not_started"),
    ("running", "still_running"),
    ("private-unrecognized-status", "unknown_status"),
])
def test_partial_or_unknown_state_without_trace(status, limitation) -> None:
    state = case_by_id("normal_success").initial_state
    state.investigation_status = status

    report = build_qa_report(state)

    assert report.status == ("unknown" if status.startswith("private-") else status)
    assert limitation in report.limitations
    assert report.trace is None
    markdown = render_qa_report_markdown(report)
    assert "Not provided." in markdown
    assert "private-unrecognized-status" not in report.model_dump_json() + markdown


@pytest.mark.parametrize(("trace_status", "limitation"), [
    (None, "trace_final_status_unavailable"),
    ("private-trace-status", "trace_final_status_unavailable"),
    ("human_review_required", "trace_status_mismatch"),
])
def test_missing_or_inconsistent_trace_status_does_not_override_state(trace_status, limitation) -> None:
    state, trace = investigate(case_by_id("normal_success"))
    trace.final_status = trace_status
    trace.steps.clear()

    report = build_qa_report(state, trace)

    assert report.status == "finished"
    assert report.trace.step_count == 0
    assert report.trace.final_status == ("unknown" if trace_status == "private-trace-status" else trace_status)
    assert limitation in report.limitations
    assert "private-trace-status" not in report.model_dump_json() + render_qa_report_markdown(report)


def test_normal_markdown_is_concise_and_exact() -> None:
    state, trace = investigate(case_by_id("normal_success"))

    assert render_qa_report_markdown(build_qa_report(state, trace)) == (
        "# QA investigation report\n\n"
        "Status: `finished`.\n\n"
        "Trusted scope (version 1): `task_a`.\n\n"
        "Findings: 0 recorded; 0 unrecognized.\n"
        "Decision errors: 0.\n\n"
        "## Findings\n\n"
        "No findings recorded.\n\n"
        "## Trace\n\n"
        "Recorded steps: 2; final status: `finished`; rejected steps: 0; scope changes: 0.\n\n"
        "## Limitations\n\n"
        "- Counts reflect recorded findings; full checker coverage is not established.\n"
        "- Trace is diagnostic and may omit steps.\n"
    )


def test_report_is_canonical_for_equivalent_finding_and_scope_order() -> None:
    first_state, first_trace = investigate(all_finding_case())
    second_state, second_trace = investigate(all_finding_case())
    second_state.scope_task_ids.reverse()
    second_state.issues.reverse()
    for issue in second_state.issues:
        issue.task_ids.reverse()
        issue.npc_ids.reverse()
    for step in second_trace.steps:
        step.scope_task_ids_before.reverse()
        step.scope_task_ids_after.reverse()

    first = build_qa_report(first_state, first_trace)
    second = build_qa_report(second_state, second_trace)

    assert first == second
    assert first.model_dump_json() == second.model_dump_json()
    assert render_qa_report_markdown(first) == render_qa_report_markdown(second)
    assert "\r" not in render_qa_report_markdown(first)


def test_report_generation_is_passive_and_copies_input_collections(monkeypatch) -> None:
    cases = build_deterministic_evaluation_cases()
    evaluation_before = run_evaluation_suite(cases)
    state, trace = investigate(case_by_id("authorized_dynamic_scope_expansion"))
    state_before = state.model_dump()
    trace_before = deepcopy(trace.__dict__)

    def forbidden(*args, **kwargs):
        pytest.fail("Report generation must not execute, record, or access external data")

    with monkeypatch.context() as guard:
        guard.setattr(orchestration, "run_agent_investigation", forbidden)
        guard.setattr(orchestration, "execute_action", forbidden)
        guard.setattr(ToolRegistry, "get_tool", forbidden)
        guard.setattr(ScriptedProvider, "generate_next_action", forbidden)
        guard.setattr(DeepSeekProvider, "__init__", forbidden)
        guard.setattr(DeepSeekProvider, "generate_next_action", forbidden)
        guard.setattr(InMemoryInvestigationTraceRecorder, "record_step", forbidden)
        guard.setattr(InMemoryInvestigationTraceRecorder, "record_final_status", forbidden)
        for name in ["detect_dependency_cycles", "detect_missing_dependencies",
                     "detect_potential_npc_conflicts", "check_npc_runtime_requirements"]:
            guard.setattr(checkers, name, forbidden)
        guard.setattr("builtins.open", forbidden)
        guard.setattr("io.open", forbidden)
        guard.setattr("os.getenv", forbidden)
        guard.setattr("socket.socket", forbidden)
        report = build_qa_report(state, trace)
        markdown = render_qa_report_markdown(report)
        assert build_qa_report(state, trace) == report
        assert render_qa_report_markdown(report) == markdown

    assert state.model_dump() == state_before
    assert trace.__dict__ == trace_before
    assert run_evaluation_suite(cases) == evaluation_before
    report_before = report.model_dump()
    state.scope_task_ids.clear()
    state.issues.clear()
    state.decision_errors.append("later error")
    trace.steps[1].scope_task_ids_after.clear()
    trace.steps.clear()
    trace.final_status = None
    assert report.model_dump() == report_before
    assert render_qa_report_markdown(report) == markdown


def test_provider_action_error_and_tool_payloads_never_cross_report_boundary() -> None:
    case = case_by_id("normal_success")
    proposed = action("call_tool", tool_name="private-tool-name", reason="private-provider-reason")
    proposed.tool_args = {"api_key": "private-credential", "prompt": "private-prompt"}
    proposed.expand_task_ids = ["private-rejected-id"]
    case.scripted_actions = [proposed, action("finish", reason="private-completion")]
    state, trace = investigate(case)
    state.investigation_goal = "private-goal"
    state.decision_errors.append("private-error private-credential private-completion")
    state.called_tool_names.append("private-called-tool")
    state.called_tool_scope_versions["private-tool-history"] = [1]
    state.expandable_task_ids = ["private-expandable-id"]
    state.impact_analysis.changed_task_ids = ["private-impact-id"]
    state.impact_analysis.affected_npc_ids = ["private-impact-npc"]
    state.issues = [
        ValidationIssue(
            issue_type="npc_location_mismatch", checker_name="npc_runtime_checker",
            message="private-tool-message",
            task_ids=["task_a", "private-rejected-id"], npc_ids=["private-tool-npc"],
            evidence={"required_location": "private-required-location",
                      "actual_location": "private-actual-location",
                      "raw_result": {"completion": "private-completion",
                                     "api_key": "private-credential"}},
        ),
        ValidationIssue(issue_type="private-issue-type", checker_name="npc_runtime_checker",
                        message="private-unknown-message"),
        ValidationIssue(issue_type="dependency_cycle", checker_name="private-checker-name",
                        message="private-mismatched-message"),
    ]
    trace.steps[0].action.trusted_tool_name = "private-trace-tool"
    trace.steps[0].investigation_status_before = "private-before-status"
    trace.steps[0].investigation_status_after = "private-after-status"
    trace.steps[0].new_issue_types = ["private-trace-type"]
    trace.steps[0].scope_task_ids_after.append("private-trace-id")
    trace.final_status = "private-trace-final-status"

    report = build_qa_report(state, trace)

    assert report.status == "finished"
    assert report.scope_task_ids == ("task_a",)
    assert report.issue_count == 3
    assert report.unrecognized_issue_count == 2
    assert report.issue_type_counts == {"npc_location_mismatch": 1}
    assert report.findings[0].scoped_task_ids == ("task_a",)
    assert report.findings[0].unscoped_task_count == 1
    assert report.findings[0].npc_count == 1
    assert report.decision_error_count == 2
    assert report.trace.rejected_step_count == 1
    assert report.trace.scope_changes == ()
    assert "unrecognized_findings" in report.limitations
    output = repr(report.model_dump()) + report.model_dump_json() + render_qa_report_markdown(report)
    assert "private-" not in output
    for forbidden_field in ["reason", "tool_args", "evidence", "message", "decision_errors",
                            "expand_task_ids", "api_key", "prompt", "completion", "raw_result"]:
        assert forbidden_field not in output


def test_unknown_findings_are_not_reported_as_no_findings() -> None:
    state = case_by_id("normal_success").initial_state
    state.issues.append(ValidationIssue(
        issue_type="private-type", checker_name="private-checker", message="private-message",
    ))

    report = build_qa_report(state)
    markdown = render_qa_report_markdown(report)

    assert report.issue_count == report.unrecognized_issue_count == 1
    assert report.findings == ()
    assert "No recognized findings recorded." in markdown
    assert "No findings recorded." not in markdown
    assert "private-" not in report.model_dump_json() + markdown


@pytest.mark.parametrize(("action_type", "marker"), [
    ("expand_scope", "private-rejected-task-id"),
    ("expand_scope", "task_x"),
    ("call_tool", "private-unknown-tool-name"),
])
def test_rejected_ids_and_names_in_real_decision_errors_are_counts_only(action_type, marker) -> None:
    case = case_by_id("rejected_unauthorized_scope_expansion")
    case.scripted_actions[1] = action(
        action_type,
        expand_task_ids=[marker] if action_type == "expand_scope" else None,
        tool_name=marker if action_type == "call_tool" else None,
    )
    state, trace = investigate(case)
    assert marker in state.decision_errors[0]

    report = build_qa_report(state, trace)

    assert report.decision_error_count == report.trace.rejected_step_count == 1
    assert report.scope_task_ids == ("task_a", "task_b")
    assert report.trace.scope_changes == ()
    assert marker not in report.model_dump_json() + render_qa_report_markdown(report)


def test_trusted_task_labels_cannot_break_markdown_structure() -> None:
    state = case_by_id("normal_success").initial_state
    state.scope_task_ids = ["task`|<script>\n# forged heading\r\t\u202e"]

    markdown = render_qa_report_markdown(build_qa_report(state))

    assert "&#96;" in markdown
    assert "&#124;" in markdown
    assert "&lt;script&gt;" in markdown
    assert "\\n# forged heading" in markdown
    assert "\n# forged heading" not in markdown
    assert "\r" not in markdown
    assert "\t" not in markdown
    assert "\u202e" not in markdown


def test_readme_offline_example_matches_documented_output(capsys, monkeypatch) -> None:
    def no_live_provider(*args, **kwargs):
        pytest.fail("README example must stay offline")

    monkeypatch.setattr(DeepSeekProvider, "__init__", no_live_provider)
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    example = readme.split("```python\n", 1)[1].split("```", 1)[0]
    documented_output = readme.split("```markdown\n", 1)[1].split("```", 1)[0]

    exec(compile(example, "README.md", "exec"), {})

    assert capsys.readouterr().out == documented_output
