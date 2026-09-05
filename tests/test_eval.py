from game_qa_agent.eval import (
    AgentEvaluationCase,
    build_deterministic_evaluation_cases,
    run_evaluation_case,
    run_evaluation_suite,
)
from game_qa_agent.models import (
    GameRuntimeState,
    NPCRequirement,
    NPCRuntimeState,
    NextActionSpec,
    Task,
)


def case_by_id(case_id: str) -> AgentEvaluationCase:
    return next(
        case
        for case in build_deterministic_evaluation_cases()
        if case.case_id == case_id
    )


def test_case_observation_uses_real_dynamic_scope_and_tools() -> None:
    result = run_evaluation_case(
        case_by_id("authorized_dynamic_scope_expansion")
    )

    assert result.outcome == "passed"
    assert result.harness_error is None
    assert result.evidence is not None
    assert result.evidence.actual_final_status == "finished"
    assert result.evidence.actual_issue_types == [
        "npc_location_mismatch",
        "npc_occupied_by_other_task",
        "potential_npc_conflict",
    ]
    assert result.evidence.actual_trusted_scope_task_ids == [
        "task_a", "task_b", "task_event_7"
    ]
    assert result.evidence.actual_decision_error_count == 0
    assert result.failed_expectations == []


def test_behavioral_expectation_failure_keeps_valid_observation() -> None:
    case = case_by_id("normal_success")
    case.expected_final_status = "human_review_required"

    result = run_evaluation_case(case)

    assert result.outcome == "failed_behavior"
    assert result.evidence is not None
    assert result.evidence.actual_final_status == "finished"
    assert result.harness_error is None
    assert result.failed_expectations == ["final_status"]
    final_status_check = next(
        check
        for check in result.expectation_results
        if check.expectation == "final_status"
    )
    assert final_status_check.expected == "human_review_required"
    assert final_status_check.actual == "finished"
    assert final_status_check.passed is False


def test_harness_error_is_not_a_behavioral_failure() -> None:
    case = case_by_id("normal_success")
    case.scripted_actions = []

    result = run_evaluation_case(case)

    assert result.outcome == "harness_error"
    assert result.evidence is None
    assert result.expectation_results == []
    assert result.failed_expectations == []
    assert result.harness_error is not None
    assert result.harness_error.stage == "investigation_execution"
    assert result.harness_error.exception_type == "IndexError"


def test_suite_summary_counts_only_valid_observations_as_evaluated() -> None:
    passing_case = case_by_id("normal_success")
    failing_case = case_by_id("normal_success")
    failing_case.case_id = "wrong_status_expectation"
    failing_case.expected_final_status = "clarification_required"
    harness_error_case = case_by_id("normal_success")
    harness_error_case.case_id = "missing_scripted_actions"
    harness_error_case.scripted_actions = []

    summary = run_evaluation_suite([
        passing_case, failing_case, harness_error_case
    ])

    assert summary.total_case_count == 3
    assert summary.evaluated_case_count == 2
    assert summary.passed_case_count == 1
    assert summary.failed_behavioral_case_count == 1
    assert summary.harness_error_case_count == 1
    assert [result.outcome for result in summary.case_results] == [
        "passed", "failed_behavior", "harness_error"
    ]


def test_trace_facts_are_evidence_and_expectations() -> None:
    result = run_evaluation_case(case_by_id("max_step_exhaustion"))

    assert result.outcome == "passed"
    assert result.evidence is not None
    assert result.evidence.actual_final_status == "max_steps_exceeded"
    assert result.evidence.trace_step_count == 1
    assert result.evidence.trace_final_status == "max_steps_exceeded"
    checks = {check.expectation: check for check in result.expectation_results}
    assert checks["trace_step_count"].passed is True
    assert checks["trace_final_status"].passed is True


def test_eval_evidence_excludes_provider_text_and_unsafe_action_payloads() -> None:
    case = case_by_id("max_step_exhaustion")
    case.scripted_actions = [NextActionSpec(
        action_type="call_tool",
        tool_name="unsafe-tool-name-marker",
        tool_args={"credential": "unsafe-payload-marker"},
        expand_task_ids=["unsafe-expansion-marker"],
        reason="unsafe-reason-marker",
    )]

    result = run_evaluation_case(case)

    assert result.outcome == "passed"
    serialized_evidence = result.model_dump_json()
    for marker in [
        "unsafe-tool-name-marker",
        "unsafe-payload-marker",
        "unsafe-expansion-marker",
        "unsafe-reason-marker",
    ]:
        assert marker not in serialized_evidence
    assert "decision_errors" not in serialized_evidence
    assert "tool_args" not in serialized_evidence


def test_eval_evidence_excludes_raw_tool_issue_payloads() -> None:
    case = case_by_id("normal_success")
    case.tasks = [Task(
        id="task_a",
        npc_requirements=[NPCRequirement(
            npc_id="npc_a", location="village", state="available"
        )],
    )]
    case.runtime_state = GameRuntimeState(
        task_statuses={"task_a": "active"},
        npc_states={"npc_a": NPCRuntimeState(
            location="unsafe-tool-output-marker",
            state="available",
        )},
    )
    case.scripted_actions = [
        NextActionSpec(
            action_type="call_tool",
            tool_name="npc_runtime_checker",
            reason="unsafe-provider-reason-marker",
        ),
        NextActionSpec(action_type="finish", reason="evaluation complete"),
    ]
    case.expected_issue_types = ["npc_location_mismatch"]

    result = run_evaluation_case(case)

    assert result.outcome == "passed"
    serialized_evidence = result.model_dump_json()
    assert "npc_location_mismatch" in serialized_evidence
    assert "unsafe-tool-output-marker" not in serialized_evidence
    assert "unsafe-provider-reason-marker" not in serialized_evidence
    assert "required_location" not in serialized_evidence
    assert "actual_location" not in serialized_evidence


def test_representative_evaluation_suite_is_deterministic_and_complete() -> None:
    cases = build_deterministic_evaluation_cases()

    first = run_evaluation_suite(cases)
    second = run_evaluation_suite(cases)

    assert [case.case_id for case in cases] == [
        "normal_success",
        "authorized_dynamic_scope_expansion",
        "rejected_unauthorized_scope_expansion",
        "max_step_exhaustion",
    ]
    assert first == second
    assert first.total_case_count == 4
    assert first.evaluated_case_count == 4
    assert first.passed_case_count == 4
    assert first.failed_behavioral_case_count == 0
    assert first.harness_error_case_count == 0
    assert all(
        case.initial_state.investigation_status == "start" for case in cases
    )
