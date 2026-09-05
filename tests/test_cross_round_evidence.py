import pytest

from game_qa_agent.analysis import build_task_index
from game_qa_agent.eval import build_deterministic_evaluation_cases
from game_qa_agent.orchestration import run_agent_investigation
from game_qa_agent.report import build_qa_report
from game_qa_agent.scenarios import (
    build_dynamic_scope_state, case_a_runtime_state, dynamic_scope_tasks,
)
from game_qa_agent.tools import build_default_tool_registry
from game_qa_agent.trace import InMemoryInvestigationTraceRecorder

from conftest import ScriptedProvider, action


class ObservingProvider(ScriptedProvider):
    def __init__(self, actions):
        super().__init__(actions)
        self.contexts = []

    def generate_next_action(self, context):
        self.contexts.append(context)
        return super().generate_next_action(context)


def test_repeat_rejection_expansion_rerun_and_report_are_trace_independent() -> None:
    class FailingRecorder:
        calls = 0

        def record_step(self, step):
            self.calls += 1
            raise RuntimeError("offline recorder failure")

        def record_final_status(self, status):
            self.calls += 1
            raise RuntimeError("offline recorder failure")

    tasks = dynamic_scope_tasks()
    actions = [
        action("call_tool", tool_name="npc_runtime_checker"),
        action("call_tool", tool_name="npc_static_conflict_checker"),
        action("call_tool", tool_name="npc_static_conflict_checker"),
        action("expand_scope", expand_task_ids=["task_event_7"]),
        action("call_tool", tool_name="npc_static_conflict_checker"),
        action("finish"),
    ]
    trace = InMemoryInvestigationTraceRecorder()
    failing_trace = FailingRecorder()
    results, contexts = [], []
    for recorder in (None, trace, failing_trace):
        provider = ObservingProvider(actions)
        results.append(run_agent_investigation(
            build_dynamic_scope_state(tasks), build_default_tool_registry(),
            build_task_index(tasks), case_a_runtime_state(), provider,
            max_steps=6, trace_recorder=recorder,
        ))
        contexts.append(provider.contexts)

    assert results[0] == results[1] == results[2]
    assert contexts[0] == contexts[1] == contexts[2]
    state = results[0]
    assert state.investigation_status == "finished"
    assert state.scope_version == 2
    assert state.called_tool_scope_versions == {
        "npc_runtime_checker": [1], "npc_static_conflict_checker": [1, 2],
    }
    assert [record.model_dump() for record in state.tool_executions] == [
        {"execution_number": 1, "tool_name": "npc_runtime_checker",
         "scope_version": 1, "status": "succeeded", "issue_indices": (0, 1)},
        {"execution_number": 2, "tool_name": "npc_static_conflict_checker",
         "scope_version": 1, "status": "succeeded", "issue_indices": ()},
        {"execution_number": 3, "tool_name": "npc_static_conflict_checker",
         "scope_version": 2, "status": "succeeded", "issue_indices": (2,)},
    ]
    assert all("tool_executions" not in context.model_dump() for context in contexts[0])
    assert len(state.decision_errors) == 1
    assert contexts[0][3].last_decision_rejection == "tool_already_called"
    for round_index, blocked in ((2, True), (4, False)):
        capability = next(
            tool for tool in contexts[0][round_index].active_tools
            if tool.name == "npc_static_conflict_checker"
        )
        assert capability.blocked_by_call_history is blocked
    report = build_qa_report(state)
    assert report == build_qa_report(results[1]) == build_qa_report(results[2])
    assert report.issue_type_counts == {
        "npc_location_mismatch": 1,
        "npc_occupied_by_other_task": 1,
        "potential_npc_conflict": 1,
    }
    conflict = next(
        finding for finding in report.findings
        if finding.issue_type == "potential_npc_conflict"
    )
    assert conflict.scoped_task_ids == ("task_b", "task_event_7")
    assert "recorded_findings_only" in report.limitations
    assert [step.outcome for step in trace.steps] == [
        "processed", "processed", "rejected", "processed", "processed", "terminal",
    ]
    assert trace.final_status == "finished"
    assert failing_trace.calls == 7


def test_separate_investigations_do_not_share_scope_history_or_rejection_context() -> None:
    tasks = dynamic_scope_tasks()
    registry = build_default_tool_registry()
    registry_before = dict(registry.tools)
    first_provider = ObservingProvider([
        action("call_tool", tool_name="npc_runtime_checker"),
        action("expand_scope", expand_task_ids=["task_event_7"]),
        action("call_tool", tool_name="private-rejected-tool"),
    ])
    first = run_agent_investigation(
        build_dynamic_scope_state(tasks), registry, build_task_index(tasks),
        case_a_runtime_state(), first_provider, max_steps=3,
    )
    assert first.scope_version == 2
    assert first.last_decision_rejection == "unknown_tool"
    assert len(first.decision_errors) == 1
    first_before = first.model_dump_json()
    first_contexts_before = [context.model_dump_json() for context in first_provider.contexts]

    second_provider = ObservingProvider([
        action("call_tool", tool_name="npc_runtime_checker"), action("finish"),
    ])
    second_initial = build_dynamic_scope_state(tasks)
    second_initial.investigation_goal = "Independent second investigation."
    second = run_agent_investigation(
        second_initial, registry, build_task_index(tasks), case_a_runtime_state(),
        second_provider, max_steps=2,
    )

    initial_context = second_provider.contexts[0]
    assert initial_context.investigation_goal == "Independent second investigation."
    assert initial_context.scope_task_ids == ("task_a", "task_b")
    assert initial_context.expandable_task_ids == ()
    assert initial_context.scope_version == 1
    assert initial_context.last_decision_rejection is None
    assert initial_context.decision_error_count == 0
    assert initial_context.finding_counts == ()
    assert all(
        not tool.blocked_by_call_history and tool.last_called_scope_version is None
        for tool in initial_context.active_tools
    )
    assert second.investigation_status == "finished"
    assert second.scope_version == 1
    assert second.called_tool_scope_versions == {"npc_runtime_checker": [1]}
    assert len(first.tool_executions) == len(second.tool_executions) == 1
    assert second.tool_executions[0].execution_number == 1
    assert second.decision_errors == []
    assert len(second.issues) == 2
    assert first.model_dump_json() == first_before
    assert [context.model_dump_json() for context in first_provider.contexts] == (
        first_contexts_before
    )
    assert registry.tools == registry_before


@pytest.mark.parametrize(("max_steps", "scope_version", "issue_count"), [
    (1, 1, 2), (2, 2, 2), (3, 2, 3), (4, 2, 3),
])
def test_dynamic_investigation_prefix_is_incomplete_until_finish(
    max_steps, scope_version, issue_count,
) -> None:
    case = build_deterministic_evaluation_cases()[1]
    provider = ScriptedProvider(case.scripted_actions)
    trace = InMemoryInvestigationTraceRecorder()
    state = run_agent_investigation(
        case.initial_state, build_default_tool_registry(), build_task_index(case.tasks),
        case.runtime_state, provider, max_steps=max_steps, trace_recorder=trace,
    )

    expected_status = "finished" if max_steps == 4 else "max_steps_exceeded"
    assert state.investigation_status == trace.final_status == expected_status
    assert state.scope_version == scope_version
    assert len(state.issues) == issue_count
    assert len(trace.steps) == max_steps
    assert len(provider.actions) == 4 - max_steps
    report = build_qa_report(state, trace)
    assert report.status == expected_status
    assert ("max_steps_exceeded" in report.limitations) is (max_steps < 4)
    assert "recorded_findings_only" in report.limitations
