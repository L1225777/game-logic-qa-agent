import pytest

from game_qa_agent.analysis import build_task_index
from game_qa_agent.context import ProviderDecisionContext
from game_qa_agent.models import (
    AgentInvestigationState,
    GameRuntimeState,
    ImpactAnalysisResult,
    NextActionSpec,
    Task,
)
from game_qa_agent.orchestration import run_agent_investigation
from game_qa_agent.scenarios import (
    build_dynamic_scope_state,
    case_a_runtime_state,
    dynamic_scope_tasks,
)
from game_qa_agent.tools import build_default_tool_registry
from game_qa_agent.trace import InMemoryInvestigationTraceRecorder

from conftest import ScriptedProvider, action


def make_state() -> AgentInvestigationState:
    return AgentInvestigationState(
        investigation_goal="trace test",
        impact_analysis=ImpactAnalysisResult(affected_task_ids=["task_a"]),
        scope_task_ids=["task_a"],
    )


def test_normal_investigation_trace_is_deterministic() -> None:
    recorder = InMemoryInvestigationTraceRecorder()
    result = run_agent_investigation(
        make_state(),
        build_default_tool_registry(),
        build_task_index([Task(id="task_a")]),
        GameRuntimeState(),
        ScriptedProvider([
            action("call_tool", tool_name="dependency_reference_checker"),
            action("finish"),
        ]),
        trace_recorder=recorder,
    )

    assert result.investigation_status == "finished"
    assert [step.action.action_type for step in recorder.steps] == [
        "call_tool", "finish"
    ]
    assert [step.step_number for step in recorder.steps] == [1, 2]
    assert [step.investigation_status_before for step in recorder.steps] == [
        "start", "running"
    ]
    assert [step.investigation_status_after for step in recorder.steps] == [
        "running", "finished"
    ]
    assert [step.outcome for step in recorder.steps] == ["processed", "terminal"]
    assert recorder.final_status == "finished"


def test_dynamic_scope_trace_records_authorized_scope_change() -> None:
    tasks = dynamic_scope_tasks()
    recorder = InMemoryInvestigationTraceRecorder()
    run_agent_investigation(
        build_dynamic_scope_state(tasks),
        build_default_tool_registry(),
        build_task_index(tasks),
        case_a_runtime_state(),
        ScriptedProvider([
            action("call_tool", tool_name="npc_runtime_checker"),
            action("expand_scope", expand_task_ids=["task_event_7"]),
            action("finish"),
        ]),
        trace_recorder=recorder,
    )

    expansion = recorder.steps[1]
    assert recorder.steps[0].new_issue_types == [
        "npc_location_mismatch", "npc_occupied_by_other_task"
    ]
    assert expansion.scope_version_before == 1
    assert expansion.scope_version_after == 2
    assert expansion.scope_task_ids_before == ["task_a", "task_b"]
    assert expansion.scope_task_ids_after == [
        "task_a", "task_b", "task_event_7"
    ]
    assert expansion.outcome == "processed"


def test_rejected_scope_expansion_trace_preserves_trusted_scope() -> None:
    tasks = dynamic_scope_tasks()
    recorder = InMemoryInvestigationTraceRecorder()
    result = run_agent_investigation(
        build_dynamic_scope_state(tasks),
        build_default_tool_registry(),
        build_task_index(tasks),
        case_a_runtime_state(),
        ScriptedProvider([
            action("call_tool", tool_name="npc_runtime_checker"),
            action("expand_scope", expand_task_ids=["task_c"]),
            action("finish"),
        ]),
        trace_recorder=recorder,
    )

    rejected = recorder.steps[1]
    assert rejected.outcome == "rejected"
    assert rejected.new_decision_error_count == 1
    assert rejected.action.requested_expansion_count == 1
    assert "task_c" not in repr(recorder.__dict__)
    assert rejected.scope_version_before == rejected.scope_version_after == 1
    assert rejected.scope_task_ids_before == rejected.scope_task_ids_after == [
        "task_a", "task_b"
    ]
    assert result.scope_task_ids == ["task_a", "task_b"]


def test_trace_recorder_failure_does_not_change_agent_result() -> None:
    class FailingRecorder:
        def record_step(self, step: object) -> None:
            raise RuntimeError("trace unavailable")

        def record_final_status(self, status: object) -> None:
            raise RuntimeError("trace unavailable")

    tasks = [Task(id="task_a")]
    actions = [
        action("call_tool", tool_name="dependency_reference_checker"),
        action("finish"),
    ]
    without_trace = run_agent_investigation(
        make_state(), build_default_tool_registry(), build_task_index(tasks),
        GameRuntimeState(), ScriptedProvider(actions),
    )
    with_failing_trace = run_agent_investigation(
        make_state(), build_default_tool_registry(), build_task_index(tasks),
        GameRuntimeState(), ScriptedProvider(actions),
        trace_recorder=FailingRecorder(),
    )

    assert with_failing_trace == without_trace


def test_trace_records_max_step_final_status() -> None:
    recorder = InMemoryInvestigationTraceRecorder()
    run_agent_investigation(
        make_state(),
        build_default_tool_registry(),
        build_task_index([Task(id="task_a")]),
        GameRuntimeState(),
        ScriptedProvider([action("call_tool", tool_name="unknown_tool")]),
        max_steps=1,
        trace_recorder=recorder,
    )

    assert recorder.steps[0].outcome == "rejected"
    assert recorder.steps[0].investigation_status_after == "running"
    assert recorder.final_status == "max_steps_exceeded"


def test_trace_does_not_retain_tool_arguments() -> None:
    recorder = InMemoryInvestigationTraceRecorder()
    proposed_action = action(
        "call_tool", tool_name="dependency_reference_checker"
    )
    proposed_action.tool_args = {"credential": "must-not-be-recorded"}
    run_agent_investigation(
        make_state(),
        build_default_tool_registry(),
        build_task_index([Task(id="task_a")]),
        GameRuntimeState(),
        ScriptedProvider([proposed_action]),
        max_steps=1,
        trace_recorder=recorder,
    )

    recorded_step = recorder.steps[0]
    assert recorded_step.outcome == "rejected"
    assert recorded_step.action.had_tool_args is True
    assert "must-not-be-recorded" not in recorded_step.model_dump_json()
    assert recorded_step.new_decision_error_count == 1
    assert "reason" not in recorded_step.action.model_dump()
    assert "tool_args" not in recorded_step.action.model_dump()
    assert "expand_task_ids" not in recorded_step.action.model_dump()
    assert "new_decision_errors" not in recorded_step.model_dump()


@pytest.mark.parametrize(
    ("reason", "tool_name", "marker"),
    [
        ("reason-sensitive-marker", "unknown_tool", "reason-sensitive-marker"),
        (
            "ordinary reason",
            "unknown-tool-sensitive-marker",
            "unknown-tool-sensitive-marker",
        ),
    ],
)
def test_trace_excludes_provider_controlled_text(
    reason: str, tool_name: str, marker: str
) -> None:
    recorder = InMemoryInvestigationTraceRecorder()
    run_agent_investigation(
        make_state(),
        build_default_tool_registry(),
        build_task_index([Task(id="task_a")]),
        GameRuntimeState(),
        ScriptedProvider([
            action("call_tool", tool_name=tool_name, reason=reason)
        ]),
        max_steps=1,
        trace_recorder=recorder,
    )

    recorded_step = recorder.steps[0]
    assert marker not in repr(recorder.__dict__)
    assert marker not in recorded_step.model_dump_json()
    assert recorded_step.action.tool_name_supplied is True
    assert recorded_step.action.trusted_tool_name is None
    assert recorded_step.new_decision_error_count == 1


@pytest.mark.parametrize("max_steps", [0, -1])
def test_invalid_max_steps_does_not_call_provider_or_recorder(
    max_steps: int,
) -> None:
    class SpyProvider:
        def __init__(self) -> None:
            self.call_count = 0

        def generate_next_action(
            self, context: ProviderDecisionContext
        ) -> NextActionSpec:
            self.call_count += 1
            return action("finish")

    class SpyRecorder:
        def __init__(self) -> None:
            self.call_count = 0

        def record_step(self, step: object) -> None:
            self.call_count += 1

        def record_final_status(self, status: object) -> None:
            self.call_count += 1

    provider = SpyProvider()
    recorder = SpyRecorder()

    with pytest.raises(ValueError, match="max_steps must be at least 1"):
        run_agent_investigation(
            make_state(),
            build_default_tool_registry(),
            build_task_index([Task(id="task_a")]),
            GameRuntimeState(),
            provider,
            max_steps=max_steps,
            trace_recorder=recorder,
        )

    assert provider.call_count == 0
    assert recorder.call_count == 0


def test_provider_context_never_contains_trace_data() -> None:
    class InspectingProvider:
        def __init__(self) -> None:
            self.observed_contexts: list[dict[str, object]] = []

        def generate_next_action(
            self, context: ProviderDecisionContext
        ) -> NextActionSpec:
            self.observed_contexts.append(context.model_dump())
            if len(self.observed_contexts) == 1:
                return action(
                    "call_tool", tool_name="dependency_reference_checker"
                )
            return action("finish")

    provider = InspectingProvider()
    recorder = InMemoryInvestigationTraceRecorder()
    run_agent_investigation(
        make_state(),
        build_default_tool_registry(),
        build_task_index([Task(id="task_a")]),
        GameRuntimeState(),
        provider,
        trace_recorder=recorder,
    )

    assert len(provider.observed_contexts) == 2
    assert len(recorder.steps) == 2
    for observed_context in provider.observed_contexts:
        assert not any("trace" in key.lower() for key in observed_context)
