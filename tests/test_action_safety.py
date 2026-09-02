import pytest

from game_qa_agent.analysis import build_task_index, compute_expandable_task_ids
from game_qa_agent.models import (
    AgentInvestigationState,
    GameRuntimeState,
    ImpactAnalysisResult,
    Task,
    ValidationIssue,
)
from game_qa_agent.orchestration import execute_action, run_agent_investigation
from game_qa_agent.tools import build_default_tool_registry

from conftest import ScriptedProvider, action


def make_state() -> AgentInvestigationState:
    return AgentInvestigationState(
        investigation_goal="test",
        impact_analysis=ImpactAnalysisResult(affected_task_ids=["task_a"]),
        scope_task_ids=["task_a"],
    )


@pytest.mark.parametrize("tool_name", ["npc_runtime_checker", "unknown_tool"])
def test_repeated_or_unknown_tool_is_rejected(tool_name: str) -> None:
    state = make_state()
    if tool_name == "npc_runtime_checker":
        state.called_tool_names.append(tool_name)
    result = execute_action(
        action("call_tool", tool_name=tool_name), state,
        build_default_tool_registry(), build_task_index([Task(id="task_a")]),
        GameRuntimeState(),
    )

    assert len(result.decision_errors) == 1
    assert result.scope_task_ids == ["task_a"]


def test_empty_expansion_is_rejected() -> None:
    state = make_state()
    state.expandable_task_ids = ["task_b"]

    result = execute_action(
        action("expand_scope"), state, build_default_tool_registry(),
        build_task_index([Task(id="task_a"), Task(id="task_b")]),
        GameRuntimeState(),
    )

    assert result.scope_task_ids == ["task_a"]
    assert result.scope_version == 1
    assert len(result.decision_errors) == 1


def test_already_scoped_ids_are_not_expandable() -> None:
    state = make_state()
    state.issues.append(ValidationIssue(
        issue_type="evidence", message="test", task_ids=["task_a", "task_b"],
        checker_name="test_checker",
    ))

    assert compute_expandable_task_ids(
        state, build_task_index([Task(id="task_a"), Task(id="task_b")])
    ) == ["task_b"]


def test_max_step_exhaustion_has_explicit_terminal_status() -> None:
    state = make_state()
    provider = ScriptedProvider([
        action("call_tool", tool_name="unknown_tool"),
    ])

    result = run_agent_investigation(
        state,
        build_default_tool_registry(),
        build_task_index([Task(id="task_a")]),
        GameRuntimeState(),
        provider,
        max_steps=1,
    )

    assert result.investigation_status == "max_steps_exceeded"
    assert len(result.decision_errors) == 1
