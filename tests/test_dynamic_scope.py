from game_qa_agent.analysis import build_task_index
from game_qa_agent.orchestration import run_agent_investigation
from game_qa_agent.scenarios import (
    build_dynamic_scope_state,
    case_a_runtime_state,
    case_b_runtime_state,
    dynamic_scope_tasks,
)
from game_qa_agent.tools import build_default_tool_registry

from conftest import ScriptedProvider, action


def test_dynamic_scope_case_a_expands_only_from_runtime_evidence() -> None:
    tasks = dynamic_scope_tasks()
    state = build_dynamic_scope_state(tasks)
    assert state.scope_task_ids == ["task_a", "task_b"]
    provider = ScriptedProvider([
        action("call_tool", tool_name="npc_runtime_checker"),
        action("expand_scope", expand_task_ids=["task_event_7"]),
        action("call_tool", tool_name="npc_static_conflict_checker"),
        action("finish"),
    ])

    result = run_agent_investigation(
        state, build_default_tool_registry(), build_task_index(tasks),
        case_a_runtime_state(), provider,
    )

    assert result.scope_task_ids == ["task_a", "task_b", "task_event_7"]
    assert result.scope_version == 2
    assert result.expandable_task_ids == []
    assert "task_x" not in result.scope_task_ids
    occupation = next(
        issue for issue in result.issues
        if issue.issue_type == "npc_occupied_by_other_task"
    )
    assert occupation.task_ids == ["task_b", "task_event_7"]


def test_case_a_intermediate_expandable_ids_are_exact() -> None:
    tasks = dynamic_scope_tasks()
    provider = ScriptedProvider([
        action("call_tool", tool_name="npc_runtime_checker"),
    ])

    result = run_agent_investigation(
        build_dynamic_scope_state(tasks), build_default_tool_registry(),
        build_task_index(tasks), case_a_runtime_state(), provider, max_steps=1,
    )

    assert result.expandable_task_ids == ["task_event_7"]
    assert "task_x" not in result.expandable_task_ids


def test_dynamic_scope_case_b_does_not_expand() -> None:
    tasks = dynamic_scope_tasks()
    provider = ScriptedProvider([
        action("call_tool", tool_name="npc_runtime_checker"),
        action("finish"),
    ])

    result = run_agent_investigation(
        build_dynamic_scope_state(tasks), build_default_tool_registry(),
        build_task_index(tasks), case_b_runtime_state(), provider,
    )

    assert result.expandable_task_ids == []
    assert result.scope_task_ids == ["task_a", "task_b"]
    assert result.scope_version == 1


def test_adversarial_task_c_expansion_is_rejected_without_crashing() -> None:
    tasks = dynamic_scope_tasks()
    provider = ScriptedProvider([
        action("call_tool", tool_name="npc_runtime_checker"),
        action("expand_scope", expand_task_ids=["task_c"]),
        action("finish"),
    ])

    result = run_agent_investigation(
        build_dynamic_scope_state(tasks), build_default_tool_registry(),
        build_task_index(tasks), case_a_runtime_state(), provider,
    )

    assert result.expandable_task_ids == ["task_event_7"]
    assert result.scope_task_ids == ["task_a", "task_b"]
    assert result.scope_version == 1
    assert len(result.decision_errors) == 1
    assert "task_c" in result.decision_errors[0]
    assert result.investigation_status == "finished"
