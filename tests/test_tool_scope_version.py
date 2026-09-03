from game_qa_agent.analysis import build_task_index
from game_qa_agent.models import GameRuntimeState, Task, ValidationIssue
from game_qa_agent.orchestration import execute_action
from game_qa_agent.tools import ToolRegistry

from conftest import action
from test_action_safety import make_state


def test_same_tool_twice_at_same_scope_version_is_rejected() -> None:
    executions = []

    def checker(tasks):
        executions.append([task.id for task in tasks])
        return []

    registry = ToolRegistry()
    registry.register_tool("npc_static_conflict_checker", checker)
    state = make_state()
    task_index = build_task_index([Task(id="task_a")])

    execute_action(
        action("call_tool", tool_name="npc_static_conflict_checker"),
        state, registry, task_index, GameRuntimeState(),
    )
    execute_action(
        action("call_tool", tool_name="npc_static_conflict_checker"),
        state, registry, task_index, GameRuntimeState(),
    )

    assert executions == [["task_a"]]
    assert state.scope_version == 1
    assert len(state.decision_errors) == 1


def test_same_tool_can_rerun_after_expansion_with_new_trusted_scope() -> None:
    executions = []

    def checker(tasks):
        executions.append([task.id for task in tasks])
        if len(executions) == 1:
            return [ValidationIssue(
                issue_type="external_task_evidence",
                message="Evidence references task_b.",
                task_ids=["task_a", "task_b"],
                checker_name="npc_static_conflict_checker",
            )]
        return []

    registry = ToolRegistry()
    registry.register_tool("npc_static_conflict_checker", checker)
    state = make_state()
    task_index = build_task_index([Task(id="task_a"), Task(id="task_b")])

    execute_action(
        action("call_tool", tool_name="npc_static_conflict_checker"),
        state, registry, task_index, GameRuntimeState(),
    )
    assert state.expandable_task_ids == ["task_b"]
    execute_action(
        action("expand_scope", expand_task_ids=["task_b"]),
        state, registry, task_index, GameRuntimeState(),
    )
    execute_action(
        action("call_tool", tool_name="npc_static_conflict_checker"),
        state, registry, task_index, GameRuntimeState(),
    )

    assert state.scope_version == 2
    assert executions == [["task_a"], ["task_a", "task_b"]]
    assert state.decision_errors == []
