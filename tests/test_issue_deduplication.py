from game_qa_agent.analysis import build_task_index
from game_qa_agent.models import GameRuntimeState, Task, ValidationIssue
from game_qa_agent.orchestration import execute_action
from game_qa_agent.tools import ToolRegistry

from conftest import action
from test_action_safety import make_state


def test_checker_rerun_deduplicates_old_issue_and_preserves_new_issue() -> None:
    executions = []
    old_issue = ValidationIssue(
        issue_type="old_scope_finding",
        message="task_a has an existing finding.",
        task_ids=["task_a"],
        evidence={"fact": "old"},
        checker_name="npc_static_conflict_checker",
    )
    expansion_evidence = ValidationIssue(
        issue_type="external_task_evidence",
        message="Evidence references task_b.",
        task_ids=["task_a", "task_b"],
        evidence={"source": "trusted"},
        checker_name="npc_static_conflict_checker",
    )
    new_scope_issue = ValidationIssue(
        issue_type="new_scope_finding",
        message="Expanded scope reveals a distinct finding.",
        task_ids=["task_a", "task_b"],
        evidence={"fact": "new"},
        checker_name="npc_static_conflict_checker",
    )

    def checker(tasks):
        executions.append([task.id for task in tasks])
        if len(executions) == 1:
            return [old_issue, expansion_evidence]
        return [old_issue, new_scope_issue]

    registry = ToolRegistry()
    registry.register_tool("npc_static_conflict_checker", checker)
    state = make_state()
    task_index = build_task_index([Task(id="task_a"), Task(id="task_b")])

    execute_action(
        action("call_tool", tool_name="npc_static_conflict_checker"),
        state, registry, task_index, GameRuntimeState(),
    )
    execute_action(
        action("expand_scope", expand_task_ids=["task_b"]),
        state, registry, task_index, GameRuntimeState(),
    )
    execute_action(
        action("call_tool", tool_name="npc_static_conflict_checker"),
        state, registry, task_index, GameRuntimeState(),
    )

    assert executions == [["task_a"], ["task_a", "task_b"]]
    assert state.issues.count(old_issue) == 1
    assert state.issues == [old_issue, expansion_evidence, new_scope_issue]
