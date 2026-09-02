from game_qa_agent.analysis import build_task_index
from game_qa_agent.models import GameRuntimeState, NextActionSpec, Task
from game_qa_agent.orchestration import execute_action
from game_qa_agent.providers import normalize_provider_action
from game_qa_agent.tools import ToolRegistry

from test_action_safety import make_state


def test_overpopulated_expand_scope_is_canonical_and_preserves_task_id() -> None:
    raw = NextActionSpec(
        action_type="expand_scope",
        tool_name="npc_runtime_checker",
        tool_args={"untrusted": True},
        expand_task_ids=["task_event_7"],
        reason="runtime evidence identified an external task",
    )

    normalized = normalize_provider_action(raw)

    assert normalized == NextActionSpec(
        action_type="expand_scope",
        expand_task_ids=["task_event_7"],
        reason=raw.reason,
    )


def test_overpopulated_finish_is_canonical() -> None:
    raw = NextActionSpec(
        action_type="finish",
        tool_name="npc_runtime_checker",
        tool_args={"untrusted": True},
        expand_task_ids=["task_event_7"],
        reason="no more evidence-backed work remains",
    )

    normalized = normalize_provider_action(raw)

    assert normalized == NextActionSpec(
        action_type="finish",
        reason=raw.reason,
    )


def test_call_tool_discards_llm_args_and_execution_uses_trusted_inputs() -> None:
    captured_inputs = {}

    def capturing_runtime_checker(**kwargs):
        captured_inputs.update(kwargs)
        return []

    registry = ToolRegistry()
    registry.register_tool("npc_runtime_checker", capturing_runtime_checker)
    raw = NextActionSpec(
        action_type="call_tool",
        tool_name="npc_runtime_checker",
        tool_args={"tasks": ["invented"], "game_runtime_state": "untrusted"},
        expand_task_ids=["task_c"],
        reason="check runtime",
    )
    normalized = normalize_provider_action(raw)
    state = make_state()
    trusted_task = Task(id="task_a")
    trusted_runtime = GameRuntimeState(task_statuses={"task_a": "active"})

    execute_action(
        normalized,
        state,
        registry,
        build_task_index([trusted_task]),
        trusted_runtime,
    )

    assert normalized.tool_args == {}
    assert normalized.expand_task_ids == []
    assert captured_inputs == {
        "tasks": [trusted_task],
        "game_runtime_state": trusted_runtime,
    }
