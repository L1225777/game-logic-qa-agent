from collections.abc import Callable
from typing import Any

from .analysis import get_scoped_tasks
from .checkers import (
    check_npc_runtime_requirements,
    detect_dependency_cycles,
    detect_missing_dependencies,
    detect_potential_npc_conflicts,
)
from .models import AgentInvestigationState, GameRuntimeState, Task, ValidationIssue


ToolFunction = Callable[..., list[ValidationIssue]]


_TOOL_INPUT_KINDS = {
    "dependency_reference_checker": "dependencies",
    "dependency_cycle_checker": "scoped_tasks",
    "npc_static_conflict_checker": "scoped_tasks",
    "npc_runtime_checker": "runtime",
}


def _tool_input_kind(name: str) -> str:
    if name not in _TOOL_INPUT_KINDS:
        raise ValueError("Tool has no trusted input contract.")
    return _TOOL_INPUT_KINDS[name]


class ToolRegistry:
    def __init__(self) -> None:
        self.tools: dict[str, ToolFunction] = {}

    def register_tool(self, name: str, tool_function: ToolFunction) -> None:
        _tool_input_kind(name)
        self.tools[name] = tool_function

    def get_tool(self, name: str) -> ToolFunction:
        return self.tools[name]

    def active_tool_names(self) -> tuple[str, ...]:
        # Validate direct edits to the public registry as well as registration.
        for name in self.tools:
            _tool_input_kind(name)
        return tuple(sorted(self.tools))


def build_default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_tool("dependency_cycle_checker", detect_dependency_cycles)
    registry.register_tool("dependency_reference_checker", detect_missing_dependencies)
    registry.register_tool("npc_static_conflict_checker", detect_potential_npc_conflicts)
    registry.register_tool("npc_runtime_checker", check_npc_runtime_requirements)
    return registry


def tool_call_is_blocked(tool_name: str, state: AgentInvestigationState) -> bool:
    """The controller's existing per-scope and legacy call-history rule."""
    return (
        state.scope_version in state.called_tool_scope_versions.get(tool_name, [])
        or (
            tool_name in state.called_tool_names
            and tool_name not in state.called_tool_scope_versions
        )
    )


def build_tool_inputs(
    tool_name: str,
    state: AgentInvestigationState,
    full_task_index: dict[str, Task],
    full_game_runtime_state: GameRuntimeState,
) -> dict[str, Any]:
    input_kind = _tool_input_kind(tool_name)
    scoped_tasks = get_scoped_tasks(state.scope_task_ids, full_task_index)
    if input_kind == "dependencies":
        return {"target_tasks": scoped_tasks, "full_task_index": full_task_index}
    if input_kind == "runtime":
        return {"tasks": scoped_tasks, "game_runtime_state": full_game_runtime_state}
    return {"tasks": scoped_tasks}
