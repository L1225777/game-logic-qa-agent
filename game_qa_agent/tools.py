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


class ToolRegistry:
    def __init__(self) -> None:
        self.tools: dict[str, ToolFunction] = {}

    def register_tool(self, name: str, tool_function: ToolFunction) -> None:
        self.tools[name] = tool_function

    def get_tool(self, name: str) -> ToolFunction:
        return self.tools[name]


def build_default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_tool("dependency_cycle_checker", detect_dependency_cycles)
    registry.register_tool("dependency_reference_checker", detect_missing_dependencies)
    registry.register_tool("npc_static_conflict_checker", detect_potential_npc_conflicts)
    registry.register_tool("npc_runtime_checker", check_npc_runtime_requirements)
    return registry


def build_tool_inputs(
    tool_name: str,
    state: AgentInvestigationState,
    full_task_index: dict[str, Task],
    full_game_runtime_state: GameRuntimeState,
) -> dict[str, Any]:
    scoped_tasks = get_scoped_tasks(state.scope_task_ids, full_task_index)
    if tool_name == "dependency_reference_checker":
        return {"target_tasks": scoped_tasks, "full_task_index": full_task_index}
    if tool_name in {"dependency_cycle_checker", "npc_static_conflict_checker"}:
        return {"tasks": scoped_tasks}
    if tool_name == "npc_runtime_checker":
        return {"tasks": scoped_tasks, "game_runtime_state": full_game_runtime_state}
    raise ValueError(f"No tool input builder for '{tool_name}'.")
