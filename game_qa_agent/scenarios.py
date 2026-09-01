from .analysis import analyze_initial_change_impact, build_task_index
from .models import (
    AgentInvestigationState,
    GameRuntimeState,
    NPCRequirement,
    NPCRuntimeState,
    Task,
)


def dynamic_scope_tasks() -> list[Task]:
    return [
        Task(id="task_a"),
        Task(id="task_b", dependencies=["task_a"], npc_requirements=[
            NPCRequirement(npc_id="blacksmith", location="village", state="available")
        ]),
        Task(id="task_event_7", npc_requirements=[
            NPCRequirement(npc_id="blacksmith", location="town", state="available")
        ]),
        Task(id="task_x", npc_requirements=[
            NPCRequirement(npc_id="healer", location="forest", state="available")
        ]),
    ]


def case_a_runtime_state() -> GameRuntimeState:
    return GameRuntimeState(
        task_statuses={"task_a": "inactive", "task_b": "active",
                       "task_event_7": "active", "task_x": "active"},
        npc_states={
            "blacksmith": NPCRuntimeState(
                location="town", state="available", occupied_by_task_id="task_event_7"
            ),
            "healer": NPCRuntimeState(
                location="castle", state="available", occupied_by_task_id="task_x"
            ),
        },
    )


def case_b_runtime_state() -> GameRuntimeState:
    return GameRuntimeState(
        task_statuses={"task_a": "inactive", "task_b": "active",
                       "task_event_7": "inactive", "task_x": "active"},
        npc_states={
            "blacksmith": NPCRuntimeState(
                location="village", state="available", occupied_by_task_id=None
            ),
            "healer": NPCRuntimeState(
                location="castle", state="available", occupied_by_task_id="task_x"
            ),
        },
    )


def build_dynamic_scope_state(tasks: list[Task]) -> AgentInvestigationState:
    impact = analyze_initial_change_impact(tasks, ["task_a"])
    return AgentInvestigationState(
        investigation_goal="Investigate evidence-backed dynamic scope expansion.",
        impact_analysis=impact,
        scope_task_ids=impact.affected_task_ids,
    )


def dynamic_scope_index() -> dict[str, Task]:
    return build_task_index(dynamic_scope_tasks())
