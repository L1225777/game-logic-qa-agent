import networkx as nx

from .models import GameRuntimeState, Task, ValidationIssue


def detect_dependency_cycles(tasks: list[Task]) -> list[ValidationIssue]:
    graph = nx.DiGraph()
    graph.add_nodes_from(task.id for task in tasks)
    for task in tasks:
        for dependency in task.dependencies:
            graph.add_edge(dependency, task.id)
    try:
        cycle_edges = nx.find_cycle(graph)
    except nx.NetworkXNoCycle:
        return []
    cycle_task_ids = sorted({node for edge in cycle_edges for node in edge})
    return [ValidationIssue(
        issue_type="dependency_cycle",
        message="A dependency cycle was detected.",
        task_ids=cycle_task_ids,
        checker_name="dependency_cycle_checker",
    )]


def detect_missing_dependencies(
    target_tasks: list[Task], full_task_index: dict[str, Task]
) -> list[ValidationIssue]:
    issues = []
    for task in target_tasks:
        for dependency in task.dependencies:
            if dependency not in full_task_index:
                issues.append(ValidationIssue(
                    issue_type="missing_dependency",
                    message=f"Task '{task.id}' depends on missing task '{dependency}'.",
                    task_ids=[task.id],
                    evidence={"missing_dependency_id": dependency},
                    checker_name="dependency_reference_checker",
                ))
    return issues


def tasks_may_overlap(first_task: Task, second_task: Task) -> bool:
    return not (
        first_task.exclusive_group is not None
        and first_task.exclusive_group == second_task.exclusive_group
    )


def detect_potential_npc_conflicts(tasks: list[Task]) -> list[ValidationIssue]:
    issues = []
    for first_index, first_task in enumerate(tasks):
        for second_task in tasks[first_index + 1:]:
            if not tasks_may_overlap(first_task, second_task):
                continue
            for first_requirement in first_task.npc_requirements:
                for second_requirement in second_task.npc_requirements:
                    if (
                        first_requirement.npc_id == second_requirement.npc_id
                        and first_requirement.location != second_requirement.location
                    ):
                        issues.append(ValidationIssue(
                            issue_type="potential_npc_conflict",
                            message=(f"Tasks '{first_task.id}' and '{second_task.id}' may "
                                     f"overlap and require NPC '{first_requirement.npc_id}' "
                                     "in different locations."),
                            task_ids=[first_task.id, second_task.id],
                            npc_ids=[first_requirement.npc_id],
                            evidence={
                                "first_required_location": first_requirement.location,
                                "second_required_location": second_requirement.location,
                            },
                            checker_name="npc_static_conflict_checker",
                        ))
    return issues


def check_npc_runtime_requirements(
    tasks: list[Task], game_runtime_state: GameRuntimeState
) -> list[ValidationIssue]:
    issues = []
    for task in tasks:
        if game_runtime_state.task_statuses.get(task.id) != "active":
            continue
        for requirement in task.npc_requirements:
            runtime = game_runtime_state.npc_states.get(requirement.npc_id)
            if runtime is None:
                issues.append(ValidationIssue(
                    issue_type="missing_npc_runtime_state",
                    message=f"No runtime state found for NPC '{requirement.npc_id}'.",
                    task_ids=[task.id], npc_ids=[requirement.npc_id],
                    checker_name="npc_runtime_checker",
                ))
                continue
            if runtime.location != requirement.location:
                issues.append(ValidationIssue(
                    issue_type="npc_location_mismatch",
                    message=(f"Active task '{task.id}' requires NPC '{requirement.npc_id}' "
                             f"at '{requirement.location}', but the NPC is currently at "
                             f"'{runtime.location}'."),
                    task_ids=[task.id], npc_ids=[requirement.npc_id],
                    evidence={"required_location": requirement.location,
                              "actual_location": runtime.location},
                    checker_name="npc_runtime_checker",
                ))
            if runtime.state != requirement.state:
                issues.append(ValidationIssue(
                    issue_type="npc_state_mismatch",
                    message=(f"Active task '{task.id}' requires NPC '{requirement.npc_id}' "
                             f"in state '{requirement.state}', but the NPC is currently "
                             f"in state '{runtime.state}'."),
                    task_ids=[task.id], npc_ids=[requirement.npc_id],
                    evidence={"required_state": requirement.state,
                              "actual_state": runtime.state},
                    checker_name="npc_runtime_checker",
                ))
            if runtime.occupied_by_task_id is not None and runtime.occupied_by_task_id != task.id:
                issues.append(ValidationIssue(
                    issue_type="npc_occupied_by_other_task",
                    message=(f"NPC '{requirement.npc_id}' needed by active task '{task.id}' "
                             f"is currently occupied by task '{runtime.occupied_by_task_id}'."),
                    task_ids=[task.id, runtime.occupied_by_task_id],
                    npc_ids=[requirement.npc_id],
                    evidence={"occupied_by_task_id": runtime.occupied_by_task_id},
                    checker_name="npc_runtime_checker",
                ))
    return issues
