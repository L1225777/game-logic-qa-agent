import networkx as nx

from .models import AgentInvestigationState, ImpactAnalysisResult, Task


def build_task_index(tasks: list[Task]) -> dict[str, Task]:
    return {task.id: task for task in tasks}


def analyze_initial_change_impact(
    tasks: list[Task], changed_task_ids: list[str]
) -> ImpactAnalysisResult:
    task_index = build_task_index(tasks)
    graph = nx.DiGraph()
    graph.add_nodes_from(task_index)
    for task in tasks:
        for dependency in task.dependencies:
            if dependency in task_index:
                graph.add_edge(dependency, task.id)

    affected = {task_id for task_id in changed_task_ids if task_id in task_index}
    for task_id in list(affected):
        affected.update(nx.ancestors(graph, task_id))
        affected.update(nx.descendants(graph, task_id))

    affected_npcs = {
        requirement.npc_id
        for task_id in affected
        for requirement in task_index[task_id].npc_requirements
    }
    return ImpactAnalysisResult(
        changed_task_ids=sorted(changed_task_ids),
        affected_task_ids=sorted(affected),
        affected_npc_ids=sorted(affected_npcs),
    )


def get_scoped_tasks(
    scope_task_ids: list[str], full_task_index: dict[str, Task]
) -> list[Task]:
    return [
        full_task_index[task_id]
        for task_id in scope_task_ids
        if task_id in full_task_index
    ]


def compute_expandable_task_ids(
    state: AgentInvestigationState, full_task_index: dict[str, Task]
) -> list[str]:
    evidence_task_ids = {
        task_id for issue in state.issues for task_id in issue.task_ids
    }
    current_scope = set(state.scope_task_ids)
    return sorted(
        task_id
        for task_id in evidence_task_ids
        if task_id in full_task_index and task_id not in current_scope
    )


def refresh_expandable_task_ids(
    state: AgentInvestigationState, full_task_index: dict[str, Task]
) -> None:
    state.expandable_task_ids = compute_expandable_task_ids(state, full_task_index)
