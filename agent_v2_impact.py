import os
from typing import Any, Literal

import networkx as nx
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.shared_params.response_format_json_object import (
    ResponseFormatJSONObject,
)
from pydantic import BaseModel, Field


load_dotenv()


# =========================================================
# 1. Game configuration and runtime models
# =========================================================

TaskStatus = Literal[
    "inactive",
    "active",
    "paused",
    "completed",
]


class NPCRequirement(BaseModel):
    npc_id: str
    location: str
    state: str


class Task(BaseModel):
    id: str
    dependencies: list[str] = Field(default_factory=list)
    npc_requirements: list[NPCRequirement] = Field(default_factory=list)
    exclusive_group: str | None = None


class NPCRuntimeState(BaseModel):
    location: str
    state: str


class GameRuntimeState(BaseModel):
    task_statuses: dict[str, TaskStatus] = Field(default_factory=dict)
    npc_states: dict[str, NPCRuntimeState] = Field(default_factory=dict)


class ValidationIssue(BaseModel):
    issue_type: str
    message: str
    task_ids: list[str] = Field(default_factory=list)
    npc_ids: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    checker_name: str


# =========================================================
# 2. Impact Analysis
# =========================================================

class ImpactAnalysisResult(BaseModel):
    changed_task_ids: list[str] = Field(default_factory=list)
    affected_task_ids: list[str] = Field(default_factory=list)
    affected_npc_ids: list[str] = Field(default_factory=list)


def build_task_index(
    tasks: list[Task],
) -> dict[str, Task]:

    return {
        task.id: task
        for task in tasks
    }


def analyze_change_impact(
    tasks: list[Task],
    changed_task_ids: list[str],
) -> ImpactAnalysisResult:

    task_by_id = build_task_index(tasks)

    dependency_graph = nx.DiGraph()

    for task in tasks:
        dependency_graph.add_node(task.id)

    for task in tasks:
        for dependency in task.dependencies:

            if dependency in task_by_id:
                dependency_graph.add_edge(
                    dependency,
                    task.id,
                )

    affected_task_ids = {
        task_id
        for task_id in changed_task_ids
        if task_id in task_by_id
    }

    scope_changed = True

    while scope_changed:

        previous_affected_task_ids = set(
            affected_task_ids
        )

        # Dependency-related tasks
        for task_id in list(
            affected_task_ids
        ):

            affected_task_ids.update(
                nx.ancestors(
                    dependency_graph,
                    task_id,
                )
            )

            affected_task_ids.update(
                nx.descendants(
                    dependency_graph,
                    task_id,
                )
            )

        # NPCs used by currently affected tasks
        affected_npc_ids = {
            npc_requirement.npc_id
            for task_id in affected_task_ids
            for npc_requirement
            in task_by_id[
                task_id
            ].npc_requirements
        }

        # Exclusive groups related to affected tasks
        affected_exclusive_groups = {
            task_by_id[
                task_id
            ].exclusive_group
            for task_id in affected_task_ids
            if task_by_id[
                task_id
            ].exclusive_group is not None
        }

        # Expand to tasks sharing NPCs or exclusive groups
        for task in tasks:

            shares_affected_npc = any(
                npc_requirement.npc_id
                in affected_npc_ids
                for npc_requirement
                in task.npc_requirements
            )

            shares_exclusive_group = (
                task.exclusive_group is not None
                and task.exclusive_group
                in affected_exclusive_groups
            )

            if (
                shares_affected_npc
                or shares_exclusive_group
            ):
                affected_task_ids.add(
                    task.id
                )

        scope_changed = (
            affected_task_ids
            != previous_affected_task_ids
        )

    final_affected_npc_ids = {
        npc_requirement.npc_id
        for task_id in affected_task_ids
        for npc_requirement
        in task_by_id[
            task_id
        ].npc_requirements
    }

    return ImpactAnalysisResult(
        changed_task_ids=sorted(
            changed_task_ids
        ),
        affected_task_ids=sorted(
            affected_task_ids
        ),
        affected_npc_ids=sorted(
            final_affected_npc_ids
        ),
    )


def build_scoped_tasks(
    tasks: list[Task],
    impact_analysis_result: ImpactAnalysisResult,
) -> list[Task]:

    affected_task_ids = set(
        impact_analysis_result.affected_task_ids
    )

    return [
        task
        for task in tasks
        if task.id in affected_task_ids
    ]


def build_scoped_runtime_state(
    game_runtime_state: GameRuntimeState,
    impact_analysis_result: ImpactAnalysisResult,
) -> GameRuntimeState:

    affected_task_ids = set(
        impact_analysis_result.affected_task_ids
    )

    affected_npc_ids = set(
        impact_analysis_result.affected_npc_ids
    )

    scoped_task_statuses = {
        task_id: task_status
        for task_id, task_status
        in game_runtime_state.task_statuses.items()
        if task_id in affected_task_ids
    }

    scoped_npc_states = {
        npc_id: npc_runtime_state
        for npc_id, npc_runtime_state
        in game_runtime_state.npc_states.items()
        if npc_id in affected_npc_ids
    }

    return GameRuntimeState(
        task_statuses=scoped_task_statuses,
        npc_states=scoped_npc_states,
    )


# =========================================================
# 3. Dependency checkers
# =========================================================

def detect_dependency_cycles(
    tasks: list[Task],
) -> list[ValidationIssue]:

    graph = nx.DiGraph()

    for task in tasks:
        graph.add_node(
            task.id
        )

    for task in tasks:
        for dependency in task.dependencies:

            graph.add_edge(
                dependency,
                task.id,
            )

    try:

        cycle_edges = nx.find_cycle(
            graph
        )

        cycle_task_ids = list(
            {
                node
                for edge in cycle_edges
                for node in edge
            }
        )

        return [
            ValidationIssue(
                issue_type="dependency_cycle",
                message=(
                    "A dependency cycle was detected."
                ),
                task_ids=cycle_task_ids,
                checker_name=(
                    "dependency_cycle_checker"
                ),
            )
        ]

    except nx.NetworkXNoCycle:

        return []


def detect_missing_dependencies(
    target_tasks: list[Task],
    full_task_index: dict[str, Task],
) -> list[ValidationIssue]:

    validation_issues = []

    for task in target_tasks:

        for dependency in task.dependencies:

            if dependency not in full_task_index:

                validation_issues.append(
                    ValidationIssue(
                        issue_type=(
                            "missing_dependency"
                        ),
                        message=(
                            f"Task '{task.id}' depends on missing task "
                            f"'{dependency}'."
                        ),
                        task_ids=[
                            task.id
                        ],
                        evidence={
                            "missing_dependency_id":
                                dependency,
                        },
                        checker_name=(
                            "dependency_reference_checker"
                        ),
                    )
                )

    return validation_issues


def verify_dependency_reference_source_of_truth_boundary() -> None:

    scoped_target_task = Task(
        id="scope_test_task",
        dependencies=[
            "global_existing_task"
        ],
    )

    full_task_index = build_task_index(
        [
            scoped_target_task,
            Task(
                id="global_existing_task"
            ),
        ]
    )

    existing_reference_issues = (
        detect_missing_dependencies(
            target_tasks=[
                scoped_target_task
            ],
            full_task_index=(
                full_task_index
            ),
        )
    )

    assert (
        existing_reference_issues
        == []
    )

    missing_reference_task = Task(
        id="missing_reference_test_task",
        dependencies=[
            "actually_missing_task"
        ],
    )

    missing_reference_issues = (
        detect_missing_dependencies(
            target_tasks=[
                missing_reference_task
            ],
            full_task_index=(
                full_task_index
            ),
        )
    )

    assert (
        len(
            missing_reference_issues
        )
        == 1
    )

    assert (
        missing_reference_issues[
            0
        ].issue_type
        == "missing_dependency"
    )


# =========================================================
# 4. Static NPC conflict checker
# =========================================================

def tasks_may_overlap(
    first_task: Task,
    second_task: Task,
) -> bool:

    if (
        first_task.exclusive_group
        is not None
        and first_task.exclusive_group
        == second_task.exclusive_group
    ):
        return False

    return True


def detect_potential_npc_conflicts(
    tasks: list[Task],
) -> list[ValidationIssue]:

    validation_issues = []

    for first_index in range(
        len(tasks)
    ):

        for second_index in range(
            first_index + 1,
            len(tasks),
        ):

            first_task = tasks[
                first_index
            ]

            second_task = tasks[
                second_index
            ]

            if not tasks_may_overlap(
                first_task,
                second_task,
            ):
                continue

            for first_requirement in (
                first_task.npc_requirements
            ):

                for second_requirement in (
                    second_task.npc_requirements
                ):

                    if (
                        first_requirement.npc_id
                        != second_requirement.npc_id
                    ):
                        continue

                    location_conflict = (
                        first_requirement.location
                        != second_requirement.location
                    )

                    if location_conflict:

                        validation_issues.append(
                            ValidationIssue(
                                issue_type=(
                                    "potential_npc_conflict"
                                ),
                                message=(
                                    f"Tasks '{first_task.id}' and "
                                    f"'{second_task.id}' may overlap and "
                                    f"require NPC "
                                    f"'{first_requirement.npc_id}' "
                                    f"in different locations."
                                ),
                                task_ids=[
                                    first_task.id,
                                    second_task.id,
                                ],
                                npc_ids=[
                                    first_requirement.npc_id
                                ],
                                evidence={
                                    "first_required_location":
                                        first_requirement.location,
                                    "second_required_location":
                                        second_requirement.location,
                                },
                                checker_name=(
                                    "npc_static_conflict_checker"
                                ),
                            )
                        )

    return validation_issues


# =========================================================
# 5. Runtime NPC checker
# =========================================================

def check_npc_runtime_requirements(
    tasks: list[Task],
    game_runtime_state: GameRuntimeState,
) -> list[ValidationIssue]:

    validation_issues = []

    for task in tasks:

        if (
            game_runtime_state.task_statuses.get(
                task.id
            )
            != "active"
        ):
            continue

        for npc_requirement in (
            task.npc_requirements
        ):

            npc_runtime_state = (
                game_runtime_state
                .npc_states
                .get(
                    npc_requirement.npc_id
                )
            )

            if npc_runtime_state is None:

                validation_issues.append(
                    ValidationIssue(
                        issue_type=(
                            "missing_npc_runtime_state"
                        ),
                        message=(
                            f"No runtime state found for NPC "
                            f"'{npc_requirement.npc_id}'."
                        ),
                        task_ids=[
                            task.id
                        ],
                        npc_ids=[
                            npc_requirement.npc_id
                        ],
                        checker_name=(
                            "npc_runtime_checker"
                        ),
                    )
                )

                continue

            if (
                npc_runtime_state.location
                != npc_requirement.location
            ):

                validation_issues.append(
                    ValidationIssue(
                        issue_type=(
                            "npc_location_mismatch"
                        ),
                        message=(
                            f"Active task '{task.id}' requires NPC "
                            f"'{npc_requirement.npc_id}' at "
                            f"'{npc_requirement.location}', but the NPC "
                            f"is currently at "
                            f"'{npc_runtime_state.location}'."
                        ),
                        task_ids=[
                            task.id
                        ],
                        npc_ids=[
                            npc_requirement.npc_id
                        ],
                        evidence={
                            "required_location":
                                npc_requirement.location,
                            "actual_location":
                                npc_runtime_state.location,
                        },
                        checker_name=(
                            "npc_runtime_checker"
                        ),
                    )
                )

            if (
                npc_runtime_state.state
                != npc_requirement.state
            ):

                validation_issues.append(
                    ValidationIssue(
                        issue_type=(
                            "npc_state_mismatch"
                        ),
                        message=(
                            f"Active task '{task.id}' requires NPC "
                            f"'{npc_requirement.npc_id}' in state "
                            f"'{npc_requirement.state}', but the NPC "
                            f"is currently in state "
                            f"'{npc_runtime_state.state}'."
                        ),
                        task_ids=[
                            task.id
                        ],
                        npc_ids=[
                            npc_requirement.npc_id
                        ],
                        evidence={
                            "required_state":
                                npc_requirement.state,
                            "actual_state":
                                npc_runtime_state.state,
                        },
                        checker_name=(
                            "npc_runtime_checker"
                        ),
                    )
                )

    return validation_issues


# =========================================================
# 6. Agent Tools
# =========================================================

def dependency_cycle_checker_tool(
    tasks: list[Task],
) -> list[ValidationIssue]:

    return detect_dependency_cycles(
        tasks
    )


def dependency_reference_checker_tool(
    target_tasks: list[Task],
    full_task_index: dict[str, Task],
) -> list[ValidationIssue]:

    return detect_missing_dependencies(
        target_tasks=target_tasks,
        full_task_index=full_task_index,
    )


def npc_static_conflict_checker_tool(
    tasks: list[Task],
) -> list[ValidationIssue]:

    return detect_potential_npc_conflicts(
        tasks
    )


def npc_runtime_checker_tool(
    tasks: list[Task],
    game_runtime_state: GameRuntimeState,
) -> list[ValidationIssue]:

    return check_npc_runtime_requirements(
        tasks,
        game_runtime_state,
    )


# =========================================================
# 7. Tool Registry
# =========================================================

class ToolRegistry:

    def __init__(self):

        self.tools = {}

    def register_tool(
        self,
        name,
        tool_function,
    ):

        self.tools[
            name
        ] = tool_function

    def get_tool(
        self,
        name,
    ):

        return self.tools[
            name
        ]


# =========================================================
# 8. Agent Action
# =========================================================

NextActionType = Literal[
    "call_tool",
    "clarify",
    "human_review",
    "finish",
]


class NextActionSpec(BaseModel):
    action_type: NextActionType

    tool_name: str | None = None

    tool_args: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )

    reason: str


# =========================================================
# 9. Agent Investigation State
# =========================================================

class AgentInvestigationState(BaseModel):
    investigation_goal: str

    impact_analysis: (
        ImpactAnalysisResult
    )

    called_tool_names: list[
        str
    ] = Field(
        default_factory=list
    )

    issues: list[
        ValidationIssue
    ] = Field(
        default_factory=list
    )

    investigation_status: str = (
        "start"
    )


# =========================================================
# 10. DeepSeek Provider
# =========================================================

class DeepSeekProvider:

    def __init__(self):

        api_key = os.getenv(
            "DEEPSEEK_API_KEY"
        )

        if not api_key:

            raise ValueError(
                "DEEPSEEK_API_KEY was not found in the environment."
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url=(
                "https://api.deepseek.com"
            ),
        )

    def generate_next_action(
        self,
        agent_investigation_state:
        AgentInvestigationState,
    ) -> NextActionSpec:

        system_message: (
            ChatCompletionSystemMessageParam
        ) = {
            "role": "system",
            "content": (
                "You are the decision component of a Game QA Agent. "

                "The program has already used deterministic impact "
                "analysis to narrow the initial investigation scope. "

                "You receive impact-analysis metadata and tool evidence. "
                "Raw task configuration and GameRuntimeState remain "
                "trusted program data and are supplied directly to "
                "deterministic tools. "

                "Return exactly one JSON object. "

                "action_type MUST be exactly one of: "
                "'call_tool', 'clarify', 'human_review', 'finish'. "

                "Available tools: "

                "1. dependency_reference_checker: "
                "checks dependency references for scoped target tasks "
                "against the full task configuration source of truth. "

                "2. dependency_cycle_checker: "
                "checks whether task dependencies contain a cycle "
                "inside the current dependency investigation data. "

                "3. npc_static_conflict_checker: "
                "checks configuration-level potential NPC conflicts "
                "between tasks that may overlap. "
                "A static issue is a potential risk, not proof "
                "of a runtime conflict. "

                "4. npc_runtime_checker: "
                "checks whether current active tasks' NPC requirements "
                "match the scoped GameRuntimeState. "

                "Choose the next action using the investigation goal, "
                "impact analysis, called tools, and intermediate issues. "

                "Do not call irrelevant tools. "
                "Do not call the same tool more than once. "

                "Use runtime evidence when needed to determine whether "
                "a static NPC risk is affecting the current runtime. "

                "Finish when enough evidence has been collected. "

                "When calling a tool, tool_args must be {}. "

                "When action_type is finish, tool_name must be null "
                "and tool_args must be {}. "

                "The JSON must contain action_type, tool_name, "
                "tool_args, and reason."
            ),
        }

        user_message: (
            ChatCompletionUserMessageParam
        ) = {
            "role": "user",
            "content": (
                "Current AgentInvestigationState:\n"
                f"{agent_investigation_state.model_dump_json()}"
            ),
        }

        messages: list[
            ChatCompletionMessageParam
        ] = [
            system_message,
            user_message,
        ]

        response_format: (
            ResponseFormatJSONObject
        ) = {
            "type": "json_object"
        }

        response = (
            self.client
            .chat
            .completions
            .create(
                model=(
                    "deepseek-v4-pro"
                ),
                messages=messages,
                response_format=(
                    response_format
                ),
                stream=False,
            )
        )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        if content is None:

            raise ValueError(
                "DeepSeek returned empty response content."
            )

        return (
            NextActionSpec
            .model_validate_json(
                content
            )
        )


# =========================================================
# 11. Agent Decision
# =========================================================

def choose_next_action(
    agent_investigation_state:
    AgentInvestigationState,
    llm_provider:
    DeepSeekProvider,
) -> NextActionSpec:

    return (
        llm_provider
        .generate_next_action(
            agent_investigation_state
        )
    )


# =========================================================
# 12. Execute Agent Action
# =========================================================

def execute_action(
    next_action_spec:
    NextActionSpec,

    agent_investigation_state:
    AgentInvestigationState,

    tool_registry:
    ToolRegistry,

    tool_inputs_by_name:
    dict[
        str,
        dict[
            str,
            Any,
        ],
    ],
) -> AgentInvestigationState:

    if (
        next_action_spec.action_type
        == "call_tool"
    ):

        if (
            next_action_spec.tool_name
            is None
        ):

            raise ValueError(
                "tool_name is required."
            )

        tool_function = (
            tool_registry
            .get_tool(
                next_action_spec.tool_name
            )
        )

        tool_inputs = (
            tool_inputs_by_name[
                next_action_spec.tool_name
            ]
        )

        validation_issues = (
            tool_function(
                **tool_inputs
            )
        )

        agent_investigation_state.called_tool_names.append(
            next_action_spec.tool_name
        )

        agent_investigation_state.issues.extend(
            validation_issues
        )

        agent_investigation_state.investigation_status = (
            "running"
        )

    elif (
        next_action_spec.action_type
        == "finish"
    ):

        agent_investigation_state.investigation_status = (
            "finished"
        )

    return (
        agent_investigation_state
    )


# =========================================================
# 13. Agent Investigation Loop
# =========================================================

def run_agent_investigation(
    agent_investigation_state:
    AgentInvestigationState,

    tool_registry:
    ToolRegistry,

    tool_inputs_by_name:
    dict[
        str,
        dict[
            str,
            Any,
        ],
    ],

    llm_provider:
    DeepSeekProvider,

    max_steps: int = 6,
) -> AgentInvestigationState:

    for step in range(
        max_steps
    ):

        next_action_spec = (
            choose_next_action(
                agent_investigation_state,
                llm_provider,
            )
        )

        print(
            f"\nStep {step + 1}:"
        )

        print(
            next_action_spec
        )

        agent_investigation_state = (
            execute_action(
                next_action_spec,
                agent_investigation_state,
                tool_registry,
                tool_inputs_by_name,
            )
        )

        if (
            agent_investigation_state
            .investigation_status
            == "finished"
        ):
            break

    return (
        agent_investigation_state
    )


# =========================================================
# 14. Register Tools
# =========================================================

game_qa_tool_registry = (
    ToolRegistry()
)

game_qa_tool_registry.register_tool(
    "dependency_cycle_checker",
    dependency_cycle_checker_tool,
)

game_qa_tool_registry.register_tool(
    "dependency_reference_checker",
    dependency_reference_checker_tool,
)

game_qa_tool_registry.register_tool(
    "npc_static_conflict_checker",
    npc_static_conflict_checker_tool,
)

game_qa_tool_registry.register_tool(
    "npc_runtime_checker",
    npc_runtime_checker_tool,
)


deepseek_provider = (
    DeepSeekProvider()
)


# =========================================================
# 15. Full game configuration
# =========================================================

full_tasks = [

    Task(
        id="task_a",
        npc_requirements=[
            NPCRequirement(
                npc_id=(
                    "blacksmith"
                ),
                location=(
                    "village"
                ),
                state=(
                    "available"
                ),
            )
        ],
    ),

    Task(
        id="task_b",
        npc_requirements=[
            NPCRequirement(
                npc_id=(
                    "blacksmith"
                ),
                location=(
                    "town"
                ),
                state=(
                    "available"
                ),
            )
        ],
    ),

    Task(
        id="task_c",
        dependencies=[
            "task_a"
        ],
    ),

    Task(
        id="task_x",
        npc_requirements=[
            NPCRequirement(
                npc_id=(
                    "healer"
                ),
                location=(
                    "forest"
                ),
                state=(
                    "available"
                ),
            )
        ],
    ),
]


full_task_index = (
    build_task_index(
        full_tasks
    )
)


full_game_runtime_state = (
    GameRuntimeState(
        task_statuses={
            "task_a":
                "active",

            "task_b":
                "active",

            "task_c":
                "inactive",

            "task_x":
                "active",
        },

        npc_states={
            "blacksmith":
                NPCRuntimeState(
                    location=(
                        "village"
                    ),
                    state=(
                        "available"
                    ),
                ),

            "healer":
                NPCRuntimeState(
                    location=(
                        "castle"
                    ),
                    state=(
                        "available"
                    ),
                ),
        },
    )
)


# =========================================================
# 16. Verify source-of-truth boundary
# =========================================================

verify_dependency_reference_source_of_truth_boundary()


print(
    "\nDependency reference source-of-truth boundary: PASSED"
)


# =========================================================
# 17. Config diff / changed tasks
# =========================================================

changed_task_ids = [
    "task_a"
]


# =========================================================
# 18. Run deterministic Impact Analysis
# =========================================================

impact_analysis_result = (
    analyze_change_impact(
        tasks=full_tasks,
        changed_task_ids=(
            changed_task_ids
        ),
    )
)


print(
    "\nImpact Analysis:"
)

print(
    impact_analysis_result
)


assert set(
    impact_analysis_result
    .affected_task_ids
) == {
    "task_a",
    "task_b",
    "task_c",
}


assert (
    impact_analysis_result
    .affected_npc_ids
    == [
        "blacksmith"
    ]
)


# =========================================================
# 19. Build initial scoped context
# =========================================================

scoped_tasks = (
    build_scoped_tasks(
        tasks=full_tasks,
        impact_analysis_result=(
            impact_analysis_result
        ),
    )
)


scoped_game_runtime_state = (
    build_scoped_runtime_state(
        game_runtime_state=(
            full_game_runtime_state
        ),
        impact_analysis_result=(
            impact_analysis_result
        ),
    )
)


print(
    "\nFull task IDs:"
)

print(
    [
        task.id
        for task in full_tasks
    ]
)


print(
    "\nScoped task IDs:"
)

print(
    [
        task.id
        for task in scoped_tasks
    ]
)


print(
    "\nScoped runtime NPC IDs:"
)

print(
    list(
        scoped_game_runtime_state
        .npc_states
        .keys()
    )
)


# =========================================================
# 20. Tool inputs
# =========================================================

tool_inputs_by_name = {

    "dependency_cycle_checker": {
        "tasks":
            scoped_tasks,
    },

    "dependency_reference_checker": {
        "target_tasks":
            scoped_tasks,

        "full_task_index":
            full_task_index,
    },

    "npc_static_conflict_checker": {
        "tasks":
            scoped_tasks,
    },

    "npc_runtime_checker": {
        "tasks":
            scoped_tasks,

        "game_runtime_state":
            scoped_game_runtime_state,
    },
}


# =========================================================
# 21. Start Agent investigation
# =========================================================

agent_investigation_state = (
    AgentInvestigationState(
        investigation_goal=(
            "Validate the NPC-related QA impact of the changed "
            "task configuration inside the affected scope."
        ),

        impact_analysis=(
            impact_analysis_result
        ),
    )
)


final_agent_investigation_state = (
    run_agent_investigation(
        agent_investigation_state=(
            agent_investigation_state
        ),

        tool_registry=(
            game_qa_tool_registry
        ),

        tool_inputs_by_name=(
            tool_inputs_by_name
        ),

        llm_provider=(
            deepseek_provider
        ),
    )
)


print(
    "\nCalled tool names:"
)

print(
    final_agent_investigation_state
    .called_tool_names
)


print(
    "\nIssues:"
)

print(
    final_agent_investigation_state
    .issues
)


print(
    "\nInvestigation status:"
)

print(
    final_agent_investigation_state
    .investigation_status
)