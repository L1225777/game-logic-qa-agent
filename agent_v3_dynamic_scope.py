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
    occupied_by_task_id: str | None = None


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
# 2. Initial Impact Analysis
# =========================================================

class ImpactAnalysisResult(BaseModel):
    changed_task_ids: list[str] = Field(default_factory=list)
    affected_task_ids: list[str] = Field(default_factory=list)
    affected_npc_ids: list[str] = Field(default_factory=list)


def build_task_index(tasks: list[Task]) -> dict[str, Task]:
    return {
        task.id: task
        for task in tasks
    }


def analyze_initial_change_impact(
    tasks: list[Task],
    changed_task_ids: list[str],
) -> ImpactAnalysisResult:

    task_index = build_task_index(tasks)

    dependency_graph = nx.DiGraph()

    for task in tasks:
        dependency_graph.add_node(task.id)

    for task in tasks:
        for dependency in task.dependencies:
            if dependency in task_index:
                dependency_graph.add_edge(
                    dependency,
                    task.id,
                )

    affected_task_ids = {
        task_id
        for task_id in changed_task_ids
        if task_id in task_index
    }

    for task_id in list(affected_task_ids):

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

    affected_npc_ids = {
        npc_requirement.npc_id
        for task_id in affected_task_ids
        for npc_requirement
        in task_index[task_id].npc_requirements
    }

    return ImpactAnalysisResult(
        changed_task_ids=sorted(
            changed_task_ids
        ),
        affected_task_ids=sorted(
            affected_task_ids
        ),
        affected_npc_ids=sorted(
            affected_npc_ids
        ),
    )


def get_scoped_tasks(
    scope_task_ids: list[str],
    full_task_index: dict[str, Task],
) -> list[Task]:

    return [
        full_task_index[task_id]
        for task_id in scope_task_ids
        if task_id in full_task_index
    ]


# =========================================================
# 3. Dependency checkers
# =========================================================

def detect_dependency_cycles(
    tasks: list[Task],
) -> list[ValidationIssue]:

    graph = nx.DiGraph()

    for task in tasks:
        graph.add_node(task.id)

    for task in tasks:
        for dependency in task.dependencies:
            graph.add_edge(
                dependency,
                task.id,
            )

    try:
        cycle_edges = nx.find_cycle(graph)

        cycle_task_ids = sorted(
            {
                node
                for edge in cycle_edges
                for node in edge
            }
        )

        return [
            ValidationIssue(
                issue_type="dependency_cycle",
                message="A dependency cycle was detected.",
                task_ids=cycle_task_ids,
                checker_name="dependency_cycle_checker",
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
                        issue_type="missing_dependency",
                        message=(
                            f"Task '{task.id}' depends on missing task "
                            f"'{dependency}'."
                        ),
                        task_ids=[task.id],
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


# =========================================================
# 4. Static NPC conflict checker
# =========================================================

def tasks_may_overlap(
    first_task: Task,
    second_task: Task,
) -> bool:

    if (
        first_task.exclusive_group is not None
        and first_task.exclusive_group
        == second_task.exclusive_group
    ):
        return False

    return True


def detect_potential_npc_conflicts(
    tasks: list[Task],
) -> list[ValidationIssue]:

    validation_issues = []

    for first_index in range(len(tasks)):

        for second_index in range(
            first_index + 1,
            len(tasks),
        ):

            first_task = tasks[first_index]
            second_task = tasks[second_index]

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

                    if (
                        first_requirement.location
                        != second_requirement.location
                    ):

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
            game_runtime_state
            .task_statuses
            .get(task.id)
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

            occupied_by_task_id = (
                npc_runtime_state
                .occupied_by_task_id
            )

            if (
                occupied_by_task_id is not None
                and occupied_by_task_id
                != task.id
            ):

                validation_issues.append(
                    ValidationIssue(
                        issue_type=(
                            "npc_occupied_by_other_task"
                        ),
                        message=(
                            f"NPC '{npc_requirement.npc_id}' needed by "
                            f"active task '{task.id}' is currently occupied "
                            f"by task '{occupied_by_task_id}'."
                        ),
                        task_ids=[
                            task.id,
                            occupied_by_task_id,
                        ],
                        npc_ids=[
                            npc_requirement.npc_id
                        ],
                        evidence={
                            "occupied_by_task_id":
                                occupied_by_task_id,
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
        tasks=tasks,
        game_runtime_state=game_runtime_state,
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
        self.tools[name] = tool_function

    def get_tool(
        self,
        name,
    ):
        return self.tools[name]


# =========================================================
# 8. Agent Action
# =========================================================

NextActionType = Literal[
    "call_tool",
    "expand_scope",
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

    expand_task_ids: list[str] = Field(
        default_factory=list
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

    scope_task_ids: list[str] = Field(
        default_factory=list
    )

    expandable_task_ids: list[str] = Field(
        default_factory=list
    )

    scope_version: int = 1

    called_tool_names: list[str] = Field(
        default_factory=list
    )

    issues: list[ValidationIssue] = Field(
        default_factory=list
    )

    decision_errors: list[str] = Field(
        default_factory=list
    )

    investigation_status: str = "start"


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

                "The program has already created a narrow initial "
                "investigation scope using deterministic impact analysis. "

                "The full game configuration and GameRuntimeState remain "
                "trusted program-side sources of truth. "

                "Return exactly one JSON object. "

                "action_type MUST be exactly one of: "
                "'call_tool', 'expand_scope', 'clarify', "
                "'human_review', 'finish'. "

                "Available tools: "

                "1. dependency_reference_checker: checks dependency "
                "references for scoped target tasks against the full task "
                "source of truth. "

                "2. dependency_cycle_checker: checks dependency cycles "
                "inside the current scope. "

                "3. npc_static_conflict_checker: checks potential NPC "
                "configuration conflicts among tasks currently in scope. "

                "4. npc_runtime_checker: checks active scoped tasks against "
                "trusted GameRuntimeState and can reveal an external task "
                "currently occupying a required NPC. "

                "The AgentInvestigationState contains expandable_task_ids. "
                "That list is computed deterministically by the program from "
                "tool evidence and the full source of truth. "

                "You may choose action_type='expand_scope' ONLY when "
                "expandable_task_ids is non-empty. "

                "expand_task_ids MUST be a non-empty subset of the current "
                "expandable_task_ids. Never invent task IDs. "

                "If expandable_task_ids is empty, NEVER choose expand_scope. "

                "For this controlled test, if npc_runtime_checker has not "
                "been called yet, call it first because the goal asks whether "
                "runtime evidence reveals an external task. "

                "After an external task is added to scope, use "
                "npc_static_conflict_checker if needed to inspect the "
                "configuration relationship. "

                "If npc_runtime_checker has already been called and "
                "expandable_task_ids is empty, no evidence-backed external "
                "task is available to load. Finish unless another relevant "
                "tool is clearly needed. "

                "Dependency tools are irrelevant unless dependency evidence "
                "appears. "

                "Do not call the same tool more than once in this "
                "controlled test. "

                "If decision_errors is non-empty, correct the previous "
                "invalid decision and obey the current state constraints. "

                "When action_type='call_tool', tool_name must name an "
                "available tool, tool_args must be {}, and "
                "expand_task_ids must be []. "

                "When action_type='expand_scope', tool_name must be null, "
                "tool_args must be {}, and expand_task_ids must be a valid "
                "subset of expandable_task_ids. "

                "When action_type='finish', tool_name must be null, "
                "tool_args must be {}, and expand_task_ids must be []. "

                "The JSON must contain action_type, tool_name, tool_args, "
                "expand_task_ids, and reason."
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
# 12. Build Tool Inputs From Current Scope
# =========================================================

def build_tool_inputs(
    tool_name: str,

    agent_investigation_state:
    AgentInvestigationState,

    full_task_index:
    dict[str, Task],

    full_game_runtime_state:
    GameRuntimeState,
) -> dict[str, Any]:

    scoped_tasks = (
        get_scoped_tasks(
            scope_task_ids=(
                agent_investigation_state
                .scope_task_ids
            ),
            full_task_index=(
                full_task_index
            ),
        )
    )

    if (
        tool_name
        == "dependency_reference_checker"
    ):

        return {
            "target_tasks":
                scoped_tasks,

            "full_task_index":
                full_task_index,
        }

    if (
        tool_name
        == "dependency_cycle_checker"
    ):

        return {
            "tasks":
                scoped_tasks,
        }

    if (
        tool_name
        == "npc_static_conflict_checker"
    ):

        return {
            "tasks":
                scoped_tasks,
        }

    if (
        tool_name
        == "npc_runtime_checker"
    ):

        return {
            "tasks":
                scoped_tasks,

            "game_runtime_state":
                full_game_runtime_state,
        }

    raise ValueError(
        f"No tool input builder for '{tool_name}'."
    )


# =========================================================
# 13. Deterministic scope-expansion candidates
# =========================================================

def compute_expandable_task_ids(
    agent_investigation_state:
    AgentInvestigationState,

    full_task_index:
    dict[str, Task],
) -> list[str]:

    evidence_task_ids = {
        task_id
        for issue
        in agent_investigation_state.issues
        for task_id
        in issue.task_ids
    }

    current_scope = set(
        agent_investigation_state
        .scope_task_ids
    )

    return sorted(
        task_id
        for task_id
        in evidence_task_ids
        if (
            task_id in full_task_index
            and task_id not in current_scope
        )
    )


def refresh_expandable_task_ids(
    agent_investigation_state:
    AgentInvestigationState,

    full_task_index:
    dict[str, Task],
) -> None:

    agent_investigation_state.expandable_task_ids = (
        compute_expandable_task_ids(
            agent_investigation_state=(
                agent_investigation_state
            ),
            full_task_index=(
                full_task_index
            ),
        )
    )


# =========================================================
# 14. Safe rejection of invalid LLM actions
# =========================================================

def reject_agent_action(
    agent_investigation_state:
    AgentInvestigationState,

    message: str,
) -> AgentInvestigationState:

    agent_investigation_state.decision_errors.append(
        message
    )

    agent_investigation_state.investigation_status = (
        "running"
    )

    return (
        agent_investigation_state
    )


# =========================================================
# 15. Execute Agent Action
# =========================================================

def execute_action(
    next_action_spec:
    NextActionSpec,

    agent_investigation_state:
    AgentInvestigationState,

    tool_registry:
    ToolRegistry,

    full_task_index:
    dict[str, Task],

    full_game_runtime_state:
    GameRuntimeState,
) -> AgentInvestigationState:

    if (
        next_action_spec.action_type
        == "call_tool"
    ):

        if (
            next_action_spec.tool_name
            is None
        ):

            return reject_agent_action(
                agent_investigation_state,
                (
                    "Rejected call_tool because "
                    "tool_name was missing."
                ),
            )

        if (
            next_action_spec.tool_name
            in agent_investigation_state
            .called_tool_names
        ):

            return reject_agent_action(
                agent_investigation_state,
                (
                    f"Rejected repeated tool call "
                    f"'{next_action_spec.tool_name}'."
                ),
            )

        try:

            tool_function = (
                tool_registry
                .get_tool(
                    next_action_spec.tool_name
                )
            )

        except KeyError:

            return reject_agent_action(
                agent_investigation_state,
                (
                    f"Rejected unknown tool "
                    f"'{next_action_spec.tool_name}'."
                ),
            )

        tool_inputs = (
            build_tool_inputs(
                tool_name=(
                    next_action_spec.tool_name
                ),
                agent_investigation_state=(
                    agent_investigation_state
                ),
                full_task_index=(
                    full_task_index
                ),
                full_game_runtime_state=(
                    full_game_runtime_state
                ),
            )
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

        refresh_expandable_task_ids(
            agent_investigation_state=(
                agent_investigation_state
            ),
            full_task_index=(
                full_task_index
            ),
        )

        agent_investigation_state.investigation_status = (
            "running"
        )

    elif (
        next_action_spec.action_type
        == "expand_scope"
    ):

        allowed_task_ids = set(
            agent_investigation_state
            .expandable_task_ids
        )

        requested_task_ids = set(
            next_action_spec
            .expand_task_ids
        )

        if not allowed_task_ids:

            return reject_agent_action(
                agent_investigation_state,
                (
                    "Rejected expand_scope because "
                    "expandable_task_ids is empty."
                ),
            )

        if not requested_task_ids:

            return reject_agent_action(
                agent_investigation_state,
                (
                    "Rejected expand_scope because "
                    "expand_task_ids was empty."
                ),
            )

        if not requested_task_ids.issubset(
            allowed_task_ids
        ):

            return reject_agent_action(
                agent_investigation_state,
                (
                    "Rejected expand_scope because requested task IDs "
                    f"{sorted(requested_task_ids)} are not a subset of "
                    f"expandable_task_ids "
                    f"{agent_investigation_state.expandable_task_ids}."
                ),
            )

        current_scope = set(
            agent_investigation_state
            .scope_task_ids
        )

        agent_investigation_state.scope_task_ids = sorted(
            current_scope.union(
                requested_task_ids
            )
        )

        agent_investigation_state.scope_version += 1

        refresh_expandable_task_ids(
            agent_investigation_state=(
                agent_investigation_state
            ),
            full_task_index=(
                full_task_index
            ),
        )

        agent_investigation_state.investigation_status = (
            "running"
        )

    elif (
        next_action_spec.action_type
        == "clarify"
    ):

        agent_investigation_state.investigation_status = (
            "clarification_required"
        )

    elif (
        next_action_spec.action_type
        == "human_review"
    ):

        agent_investigation_state.investigation_status = (
            "human_review_required"
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
# 16. Agent Investigation Loop
# =========================================================

def run_agent_investigation(
    agent_investigation_state:
    AgentInvestigationState,

    tool_registry:
    ToolRegistry,

    full_task_index:
    dict[str, Task],

    full_game_runtime_state:
    GameRuntimeState,

    llm_provider:
    DeepSeekProvider,

    max_steps: int = 8,
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

        previous_error_count = len(
            agent_investigation_state
            .decision_errors
        )

        agent_investigation_state = (
            execute_action(
                next_action_spec=(
                    next_action_spec
                ),
                agent_investigation_state=(
                    agent_investigation_state
                ),
                tool_registry=(
                    tool_registry
                ),
                full_task_index=(
                    full_task_index
                ),
                full_game_runtime_state=(
                    full_game_runtime_state
                ),
            )
        )

        print(
            "Current scope:",
            agent_investigation_state
            .scope_task_ids,
        )

        print(
            "Expandable task IDs:",
            agent_investigation_state
            .expandable_task_ids,
        )

        if (
            len(
                agent_investigation_state
                .decision_errors
            )
            > previous_error_count
        ):

            print(
                "Decision rejected:",
                agent_investigation_state
                .decision_errors[-1],
            )

        if (
            agent_investigation_state
            .investigation_status
            in {
                "finished",
                "clarification_required",
                "human_review_required",
            }
        ):
            break

    return (
        agent_investigation_state
    )


# =========================================================
# 17. Register Tools
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
# 18. Run one controlled scope-expansion case
# =========================================================

def run_scope_expansion_case(
    case_name: str,

    full_tasks:
    list[Task],

    full_game_runtime_state:
    GameRuntimeState,

    changed_task_ids:
    list[str],
) -> None:

    print(
        "\n"
        + "=" * 70
    )

    print(
        case_name
    )

    print(
        "=" * 70
    )

    full_task_index = (
        build_task_index(
            full_tasks
        )
    )

    impact_analysis_result = (
        analyze_initial_change_impact(
            tasks=(
                full_tasks
            ),
            changed_task_ids=(
                changed_task_ids
            ),
        )
    )

    print(
        "\nInitial Impact Analysis:"
    )

    print(
        impact_analysis_result
    )

    agent_investigation_state = (
        AgentInvestigationState(
            investigation_goal=(
                "Investigate whether current runtime evidence from "
                "the affected scope reveals an external task that "
                "needs to be loaded into the investigation, then "
                "check any relevant NPC configuration relationship."
            ),

            impact_analysis=(
                impact_analysis_result
            ),

            scope_task_ids=(
                impact_analysis_result
                .affected_task_ids
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

            full_task_index=(
                full_task_index
            ),

            full_game_runtime_state=(
                full_game_runtime_state
            ),

            llm_provider=(
                deepseek_provider
            ),
        )
    )

    print(
        "\nFinal scope:"
    )

    print(
        final_agent_investigation_state
        .scope_task_ids
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
        "\nDecision errors:"
    )

    print(
        final_agent_investigation_state
        .decision_errors
    )

    print(
        "\nInvestigation status:"
    )

    print(
        final_agent_investigation_state
        .investigation_status
    )


# =========================================================
# 19. Case A
# Runtime evidence reveals an out-of-scope task.
# =========================================================

case_a_tasks = [

    Task(
        id="task_a",
    ),

    Task(
        id="task_b",

        dependencies=[
            "task_a"
        ],

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
        id="task_event_7",

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


case_a_runtime_state = (
    GameRuntimeState(
        task_statuses={
            "task_a":
                "inactive",

            "task_b":
                "active",

            "task_event_7":
                "active",

            "task_x":
                "active",
        },

        npc_states={
            "blacksmith":
                NPCRuntimeState(
                    location=(
                        "town"
                    ),
                    state=(
                        "available"
                    ),
                    occupied_by_task_id=(
                        "task_event_7"
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
                    occupied_by_task_id=(
                        "task_x"
                    ),
                ),
        },
    )
)


# =========================================================
# 20. Case B
# No external runtime evidence.
# =========================================================

case_b_tasks = (
    case_a_tasks
)


case_b_runtime_state = (
    GameRuntimeState(
        task_statuses={
            "task_a":
                "inactive",

            "task_b":
                "active",

            "task_event_7":
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
                    occupied_by_task_id=None,
                ),

            "healer":
                NPCRuntimeState(
                    location=(
                        "castle"
                    ),
                    state=(
                        "available"
                    ),
                    occupied_by_task_id=(
                        "task_x"
                    ),
                ),
        },
    )
)


# =========================================================
# 21. Run A/B
# =========================================================

run_scope_expansion_case(
    case_name=(
        "CASE A - runtime evidence should expand scope"
    ),

    full_tasks=(
        case_a_tasks
    ),

    full_game_runtime_state=(
        case_a_runtime_state
    ),

    changed_task_ids=[
        "task_a"
    ],
)


run_scope_expansion_case(
    case_name=(
        "CASE B - no external runtime evidence"
    ),

    full_tasks=(
        case_b_tasks
    ),

    full_game_runtime_state=(
        case_b_runtime_state
    ),

    changed_task_ids=[
        "task_a"
    ],
)