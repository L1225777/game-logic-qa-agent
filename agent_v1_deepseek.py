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
# 1. 游戏配置与 Runtime 数据
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
# 2. Dependency checkers
# =========================================================

def detect_dependency_cycles(
    tasks: list[Task],
) -> list[ValidationIssue]:

    graph = nx.DiGraph()

    for task in tasks:
        graph.add_node(task.id)

    for task in tasks:
        for dependency in task.dependencies:
            graph.add_edge(dependency, task.id)

    try:
        cycle_edges = nx.find_cycle(graph)

        cycle_task_ids = list(
            {node for edge in cycle_edges for node in edge}
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
    tasks: list[Task],
) -> list[ValidationIssue]:

    existing_task_ids = {task.id for task in tasks}
    validation_issues = []

    for task in tasks:
        for dependency in task.dependencies:

            if dependency not in existing_task_ids:
                validation_issues.append(
                    ValidationIssue(
                        issue_type="missing_dependency",
                        message=(
                            f"Task '{task.id}' depends on missing task "
                            f"'{dependency}'."
                        ),
                        task_ids=[task.id],
                        checker_name="dependency_reference_checker",
                    )
                )

    return validation_issues


# =========================================================
# 3. Static NPC conflict checker
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

            for first_requirement in first_task.npc_requirements:

                for second_requirement in second_task.npc_requirements:

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
                                issue_type="potential_npc_conflict",
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
# 4. Runtime NPC checker
# =========================================================

def check_npc_runtime_requirements(
    tasks: list[Task],
    game_runtime_state: GameRuntimeState,
) -> list[ValidationIssue]:

    validation_issues = []

    for task in tasks:

        if (
            game_runtime_state.task_statuses.get(task.id)
            != "active"
        ):
            continue

        for npc_requirement in task.npc_requirements:

            npc_runtime_state = (
                game_runtime_state.npc_states.get(
                    npc_requirement.npc_id
                )
            )

            if npc_runtime_state is None:
                validation_issues.append(
                    ValidationIssue(
                        issue_type="missing_npc_runtime_state",
                        message=(
                            f"No runtime state found for NPC "
                            f"'{npc_requirement.npc_id}'."
                        ),
                        task_ids=[task.id],
                        npc_ids=[npc_requirement.npc_id],
                        checker_name="npc_runtime_checker",
                    )
                )
                continue

            if (
                npc_runtime_state.location
                != npc_requirement.location
            ):
                validation_issues.append(
                    ValidationIssue(
                        issue_type="npc_location_mismatch",
                        message=(
                            f"Active task '{task.id}' requires NPC "
                            f"'{npc_requirement.npc_id}' at "
                            f"'{npc_requirement.location}', but the NPC "
                            f"is currently at "
                            f"'{npc_runtime_state.location}'."
                        ),
                        task_ids=[task.id],
                        npc_ids=[npc_requirement.npc_id],
                        evidence={
                            "required_location":
                                npc_requirement.location,
                            "actual_location":
                                npc_runtime_state.location,
                        },
                        checker_name="npc_runtime_checker",
                    )
                )

            if (
                npc_runtime_state.state
                != npc_requirement.state
            ):
                validation_issues.append(
                    ValidationIssue(
                        issue_type="npc_state_mismatch",
                        message=(
                            f"Active task '{task.id}' requires NPC "
                            f"'{npc_requirement.npc_id}' in state "
                            f"'{npc_requirement.state}', but the NPC "
                            f"is currently in state "
                            f"'{npc_runtime_state.state}'."
                        ),
                        task_ids=[task.id],
                        npc_ids=[npc_requirement.npc_id],
                        evidence={
                            "required_state":
                                npc_requirement.state,
                            "actual_state":
                                npc_runtime_state.state,
                        },
                        checker_name="npc_runtime_checker",
                    )
                )

    return validation_issues


# =========================================================
# 5. Agent Tools
# =========================================================

def dependency_cycle_checker_tool(
    tasks: list[Task],
) -> list[ValidationIssue]:

    return detect_dependency_cycles(tasks)


def dependency_reference_checker_tool(
    tasks: list[Task],
) -> list[ValidationIssue]:

    return detect_missing_dependencies(tasks)


def npc_static_conflict_checker_tool(
    tasks: list[Task],
) -> list[ValidationIssue]:

    return detect_potential_npc_conflicts(tasks)


def npc_runtime_checker_tool(
    tasks: list[Task],
    game_runtime_state: GameRuntimeState,
) -> list[ValidationIssue]:

    return check_npc_runtime_requirements(
        tasks,
        game_runtime_state,
    )


# =========================================================
# 6. Tool Registry
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

    def get_tool(self, name):
        return self.tools[name]


# =========================================================
# 7. Agent Action
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
    tool_args: dict[str, Any] = Field(default_factory=dict)
    reason: str


# =========================================================
# 8. Agent Investigation State
# =========================================================

class AgentInvestigationState(BaseModel):
    investigation_goal: str
    called_tool_names: list[str] = Field(default_factory=list)
    issues: list[ValidationIssue] = Field(default_factory=list)
    investigation_status: str = "start"


# =========================================================
# 9. DeepSeek Provider
# =========================================================

class DeepSeekProvider:

    def __init__(self):

        api_key = os.getenv("DEEPSEEK_API_KEY")

        if not api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY was not found in the environment."
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )

    def generate_next_action(
        self,
        agent_investigation_state: AgentInvestigationState,
    ) -> NextActionSpec:

        system_message: ChatCompletionSystemMessageParam = {
            "role": "system",
            "content": (
                "You are the decision component of a Game QA Agent. "
                "Return exactly one JSON object. "

                "action_type MUST be exactly one of: "
                "'call_tool', 'clarify', 'human_review', 'finish'. "

                "Available tools: "

                "1. dependency_reference_checker: "
                "checks whether dependency references point to "
                "tasks that exist. "

                "2. dependency_cycle_checker: "
                "checks whether task dependencies contain a cycle. "

                "3. npc_static_conflict_checker: "
                "checks configuration-level potential NPC conflicts "
                "between tasks that may overlap. "
                "A result from this checker is a potential static "
                "risk, not proof of a runtime conflict. "

                "4. npc_runtime_checker: "
                "checks whether current active tasks' NPC "
                "requirements match GameRuntimeState. "
                "This tool can help determine whether a relevant "
                "NPC risk is actually affecting the current runtime. "

                "Choose the next action using the investigation goal, "
                "the tools already called, and all intermediate "
                "issues discovered so far. "

                "Do not call irrelevant tools. "
                "Do not call the same tool more than once. "

                "Dependency tools are not useful unless the "
                "investigation concerns dependencies. "

                "Do not treat a static potential NPC conflict as "
                "proof of a runtime conflict. "

                "Use runtime evidence when it is necessary to "
                "determine whether a static risk matters now. "

                "Finish when enough evidence has been collected. "

                "When calling a tool, tool_args must be {}. "

                "When action_type is finish, tool_name must be null "
                "and tool_args must be {}. "

                "The JSON must contain action_type, tool_name, "
                "tool_args, and reason."
            ),
        }

        user_message: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": (
                "Current AgentInvestigationState:\n"
                f"{agent_investigation_state.model_dump_json()}"
            ),
        }

        messages: list[ChatCompletionMessageParam] = [
            system_message,
            user_message,
        ]

        response_format: ResponseFormatJSONObject = {
            "type": "json_object"
        }

        response = self.client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=messages,
            response_format=response_format,
            stream=False,
        )

        content = response.choices[0].message.content

        if content is None:
            raise ValueError(
                "DeepSeek returned empty response content."
            )

        return NextActionSpec.model_validate_json(content)


# =========================================================
# 10. Agent Decision
# =========================================================

def choose_next_action(
    agent_investigation_state: AgentInvestigationState,
    llm_provider: DeepSeekProvider,
) -> NextActionSpec:

    return llm_provider.generate_next_action(
        agent_investigation_state
    )


# =========================================================
# 11. Execute Agent Action
# =========================================================

def execute_action(
    next_action_spec: NextActionSpec,
    agent_investigation_state: AgentInvestigationState,
    tool_registry: ToolRegistry,
    tool_inputs_by_name: dict[str, dict[str, Any]],
) -> AgentInvestigationState:

    if next_action_spec.action_type == "call_tool":

        if next_action_spec.tool_name is None:
            raise ValueError("tool_name is required.")

        tool_function = tool_registry.get_tool(
            next_action_spec.tool_name
        )

        tool_inputs = tool_inputs_by_name[
            next_action_spec.tool_name
        ]

        validation_issues = tool_function(
            **tool_inputs
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

    elif next_action_spec.action_type == "finish":

        agent_investigation_state.investigation_status = (
            "finished"
        )

    return agent_investigation_state


# =========================================================
# 12. Agent Investigation Loop
# =========================================================

def run_agent_investigation(
    agent_investigation_state: AgentInvestigationState,
    tool_registry: ToolRegistry,
    tool_inputs_by_name: dict[str, dict[str, Any]],
    llm_provider: DeepSeekProvider,
    max_steps: int = 6,
) -> AgentInvestigationState:

    for step in range(max_steps):

        next_action_spec = choose_next_action(
            agent_investigation_state,
            llm_provider,
        )

        print(f"\nStep {step + 1}:")
        print(next_action_spec)

        agent_investigation_state = execute_action(
            next_action_spec,
            agent_investigation_state,
            tool_registry,
            tool_inputs_by_name,
        )

        if (
            agent_investigation_state.investigation_status
            == "finished"
        ):
            break

    return agent_investigation_state


# =========================================================
# 13. Register Tools
# =========================================================

game_qa_tool_registry = ToolRegistry()

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


deepseek_provider = DeepSeekProvider()


# =========================================================
# 14. Run one controlled test case
# =========================================================

def run_test_case(
    case_name: str,
    tasks: list[Task],
    game_runtime_state: GameRuntimeState,
):

    print("\n" + "=" * 70)
    print(case_name)
    print("=" * 70)

    tool_inputs_by_name = {
        "dependency_cycle_checker": {
            "tasks": tasks,
        },
        "dependency_reference_checker": {
            "tasks": tasks,
        },
        "npc_static_conflict_checker": {
            "tasks": tasks,
        },
        "npc_runtime_checker": {
            "tasks": tasks,
            "game_runtime_state": game_runtime_state,
        },
    }

    agent_investigation_state = AgentInvestigationState(
        investigation_goal=(
            "Investigate whether the NPC task configuration contains "
            "a potential conflict, and determine whether any such risk "
            "is currently affecting the runtime."
        )
    )

    final_agent_investigation_state = run_agent_investigation(
        agent_investigation_state=agent_investigation_state,
        tool_registry=game_qa_tool_registry,
        tool_inputs_by_name=tool_inputs_by_name,
        llm_provider=deepseek_provider,
    )

    print("\nCalled tool names:")
    print(
        final_agent_investigation_state.called_tool_names
    )

    print("\nIssues:")
    print(
        final_agent_investigation_state.issues
    )

    print("\nInvestigation status:")
    print(
        final_agent_investigation_state.investigation_status
    )


# =========================================================
# 15. Case A
# =========================================================

case_a_tasks = [
    Task(
        id="task_a",
        npc_requirements=[
            NPCRequirement(
                npc_id="blacksmith",
                location="village",
                state="available",
            )
        ],
    ),
    Task(
        id="task_b",
        npc_requirements=[
            NPCRequirement(
                npc_id="blacksmith",
                location="town",
                state="available",
            )
        ],
    ),
]


case_a_runtime_state = GameRuntimeState(
    task_statuses={
        "task_a": "active",
        "task_b": "active",
    },
    npc_states={
        "blacksmith": NPCRuntimeState(
            location="village",
            state="available",
        )
    },
)


# =========================================================
# 16. Case B
# =========================================================

case_b_tasks = [
    Task(
        id="task_a",
        npc_requirements=[
            NPCRequirement(
                npc_id="blacksmith",
                location="village",
                state="available",
            )
        ],
        exclusive_group="story_choice_1",
    ),
    Task(
        id="task_b",
        npc_requirements=[
            NPCRequirement(
                npc_id="blacksmith",
                location="town",
                state="available",
            )
        ],
        exclusive_group="story_choice_1",
    ),
]


case_b_runtime_state = GameRuntimeState(
    task_statuses={
        "task_a": "active",
        "task_b": "inactive",
    },
    npc_states={
        "blacksmith": NPCRuntimeState(
            location="village",
            state="available",
        )
    },
)


# =========================================================
# 17. Run A/B
# =========================================================

run_test_case(
    "CASE A - potential static conflict",
    case_a_tasks,
    case_a_runtime_state,
)

run_test_case(
    "CASE B - mutually exclusive tasks",
    case_b_tasks,
    case_b_runtime_state,
)