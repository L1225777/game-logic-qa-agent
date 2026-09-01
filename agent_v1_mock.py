from typing import Any, Literal

import networkx as nx
from pydantic import BaseModel, Field


# =========================================================
# 1. 游戏任务数据
# =========================================================

class Task(BaseModel):
    id: str
    dependencies: list[str] = Field(default_factory=list)


class ValidationIssue(BaseModel):
    issue_type: str
    message: str
    task_ids: list[str] = Field(default_factory=list)
    checker_name: str


# =========================================================
# 2. 真实 deterministic checker
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


# =========================================================
# 3. Agent 可以调用的 Tool
# =========================================================

def dependency_cycle_checker_tool(
    tasks: list[Task],
) -> list[ValidationIssue]:

    return detect_dependency_cycles(tasks)


# =========================================================
# 4. Tool Registry
# =========================================================

class ToolRegistry:

    def __init__(self):
        self.tools = {}

    def register_tool(self, name, tool_function):
        self.tools[name] = tool_function

    def get_tool(self, name):
        return self.tools[name]


# =========================================================
# 5. Agent 下一步动作
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
# 6. Agent 单次 QA 调查状态
# =========================================================

class AgentInvestigationState(BaseModel):
    called_tool_names: list[str] = Field(default_factory=list)
    issues: list[ValidationIssue] = Field(default_factory=list)
    investigation_status: str = "start"


# =========================================================
# 7. 临时替代真实 LLM
# =========================================================

class MockLLMProvider:

    def generate_next_action(
        self,
        agent_investigation_state: AgentInvestigationState,
        tasks: list[Task],
    ) -> NextActionSpec:

        if (
            "dependency_cycle_checker"
            not in agent_investigation_state.called_tool_names
        ):

            return NextActionSpec(
                action_type="call_tool",
                tool_name="dependency_cycle_checker",
                tool_args={"tasks": tasks},
                reason="Check for dependency cycles.",
            )

        return NextActionSpec(
            action_type="finish",
            reason="Dependency cycle checking is complete.",
        )


# =========================================================
# 8. 选择下一步动作
# =========================================================

def choose_next_action(
    agent_investigation_state: AgentInvestigationState,
    tasks: list[Task],
    llm_provider,
) -> NextActionSpec:

    return llm_provider.generate_next_action(
        agent_investigation_state,
        tasks,
    )


# =========================================================
# 9. 执行下一步动作
# =========================================================

def execute_action(
    next_action_spec: NextActionSpec,
    agent_investigation_state: AgentInvestigationState,
    tool_registry: ToolRegistry,
) -> AgentInvestigationState:

    if next_action_spec.action_type == "call_tool":

        if next_action_spec.tool_name is None:
            raise ValueError("tool_name is required.")

        tool_function = tool_registry.get_tool(
            next_action_spec.tool_name
        )

        validation_issues = tool_function(
            **next_action_spec.tool_args
        )

        agent_investigation_state.called_tool_names.append(
            next_action_spec.tool_name
        )

        agent_investigation_state.issues.extend(
            validation_issues
        )

        agent_investigation_state.investigation_status = "running"

    elif next_action_spec.action_type == "finish":

        agent_investigation_state.investigation_status = "finished"

    return agent_investigation_state


# =========================================================
# 10. Agent QA investigation 总调度循环
# =========================================================

def run_agent_investigation(
    agent_investigation_state: AgentInvestigationState,
    tasks: list[Task],
    tool_registry: ToolRegistry,
    llm_provider,
    max_steps: int = 5,
) -> AgentInvestigationState:

    for _ in range(max_steps):

        next_action_spec = choose_next_action(
            agent_investigation_state,
            tasks,
            llm_provider,
        )

        agent_investigation_state = execute_action(
            next_action_spec,
            agent_investigation_state,
            tool_registry,
        )

        if (
            agent_investigation_state.investigation_status
            == "finished"
        ):
            break

    return agent_investigation_state


# =========================================================
# 11. 测试数据
# =========================================================

tasks = [
    Task(
        id="task_a",
        dependencies=["task_b"],
    ),
    Task(
        id="task_b",
        dependencies=["task_a"],
    ),
]


# =========================================================
# 12. 注册 Tool
# =========================================================

tool_registry = ToolRegistry()

tool_registry.register_tool(
    "dependency_cycle_checker",
    dependency_cycle_checker_tool,
)


# =========================================================
# 13. 启动 Agent QA investigation
# =========================================================

agent_investigation_state = AgentInvestigationState()
mock_llm_provider = MockLLMProvider()

final_agent_investigation_state = run_agent_investigation(
    agent_investigation_state=agent_investigation_state,
    tasks=tasks,
    tool_registry=tool_registry,
    llm_provider=mock_llm_provider,
)


print("Called tool names:")
print(final_agent_investigation_state.called_tool_names)

print("\nIssues:")
print(final_agent_investigation_state.issues)

print("\nInvestigation status:")
print(final_agent_investigation_state.investigation_status)