from collections import deque

from game_qa_agent.models import AgentInvestigationState, NextActionSpec


class ScriptedProvider:
    def __init__(self, actions: list[NextActionSpec]) -> None:
        self.actions = deque(actions)

    def generate_next_action(
        self, state: AgentInvestigationState
    ) -> NextActionSpec:
        return self.actions.popleft()


def action(
    action_type: str,
    *,
    tool_name: str | None = None,
    expand_task_ids: list[str] | None = None,
    reason: str = "scripted test decision",
) -> NextActionSpec:
    return NextActionSpec(
        action_type=action_type,
        tool_name=tool_name,
        expand_task_ids=expand_task_ids or [],
        reason=reason,
    )
