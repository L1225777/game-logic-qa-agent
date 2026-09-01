import os
from typing import Protocol

from .models import AgentInvestigationState, NextActionSpec


class NextActionProvider(Protocol):
    def generate_next_action(
        self, state: AgentInvestigationState
    ) -> NextActionSpec: ...


class DeepSeekProvider:
    """Real provider. Construction and use are deliberately explicit."""

    def __init__(self, api_key: str | None = None) -> None:
        from openai import OpenAI

        resolved_api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not resolved_api_key:
            raise ValueError("DEEPSEEK_API_KEY was not found in the environment.")
        self.client = OpenAI(
            api_key=resolved_api_key,
            base_url="https://api.deepseek.com",
        )

    def generate_next_action(
        self, state: AgentInvestigationState
    ) -> NextActionSpec:
        system_message = {
            "role": "system",
            "content": (
                "You are the decision component of a Game QA Agent. The program "
                "creates trusted tool inputs and controls scope. Return one JSON "
                "object with action_type, tool_name, tool_args, expand_task_ids, "
                "and reason. action_type must be call_tool, expand_scope, clarify, "
                "human_review, or finish. Available tools are "
                "dependency_reference_checker, dependency_cycle_checker, "
                "npc_static_conflict_checker, and npc_runtime_checker. Only choose "
                "expand_scope when expand_task_ids is a non-empty subset of the "
                "current expandable_task_ids. Never invent task IDs. For call_tool, "
                "use empty tool_args and expand_task_ids. Do not repeat tools. If "
                "decision_errors is non-empty, correct the rejected decision."
            ),
        }
        user_message = {
            "role": "user",
            "content": f"Current AgentInvestigationState:\n{state.model_dump_json()}",
        }
        response = self.client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[system_message, user_message],
            response_format={"type": "json_object"},
            stream=False,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("DeepSeek returned empty response content.")
        return NextActionSpec.model_validate_json(content)
