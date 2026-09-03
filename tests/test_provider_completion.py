from types import SimpleNamespace

import pytest

from game_qa_agent.models import AgentInvestigationState, ImpactAnalysisResult
from game_qa_agent.providers import DeepSeekProvider


VALID_ACTION_JSON = (
    '{"action_type":"finish","tool_name":null,"tool_args":{},'
    '"expand_task_ids":[],"reason":"investigation complete"}'
)


class FakeCompletions:
    def __init__(self, finish_reason: str | None) -> None:
        self.finish_reason = finish_reason

    def create(self, **kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason=self.finish_reason,
            message=SimpleNamespace(content=VALID_ACTION_JSON),
        )])


def provider_with_finish_reason(finish_reason: str | None) -> DeepSeekProvider:
    provider = DeepSeekProvider.__new__(DeepSeekProvider)
    provider.client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions(finish_reason))
    )
    return provider


def provider_with_empty_choices() -> DeepSeekProvider:
    provider = DeepSeekProvider.__new__(DeepSeekProvider)
    provider.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(choices=[])
            )
        )
    )
    return provider


def investigation_state() -> AgentInvestigationState:
    return AgentInvestigationState(
        investigation_goal="test completion acceptance",
        impact_analysis=ImpactAnalysisResult(),
    )


def test_stop_completion_is_accepted_before_action_parsing() -> None:
    action = provider_with_finish_reason("stop").generate_next_action(
        investigation_state()
    )

    assert action.action_type == "finish"
    assert action.reason == "investigation complete"


@pytest.mark.parametrize("finish_reason", ["length", "unexpected", None])
def test_non_stop_completion_is_rejected_even_when_action_json_is_valid(
    finish_reason: str | None,
) -> None:
    with pytest.raises(ValueError, match="completion was not accepted"):
        provider_with_finish_reason(finish_reason).generate_next_action(
            investigation_state()
        )


def test_empty_choices_is_rejected_as_controlled_provider_failure() -> None:
    with pytest.raises(ValueError, match="no completion choices"):
        provider_with_empty_choices().generate_next_action(investigation_state())
