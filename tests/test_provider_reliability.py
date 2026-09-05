from collections import deque
from dataclasses import asdict
from json import JSONDecodeError
from types import SimpleNamespace

import httpx
import openai
import pytest

import game_qa_agent.eval as evaluation
from game_qa_agent.analysis import build_task_index
from game_qa_agent.models import NextActionSpec
from game_qa_agent.orchestration import run_agent_investigation
from game_qa_agent.providers import DeepSeekProvider, ProviderError
from game_qa_agent.report import build_qa_report, render_qa_report_markdown
from game_qa_agent.tools import build_default_tool_registry
from game_qa_agent.trace import InMemoryInvestigationTraceRecorder

from conftest import ScriptedProvider
from test_provider_completion import VALID_ACTION_JSON, investigation_state


@pytest.fixture(autouse=True)
def forbid_real_sleep(monkeypatch):
    def forbidden(seconds):
        pytest.fail("Provider tests must inject waiting")

    monkeypatch.setattr("time.sleep", forbidden)


def completion(content=VALID_ACTION_JSON, finish_reason="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish_reason,
        message=SimpleNamespace(content=content),
    )])


def status_error(status=429, *, body=None, headers=None):
    request = httpx.Request(
        "POST", "https://offline.invalid/chat/completions",
        headers={"Authorization": "Bearer private-credential-marker"},
        content=b"private-request-marker",
    )
    return openai.APIStatusError(
        "private-provider-error-marker",
        response=httpx.Response(status, headers=headers, request=request),
        body=body,
    )


class ScriptedCompletions:
    def __init__(self, outcomes):
        self.outcomes = deque(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.popleft()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def scripted_provider(*outcomes):
    requests = ScriptedCompletions(outcomes)
    waits = []
    provider = DeepSeekProvider.__new__(DeepSeekProvider)
    provider.client = SimpleNamespace(chat=SimpleNamespace(completions=requests))
    provider._wait = waits.append
    return provider, requests.calls, waits


def test_sdk_does_not_retry_quota_exhaustion(monkeypatch) -> None:
    requests = []
    waits = []

    def respond(request):
        requests.append(request)
        return httpx.Response(429, json={"error": {
            "code": "insufficient_quota",
            "type": "insufficient_quota",
            "message": "private-provider-error-marker",
        }})

    sdk_client = openai.OpenAI
    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        monkeypatch.setattr(
            openai, "OpenAI",
            lambda **kwargs: sdk_client(http_client=http_client, **kwargs),
        )
        monkeypatch.setattr("openai._base_client.time.sleep", waits.append)
        provider = DeepSeekProvider(api_key="offline-test-key")

        with pytest.raises(ProviderError) as caught:
            provider.generate_next_action(investigation_state())

    assert len(requests) == 1
    assert waits == []
    assert caught.value.failure.category == "quota_billing"
    assert caught.value.attempts == 1


def test_transient_rate_limit_retries_the_same_request_then_succeeds() -> None:
    provider, calls, waits = scripted_provider(
        status_error(body={"code": "rate_limit_exceeded"}), completion(),
    )
    state = investigation_state()
    before = state.model_dump_json()

    action = provider.generate_next_action(state)

    assert action == NextActionSpec.model_validate_json(VALID_ACTION_JSON)
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert waits == [0.5]
    assert state.model_dump_json() == before


def test_retry_budget_exhaustion_exposes_last_safe_failure() -> None:
    provider, calls, waits = scripted_provider(
        status_error(429), status_error(503), status_error(502), completion(),
    )

    with pytest.raises(ProviderError) as caught:
        provider.generate_next_action(investigation_state())

    assert len(calls) == 3
    assert waits == [0.5, 1.0]
    assert caught.value.attempts == 3
    assert asdict(caught.value.failure) == {
        "category": "transport_service",
        "retryable": True,
        "status_code": 502,
        "retry_after_seconds": None,
    }


@pytest.mark.parametrize("body", [
    {"code": "insufficient_quota"},
    {"type": "quota_exceeded"},
    {"code": "billing_hard_limit_reached"},
    {"type": "subscription_limit_exceeded"},
    {"code": "usage_limit_reached"},
    {"error": {"code": "insufficient_quota"}},
    {"message": "You exceeded your current quota."},
    {"message": "Insufficient Balance"},
    "Quota exceeded",
])
def test_quota_429_fails_fast_despite_retry_headers(body) -> None:
    provider, calls, waits = scripted_provider(status_error(
        body=body,
        headers={"Retry-After": "2", "x-should-retry": "true"},
    ), completion())

    with pytest.raises(ProviderError) as caught:
        provider.generate_next_action(investigation_state())

    assert len(calls) == 1
    assert waits == []
    assert caught.value.failure.category == "quota_billing"
    assert caught.value.failure.retryable is False
    assert caught.value.failure.retry_after_seconds is None
    assert caught.value.attempts == 1


@pytest.mark.parametrize("status", [429, 503])
def test_explicit_do_not_retry_overrides_retryable_status(status) -> None:
    provider, calls, waits = scripted_provider(status_error(
        status, headers={"x-should-retry": "false", "Retry-After": "2"},
    ), completion())

    with pytest.raises(ProviderError) as caught:
        provider.generate_next_action(investigation_state())

    assert len(calls) == 1
    assert waits == []
    assert caught.value.failure.retryable is False
    assert caught.value.failure.retry_after_seconds is None


def test_request_failure_does_not_export_sdk_text_payloads_or_chain() -> None:
    provider, calls, waits = scripted_provider(status_error(401, body={
        "message": "private-provider-body-marker",
        "code": "private-provider-code-marker",
        "type": "private-provider-type-marker",
    }))

    with pytest.raises(ProviderError) as caught:
        provider.generate_next_action(investigation_state())

    error = caught.value
    assert "private-" not in str(error)
    assert "private-" not in repr(error)
    assert "private-" not in repr(vars(error))
    assert error.__cause__ is None
    assert error.__context__ is None
    traceback = error.__traceback__
    while traceback is not None:
        if traceback.tb_frame.f_code.co_filename.endswith("providers.py"):
            assert set(traceback.tb_frame.f_locals) == {"result"}
        traceback = traceback.tb_next
    assert asdict(error.failure) == {
        "category": "non_retryable",
        "retryable": False,
        "status_code": 401,
        "retry_after_seconds": None,
    }
    assert len(calls) == 1
    assert waits == []


@pytest.mark.parametrize("response", [
    completion(finish_reason="length"),
    completion(content=None),
    completion(content=""),
    completion(content="not json"),
    completion(content='{"action_type":"finish"}'),
    completion(content='{"action_type":"invented","reason":"bad"}'),
    completion(content='{"action_type":"finish","reason":17}'),
    completion(content='{"action_type":"finish","reason":"ok","tool_args":[]}'),
    SimpleNamespace(choices=[]),
    SimpleNamespace(choices=[None]),
    SimpleNamespace(choices="not a choice list"),
    SimpleNamespace(),
    None,
])
def test_invalid_completion_is_normalized_without_retry(response) -> None:
    provider, calls, waits = scripted_provider(response, completion())

    with pytest.raises(ProviderError) as caught:
        provider.generate_next_action(investigation_state())

    assert len(calls) == 1
    assert waits == []
    assert caught.value.failure.category == "invalid_response"
    assert caught.value.failure.retryable is False
    assert caught.value.attempts == 1


def test_successful_first_attempt_preserves_request_and_action_normalization() -> None:
    provider, calls, waits = scripted_provider(completion(content=(
        '{"action_type":"call_tool","tool_name":"npc_runtime_checker",'
        '"tool_args":{"untrusted":true},"expand_task_ids":["invented"],'
        '"reason":"check runtime"}'
    )))
    state = investigation_state()
    before = state.model_dump_json()

    action = provider.generate_next_action(state)

    assert action == NextActionSpec(
        action_type="call_tool", tool_name="npc_runtime_checker", reason="check runtime",
    )
    assert len(calls) == 1
    assert waits == []
    assert calls[0]["model"] == "deepseek-v4-pro"
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert calls[0]["stream"] is False
    assert calls[0]["messages"][0]["role"] == "system"
    assert calls[0]["messages"][1] == {
        "role": "user", "content": f"Current AgentInvestigationState:\n{before}",
    }
    assert state.model_dump_json() == before


@pytest.mark.parametrize("status", [408, 409, 500, 502, 503, 504])
def test_temporary_service_failures_retry(status) -> None:
    provider, calls, waits = scripted_provider(status_error(status), completion())

    assert provider.generate_next_action(investigation_state()).action_type == "finish"
    assert len(calls) == 2
    assert waits == [0.5]


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 422, 501, 505])
def test_permanent_status_does_not_retry_even_with_positive_hints(status) -> None:
    provider, calls, waits = scripted_provider(status_error(
        status, headers={"x-should-retry": "true", "Retry-After": "1"},
    ), completion())

    with pytest.raises(ProviderError) as caught:
        provider.generate_next_action(investigation_state())

    assert len(calls) == 1
    assert waits == []
    assert caught.value.failure.status_code == status
    assert caught.value.failure.retryable is False
    assert caught.value.failure.retry_after_seconds is None


@pytest.mark.parametrize("error_type", [openai.APIConnectionError, openai.APITimeoutError])
def test_transport_failures_retry_without_exporting_request(error_type) -> None:
    request = httpx.Request("POST", "https://offline.invalid")
    provider, calls, waits = scripted_provider(
        error_type(request=request), error_type(request=request),
        error_type(request=request), completion(),
    )

    with pytest.raises(ProviderError) as caught:
        provider.generate_next_action(investigation_state())

    assert len(calls) == 3
    assert waits == [0.5, 1.0]
    assert caught.value.failure.category == "transport_service"
    assert caught.value.failure.retryable is True
    assert caught.value.failure.status_code is None
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None


@pytest.mark.parametrize(("header", "expected_wait"), [
    ("2.5", 2.5), ("0", 0.0), ("9999", 5.0),
    ("-1", 0.5), ("nan", 0.5), ("inf", 0.5), ("nonsense", 0.5),
    ("Wed, 21 Oct 2015 07:28:00 GMT", 0.5),
])
def test_retry_after_is_bounded_and_only_controls_waiting(header, expected_wait) -> None:
    provider, calls, waits = scripted_provider(
        status_error(headers={"Retry-After": header}), completion(),
    )

    provider.generate_next_action(investigation_state())

    assert len(calls) == 2
    assert waits == [expected_wait]


@pytest.mark.parametrize("last_outcome", [
    status_error(body={"code": "quota_exceeded"}),
    completion(content="private-invalid-completion-marker"),
    status_error(503, headers={"x-should-retry": "false"}),
])
def test_retry_stops_immediately_when_failure_becomes_permanent(last_outcome) -> None:
    provider, calls, waits = scripted_provider(status_error(), last_outcome, completion())

    with pytest.raises(ProviderError) as caught:
        provider.generate_next_action(investigation_state())

    assert len(calls) == 2
    assert waits == [0.5]
    assert caught.value.attempts == 2
    assert caught.value.failure.retryable is False


@pytest.mark.parametrize("error", [
    openai.APIResponseValidationError(
        response=httpx.Response(200, request=httpx.Request("POST", "https://offline.invalid"),
                                headers={"x-should-retry": "true", "Retry-After": "1"}),
        body={"private-provider-body-marker": "invalid response"},
    ),
    JSONDecodeError("private-json-error-marker", "private-completion-marker", 0),
])
def test_sdk_response_decode_and_validation_failures_are_not_transient(error) -> None:
    provider, calls, waits = scripted_provider(error, completion())

    with pytest.raises(ProviderError) as caught:
        provider.generate_next_action(investigation_state())

    assert len(calls) == 1
    assert waits == []
    assert caught.value.failure.category == "invalid_response"
    assert caught.value.failure.retryable is False
    assert "private-" not in repr(vars(caught.value))
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None


@pytest.mark.parametrize("eventually_succeeds", [True, False])
def test_real_sdk_obeys_one_outer_request_budget(monkeypatch, eventually_succeeds) -> None:
    requests = []
    waits = []

    def respond(request):
        requests.append(request)
        if eventually_succeeds and len(requests) == 3:
            return httpx.Response(200, json={
                "id": "offline", "object": "chat.completion", "created": 0,
                "model": "deepseek-v4-pro", "choices": [{
                    "index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": VALID_ACTION_JSON},
                }],
            })
        return httpx.Response(429, json={"error": {"code": "rate_limit_exceeded"}})

    sdk_client = openai.OpenAI
    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        monkeypatch.setattr(
            openai, "OpenAI",
            lambda **kwargs: sdk_client(http_client=http_client, **kwargs),
        )
        provider = DeepSeekProvider(api_key="offline-test-key", wait=waits.append)

        if eventually_succeeds:
            assert provider.generate_next_action(investigation_state()).action_type == "finish"
        else:
            with pytest.raises(ProviderError) as caught:
                provider.generate_next_action(investigation_state())
            assert caught.value.attempts == 3
        assert provider.client.max_retries == 0

    assert len(requests) == 3
    assert waits == [0.5, 1.0]
    assert len({request.content for request in requests}) == 1


def run_case_with_provider(case, provider):
    trace = InMemoryInvestigationTraceRecorder()
    state = run_agent_investigation(
        case.initial_state.model_copy(deep=True), build_default_tool_registry(),
        build_task_index(case.tasks), case.runtime_state, provider,
        max_steps=case.max_steps, trace_recorder=trace,
    )
    return state, trace


@pytest.mark.parametrize("case", evaluation.build_deterministic_evaluation_cases(),
                         ids=lambda case: case.case_id)
def test_retries_preserve_controller_trace_eval_and_report(case, monkeypatch) -> None:
    expected_state, expected_trace = run_case_with_provider(
        case, ScriptedProvider(case.scripted_actions),
    )
    expected_eval = evaluation.run_evaluation_case(case)

    def recovering_provider(actions):
        outcomes = []
        for action in actions:
            outcomes.extend((status_error(), completion(content=action.model_dump_json())))
        return scripted_provider(*outcomes)

    provider, calls, waits = recovering_provider(case.scripted_actions)
    state, trace = run_case_with_provider(case, provider)
    monkeypatch.setattr(
        evaluation, "_ScriptedEvaluationProvider",
        lambda actions: recovering_provider(actions)[0],
    )

    assert state == expected_state
    assert trace.steps == expected_trace.steps
    assert trace.final_status == expected_trace.final_status
    assert evaluation.run_evaluation_case(case) == expected_eval
    report = build_qa_report(state, trace)
    assert report == build_qa_report(expected_state, expected_trace)
    assert render_qa_report_markdown(report) == render_qa_report_markdown(
        build_qa_report(expected_state, expected_trace)
    )
    assert len(calls) == 2 * len(case.scripted_actions)
    assert waits == [0.5] * len(case.scripted_actions)


def test_exhausted_request_preserves_prior_investigation_progress(monkeypatch) -> None:
    case = evaluation.build_deterministic_evaluation_cases()[0]
    state = case.initial_state.model_copy(deep=True)
    trace = InMemoryInvestigationTraceRecorder()

    def failing_provider(actions):
        return scripted_provider(
            completion(content=actions[0].model_dump_json()),
            status_error(), status_error(), status_error(),
        )

    provider, calls, waits = failing_provider(case.scripted_actions)
    with pytest.raises(ProviderError):
        run_agent_investigation(
            state, build_default_tool_registry(), build_task_index(case.tasks),
            case.runtime_state, provider, max_steps=2, trace_recorder=trace,
        )

    assert state.investigation_status == "running"
    assert state.called_tool_scope_versions == {"dependency_reference_checker": [1]}
    assert state.scope_task_ids == case.initial_state.scope_task_ids
    assert state.scope_version == case.initial_state.scope_version
    assert state.decision_errors == []
    assert len(trace.steps) == 1
    assert trace.final_status is None
    assert len(calls) == 4
    assert waits == [0.5, 1.0]

    monkeypatch.setattr(
        evaluation, "_ScriptedEvaluationProvider",
        lambda actions: failing_provider(actions)[0],
    )
    result = evaluation.run_evaluation_case(case)
    assert result.outcome == "harness_error"
    assert result.harness_error.exception_type == "ProviderError"
    assert result.harness_error.stage == "investigation_execution"
    assert result.evidence is None
    diagnostics = result.model_dump_json() + render_qa_report_markdown(build_qa_report(state, trace))
    diagnostics += "".join(step.model_dump_json() for step in trace.steps)
    assert "private-" not in diagnostics
