"""H5a acceptance: every provider is scripted or an SDK MockTransport."""

from copy import deepcopy
import json
from types import SimpleNamespace

import httpx
import openai
import pytest

from conftest import ScriptedProvider, action
from test_evidence_package import rewrite_artifact
from test_provider_reliability import completion, scripted_provider, status_error


def scripts():
    call = lambda name: action("call_tool", tool_name=name)
    finish = action("finish", reason="synthetic-private-final-answer")
    expand = action("expand_scope", expand_task_ids=["task_event_7"])
    return {
        "missing_dependency": [call("dependency_reference_checker"), finish],
        "dependency_cycle": [call("dependency_cycle_checker"), finish],
        "runtime_no_findings": [call("npc_runtime_checker"), finish],
        "dynamic_expansion": [call("npc_runtime_checker"), expand,
                              call("npc_static_conflict_checker"), finish],
        "expanded_scope_rerun": [call("npc_runtime_checker"), expand,
                                 call("npc_runtime_checker"), finish],
        "active_tool_selection": [call("dependency_reference_checker"), finish],
        "scope_boundary": [call("npc_runtime_checker"), finish],
        "bounded_incomplete": [call("dependency_reference_checker")],
    }


def passing_factory(slot):
    return ScriptedProvider(scripts()[slot.case_id])


def test_fixed_plan_round_trip_and_canonical_oracles(tmp_path):
    from game_qa_agent.live_eval import run_live_evidence_package, account_live_results
    from game_qa_agent.evidence import read_evidence_package

    package = run_live_evidence_package(tmp_path / "live", provider_factory=passing_factory)
    assert read_evidence_package(tmp_path / "live") == package
    assert len(package.plan.slots) == 16
    assert len({(s.case_id, s.repetition) for s in package.plan.slots}) == 16
    assert len({s.case_id for s in package.plan.slots}) == 8
    for case_id in scripts():
        slots = [s for s in package.plan.slots if s.case_id == case_id]
        assert [s.repetition for s in slots] == [1, 2]
        assert slots[0].case_fingerprint == slots[1].case_fingerprint
    counts = account_live_results(package.results)
    assert counts["planned"] == counts["completed"] == counts["behavioral_pass"] == 16
    assert counts["unscored"] == counts["provider_failures"] == 0
    assert package.plan.run.validation_only is True
    assert all(s.evidence == "complete" for s in package.results.slots)
    assert all(d.request_attempts is None for s in package.results.slots for d in s.decisions)


def test_quota_keeps_all_planned_slots_and_zero_denominator(tmp_path):
    from game_qa_agent.live_eval import run_live_evidence_package, account_live_results

    provider, calls, waits = scripted_provider(status_error(body={"code": "insufficient_quota"}))
    constructed = []

    def factory(slot):
        constructed.append(slot)
        return provider

    package = run_live_evidence_package(tmp_path / "quota", provider_factory=factory)
    first, *rest = package.results.slots
    assert first.execution == "provider_failure"
    assert first.provider_failure.category == "quota_billing"
    assert first.evaluation == "unscored"
    assert len(calls) == len(constructed) == 1
    assert waits == []
    assert len(rest) == 15
    assert all(s.execution == "not_started" and s.canonical is None for s in rest)
    counts = account_live_results(package.results)
    assert counts["planned"] == 16
    assert counts["attempted"] == 1
    assert counts["evaluated"] == 0
    report = (tmp_path / "quota" / "report.md").read_text(encoding="utf-8")
    assert "Behavioral pass / evaluated: N/A" in report
    assert "Verified behavioral pass / planned: 0/16" in report


def test_transient_retry_is_one_decision_and_does_not_repeat_tool(tmp_path):
    from game_qa_agent.live_eval import run_live_evidence_package

    actions = scripts()["missing_dependency"]
    provider, calls, waits = scripted_provider(
        status_error(), completion(actions[0].model_dump_json()),
        completion(actions[1].model_dump_json()),
    )
    package = run_live_evidence_package(
        tmp_path / "retry", provider_factory=lambda slot: provider
        if (slot.case_id, slot.repetition) == ("missing_dependency", 1) else passing_factory(slot),
    )
    slot = package.results.slots[0]
    assert slot.evaluation == "passed"
    assert len(package.results.slots) == 16 and slot.repetition == 1
    assert len(slot.canonical.tool_executions) == 1
    assert [d.logical_decision for d in slot.decisions] == [1, 2]
    assert [d.request_attempts for d in slot.decisions] == [2, 1]
    assert [d.request_retries for d in slot.decisions] == [1, 0]
    assert calls[0] == calls[1]
    assert len(calls) == 3 and waits == [0.5]


def test_self_declared_finish_cannot_pass_zero_finding_oracle(tmp_path):
    from game_qa_agent.live_eval import run_live_evidence_package

    package = run_live_evidence_package(tmp_path / "empty", provider_factory=lambda slot:
        ScriptedProvider([action("finish", reason="All checks passed")]))
    slot = next(s for s in package.results.slots if s.case_id == "runtime_no_findings")
    assert slot.execution == "completed"
    assert slot.canonical.findings == slot.canonical.tool_executions == ()
    assert slot.evaluation == "failed_behavior"
    checks = {c.expectation: c.passed for c in slot.checks}
    assert checks["issue_types"] is True
    assert checks["required_tool_executions"] is False


@pytest.mark.parametrize(("kind", "category", "attempts", "stops"), [
    ("timeout", "transport_service", 3, False),
    ("empty_choices", "invalid_response", 1, False),
    ("malformed", "invalid_response", 1, False),
    ("non_retryable", "non_retryable", 1, True),
    ("explicit_no_retry", "transport_service", 1, False),
])
def test_provider_failure_categories_preserve_slot_accounting(tmp_path, kind, category, attempts, stops):
    from game_qa_agent.live_eval import run_live_evidence_package, account_live_results

    if kind == "timeout":
        outcomes = [openai.APITimeoutError(request=httpx.Request("POST", "https://offline.invalid"))] * 3
    elif kind == "empty_choices":
        outcomes = [SimpleNamespace(choices=[])]
    elif kind == "malformed":
        outcomes = [completion("synthetic-private-completion")]
    elif kind == "explicit_no_retry":
        outcomes = [status_error(503, headers={"x-should-retry": "false", "Retry-After": "4"})]
    else:
        outcomes = [status_error(401)]
    provider, calls, waits = scripted_provider(*outcomes)
    package = run_live_evidence_package(tmp_path / kind, provider_factory=lambda slot: provider
        if (slot.case_id, slot.repetition) == ("missing_dependency", 1) else passing_factory(slot))
    first = package.results.slots[0]
    assert first.execution == "provider_failure" and first.evaluation == "unscored"
    assert first.evidence == "complete" and first.checks == ()
    assert first.provider_failure.category == category
    assert first.provider_failure.attempts == len(calls) == attempts
    assert waits == ([0.5, 1.0] if kind == "timeout" else [])
    c = account_live_results(package.results)
    assert c["planned"] == 16 and c["provider_failure_categories"] == {category: 1}
    assert c["behavioral_fail"] == 0
    assert c["evaluated"] == (0 if stops else 15)
    assert c["not_started"] == (15 if stops else 0)
    report = (tmp_path / kind / "report.md").read_text(encoding="utf-8")
    assert f"Provider category `{category}`: 1" in report
    assert "synthetic-private" not in report


def test_rejection_is_behavioral_not_operational_and_scope_is_controller_owned(tmp_path):
    from game_qa_agent.live_eval import run_live_evidence_package

    def factory(slot):
        actions = scripts()[slot.case_id]
        if slot.case_id == "active_tool_selection":
            actions = [action("call_tool", tool_name="npc_runtime_checker")] + actions
        if slot.case_id == "scope_boundary":
            actions = [action("expand_scope", expand_task_ids=["synthetic-private-rejected-id"])] + actions
        return ScriptedProvider(actions)

    package = run_live_evidence_package(tmp_path / "rejected", provider_factory=factory)
    for slot in package.results.slots:
        if slot.case_id in {"active_tool_selection", "scope_boundary"}:
            assert slot.execution == "completed" and slot.evaluation == "failed_behavior"
            assert slot.provider_failure is None
            assert slot.canonical.decision_error_count == 1
            assert len(slot.canonical.tool_executions) == 1
            assert "synthetic-private-rejected-id" not in slot.canonical.scope_task_ids
    assert "synthetic-private" not in (tmp_path / "rejected" / "results.json").read_text()


def test_plan_precedes_provider_setup_and_repetitions_have_independent_context(tmp_path, monkeypatch):
    import game_qa_agent.live_eval as live
    from game_qa_agent.evidence import _project_canonical
    from game_qa_agent.analysis import build_task_index
    from game_qa_agent.orchestration import run_agent_investigation
    from game_qa_agent.tools import build_default_tool_registry
    from game_qa_agent.trace import InMemoryInvestigationTraceRecorder

    directory = tmp_path / "frozen"
    contexts = {}
    definitions = {d.case.case_id: d for d in live.build_fixed_live_cases()}
    before = {name: d.case.model_dump_json() for name, d in definitions.items()}
    monkeypatch.setattr(live, "build_fixed_live_cases", lambda: tuple(definitions.values()))

    class Observe(ScriptedProvider):
        def __init__(self, slot):
            super().__init__(scripts()[slot.case_id])
            self.slot = slot

        def generate_next_action(self, context):
            contexts.setdefault((self.slot.case_id, self.slot.repetition), []).append(context.model_dump_json())
            return super().generate_next_action(context)

    def factory(slot):
        assert (directory / "plan.json").is_file()
        assert not (directory / "manifest.json").exists()
        assert len(json.loads((directory / "plan.json").read_bytes())["slots"]) == 16
        return Observe(slot)

    package = live.run_live_evidence_package(directory, provider_factory=factory)
    for case_id, definition in definitions.items():
        assert contexts[(case_id, 1)][0] == contexts[(case_id, 2)][0]
        initial = json.loads(contexts[(case_id, 2)][0])
        assert initial["scope_version"] == 1
        assert initial["decision_error_count"] == 0 and initial["last_decision_rejection"] is None
        assert all(c["last_called_scope_version"] is None for c in initial["active_tools"])
        case = definition.case.model_copy(deep=True)
        registry = build_default_tool_registry()
        registry.tools = {name: registry.get_tool(name) for name in definition.active_tools}
        trace = InMemoryInvestigationTraceRecorder()
        expected = run_agent_investigation(
            case.initial_state, registry, build_task_index(case.tasks), case.runtime_state,
            ScriptedProvider(scripts()[case_id]), max_steps=case.max_steps, trace_recorder=trace,
        )
        actual = next(s for s in package.results.slots if s.case_id == case_id)
        assert actual.canonical == _project_canonical(expected)
        assert len(actual.decisions) == len(trace.steps)
    assert {name: d.case.model_dump_json() for name, d in definitions.items()} == before


@pytest.mark.parametrize("failure_stage", ["factory", "tool_runtime", "tool_provider_error"])
def test_harness_failures_never_enter_behavioral_or_provider_denominator(tmp_path, monkeypatch, failure_stage):
    import game_qa_agent.live_eval as live
    from game_qa_agent.models import ValidationIssue
    from game_qa_agent.providers import ProviderError, ProviderFailure

    original = live.build_default_tool_registry

    def registry():
        tools = original()

        def partial(**kwargs):
            yield ValidationIssue(issue_type="missing_dependency", message="synthetic-private-issue",
                                  checker_name="dependency_reference_checker", task_ids=["task_a"],
                                  evidence={"nested": {"raw_output": "synthetic-private-output"}})
            if failure_stage == "tool_provider_error":
                raise ProviderError(ProviderFailure("quota_billing", False), 1, "synthetic-private-exception")
            raise RuntimeError("synthetic-private-exception")

        tools.register_tool("dependency_reference_checker", partial)
        return tools

    monkeypatch.setattr(live, "build_default_tool_registry", registry)

    def factory(slot):
        if failure_stage == "factory":
            raise RuntimeError("synthetic-private-factory")
        return passing_factory(slot)

    package = live.run_live_evidence_package(tmp_path / failure_stage, provider_factory=factory)
    first = package.results.slots[0]
    assert (first.execution, first.evaluation, first.evidence) == ("harness_failure", "unscored", "complete")
    assert first.provider_failure is None
    if failure_stage != "factory":
        assert first.canonical.investigation_status == "running"
        record, = first.canonical.tool_executions
        assert record.status == "failed" and record.issue_indices == (0,)
    assert live.account_live_results(package.results)["not_started"] == 15
    for path in (tmp_path / failure_stage).iterdir():
        assert "synthetic-private" not in path.read_text(encoding="utf-8")


def test_sdk_privacy_and_actual_configuration_are_preserved_without_network(tmp_path, monkeypatch):
    import game_qa_agent.live_eval as live
    from game_qa_agent.models import AgentInvestigationState
    from game_qa_agent.providers import DeepSeekProvider

    sdk_client = openai.OpenAI
    requests, clients = [], []

    def factory(slot):
        actions = iter(scripts()[slot.case_id])

        def respond(request):
            requests.append(request)
            decision = next(actions).model_copy(deep=True)
            decision.reason = "synthetic-private-provider-reason"
            decision.tool_args = {"nested": {"raw": "synthetic-private-args"}}
            return httpx.Response(200, json={
                "id": "synthetic-private-call-id", "object": "chat.completion", "created": 0,
                "model": "synthetic-private-returned-model", "metadata": {"raw": "synthetic-private-metadata"},
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": decision.model_dump_json()}}],
            })

        http_client = httpx.Client(transport=httpx.MockTransport(respond))
        clients.append(http_client)
        monkeypatch.setattr(openai, "OpenAI", lambda **kw: sdk_client(http_client=http_client, **kw))
        provider = DeepSeekProvider(api_key="synthetic-private-key", wait=lambda _: pytest.fail("Unexpected wait"))
        assert provider.resolved_configuration().sdk_max_retries == 0
        return provider

    def forbidden_dump(*args, **kwargs):
        pytest.fail("Full Agent state must never be serialized")

    monkeypatch.setattr(AgentInvestigationState, "model_dump", forbidden_dump)
    monkeypatch.setattr(AgentInvestigationState, "model_dump_json", forbidden_dump)
    try:
        package = live.run_live_evidence_package(tmp_path / "privacy", provider_factory=factory)
    finally:
        for client in clients:
            client.close()
    assert live.account_live_results(package.results)["behavioral_pass"] == 16
    assert all(d.returned_model is None for s in package.results.slots for d in s.decisions)
    assert all(set(request.extensions["timeout"].values()) == {package.plan.provider.timeout_seconds}
               for request in requests)
    for path in (tmp_path / "privacy").iterdir():
        assert "synthetic-private" not in path.read_text(encoding="utf-8")


@pytest.fixture
def live_package_directory(tmp_path):
    from game_qa_agent.live_eval import run_live_evidence_package

    directory = tmp_path / "integrity"
    run_live_evidence_package(directory, provider_factory=passing_factory)
    return directory


@pytest.mark.parametrize(("mutation", "code"), [
    ("missing", "slot_set_mismatch"), ("duplicate", "duplicate_slot"),
    ("unexpected", "slot_set_mismatch"), ("repetition", "duplicate_slot"),
    ("fingerprint", "case_identity_mismatch"), ("dangling", "dangling_finding_reference"),
    ("lost_execution", "unlinked_finding"), ("incomplete_pass", "required_evidence_incomplete"),
    ("wrong_run", "run_mismatch"), ("wrong_retry_count", "invalid_request_accounting"),
])
def test_live_reader_rejects_slot_provenance_and_accounting_corruption(live_package_directory, mutation, code):
    from game_qa_agent.evidence import EvidencePackageError, read_evidence_package

    def mutate(data):
        slots = data["slots"]
        if mutation == "missing":
            slots.pop()
        elif mutation == "duplicate":
            slots.append(deepcopy(slots[0]))
        elif mutation == "unexpected":
            slots[-1]["case_id"] = "unexpected"
        elif mutation == "repetition":
            slots[0]["repetition"] = 2
        elif mutation == "fingerprint":
            slots[0]["case_fingerprint"] = "0" * 64
        elif mutation == "dangling":
            slots[0]["canonical"]["tool_executions"][0]["issue_indices"] = [999]
        elif mutation == "lost_execution":
            slots[0]["canonical"]["tool_executions"] = []
        elif mutation == "incomplete_pass":
            slots[0]["evidence"] = "incomplete"
        elif mutation == "wrong_run":
            data["run"]["run_id"] = "0" * 32
        else:
            slots[0]["decisions"][0]["request_attempts"] = 2
            slots[0]["decisions"][0]["request_retries"] = 0

    rewrite_artifact(live_package_directory, "results.json", mutate)
    with pytest.raises(EvidencePackageError, match=code):
        read_evidence_package(live_package_directory)


@pytest.mark.parametrize("name", ["plan.json", "results.json", "report.md"])
def test_live_files_use_h4_hash_validation(live_package_directory, name):
    from game_qa_agent.evidence import EvidencePackageError, read_evidence_package

    path = live_package_directory / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(EvidencePackageError, match="artifact_mismatch"):
        read_evidence_package(live_package_directory)


def test_live_packages_cannot_be_joined_across_runs(live_package_directory, tmp_path):
    from game_qa_agent.live_eval import run_live_evidence_package
    from game_qa_agent.evidence import EvidencePackageError, read_evidence_package

    run_live_evidence_package(tmp_path / "other", provider_factory=passing_factory)
    other = json.loads((tmp_path / "other" / "results.json").read_bytes())
    rewrite_artifact(live_package_directory, "results.json", lambda data: data.update(other))
    with pytest.raises(EvidencePackageError, match="run_mismatch"):
        read_evidence_package(live_package_directory)


@pytest.mark.parametrize("name", ["plan.json", "results.json", "report.md", "manifest.json"])
def test_live_required_write_failure_cannot_publish(tmp_path, monkeypatch, name):
    import game_qa_agent.evidence as evidence
    from game_qa_agent.live_eval import run_live_evidence_package

    original = evidence._write_artifact
    calls = []

    def broken(directory, artifact, data):
        original(directory, artifact, data)
        if artifact == name:
            raise OSError("synthetic-private-write-failure")

    monkeypatch.setattr(evidence, "_write_artifact", broken)
    with pytest.raises(evidence.EvidencePackageError, match="publication_failed"):
        run_live_evidence_package(tmp_path / name, provider_factory=lambda slot:
            (calls.append(slot), passing_factory(slot))[1])
    if name == "plan.json":
        assert calls == []
    assert not (tmp_path / name / "manifest.json").exists()
    with pytest.raises(evidence.EvidencePackageError, match="incomplete"):
        evidence.read_evidence_package(tmp_path / name)


def test_case_identity_binds_fixture_oracle_and_goal_but_not_order():
    import game_qa_agent.live_eval as live

    definitions = live.build_fixed_live_cases()
    expected = {d.case.case_id: live._live_slot(d, 1).case_fingerprint for d in definitions}
    assert expected == {d.case.case_id: live._live_slot(d, 2).case_fingerprint for d in reversed(definitions)}
    for field in ("fixture", "oracle", "goal"):
        definition = deepcopy(definitions[0])
        if field == "fixture":
            definition.case.tasks[0].dependencies = ["different_missing_task"]
        elif field == "oracle":
            definition.case.expected_issue_types = []
        else:
            definition.case.initial_state.investigation_goal = "Different trusted local goal"
        assert live._live_slot(definition, 1).case_fingerprint != expected[definition.case.case_id]


def test_live_export_rejects_missing_canonical_evidence_even_when_tool_succeeded(tmp_path, monkeypatch):
    import game_qa_agent.live_eval as live
    from game_qa_agent.evidence import EvidencePackageError

    original = live.run_agent_investigation

    def lose_record(*args, **kwargs):
        state = original(*args, **kwargs)
        state.tool_executions.clear()
        return state

    monkeypatch.setattr(live, "run_agent_investigation", lose_record)
    with pytest.raises(EvidencePackageError, match="missing_execution_evidence"):
        live.run_live_evidence_package(tmp_path / "lost", provider_factory=passing_factory)
    assert not (tmp_path / "lost" / "manifest.json").exists()


@pytest.mark.parametrize("drift_after_first_decision", [False, True])
def test_effective_sdk_configuration_drift_stops_before_next_request(tmp_path, drift_after_first_decision):
    from game_qa_agent.live_eval import run_live_evidence_package

    actions = scripts()["missing_dependency"]
    provider, calls, waits = scripted_provider(*(completion(a.model_dump_json()) for a in actions))
    provider.client.timeout = 30.0
    provider.client.max_retries = 0 if drift_after_first_decision else 1
    create = provider.client.chat.completions.create

    def mutate_after_response(**kwargs):
        response = create(**kwargs)
        provider.client.max_retries = 1
        return response

    provider.client.chat.completions.create = mutate_after_response
    package = run_live_evidence_package(tmp_path / "drift", provider_factory=lambda slot: provider)
    first, *rest = package.results.slots
    assert first.execution == "harness_failure" and first.evaluation == "unscored"
    assert first.provider_failure is None
    assert len(calls) == int(drift_after_first_decision)
    assert len(first.canonical.tool_executions) == int(drift_after_first_decision)
    assert all(s.execution == "not_started" for s in rest)


def test_non_validation_mode_requires_real_provider_type_without_constructing_one(tmp_path, monkeypatch):
    from game_qa_agent.live_eval import run_live_evidence_package
    from game_qa_agent.providers import DeepSeekProvider

    def forbidden(*args, **kwargs):
        pytest.fail("A real provider must never be constructed implicitly")

    monkeypatch.setattr(DeepSeekProvider, "__init__", forbidden)
    package = run_live_evidence_package(tmp_path / "gated", provider_factory=passing_factory,
                                        validation_only=False)
    first = package.results.slots[0]
    assert first.execution == "harness_failure" and first.harness_stage == "provider_setup"
    assert first.decisions == ()
    assert all(s.evaluation == "unscored" for s in package.results.slots)


def test_live_reader_cannot_certify_pass_after_removing_required_zero_finding_execution(live_package_directory):
    from game_qa_agent.evidence import EvidencePackageError, read_evidence_package

    def remove(data):
        result = next(s for s in data["slots"] if s["case_id"] == "runtime_no_findings")
        result["canonical"]["tool_executions"] = []

    rewrite_artifact(live_package_directory, "results.json", remove)
    with pytest.raises(EvidencePackageError, match="inconsistent_expectation"):
        read_evidence_package(live_package_directory)
