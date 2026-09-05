from copy import deepcopy
from hashlib import sha256
import json
import os
import subprocess

from pydantic import Field
import pytest

import game_qa_agent.eval as evaluation
import game_qa_agent.evidence as evidence
from game_qa_agent.evidence import (
    EvidencePackageError, canonical_json, read_evidence_package, run_evidence_package,
)

from game_qa_agent.eval import build_deterministic_evaluation_cases, run_evaluation_suite
from game_qa_agent.models import AgentInvestigationState, NextActionSpec, Task, ValidationIssue
from game_qa_agent.tools import build_default_tool_registry


def test_offline_package_round_trip_preserves_existing_eval(tmp_path):
    cases = build_deterministic_evaluation_cases()
    before = [case.model_dump() for case in cases]
    expected = run_evaluation_suite(cases)
    directory = tmp_path / "run"

    published = run_evidence_package(directory, cases)
    reread = read_evidence_package(directory)

    assert reread == published
    assert sorted(path.name for path in directory.iterdir()) == [
        "manifest.json", "plan.json", "report.md", "results.json",
    ]
    assert [slot.evaluation for slot in reread.results.slots] == [
        result.outcome for result in expected.case_results
    ]
    assert all(slot.execution == "completed" for slot in reread.results.slots)
    assert all(slot.evidence == "complete" for slot in reread.results.slots)
    assert [case.model_dump() for case in cases] == before
    assert reread.plan.run.mode == "offline"
    assert len(reread.plan.run.source_commit) == 40
    assert reread.plan.run.source_commit == subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert set(reread.plan.run.runtime_versions) == {"python", "pydantic", "networkx", "openai"}
    assert type(reread.plan.run.source_dirty) is bool
    assert reread.plan.run.source_dirty == bool(subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        check=True, capture_output=True,
    ).stdout)
    assert all(slot.repetition == 1 for slot in reread.plan.slots)
    with pytest.raises(FileExistsError):
        run_evidence_package(directory, cases)


@pytest.fixture
def package_directory(tmp_path):
    directory = tmp_path / "package"
    run_evidence_package(directory)
    return directory


def rewrite_artifact(directory, name, change):
    """Re-hash deliberately, so semantic tests reach the validator past SHA checks."""
    document = json.loads((directory / name).read_bytes())
    change(document)
    data = canonical_json(document)
    (directory / name).write_bytes(data)
    manifest = json.loads((directory / "manifest.json").read_bytes())
    manifest["artifacts"][name] = {"size": len(data), "sha256": sha256(data).hexdigest()}
    (directory / "manifest.json").write_bytes(canonical_json(manifest))


@pytest.mark.parametrize("name", ["manifest.json", "plan.json", "results.json"])
def test_reader_rejects_unknown_schema(package_directory, name):
    if name == "manifest.json":
        document = json.loads((package_directory / name).read_bytes())
        document["schema_version"] = 999
        (package_directory / name).write_bytes(canonical_json(document))
    else:
        rewrite_artifact(package_directory, name, lambda data: data.update(schema_version=999))
    with pytest.raises(EvidencePackageError, match="unsupported_schema"):
        read_evidence_package(package_directory)


@pytest.mark.parametrize("name", ["manifest.json", "plan.json", "results.json", "report.md"])
def test_missing_required_artifact_cannot_be_completed(package_directory, name):
    (package_directory / name).unlink()
    code = "incomplete" if name == "manifest.json" else "missing_artifact"
    with pytest.raises(EvidencePackageError, match=code):
        read_evidence_package(package_directory)


@pytest.mark.parametrize("name", ["plan.json", "results.json", "report.md"])
def test_reader_detects_any_artifact_byte_change(package_directory, name):
    path = package_directory / name
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 1
    path.write_bytes(data)
    with pytest.raises(EvidencePackageError, match="artifact_mismatch"):
        read_evidence_package(package_directory)


@pytest.mark.parametrize(("mutation", "code"), [
    ("missing", "slot_set_mismatch"), ("duplicate", "duplicate_slot"),
    ("unexpected", "slot_set_mismatch"), ("fingerprint", "case_identity_mismatch"),
])
def test_slot_integrity_uses_keys_and_identity_not_counts(package_directory, mutation, code):
    def change(data):
        if mutation == "missing":
            data["slots"].pop()
        elif mutation == "duplicate":
            data["slots"][1] = deepcopy(data["slots"][0])
        elif mutation == "unexpected":
            data["slots"][1]["case_id"] = "unexpected_same_total_count"
        else:
            data["slots"][0]["case_fingerprint"] = "f" * 64
    rewrite_artifact(package_directory, "results.json", change)
    with pytest.raises(EvidencePackageError, match=code):
        read_evidence_package(package_directory)


@pytest.mark.parametrize("copied", [("results.json",), ("manifest.json",), ("results.json", "manifest.json")])
def test_cannot_join_artifacts_from_different_runs(package_directory, tmp_path, copied):
    other = tmp_path / "other"
    run_evidence_package(other)
    for name in copied:
        (other / name).write_bytes((package_directory / name).read_bytes())
    if copied == ("results.json",):
        rewrite_artifact(other, "results.json", lambda data: None)
    with pytest.raises(EvidencePackageError, match="run_mismatch|artifact_mismatch"):
        read_evidence_package(other)


@pytest.mark.parametrize(("mutation", "code"), [
    ("dangling", "dangling_finding_reference"), ("execution", "invalid_execution_reference"),
    ("unlinked", "unlinked_finding"), ("finding_index", "invalid_finding_indices"),
])
def test_canonical_references_must_remain_local_and_valid(package_directory, mutation, code):
    def change(data):
        canonical = data["slots"][1]["canonical"]
        if mutation == "dangling":
            canonical["tool_executions"][0]["issue_indices"].append(len(canonical["findings"]))
        elif mutation == "execution":
            canonical["tool_executions"][0]["execution_number"] = 20
        elif mutation == "unlinked":
            canonical["tool_executions"][0]["issue_indices"] = []
        else:
            canonical["findings"][0]["issue_index"] = 99
    rewrite_artifact(package_directory, "results.json", change)
    with pytest.raises(EvidencePackageError, match=code):
        read_evidence_package(package_directory)


@pytest.mark.parametrize("kind", ["fixture", "expectation"])
def test_case_content_changes_fingerprint_but_execution_order_does_not(tmp_path, kind):
    cases = build_deterministic_evaluation_cases()
    first = run_evidence_package(tmp_path / "first", cases)
    reverse = run_evidence_package(tmp_path / "reverse", list(reversed(cases)))
    fingerprints = lambda package: {slot.case_id: slot.case_fingerprint for slot in package.plan.slots}
    assert fingerprints(first) == fingerprints(reverse)
    changed = cases[0].model_copy(deep=True)
    if kind == "fixture":
        changed.tasks[0].dependencies.append("missing_trusted_fixture_task")
    else:
        changed.expected_final_status = "human_review_required"
    third = run_evidence_package(tmp_path / "changed", [changed])
    assert third.plan.slots[0].case_id == first.plan.slots[0].case_id
    assert third.plan.slots[0].case_fingerprint != first.plan.slots[0].case_fingerprint


def test_plan_is_written_before_execution_and_execution_uses_frozen_case(tmp_path, monkeypatch):
    cases = build_deterministic_evaluation_cases()
    expected = run_evaluation_suite(cases)
    directory = tmp_path / "frozen"
    original = evidence._run_evaluation_case_with_state
    calls = []

    def inspect(case):
        assert (directory / "plan.json").is_file()
        assert not (directory / "manifest.json").exists()
        calls.append(case.case_id)
        cases[0].scripted_actions.clear()
        cases[0].expected_final_status = "human_review_required"
        return original(case)

    monkeypatch.setattr(evidence, "_run_evaluation_case_with_state", inspect)
    package = run_evidence_package(directory, cases)
    assert calls == [slot.case_id for slot in package.plan.slots]
    assert [slot.evaluation for slot in package.results.slots] == [
        result.outcome for result in expected.case_results
    ]


@pytest.mark.parametrize("value", [object(), float("nan"), float("inf"), {1: "key"}])
def test_canonical_identity_rejects_unsupported_values(value):
    with pytest.raises(EvidencePackageError, match="unsupported_canonical_value"):
        canonical_json({"nested": [value]})


def test_canonical_json_is_utf8_sorted_and_preserves_falsy_values():
    assert canonical_json({"z": "中文", "a": [None, 0, False, []]}) == (
        '{"a":[null,0,false,[]],"z":"中文"}\n'.encode("utf-8")
    )


def test_dedup_links_and_equal_safe_findings_keep_actual_execution_identity(tmp_path, monkeypatch):
    case = build_deterministic_evaluation_cases()[1]
    case.scripted_actions = [
        NextActionSpec(action_type="call_tool", tool_name="npc_runtime_checker", reason="private"),
        NextActionSpec(action_type="expand_scope", expand_task_ids=["task_event_7"], reason="private"),
        NextActionSpec(action_type="call_tool", tool_name="npc_runtime_checker", reason="private"),
        NextActionSpec(action_type="finish", reason="private"),
    ]
    case.expected_issue_types = ["npc_location_mismatch", "npc_occupied_by_other_task"]
    package = run_evidence_package(tmp_path / "dedup", [case])
    records = package.results.slots[0].canonical.tool_executions
    assert [(r.execution_number, r.scope_version, r.issue_indices) for r in records] == [
        (1, 1, (0, 1)), (2, 2, (0, 1)),
    ]

    def registry():
        active = build_default_tool_registry()
        active.register_tool("dependency_reference_checker", lambda **kwargs: [
            ValidationIssue(issue_type="missing_dependency", checker_name="dependency_reference_checker",
                            task_ids=["task_a"], message=f"private-{index}", evidence={"raw": index})
            for index in range(2)
        ])
        return active

    monkeypatch.setattr(evaluation, "build_default_tool_registry", registry)
    case = build_deterministic_evaluation_cases()[0]
    case.expected_issue_types = ["missing_dependency", "missing_dependency"]
    package = run_evidence_package(tmp_path / "equal-safe", [case])
    canonical = package.results.slots[0].canonical
    assert canonical.findings[0].detail == canonical.findings[1].detail
    assert [f.issue_index for f in canonical.findings] == [0, 1]
    assert canonical.tool_executions[0].issue_indices == (0, 1)


def test_package_privacy_covers_nested_fields_private_actions_and_partial_failure(tmp_path, monkeypatch):
    canary = "SYNTHETIC-SECRET-CANARY"

    class IssueWithMetadata(ValidationIssue):
        metadata: dict = Field(default_factory=lambda: {"nested": canary})

    class TaskWithMetadata(Task):
        metadata: dict = Field(default_factory=lambda: {"nested": canary})

    class StateWithMetadata(AgentInvestigationState):
        private_diagnostics: dict = Field(default_factory=lambda: {"nested": canary})

    def partial(**kwargs):
        yield IssueWithMetadata(
            issue_type="missing_dependency", checker_name="dependency_reference_checker",
            task_ids=["task_a", canary], npc_ids=[canary],
            message=canary, evidence={"nested": [{"raw_tool_payload": canary}]},
        )
        raise RuntimeError(canary)

    def registry():
        active = build_default_tool_registry()
        active.register_tool("dependency_reference_checker", partial)
        return active

    monkeypatch.setattr(evaluation, "build_default_tool_registry", registry)
    cases = build_deterministic_evaluation_cases()
    cases[0].tasks = [TaskWithMetadata(id="task_a")]
    cases[0].initial_state = StateWithMetadata(**cases[0].initial_state.model_dump())
    cases[0].scripted_actions[0].reason = canary
    cases[3].scripted_actions[0] = NextActionSpec(
        action_type="call_tool", tool_name=canary, reason=canary,
        tool_args={"raw_tool_payload": {"nested": canary}}, expand_task_ids=[canary],
    )
    original_json = evidence.canonical_json

    def inspect_hash_and_export_input(value):
        assert canary not in repr(value)
        return original_json(value)

    def forbid_state_dump(*args, **kwargs):
        raise AssertionError("Whole state serialization is forbidden")

    monkeypatch.setattr(evidence, "canonical_json", inspect_hash_and_export_input)
    monkeypatch.setattr(AgentInvestigationState, "model_dump", forbid_state_dump)
    monkeypatch.setattr(AgentInvestigationState, "model_dump_json", forbid_state_dump)
    package = run_evidence_package(tmp_path / "privacy", [cases[0], cases[3]])
    slot = package.results.slots[0]
    assert (slot.execution, slot.evaluation, slot.evidence) == ("harness_error", "unscored", "complete")
    assert slot.canonical.investigation_status == "running"
    assert slot.canonical.tool_executions[0].status == "failed"
    assert slot.canonical.tool_executions[0].issue_indices == (0,)
    assert slot.canonical.findings[0].detail.unscoped_task_count == 1
    assert slot.trace_step_count is None
    for path in (tmp_path / "privacy").iterdir():
        text = path.read_text(encoding="utf-8")
        assert canary not in text
        assert "raw_tool_payload" not in text
        assert "exception_type" not in text
    # Private action text and rejected payload contents are not even hash input.
    active = build_default_tool_registry().active_tool_names()
    original = evidence._plan_slot(cases[3], active)
    cases[3].scripted_actions[0].tool_args = {"different": object()}
    cases[3].scripted_actions[0].reason = "different private reason"
    cases[3].scripted_actions[0].tool_name = "different-unknown-name"
    cases[3].scripted_actions[0].expand_task_ids = ["different-unknown-id"]
    assert evidence._plan_slot(cases[3], active).case_fingerprint == original.case_fingerprint


@pytest.mark.parametrize("name", ["plan.json", "results.json", "report.md", "manifest.json"])
@pytest.mark.parametrize("after_visible", [False, True])
def test_required_write_failure_never_confirms_publication(tmp_path, monkeypatch, name, after_visible):
    original = evidence._write_artifact

    def failing(directory, filename, data):
        if filename == name and not after_visible:
            raise OSError("SYNTHETIC-write-error")
        original(directory, filename, data)
        if filename == name:
            raise OSError("SYNTHETIC-visible-but-not-confirmed")

    monkeypatch.setattr(evidence, "_write_artifact", failing)
    directory = tmp_path / "failed"
    with pytest.raises(EvidencePackageError, match="publication_failed") as caught:
        run_evidence_package(directory)
    assert caught.value.__context__ is None
    assert not (directory / "manifest.json").exists()
    with pytest.raises(EvidencePackageError, match="incomplete"):
        read_evidence_package(directory)


def test_file_fsync_precedes_replace_and_manifest_is_last(tmp_path, monkeypatch):
    fsync, replace = os.fsync, os.replace
    events = []

    def sync(fd):
        events.append("fsync")
        return fsync(fd)

    def publish(source, target):
        assert source.parent == target.parent
        events.append(target.name)
        return replace(source, target)

    monkeypatch.setattr(evidence.os, "fsync", sync)
    monkeypatch.setattr(evidence.os, "replace", publish)
    run_evidence_package(tmp_path / "fsync")
    assert events == ["fsync", "plan.json", "fsync", "results.json", "fsync", "report.md",
                      "fsync", "manifest.json"]


def test_fsync_failure_is_not_a_successful_visible_package(tmp_path, monkeypatch):
    def fail(fd):
        raise OSError("SYNTHETIC-fsync-error")
    monkeypatch.setattr(evidence.os, "fsync", fail)
    directory = tmp_path / "fsync-failed"
    with pytest.raises(EvidencePackageError, match="publication_failed"):
        run_evidence_package(directory)
    assert not (directory / "manifest.json").exists()


def test_missing_canonical_zero_finding_record_cannot_be_replaced_by_trace(tmp_path, monkeypatch):
    original = evidence._run_evaluation_case_with_state

    def lose_record(case):
        result, state = original(case)
        assert result.outcome == "passed" and result.evidence.trace_step_count == 2
        state.tool_executions.clear()
        return result, state

    monkeypatch.setattr(evidence, "_run_evaluation_case_with_state", lose_record)
    directory = tmp_path / "lost-evidence"
    with pytest.raises(EvidencePackageError, match="required_evidence_incomplete"):
        run_evidence_package(directory, build_deterministic_evaluation_cases()[:1])
    result = json.loads((directory / "results.json").read_bytes())["slots"][0]
    assert result["evaluation"] == "unscored"
    assert result["evidence"] == "invalid"
    assert result["canonical"] is None
    assert result["trace_step_count"] == 2
    assert not (directory / "manifest.json").exists()


def test_harness_errors_and_empty_suite_have_no_behavioral_denominator(tmp_path):
    case = build_deterministic_evaluation_cases()[0]
    case.scripted_actions = []
    failed = run_evidence_package(tmp_path / "harness", [case])
    assert failed.results.slots[0].execution == "harness_error"
    assert failed.results.slots[0].evaluation == "unscored"
    assert failed.results.slots[0].canonical.tool_executions == ()
    empty = run_evidence_package(tmp_path / "empty", [])
    assert empty.results.planned_case_count == 0
    for directory in (tmp_path / "harness", tmp_path / "empty"):
        report = (directory / "report.md").read_text(encoding="utf-8")
        assert "N/A (no evaluated cases)" in report
        assert "0%" not in report


def test_trace_absence_does_not_invalidate_canonical_evidence(tmp_path):
    case = build_deterministic_evaluation_cases()[0]
    case.trace_enabled = False
    case.expected_trace_step_count = case.expected_trace_final_status = None
    package = run_evidence_package(tmp_path / "no-trace", [case])
    slot = package.results.slots[0]
    assert slot.evaluation == "passed" and slot.evidence == "complete"
    assert slot.trace_step_count is slot.trace_final_status is None
    assert slot.canonical.tool_executions[0].status == "succeeded"


def test_incomplete_evidence_cannot_certify_a_behavioral_pass(package_directory):
    rewrite_artifact(package_directory, "results.json", lambda data: data["slots"][0].update(
        evidence="incomplete", canonical=None,
    ))
    with pytest.raises(EvidencePackageError, match="required_evidence_incomplete"):
        read_evidence_package(package_directory)


def test_reader_rejects_pass_that_contradicts_canonical_observation(package_directory):
    rewrite_artifact(package_directory, "results.json", lambda data: data["slots"][0]["canonical"].update(
        decision_error_count=99,
    ))
    # Keep even the derived report and all byte hashes consistent. The oracle
    # expects zero decision errors, so the stored passing check is contradictory.
    results = evidence.EvidenceResults.model_validate_json((package_directory / "results.json").read_bytes())
    report = evidence.render_evidence_report(results).encode("utf-8")
    (package_directory / "report.md").write_bytes(report)
    manifest = json.loads((package_directory / "manifest.json").read_bytes())
    manifest["artifacts"]["report.md"] = {"size": len(report), "sha256": sha256(report).hexdigest()}
    (package_directory / "manifest.json").write_bytes(canonical_json(manifest))
    with pytest.raises(EvidencePackageError, match="inconsistent_expectation"):
        read_evidence_package(package_directory)


@pytest.mark.parametrize("field", ["schema_version", "repetition"])
def test_boolean_is_not_an_integer_identity(package_directory, field):
    def change(data):
        target = data if field == "schema_version" else data["slots"][0]
        target[field] = True
    rewrite_artifact(package_directory, "results.json", change)
    with pytest.raises(EvidencePackageError, match="invalid_schema|unsupported_schema"):
        read_evidence_package(package_directory)


def test_runtime_identity_requires_the_recorded_dependency_versions(package_directory):
    rewrite_artifact(package_directory, "plan.json", lambda data: data["run"].update(runtime_versions={}))
    rewrite_artifact(package_directory, "results.json", lambda data: data["run"].update(runtime_versions={}))
    with pytest.raises(EvidencePackageError, match="invalid_schema"):
        read_evidence_package(package_directory)


@pytest.mark.parametrize("problem", ["duplicate", "prior_history", "numeric_case_id"])
def test_invalid_plan_is_rejected_before_any_case_executes(tmp_path, monkeypatch, problem):
    cases = build_deterministic_evaluation_cases()[:1]
    if problem == "duplicate":
        cases.append(cases[0].model_copy(deep=True))
    elif problem == "prior_history":
        cases[0].initial_state.called_tool_names.append("dependency_reference_checker")
    else:
        cases[0].case_id = 123

    def forbidden(case):
        pytest.fail("Invalid plan must not execute")

    monkeypatch.setattr(evidence, "_run_evaluation_case_with_state", forbidden)
    directory = tmp_path / "invalid-plan"
    with pytest.raises(ValueError):
        run_evidence_package(directory, cases)
    assert not directory.exists()


def test_disk_validation_failure_prevents_manifest_publication(tmp_path, monkeypatch):
    original = evidence._write_artifact

    def corrupt(directory, name, data):
        original(directory, name, data)
        if name == "results.json":
            (directory / name).write_bytes(b"invalid-required-evidence")

    monkeypatch.setattr(evidence, "_write_artifact", corrupt)
    directory = tmp_path / "disk-corruption"
    with pytest.raises(EvidencePackageError, match="artifact_mismatch"):
        run_evidence_package(directory)
    assert not (directory / "manifest.json").exists()


def test_nested_unknown_fields_are_rejected_without_propagating_file_text(package_directory):
    canary = "SYNTHETIC-SECRET-IN-UNTRUSTED-FILE"
    rewrite_artifact(package_directory, "results.json", lambda data: data["slots"][1]["canonical"]
                     ["findings"][0]["detail"].update(private_metadata={"nested": canary}))
    with pytest.raises(EvidencePackageError, match="invalid_schema") as caught:
        read_evidence_package(package_directory)
    assert canary not in str(caught.value)
    assert caught.value.__context__ is caught.value.__cause__ is None


def test_package_report_uses_existing_outcomes_and_keeps_harness_out_of_denominator(tmp_path, monkeypatch):
    from game_qa_agent.providers import DeepSeekProvider

    def forbidden(*args, **kwargs):
        pytest.fail("Evidence must use only the scripted provider")

    monkeypatch.setattr(DeepSeekProvider, "__init__", forbidden)
    case = build_deterministic_evaluation_cases()[0]
    failing = case.model_copy(deep=True, update={"case_id": "wrong_expectation"})
    failing.expected_final_status = "human_review_required"
    harness = case.model_copy(deep=True, update={"case_id": "empty_script", "scripted_actions": []})
    directory = tmp_path / "mixed"
    package = run_evidence_package(directory, [case, failing, harness])
    assert [slot.evaluation for slot in package.results.slots] == ["passed", "failed_behavior", "unscored"]
    report = (directory / "report.md").read_text(encoding="utf-8")
    assert "Planned: 3; evaluated: 2; behavioral pass: 1" in report
    assert "behavioral fail: 1; harness/internal error: 1; unscored: 1" in report
    assert "Behavioral pass fraction: 1/2." in report
    assert "Required evidence complete: 3/3." in report


@pytest.mark.parametrize("field", ["schema_version", "repetition", "issue_indices"])
def test_reader_does_not_default_missing_required_evidence_fields(package_directory, field):
    def change(data):
        target = data
        if field == "repetition":
            target = data["slots"][0]
        elif field == "issue_indices":
            target = data["slots"][0]["canonical"]["tool_executions"][0]
        del target[field]
    rewrite_artifact(package_directory, "results.json", change)
    with pytest.raises(EvidencePackageError, match="invalid_schema|missing_execution_evidence"):
        read_evidence_package(package_directory)
