"""Version-bound local evidence for trusted, non-sensitive scripted Eval cases.

This is an export/validation boundary, not replay or authentication. Only fresh
cases are accepted; private execution state never becomes a serialized artifact.
"""

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from .eval import (
    AgentEvaluationCase, AgentEvaluationCaseResult, EvaluationExpectationName,
    EvaluationHarnessStage, _expectation_result, _run_evaluation_case_with_state,
    build_deterministic_evaluation_cases,
)
from .models import AgentInvestigationState, NextActionType, ToolExecutionRecord
from .report import (
    QAReportCheckerName, QAReportFinding, QAReportIssueType, QAReportStatus,
    build_qa_report,
)
from .tools import build_default_tool_registry
from .trace import InvestigationFinalStatus


_ARTIFACTS = ("plan.json", "results.json", "report.md")
_SOURCE_ROOT = Path(__file__).resolve().parents[1]
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
CaseID = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")]
PositiveInt = Annotated[int, Field(ge=1)]
Count = Annotated[int, Field(ge=0)]
VersionOne = Annotated[int, Field(ge=1, le=1)]


class EvidencePackageError(ValueError):
    """Fixed local codes only. Failure is never publication confirmation.

    Files may be visible after a failed write/replace/readback. This exception
    does not claim rollback or directory durability.
    """

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _EvidenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class RunIdentity(_EvidenceModel):
    run_id: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
    mode: Literal["offline"]
    source_commit: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
    source_dirty: bool
    runtime_versions: dict[
        Literal["python", "pydantic", "networkx", "openai"],
        Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9.+_-]{1,64}$")],
    ]

    @field_validator("runtime_versions")
    @classmethod
    def required_versions(cls, value: dict) -> dict:
        if set(value) != {"python", "pydantic", "networkx", "openai"}:
            raise ValueError("Required runtime versions are missing.")
        return value


class ScriptedActionConfig(_EvidenceModel):
    action_type: NextActionType
    tool_name: QAReportCheckerName | Literal["unrecognized"] | None
    has_tool_args: bool
    known_expand_task_ids: tuple[str, ...]
    unknown_expand_task_count: Count


class OfflineOracle(_EvidenceModel):
    final_status: InvestigationFinalStatus
    issue_types: tuple[QAReportIssueType, ...]
    trusted_scope_task_ids: tuple[str, ...]
    decision_error_count: Count
    trace_step_count: Count | None
    trace_final_status: InvestigationFinalStatus | None


class PlannedSlot(_EvidenceModel):
    case_id: CaseID
    repetition: VersionOne
    case_fingerprint: Digest
    fixture_fingerprint: Digest
    oracle_id: Digest
    oracle: OfflineOracle
    known_task_ids: tuple[str, ...]
    initial_scope_task_ids: tuple[str, ...]
    initial_expandable_task_ids: tuple[str, ...]
    max_steps: PositiveInt
    trace_enabled: bool
    scripted_actions: tuple[ScriptedActionConfig, ...]


class EvidencePlan(_EvidenceModel):
    schema_version: VersionOne
    run: RunIdentity
    provider: Literal["scripted"]
    active_tools: tuple[QAReportCheckerName, ...]
    slots: tuple[PlannedSlot, ...]


class CanonicalFinding(_EvidenceModel):
    issue_index: Count
    # None means an unrecognized issue/checker pair, never an absent finding.
    detail: QAReportFinding | None


class CanonicalEvidence(_EvidenceModel):
    investigation_status: QAReportStatus
    scope_task_ids: tuple[str, ...]
    scope_version: PositiveInt
    decision_error_count: Count
    tool_executions: tuple[ToolExecutionRecord, ...]
    findings: tuple[CanonicalFinding, ...]


class ExpectationCheck(_EvidenceModel):
    expectation: EvaluationExpectationName
    passed: bool


class EvidenceSlotResult(_EvidenceModel):
    case_id: CaseID
    repetition: VersionOne
    case_fingerprint: Digest
    execution: Literal["completed", "harness_error"]
    evaluation: Literal["passed", "failed_behavior", "unscored"]
    evidence: Literal["complete", "incomplete", "invalid"]
    canonical: CanonicalEvidence | None
    checks: tuple[ExpectationCheck, ...] = ()
    harness_stage: EvaluationHarnessStage | None = None
    # Optional diagnostics, never a substitute for canonical execution evidence.
    trace_step_count: Count | None = None
    trace_final_status: InvestigationFinalStatus | None = None


class EvidenceResults(_EvidenceModel):
    schema_version: VersionOne
    run: RunIdentity
    planned_case_count: Count
    slots: tuple[EvidenceSlotResult, ...]


class ArtifactDigest(_EvidenceModel):
    size: Count
    sha256: Digest


class EvidenceManifest(_EvidenceModel):
    schema_version: VersionOne
    run_id: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
    completion: Literal["complete"]
    durability: Literal["files_fsynced_directory_not_confirmed"]
    artifacts: dict[str, ArtifactDigest]


@dataclass(frozen=True)
class EvidencePackage:
    plan: EvidencePlan
    results: EvidenceResults
    manifest: EvidenceManifest


def canonical_json(value: object) -> bytes:
    """UTF-8, sorted string keys, compact JSON, LF; no coercion or non-finites."""
    def check(item: object) -> None:
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is list:
            for child in item:
                check(child)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for child in item.values():
                check(child)
            return
        raise EvidencePackageError("unsupported_canonical_value")

    check(value)
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _digest(value: object) -> str:
    return sha256(canonical_json(value)).hexdigest()


def _run_identity() -> RunIdentity:
    # No shell, environment-file loading, or Git content/patch collection.
    commit = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"], cwd=_SOURCE_ROOT,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=_SOURCE_ROOT, check=True, capture_output=True,
    ).stdout)
    return RunIdentity(
        run_id=uuid4().hex, mode="offline", source_commit=commit, source_dirty=dirty,
        runtime_versions={"python": platform.python_version(), **{
            name: version(name) for name in ("pydantic", "networkx", "openai")
        }},
    )


def _fixture_definition(case: AgentEvaluationCase) -> dict:
    """Only trusted non-sensitive program fixtures, never observations/Tool output.

    This hash is version identity, not redaction. Callers must not supply secrets
    as fixture identifiers/values. Private state and arbitrary metadata are not
    part of this explicit definition.
    """
    return {
        "tasks": [{
            "id": task.id, "dependencies": list(task.dependencies),
            "exclusive_group": task.exclusive_group,
            "npc_requirements": [{
                "npc_id": req.npc_id, "location": req.location, "state": req.state,
            } for req in task.npc_requirements],
        } for task in case.tasks],
        "runtime": {
            "task_statuses": dict(case.runtime_state.task_statuses),
            "npc_states": {npc_id: {
                "location": npc.location, "state": npc.state,
                "occupied_by_task_id": npc.occupied_by_task_id,
            } for npc_id, npc in case.runtime_state.npc_states.items()},
        },
    }


def _case_fingerprint(definition: dict, active_tools: tuple[str, ...]) -> str:
    return _digest({
        "schema_version": 1, "provider": "scripted",
        "active_tools": list(active_tools),
        "case": {key: value for key, value in definition.items()
                 if key not in ("case_fingerprint", "repetition")},
    })


def _plan_slot(case: AgentEvaluationCase, active_tools: tuple[str, ...]) -> PlannedSlot:
    state = case.initial_state
    if (state.investigation_status != "start" or state.scope_version != 1
            or state.issues or state.tool_executions or state.called_tool_names
            or state.called_tool_scope_versions or state.decision_errors
            or state.last_decision_rejection is not None):
        raise EvidencePackageError("fresh_case_required")
    known = {task.id for task in case.tasks}
    if (len(known) != len(case.tasks)
            or not set(state.scope_task_ids).issubset(known)
            or not set(state.expandable_task_ids).issubset(known)):
        raise EvidencePackageError("invalid_fixture_scope")
    oracle = OfflineOracle(
        final_status=case.expected_final_status,
        issue_types=tuple(sorted(case.expected_issue_types)),
        trusted_scope_task_ids=tuple(sorted(case.expected_trusted_scope_task_ids)),
        decision_error_count=case.expected_decision_error_count,
        trace_step_count=case.expected_trace_step_count,
        trace_final_status=case.expected_trace_final_status,
    )
    actions = tuple(ScriptedActionConfig(
        action_type=action.action_type,
        tool_name=(action.tool_name if action.tool_name in active_tools
                   else "unrecognized" if action.tool_name is not None else None),
        has_tool_args=bool(action.tool_args),
        known_expand_task_ids=tuple(sorted(set(action.expand_task_ids) & known)),
        unknown_expand_task_count=len(set(action.expand_task_ids) - known),
    ) for action in case.scripted_actions)
    definition = {
        "case_id": case.case_id, "fixture_fingerprint": _digest(_fixture_definition(case)),
        "oracle_id": _digest(oracle.model_dump(mode="json")), "oracle": oracle,
        "known_task_ids": tuple(sorted(known)),
        "initial_scope_task_ids": tuple(state.scope_task_ids),
        "initial_expandable_task_ids": tuple(state.expandable_task_ids),
        "max_steps": case.max_steps, "trace_enabled": case.trace_enabled,
        "scripted_actions": actions,
    }
    slot = PlannedSlot(repetition=1, case_fingerprint="0" * 64, **definition)
    return slot.model_copy(update={
        "case_fingerprint": _case_fingerprint(slot.model_dump(mode="json"), active_tools),
    })


def _project_canonical(state: AgentInvestigationState) -> CanonicalEvidence:
    report = build_qa_report(state)
    # Check history too: losing a zero-finding execution cannot look like not-run.
    history: dict[str, list[int]] = {}
    records = []
    for record in state.tool_executions:
        history.setdefault(record.tool_name, []).append(record.scope_version)
        records.append(ToolExecutionRecord(
            execution_number=record.execution_number, tool_name=record.tool_name,
            scope_version=record.scope_version, status=record.status,
            issue_indices=tuple(record.issue_indices),
        ))
    if history != state.called_tool_scope_versions or set(history) != set(state.called_tool_names):
        raise EvidencePackageError("missing_execution_evidence")
    findings = []
    for index, issue in enumerate(state.issues):
        # Reuse the existing Report allowlist, but retain canonical issue order.
        # Equal safe projections remain distinct; unknown pairs keep their index.
        one = build_qa_report(state.model_copy(update={"issues": [issue]}))
        detail = one.findings[0] if one.findings else None
        findings.append(CanonicalFinding(issue_index=index, detail=detail))
    return CanonicalEvidence(
        investigation_status=report.status, scope_task_ids=report.scope_task_ids,
        scope_version=report.scope_version, decision_error_count=report.decision_error_count,
        tool_executions=tuple(records), findings=tuple(findings),
    )


def _result_slot(
    planned: PlannedSlot, result: AgentEvaluationCaseResult,
    state: AgentInvestigationState | None,
) -> EvidenceSlotResult:
    canonical = None
    evidence_status = "incomplete"
    if state is not None:
        try:
            canonical = _project_canonical(state)
            evidence_status = "complete"
        except Exception:
            # An exporter failure is required-evidence failure, not behavioral failure.
            evidence_status = "invalid"
    evaluated = result.outcome != "harness_error" and evidence_status == "complete"
    observation = result.evidence
    return EvidenceSlotResult(
        case_id=planned.case_id, repetition=planned.repetition,
        case_fingerprint=planned.case_fingerprint,
        execution="harness_error" if result.outcome == "harness_error" else "completed",
        evaluation=result.outcome if evaluated else "unscored",
        evidence=evidence_status, canonical=canonical,
        checks=tuple(ExpectationCheck(expectation=check.expectation, passed=check.passed)
                     for check in result.expectation_results) if evaluated else (),
        harness_stage=result.harness_error.stage if result.harness_error is not None else None,
        trace_step_count=observation.trace_step_count if observation is not None else None,
        trace_final_status=observation.trace_final_status if observation is not None else None,
    )


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise EvidencePackageError(code)


def _slot_key(slot: PlannedSlot | EvidenceSlotResult) -> tuple[str, int]:
    return slot.case_id, slot.repetition


def _validate_canonical(evidence: CanonicalEvidence, slot: PlannedSlot, tools: tuple) -> None:
    _require(set(evidence.scope_task_ids).issubset(slot.known_task_ids)
             and set(slot.initial_scope_task_ids).issubset(evidence.scope_task_ids),
             "invalid_scope_evidence")
    _require(evidence.scope_task_ids == tuple(sorted(set(evidence.scope_task_ids))),
             "invalid_scope_evidence")
    _require([finding.issue_index for finding in evidence.findings]
             == list(range(len(evidence.findings))), "invalid_finding_indices")
    linked, calls = set(), set()
    last_version = 1
    for number, record in enumerate(evidence.tool_executions, 1):
        _require("issue_indices" in record.model_fields_set, "missing_execution_evidence")
        _require(record.execution_number == number and record.tool_name in tools
                 and last_version <= record.scope_version <= evidence.scope_version,
                 "invalid_execution_reference")
        call = record.tool_name, record.scope_version
        _require(call not in calls, "duplicate_execution")
        calls.add(call)
        last_version = record.scope_version
        _require(len(record.issue_indices) == len(set(record.issue_indices))
                 and all(0 <= index < len(evidence.findings) for index in record.issue_indices),
                 "dangling_finding_reference")
        linked.update(record.issue_indices)
    _require(linked == set(range(len(evidence.findings))), "unlinked_finding")
    for finding in evidence.findings:
        if finding.detail is not None:
            _require(set(finding.detail.scoped_task_ids).issubset(evidence.scope_task_ids),
                     "invalid_finding_scope")


def _validate_results(plan: EvidencePlan, results: EvidenceResults) -> None:
    _require(plan.run == results.run, "run_mismatch")
    keys = [_slot_key(slot) for slot in plan.slots]
    actual = [_slot_key(slot) for slot in results.slots]
    _require(len(keys) == len(set(keys)) and len(actual) == len(set(actual)), "duplicate_slot")
    _require(set(keys) == set(actual), "slot_set_mismatch")
    _require(results.planned_case_count == len(keys), "planned_count_mismatch")
    planned = {_slot_key(slot): slot for slot in plan.slots}
    for result in results.slots:
        slot = planned[_slot_key(result)]
        _require(slot.case_fingerprint == result.case_fingerprint, "case_identity_mismatch")
        _require(slot.oracle_id == _digest(slot.oracle.model_dump(mode="json"))
                 and slot.case_fingerprint == _case_fingerprint(
                     slot.model_dump(mode="json"), plan.active_tools), "case_identity_mismatch")
        _require(result.evidence == "complete" and result.canonical is not None,
                 "required_evidence_incomplete")
        _validate_canonical(result.canonical, slot, plan.active_tools)
        if result.execution == "harness_error":
            _require(result.evaluation == "unscored" and not result.checks
                     and result.harness_stage is not None, "invalid_outcome")
        else:
            names = [check.expectation for check in result.checks]
            required = {"final_status", "issue_types", "trusted_scope_task_ids", "decision_error_count"}
            if slot.oracle.trace_step_count is not None:
                required.add("trace_step_count")
            if slot.oracle.trace_final_status is not None:
                required.add("trace_final_status")
            _require(set(names) == required and len(names) == len(required), "missing_expectation")
            _require(result.harness_stage is None and result.evaluation == (
                "passed" if all(check.passed for check in result.checks) else "failed_behavior"
            ), "invalid_outcome")
            _require(all(record.status == "succeeded" for record in result.canonical.tool_executions),
                     "invalid_outcome")
            # Validate stored checks against the same safe observation, using
            # Eval's existing comparison primitive; never execute or re-score a run.
            actuals = {
                "final_status": result.canonical.investigation_status,
                "issue_types": sorted(finding.detail.issue_type if finding.detail is not None
                                      else "unrecognized" for finding in result.canonical.findings),
                "trusted_scope_task_ids": list(result.canonical.scope_task_ids),
                "decision_error_count": result.canonical.decision_error_count,
                "trace_step_count": result.trace_step_count,
                "trace_final_status": result.trace_final_status,
            }
            expected = slot.oracle.model_dump(mode="json")
            for check in result.checks:
                consistent = _expectation_result(
                    check.expectation, expected[check.expectation], actuals[check.expectation],
                )
                _require(check.passed == consistent.passed, "inconsistent_expectation")


def render_evidence_report(results: EvidenceResults) -> str:
    """Only these structured results determine all counts and presentation."""
    counts = Counter(slot.evaluation for slot in results.slots)
    harness = sum(slot.execution == "harness_error" for slot in results.slots)
    complete = sum(slot.evidence == "complete" for slot in results.slots)
    evaluated = counts["passed"] + counts["failed_behavior"]
    rate = f"{counts['passed']}/{evaluated}" if evaluated else "N/A (no evaluated cases)"
    lines = [
        "# Offline QA evidence package", "",
        f"Run: `{results.run.run_id}`; mode: `offline`; schema: {results.schema_version}.",
        f"Source: `{results.run.source_commit}`; dirty: `{str(results.run.source_dirty).lower()}`.", "",
        f"Planned: {results.planned_case_count}; evaluated: {evaluated}; behavioral pass: {counts['passed']};",
        f"behavioral fail: {counts['failed_behavior']}; harness/internal error: {harness}; unscored: {counts['unscored']}.",
        f"Behavioral pass fraction: {rate}.",
        f"Required evidence complete: {complete}/{results.planned_case_count}.", "",
        "Case | Repetition | Execution | Evaluation | Evidence | Investigation | Findings",
        "--- | --- | --- | --- | --- | --- | ---",
    ]
    for slot in results.slots:
        canonical = slot.canonical
        status = canonical.investigation_status if canonical is not None else "unavailable"
        findings = str(len(canonical.findings)) if canonical is not None else "unavailable"
        lines.append(f"{slot.case_id} | {slot.repetition} | {slot.execution} | {slot.evaluation}"
                     f" | {slot.evidence} | {status} | {findings}")
    lines.extend([
        "", "## Limitations", "",
        "- Scripted offline outcomes do not measure a real model's success rate.",
        "- Investigation completion is not QA pass or proof that every Tool ran.",
        "- Finding counts alone do not establish checker success or coverage.",
        "- Canonical indices are local to this run and case; equal safe findings are not merged.",
        "- Trace summaries are optional diagnostics, never replacement execution evidence.",
        "- Raw conversations, Tool payloads, exceptions and private state are not archived.",
        "- Hashes detect modification against this manifest; they are not signatures or authentication.",
        "- Dirty source records no patch; file fsync does not confirm directory durability.",
    ])
    return "\n".join(lines) + "\n"


def _write_artifact(directory: Path, name: str, data: bytes) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=directory, prefix=f".{name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / name)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _reject_duplicate_keys(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate_json_key")
        result[key] = value
    return result


def _decode(data: bytes, model: type[_EvidenceModel]) -> _EvidenceModel:
    code = "invalid_schema"
    try:
        document = json.loads(data.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
        if (type(document) is dict and type(document.get("schema_version")) is int
                and document["schema_version"] != 1):
            code = "unsupported_schema"
        else:
            return model.model_validate_json(canonical_json(document), strict=True)
    except Exception:
        pass
    # Pydantic/JSON exceptions may include attacker-controlled file contents.
    raise EvidencePackageError(code)


def _read_file(directory: Path, name: str) -> bytes:
    code = "incomplete" if name == "manifest.json" else "missing_artifact"
    try:
        path = directory / name
        if not path.is_symlink() and path.is_file():
            return path.read_bytes()
    except OSError:
        pass
    raise EvidencePackageError(code)


def read_evidence_package(directory: str | Path) -> EvidencePackage:
    directory = Path(directory)
    manifest = _decode(_read_file(directory, "manifest.json"), EvidenceManifest)
    _require(set(manifest.artifacts) == set(_ARTIFACTS), "artifact_set_mismatch")
    artifacts = {name: _read_file(directory, name) for name in _ARTIFACTS}
    for name, data in artifacts.items():
        expected = manifest.artifacts[name]
        _require(len(data) == expected.size and sha256(data).hexdigest() == expected.sha256,
                 "artifact_mismatch")
    plan = _decode(artifacts["plan.json"], EvidencePlan)
    results = _decode(artifacts["results.json"], EvidenceResults)
    _require(plan.run.run_id == results.run.run_id == manifest.run_id, "run_mismatch")
    _validate_results(plan, results)
    _require(artifacts["report.md"] == render_evidence_report(results).encode("utf-8"),
             "report_mismatch")
    return EvidencePackage(plan=plan, results=results, manifest=manifest)


def run_evidence_package(
    directory: str | Path, cases: list[AgentEvaluationCase] | None = None,
) -> EvidencePackage:
    """Run once into a NEW directory; success means readback validation succeeded.

    Cases must be trusted, non-sensitive local definitions. Freeze deep copies
    and write the plan before execution. No raw state or Trace crosses this API.
    On any required write/projection/validation failure, raise and leave an
    incomplete directory. Existing directories are never reused or overwritten.
    """
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError("Evidence directory must be new.")
    cases = build_deterministic_evaluation_cases() if cases is None else cases
    frozen_cases = [case.model_copy(deep=True) for case in cases]
    active_tools = build_default_tool_registry().active_tool_names()
    plan = EvidencePlan(
        schema_version=1, run=_run_identity(), provider="scripted", active_tools=active_tools,
        slots=tuple(_plan_slot(case, active_tools) for case in frozen_cases),
    )
    keys = [_slot_key(slot) for slot in plan.slots]
    _require(len(keys) == len(set(keys)), "duplicate_slot")
    directory.mkdir()
    code = "publication_failed"
    try:
        plan_bytes = canonical_json(plan.model_dump(mode="json"))
        _write_artifact(directory, "plan.json", plan_bytes)
        slots = []
        for case, planned in zip(frozen_cases, plan.slots, strict=True):
            result, state = _run_evaluation_case_with_state(case)
            slots.append(_result_slot(planned, result, state))
        results = EvidenceResults(
            schema_version=1, run=plan.run, planned_case_count=len(plan.slots), slots=tuple(slots),
        )
        artifacts = {
            "plan.json": plan_bytes,
            "results.json": canonical_json(results.model_dump(mode="json")),
            "report.md": render_evidence_report(results).encode("utf-8"),
        }
        for name in ("results.json", "report.md"):
            _write_artifact(directory, name, artifacts[name])
        _validate_results(plan, results)
        # Validate actual disk bytes before publishing the only completion marker.
        for name, data in artifacts.items():
            _require(_read_file(directory, name) == data, "artifact_mismatch")
        manifest = EvidenceManifest(
            schema_version=1, run_id=plan.run.run_id, completion="complete",
            durability="files_fsynced_directory_not_confirmed", artifacts={
                name: ArtifactDigest(size=len(data), sha256=sha256(data).hexdigest())
                for name, data in artifacts.items()
            },
        )
        _write_artifact(directory, "manifest.json", canonical_json(manifest.model_dump(mode="json")))
        return read_evidence_package(directory)
    except EvidencePackageError as error:
        code = error.code
    except Exception:
        pass
    # Even a replace that made the manifest visible and then raised is failure.
    try:
        (directory / "manifest.json").unlink(missing_ok=True)
    except OSError:
        pass
    raise EvidencePackageError(code)
