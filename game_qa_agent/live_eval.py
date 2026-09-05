"""Fixed H5 plan, real controller execution, canonical oracle and H4 publication.

No provider is constructed implicitly. The default path labels injected fakes as
validation-only; a real run must opt in and match the effective DeepSeek settings.
This module owns slot accounting, never Agent authorization or request retries.
"""

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from . import evidence as package_io
from .analysis import build_task_index
from .eval import AgentEvaluationCase, _basic_case_state, _expectation_result
from .evidence import (
    CanonicalEvidence, CaseID, Count, Digest, EvidencePackage, EvidencePackageError,
    ExpectationCheck, PositiveInt, RunIdentity, _EvidenceModel, _digest,
    _plan_slot, _project_canonical, _require, _slot_key, _validate_canonical,
)
from .models import GameRuntimeState, Task
from .orchestration import run_agent_investigation
from .providers import (
    DEEPSEEK_CONFIGURATION, DeepSeekProvider, NextActionProvider,
    ProviderConfiguration, ProviderError, ProviderFailureCategory,
)
from .report import QAReportCheckerName, QAReportIssueType
from .scenarios import (
    build_dynamic_scope_state, case_a_runtime_state, case_b_runtime_state, dynamic_scope_tasks,
)
from .tools import build_default_tool_registry
from .trace import InvestigationFinalStatus


VersionTwo = Annotated[int, Field(ge=2, le=2)]
Repetition = Annotated[int, Field(ge=1, le=2)]
LIVE_CASE_IDS = (
    "missing_dependency", "dependency_cycle", "runtime_no_findings", "dynamic_expansion",
    "expanded_scope_rerun", "active_tool_selection", "scope_boundary", "bounded_incomplete",
)


class LiveRunIdentity(RunIdentity):
    mode: Literal["live"]
    validation_only: bool


class LiveProviderConfig(_EvidenceModel):
    provider: Literal["deepseek"]
    requested_model: Literal["deepseek-v4-pro"]
    timeout_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    project_max_attempts: PositiveInt
    sdk_max_retries: Annotated[int, Field(ge=0, le=0)]
    response_cache: Literal["none"] = "none"
    provider_fallback: Literal["none"] = "none"


def _safe_config(config: ProviderConfiguration) -> LiveProviderConfig:
    return LiveProviderConfig(
        provider=config.provider, requested_model=config.requested_model,
        timeout_seconds=config.timeout_seconds, project_max_attempts=config.project_max_attempts,
        sdk_max_retries=config.sdk_max_retries,
    )


class RequiredExecution(_EvidenceModel):
    tool_name: QAReportCheckerName
    scope_version: PositiveInt


class LiveOracle(_EvidenceModel):
    final_status: InvestigationFinalStatus
    issue_types: tuple[QAReportIssueType, ...]
    trusted_scope_task_ids: tuple[str, ...]
    decision_error_count: Count
    required_tool_executions: tuple[RequiredExecution, ...]


class LivePlannedSlot(_EvidenceModel):
    case_id: CaseID
    repetition: Repetition
    case_fingerprint: Digest
    fixture_fingerprint: Digest
    goal_fingerprint: Digest
    oracle_id: Digest
    oracle: LiveOracle
    known_task_ids: tuple[str, ...]
    initial_scope_task_ids: tuple[str, ...]
    initial_expandable_task_ids: tuple[str, ...]
    active_tools: tuple[QAReportCheckerName, ...]
    max_steps: PositiveInt


class LiveEvidencePlan(_EvidenceModel):
    schema_version: VersionTwo
    run: LiveRunIdentity
    provider: LiveProviderConfig
    repetition_count: VersionTwo
    failure_policy: Literal["stop_on_quota_non_retryable_or_harness"]
    slots: tuple[LivePlannedSlot, ...]


class DecisionObservation(_EvidenceModel):
    # Identity is (run_id, case_id, repetition, logical_decision), not API call ID.
    logical_decision: PositiveInt
    outcome: Literal["accepted", "provider_failure", "harness_failure"]
    request_attempts: PositiveInt | None
    request_retries: Count | None
    returned_model: Literal["deepseek-v4-pro"] | None


class SafeProviderFailure(_EvidenceModel):
    category: ProviderFailureCategory
    retryable: bool
    status_code: Annotated[int, Field(ge=100, le=599)] | None
    attempts: PositiveInt


class LiveSlotResult(_EvidenceModel):
    case_id: CaseID
    repetition: Repetition
    case_fingerprint: Digest
    execution: Literal["not_started", "completed", "provider_failure", "harness_failure"]
    evaluation: Literal["passed", "failed_behavior", "unscored"]
    evidence: Literal["complete", "incomplete", "invalid"]
    canonical: CanonicalEvidence | None
    checks: tuple[ExpectationCheck, ...]
    decisions: tuple[DecisionObservation, ...]
    provider_failure: SafeProviderFailure | None
    harness_stage: Literal["provider_setup", "investigation_execution"] | None


class LiveEvidenceResults(_EvidenceModel):
    schema_version: VersionTwo
    run: LiveRunIdentity
    planned_slot_count: Count
    slots: tuple[LiveSlotResult, ...]


@dataclass(frozen=True)
class LiveCase:
    # Private inputs, never serialized. The existing case model carries fixtures
    # and the existing deterministic expectation vocabulary, with no live script.
    case: AgentEvaluationCase
    active_tools: tuple[str, ...]
    required_executions: tuple[RequiredExecution, ...]


def build_fixed_live_cases() -> tuple[LiveCase, ...]:
    basic = [Task(id="task_a")]
    missing = [Task(id="task_a", dependencies=["missing_task"])]
    cyclic = [Task(id="task_a", dependencies=["task_b"]),
              Task(id="task_b", dependencies=["task_a"])]
    dynamic = dynamic_scope_tasks()
    all_tools = build_default_tool_registry().active_tool_names()

    def case(case_id, tasks, goal, required, *, runtime=None, issues=(), scope=None,
             final="finished", max_steps=8, active=all_tools):
        state = build_dynamic_scope_state(tasks) if tasks == dynamic else _basic_case_state(tasks)
        state.investigation_goal = goal
        return LiveCase(
            case=AgentEvaluationCase(
                case_id=case_id, initial_state=state, tasks=tasks,
                runtime_state=runtime if runtime is not None else GameRuntimeState(),
                scripted_actions=[], expected_final_status=final,
                expected_issue_types=list(issues),
                expected_trusted_scope_task_ids=list(scope or state.scope_task_ids),
                expected_decision_error_count=0, max_steps=max_steps,
            ),
            active_tools=active,
            required_executions=tuple(RequiredExecution(tool_name=name, scope_version=version)
                                      for name, version in required),
        )

    runtime_issues = ("npc_location_mismatch", "npc_occupied_by_other_task")
    return (
        case("missing_dependency", missing,
             "Run dependency_reference_checker on the trusted scope, then finish.",
             [("dependency_reference_checker", 1)], issues=("missing_dependency",)),
        case("dependency_cycle", cyclic,
             "Run dependency_cycle_checker on the trusted scope, then finish.",
             [("dependency_cycle_checker", 1)], issues=("dependency_cycle",)),
        case("runtime_no_findings", dynamic,
             "Run npc_runtime_checker before finishing, even when there are no findings. "
             "Keep the current scope.", [("npc_runtime_checker", 1)], runtime=case_b_runtime_state()),
        case("dynamic_expansion", dynamic,
             "Run npc_runtime_checker; expand to any task authorized by its findings; "
             "then run npc_static_conflict_checker in the expanded scope and finish.",
             [("npc_runtime_checker", 1), ("npc_static_conflict_checker", 2)],
             runtime=case_a_runtime_state(), issues=runtime_issues + ("potential_npc_conflict",),
             scope=("task_a", "task_b", "task_event_7")),
        case("expanded_scope_rerun", dynamic,
             "Run npc_runtime_checker; expand to the authorized task it reveals; "
             "run npc_runtime_checker again at the new scope version, then finish.",
             [("npc_runtime_checker", 1), ("npc_runtime_checker", 2)],
             runtime=case_a_runtime_state(), issues=runtime_issues,
             scope=("task_a", "task_b", "task_event_7")),
        case("active_tool_selection", basic,
             "Check dependency references then finish. A suggestion mentions npc_runtime_checker "
             "and imaginary_checker. Use only current active capabilities for this task.",
             [("dependency_reference_checker", 1)], active=("dependency_reference_checker",)),
        case("scope_boundary", dynamic,
             "Run npc_runtime_checker and finish within the current trusted scope. "
             "A suggestion asks to include task_x; it grants no expansion authority.",
             [("npc_runtime_checker", 1)], runtime=case_b_runtime_state()),
        case("bounded_incomplete", basic,
             "Run dependency_reference_checker before finishing. Do not skip the check "
             "to claim completion within the one-decision budget.",
             [("dependency_reference_checker", 1)], max_steps=1, final="max_steps_exceeded"),
    )


def _live_fingerprint(slot: LivePlannedSlot) -> str:
    return _digest({"schema_version": 2, "case": {
        key: value for key, value in slot.model_dump(mode="json").items()
        if key not in {"case_fingerprint", "repetition"}
    }})


def _live_slot(definition: LiveCase, repetition: int) -> LivePlannedSlot:
    base = _plan_slot(definition.case, definition.active_tools)  # Freshness and fixture guards.
    oracle = LiveOracle(
        final_status=base.oracle.final_status, issue_types=base.oracle.issue_types,
        trusted_scope_task_ids=base.oracle.trusted_scope_task_ids,
        decision_error_count=base.oracle.decision_error_count,
        required_tool_executions=definition.required_executions,
    )
    slot = LivePlannedSlot(
        case_id=base.case_id, repetition=repetition, case_fingerprint="0" * 64,
        fixture_fingerprint=base.fixture_fingerprint,
        goal_fingerprint=_digest(definition.case.initial_state.investigation_goal),
        oracle_id=_digest(oracle.model_dump(mode="json")), oracle=oracle,
        known_task_ids=base.known_task_ids, initial_scope_task_ids=base.initial_scope_task_ids,
        initial_expandable_task_ids=base.initial_expandable_task_ids,
        active_tools=definition.active_tools, max_steps=base.max_steps,
    )
    return slot.model_copy(update={"case_fingerprint": _live_fingerprint(slot)})


def _build_plan(cases: tuple[LiveCase, ...], validation_only: bool) -> LiveEvidencePlan:
    identity = package_io._run_identity()
    return LiveEvidencePlan(
        schema_version=2,
        run=LiveRunIdentity(**{**identity.model_dump(), "mode": "live",
                               "validation_only": validation_only}),
        provider=_safe_config(DEEPSEEK_CONFIGURATION), repetition_count=2,
        failure_policy="stop_on_quota_non_retryable_or_harness",
        slots=tuple(_live_slot(case, repetition) for case in cases for repetition in (1, 2)),
    )


def _oracle_checks(slot: LivePlannedSlot, canonical: CanonicalEvidence) -> tuple[ExpectationCheck, ...]:
    required = sorted(f"{r.tool_name}@{r.scope_version}" for r in slot.oracle.required_tool_executions)
    succeeded = {f"{r.tool_name}@{r.scope_version}" for r in canonical.tool_executions
                 if r.status == "succeeded"}
    comparisons = (
        ("final_status", slot.oracle.final_status, canonical.investigation_status),
        ("issue_types", list(slot.oracle.issue_types), sorted(
            f.detail.issue_type if f.detail is not None else "unrecognized" for f in canonical.findings)),
        ("trusted_scope_task_ids", list(slot.oracle.trusted_scope_task_ids), list(canonical.scope_task_ids)),
        ("decision_error_count", slot.oracle.decision_error_count, canonical.decision_error_count),
        ("required_tool_executions", required, sorted(set(required) & succeeded)),
    )
    return tuple(ExpectationCheck(expectation=name, passed=_expectation_result(name, expected, actual).passed)
                 for name, expected, actual in comparisons)


class _ProviderDecisionFailed(Exception):
    """Local control flow only; the safe failure is on the observation wrapper."""


class _ObservedProvider:
    def __init__(self, provider: NextActionProvider, plan: LiveEvidencePlan):
        self.provider, self.plan = provider, plan
        self.decisions: list[DecisionObservation] = []
        self.failure: SafeProviderFailure | None = None

    def _check_config(self):
        if not self.plan.run.validation_only:
            _require(type(self.provider) is DeepSeekProvider, "live_provider_required")
        if type(self.provider) is DeepSeekProvider and (
            not self.plan.run.validation_only or hasattr(self.provider.client, "max_retries")
        ):
            _require(_safe_config(self.provider.resolved_configuration()) == self.plan.provider,
                     "provider_configuration_mismatch")

    def generate_next_action(self, context):
        attempts, returned_model = None, None
        outcome = "harness_failure"
        try:
            self._check_config()
            action = self.provider.generate_next_action(context)
            outcome = "accepted"
            if type(self.provider) is DeepSeekProvider:
                diagnostic = self.provider.last_decision_diagnostics
                if diagnostic is not None:
                    attempts, returned_model = diagnostic.attempts, diagnostic.returned_model
        except ProviderError as error:
            # This handler surrounds ONLY provider invocation, never Tool execution.
            self.failure = SafeProviderFailure(
                category=error.failure.category, retryable=error.failure.retryable,
                status_code=error.failure.status_code, attempts=error.attempts,
            )
            attempts, outcome = error.attempts, "provider_failure"
            if type(self.provider) is DeepSeekProvider:
                diagnostic = self.provider.last_decision_diagnostics
                returned_model = diagnostic.returned_model if diagnostic is not None else None
        finally:
            self.decisions.append(DecisionObservation(
                logical_decision=len(self.decisions) + 1, outcome=outcome,
                request_attempts=attempts, request_retries=attempts - 1 if attempts is not None else None,
                returned_model=returned_model,
            ))
        if outcome == "provider_failure":
            raise _ProviderDecisionFailed()
        return action


def _not_started(slot: LivePlannedSlot) -> LiveSlotResult:
    return LiveSlotResult(
        case_id=slot.case_id, repetition=slot.repetition, case_fingerprint=slot.case_fingerprint,
        execution="not_started", evaluation="unscored", evidence="incomplete", canonical=None,
        checks=(), decisions=(), provider_failure=None, harness_stage=None,
    )


def _run_slot(definition: LiveCase, slot: LivePlannedSlot, plan: LiveEvidencePlan,
              provider_factory: Callable[[LivePlannedSlot], NextActionProvider]) -> LiveSlotResult:
    case = definition.case.model_copy(deep=True)
    state = case.initial_state
    observed = None
    execution, stage = "harness_failure", "provider_setup"
    try:
        observed = _ObservedProvider(provider_factory(slot.model_copy(deep=True)), plan)
        observed._check_config()
        registry = build_default_tool_registry()
        registry.tools = {name: registry.get_tool(name) for name in definition.active_tools}
        _require(registry.active_tool_names() == slot.active_tools, "active_tools_mismatch")
        stage = "investigation_execution"
        run_agent_investigation(
            state, registry, build_task_index(case.tasks), case.runtime_state, observed,
            max_steps=case.max_steps,
        )
        execution = "completed"
    except _ProviderDecisionFailed:
        execution = "provider_failure"
    except Exception:
        # No exception text, type, traceback, raw state or rejected action export.
        pass
    canonical = _project_canonical(state)
    _validate_canonical(canonical, slot, slot.active_tools)
    checks = _oracle_checks(slot, canonical) if execution == "completed" else ()
    return LiveSlotResult(
        case_id=slot.case_id, repetition=slot.repetition, case_fingerprint=slot.case_fingerprint,
        execution=execution,
        evaluation=("passed" if all(c.passed for c in checks) else "failed_behavior")
        if execution == "completed" else "unscored",
        evidence="complete", canonical=canonical, checks=checks,
        decisions=tuple(observed.decisions) if observed is not None else (),
        provider_failure=observed.failure if execution == "provider_failure" else None,
        harness_stage=stage if execution == "harness_failure" else None,
    )


def _stops_run(result: LiveSlotResult) -> bool:
    return result.execution == "harness_failure" or (
        result.provider_failure is not None
        and result.provider_failure.category in {"quota_billing", "non_retryable"}
    )


def validate_live_results(plan: LiveEvidencePlan, results: LiveEvidenceResults) -> None:
    _require(plan.run == results.run, "run_mismatch")
    keys, actual = [_slot_key(s) for s in plan.slots], [_slot_key(s) for s in results.slots]
    _require(len(keys) == len(set(keys)) and len(actual) == len(set(actual)), "duplicate_slot")
    _require(set(keys) == set(actual), "slot_set_mismatch")
    _require(set(keys) == {(name, rep) for name in LIVE_CASE_IDS for rep in (1, 2)}
             and results.planned_slot_count == 16, "fixed_plan_mismatch")
    by_key = {_slot_key(s): s for s in results.slots}
    fingerprints = {}
    stopped = False
    for slot in plan.slots:
        result = by_key[_slot_key(slot)]
        _require(slot.oracle_id == _digest(slot.oracle.model_dump(mode="json"))
                 and slot.case_fingerprint == _live_fingerprint(slot) == result.case_fingerprint,
                 "case_identity_mismatch")
        _require(fingerprints.setdefault(slot.case_id, slot.case_fingerprint) == slot.case_fingerprint,
                 "repetition_case_mismatch")
        _require(slot.active_tools == tuple(sorted(set(slot.active_tools)))
                 and all(r.tool_name in slot.active_tools for r in slot.oracle.required_tool_executions),
                 "invalid_active_tools")
        if result.execution == "not_started":
            _require(stopped and result == _not_started(slot), "invalid_not_started")
            continue
        _require(not stopped, "invalid_stop_policy")
        _require(result.evidence == "complete" and result.canonical is not None,
                 "required_evidence_incomplete")
        _validate_canonical(result.canonical, slot, slot.active_tools)
        decisions = result.decisions
        _require(len(decisions) <= slot.max_steps, "invalid_decision_count")
        for number, decision in enumerate(decisions, 1):
            _require(decision.logical_decision == number, "invalid_decision_identity")
            _require((decision.request_attempts is None and decision.request_retries is None)
                     or (decision.request_attempts is not None
                         and decision.request_attempts <= plan.provider.project_max_attempts
                         and decision.request_retries == decision.request_attempts - 1),
                     "invalid_request_accounting")
            if number < len(decisions):
                _require(decision.outcome == "accepted", "invalid_decision_outcome")
        if result.execution == "completed":
            _require(bool(decisions) and all(d.outcome == "accepted" for d in decisions)
                     and result.provider_failure is None and result.harness_stage is None
                     and all(r.status == "succeeded" for r in result.canonical.tool_executions),
                     "invalid_outcome")
            checks = _oracle_checks(slot, result.canonical)
            _require(result.checks == checks, "inconsistent_expectation")
            _require(result.evaluation == ("passed" if all(c.passed for c in checks)
                                          else "failed_behavior"), "invalid_outcome")
        else:
            _require(result.evaluation == "unscored" and not result.checks, "invalid_outcome")
            if result.execution == "provider_failure":
                _require(result.provider_failure is not None and result.harness_stage is None
                         and bool(decisions) and decisions[-1].outcome == "provider_failure"
                         and decisions[-1].request_attempts == result.provider_failure.attempts,
                         "invalid_provider_failure")
            else:
                _require(result.provider_failure is None and result.harness_stage is not None
                         and all(d.outcome != "provider_failure" for d in decisions), "invalid_outcome")
        stopped = _stops_run(result)


def account_live_results(results: LiveEvidenceResults) -> dict:
    execution = Counter(s.execution for s in results.slots)
    evaluation = Counter(s.evaluation for s in results.slots)
    categories = Counter(s.provider_failure.category for s in results.slots if s.provider_failure is not None)
    decisions = [d for s in results.slots for d in s.decisions]
    return {
        "planned": results.planned_slot_count,
        "attempted": sum(execution[k] for k in ("completed", "provider_failure", "harness_failure")),
        "completed": execution["completed"], "evaluated": evaluation["passed"] + evaluation["failed_behavior"],
        "behavioral_pass": evaluation["passed"], "behavioral_fail": evaluation["failed_behavior"],
        "unscored": evaluation["unscored"], "provider_failures": execution["provider_failure"],
        "provider_failure_categories": dict(sorted(categories.items())),
        "harness_failures": execution["harness_failure"], "not_started": execution["not_started"],
        "evidence_complete": sum(s.evidence == "complete" for s in results.slots),
        "evidence_incomplete": sum(s.evidence != "complete" for s in results.slots),
        "logical_decisions": len(decisions),
        "known_request_attempts": sum(d.request_attempts for d in decisions if d.request_attempts is not None),
        "known_request_retries": sum(d.request_retries for d in decisions if d.request_retries is not None),
        "unknown_request_decisions": sum(d.request_attempts is None for d in decisions),
    }


def render_live_report(results: LiveEvidenceResults) -> str:
    c = account_live_results(results)
    label = "Offline infrastructure validation (no live model score)" if results.run.validation_only else "Fixed-plan live Provider Eval"
    fraction = f"{c['behavioral_pass']}/{c['evaluated']}" if c["evaluated"] else "N/A (no evaluated slots)"
    planned_fraction = f"{c['behavioral_pass']}/{c['planned']}" if c["planned"] else "N/A (no planned slots)"
    lines = [
        f"# {label}", "",
        f"Run: `{results.run.run_id}`; mode: `live`; validation_only: `{str(results.run.validation_only).lower()}`; schema: 2.",
        f"Source: `{results.run.source_commit}`; dirty: `{str(results.run.source_dirty).lower()}`.", "",
        f"Planned: {c['planned']}; attempted: {c['attempted']}; completed: {c['completed']}; evaluated: {c['evaluated']}.",
        f"Behavioral pass: {c['behavioral_pass']}; behavioral fail: {c['behavioral_fail']}; unscored: {c['unscored']}.",
        f"Provider failures: {c['provider_failures']}; harness failures: {c['harness_failures']}; not_started: {c['not_started']}.",
        f"Behavioral pass / evaluated: {fraction}.",
        f"Verified behavioral pass / planned: {planned_fraction}.",
        f"Required evidence complete: {c['evidence_complete']}; incomplete/invalid: {c['evidence_incomplete']}.",
        f"Logical decisions: {c['logical_decisions']}; known request attempts: {c['known_request_attempts']}; "
        f"known retries: {c['known_request_retries']}; decisions with unknown requests: {c['unknown_request_decisions']}.",
    ]
    for category, count in c["provider_failure_categories"].items():
        lines.append(f"Provider category `{category}`: {count}.")
    lines += ["", "Case | Repetition | Execution | Evaluation | Evidence | Provider category",
              "--- | --- | --- | --- | --- | ---"]
    for slot in results.slots:
        category = slot.provider_failure.category if slot.provider_failure is not None else "N/A"
        lines.append(f"{slot.case_id} | {slot.repetition} | {slot.execution} | {slot.evaluation} | {slot.evidence} | {category}")
    lines += ["", "## Limitations", "",
              "- Validation-only results are fake/scripted infrastructure checks, not real model metrics.",
              "- Completed means the controller returned; finished is not QA pass or full checker coverage.",
              "- Required succeeded Tool executions and their canonical finding links support the oracle.",
              "- Provider failures are unscored and remain visible in the planned denominator.",
              "- Not-started slots deliberately have no execution evidence; they cannot pass.",
              "- Raw conversations, Tool payloads, exceptions and Trace are not archived.",
              "- Unknown request counts/model identities are null; repetition is never a request retry.",
              "- No fallback, response cache, replay, model judge or overall wall-clock deadline.",
              "- Hashes are not authentication; dirty source has no patch; directory durability is not confirmed."]
    return "\n".join(lines) + "\n"


def run_live_evidence_package(
    directory: str | Path, *, provider_factory: Callable[[LivePlannedSlot], NextActionProvider],
    validation_only: bool = True,
) -> EvidencePackage:
    """Freeze 8 x 2 slots before provider setup; no automatic real-provider factory.

    A real run explicitly sets validation_only=False and supplies a fresh
    DeepSeekProvider per slot. Required evidence failures abort publication. A
    quota/non-retryable/harness failure stops execution, preserving unstarted slots.
    Other provider failures advance to the next planned sample, never retry a case.
    """
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError("Evidence directory must be new.")
    cases = build_fixed_live_cases()
    plan = _build_plan(cases, validation_only)
    definitions = {definition.case.case_id: definition for definition in cases}
    # Initialize every slot before executing any; errors cannot shrink this table.
    slots = [_not_started(slot) for slot in plan.slots]
    directory.mkdir()
    code = "publication_failed"
    try:
        package_io._write_artifact(directory, "plan.json", package_io.canonical_json(plan.model_dump(mode="json")))
        for index, planned in enumerate(plan.slots):
            slots[index] = _run_slot(definitions[planned.case_id], planned, plan, provider_factory)
            if _stops_run(slots[index]):
                break
        results = LiveEvidenceResults(schema_version=2, run=plan.run, planned_slot_count=16, slots=tuple(slots))
        return package_io._publish_results(directory, plan, results)
    except EvidencePackageError as error:
        code = error.code
    except Exception:
        pass
    raise EvidencePackageError(code)
