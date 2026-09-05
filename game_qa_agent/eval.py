from collections import deque
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .analysis import analyze_initial_change_impact, build_task_index
from .context import ProviderDecisionContext
from .models import (
    AgentInvestigationState,
    GameRuntimeState,
    NextActionSpec,
    Task,
)
from .orchestration import run_agent_investigation
from .scenarios import (
    build_dynamic_scope_state,
    case_a_runtime_state,
    dynamic_scope_tasks,
)
from .tools import build_default_tool_registry
from .trace import InMemoryInvestigationTraceRecorder, InvestigationFinalStatus


EvaluationCaseOutcome = Literal["passed", "failed_behavior", "harness_error"]
EvaluationExpectationName = Literal[
    "final_status",
    "issue_types",
    "trusted_scope_task_ids",
    "decision_error_count",
    "trace_step_count",
    "trace_final_status",
]
EvaluationHarnessStage = Literal[
    "investigation_execution", "evidence_collection", "expectation_evaluation"
]
EvaluationValue = str | int | list[str] | None


class AgentEvaluationCase(BaseModel):
    case_id: str
    initial_state: AgentInvestigationState
    tasks: list[Task]
    runtime_state: GameRuntimeState
    scripted_actions: list[NextActionSpec] = Field(repr=False)
    expected_final_status: InvestigationFinalStatus
    expected_issue_types: list[str] = Field(default_factory=list)
    expected_trusted_scope_task_ids: list[str] = Field(default_factory=list)
    expected_decision_error_count: int = 0
    max_steps: int = 8
    trace_enabled: bool = False
    expected_trace_step_count: int | None = None
    expected_trace_final_status: InvestigationFinalStatus | None = None

    @model_validator(mode="after")
    def trace_expectations_require_trace(self) -> "AgentEvaluationCase":
        if not self.trace_enabled and (
            self.expected_trace_step_count is not None
            or self.expected_trace_final_status is not None
        ):
            raise ValueError("Trace expectations require trace_enabled=True.")
        return self


class AgentEvaluationEvidence(BaseModel):
    actual_final_status: str
    actual_issue_types: list[str] = Field(default_factory=list)
    actual_trusted_scope_task_ids: list[str] = Field(default_factory=list)
    actual_decision_error_count: int
    trace_step_count: int | None = None
    trace_final_status: InvestigationFinalStatus | None = None


class AgentEvaluationExpectationResult(BaseModel):
    expectation: EvaluationExpectationName
    expected: EvaluationValue
    actual: EvaluationValue
    passed: bool


class AgentEvaluationHarnessError(BaseModel):
    stage: EvaluationHarnessStage
    exception_type: str


class AgentEvaluationCaseResult(BaseModel):
    case_id: str
    outcome: EvaluationCaseOutcome
    evidence: AgentEvaluationEvidence | None = None
    expectation_results: list[AgentEvaluationExpectationResult] = Field(
        default_factory=list
    )
    passed_expectations: list[EvaluationExpectationName] = Field(
        default_factory=list
    )
    failed_expectations: list[EvaluationExpectationName] = Field(
        default_factory=list
    )
    harness_error: AgentEvaluationHarnessError | None = None


class AgentEvaluationSummary(BaseModel):
    total_case_count: int
    evaluated_case_count: int
    passed_case_count: int
    failed_behavioral_case_count: int
    harness_error_case_count: int
    case_results: list[AgentEvaluationCaseResult] = Field(default_factory=list)


class _ScriptedEvaluationProvider:
    def __init__(self, actions: list[NextActionSpec]) -> None:
        self._actions = deque(action.model_copy(deep=True) for action in actions)

    def generate_next_action(
        self, context: ProviderDecisionContext
    ) -> NextActionSpec:
        return self._actions.popleft()


def _expectation_result(
    expectation: EvaluationExpectationName,
    expected: EvaluationValue,
    actual: EvaluationValue,
) -> AgentEvaluationExpectationResult:
    return AgentEvaluationExpectationResult(
        expectation=expectation,
        expected=expected,
        actual=actual,
        passed=expected == actual,
    )


def _evaluate_expectations(
    case: AgentEvaluationCase, evidence: AgentEvaluationEvidence
) -> list[AgentEvaluationExpectationResult]:
    results = [
        _expectation_result(
            "final_status",
            case.expected_final_status,
            evidence.actual_final_status,
        ),
        _expectation_result(
            "issue_types",
            sorted(case.expected_issue_types),
            evidence.actual_issue_types,
        ),
        _expectation_result(
            "trusted_scope_task_ids",
            sorted(case.expected_trusted_scope_task_ids),
            evidence.actual_trusted_scope_task_ids,
        ),
        _expectation_result(
            "decision_error_count",
            case.expected_decision_error_count,
            evidence.actual_decision_error_count,
        ),
    ]
    if case.expected_trace_step_count is not None:
        results.append(_expectation_result(
            "trace_step_count",
            case.expected_trace_step_count,
            evidence.trace_step_count,
        ))
    if case.expected_trace_final_status is not None:
        results.append(_expectation_result(
            "trace_final_status",
            case.expected_trace_final_status,
            evidence.trace_final_status,
        ))
    return results


def run_evaluation_case(case: AgentEvaluationCase) -> AgentEvaluationCaseResult:
    return _run_evaluation_case_with_state(case)[0]


def _run_evaluation_case_with_state(
    case: AgentEvaluationCase,
) -> tuple[AgentEvaluationCaseResult, AgentInvestigationState | None]:
    """One execution path; the private state is for downstream safe projection only."""
    stage: EvaluationHarnessStage = "investigation_execution"
    state = None
    try:
        state = case.initial_state.model_copy(deep=True)
        tasks = [task.model_copy(deep=True) for task in case.tasks]
        runtime_state = case.runtime_state.model_copy(deep=True)
        provider = _ScriptedEvaluationProvider(case.scripted_actions)
        recorder = (
            InMemoryInvestigationTraceRecorder() if case.trace_enabled else None
        )
        observation = run_agent_investigation(
            state=state,
            tool_registry=build_default_tool_registry(),
            full_task_index=build_task_index(tasks),
            full_game_runtime_state=runtime_state,
            provider=provider,
            max_steps=case.max_steps,
            trace_recorder=recorder,
        )

        stage = "evidence_collection"
        evidence = AgentEvaluationEvidence(
            actual_final_status=observation.investigation_status,
            actual_issue_types=sorted(issue.issue_type for issue in observation.issues),
            actual_trusted_scope_task_ids=sorted(observation.scope_task_ids),
            actual_decision_error_count=len(observation.decision_errors),
            trace_step_count=len(recorder.steps) if recorder is not None else None,
            trace_final_status=recorder.final_status if recorder is not None else None,
        )

        stage = "expectation_evaluation"
        expectation_results = _evaluate_expectations(case, evidence)
    except Exception as error:
        return AgentEvaluationCaseResult(
            case_id=case.case_id,
            outcome="harness_error",
            harness_error=AgentEvaluationHarnessError(
                stage=stage,
                exception_type=type(error).__name__,
            ),
        ), state

    passed_expectations = [
        result.expectation for result in expectation_results if result.passed
    ]
    failed_expectations = [
        result.expectation for result in expectation_results if not result.passed
    ]
    return AgentEvaluationCaseResult(
        case_id=case.case_id,
        outcome="failed_behavior" if failed_expectations else "passed",
        evidence=evidence,
        expectation_results=expectation_results,
        passed_expectations=passed_expectations,
        failed_expectations=failed_expectations,
    ), observation


def run_evaluation_suite(
    cases: list[AgentEvaluationCase],
) -> AgentEvaluationSummary:
    case_results = [run_evaluation_case(case) for case in cases]
    passed_case_count = sum(
        result.outcome == "passed" for result in case_results
    )
    failed_behavioral_case_count = sum(
        result.outcome == "failed_behavior" for result in case_results
    )
    harness_error_case_count = sum(
        result.outcome == "harness_error" for result in case_results
    )
    return AgentEvaluationSummary(
        total_case_count=len(case_results),
        evaluated_case_count=passed_case_count + failed_behavioral_case_count,
        passed_case_count=passed_case_count,
        failed_behavioral_case_count=failed_behavioral_case_count,
        harness_error_case_count=harness_error_case_count,
        case_results=case_results,
    )


def _action(
    action_type: Literal["call_tool", "expand_scope", "finish"],
    *,
    tool_name: str | None = None,
    expand_task_ids: list[str] | None = None,
) -> NextActionSpec:
    return NextActionSpec(
        action_type=action_type,
        tool_name=tool_name,
        expand_task_ids=expand_task_ids or [],
        reason="deterministic offline evaluation action",
    )


def _basic_case_state(tasks: list[Task]) -> AgentInvestigationState:
    impact = analyze_initial_change_impact(tasks, ["task_a"])
    return AgentInvestigationState(
        investigation_goal="Evaluate the deterministic investigation path.",
        impact_analysis=impact,
        scope_task_ids=impact.affected_task_ids,
    )


def build_deterministic_evaluation_cases() -> list[AgentEvaluationCase]:
    basic_tasks = [Task(id="task_a")]
    dynamic_tasks = dynamic_scope_tasks()
    return [
        AgentEvaluationCase(
            case_id="normal_success",
            initial_state=_basic_case_state(basic_tasks),
            tasks=basic_tasks,
            runtime_state=GameRuntimeState(),
            scripted_actions=[
                _action(
                    "call_tool", tool_name="dependency_reference_checker"
                ),
                _action("finish"),
            ],
            expected_final_status="finished",
            expected_trusted_scope_task_ids=["task_a"],
            trace_enabled=True,
            expected_trace_step_count=2,
            expected_trace_final_status="finished",
        ),
        AgentEvaluationCase(
            case_id="authorized_dynamic_scope_expansion",
            initial_state=build_dynamic_scope_state(dynamic_tasks),
            tasks=dynamic_tasks,
            runtime_state=case_a_runtime_state(),
            scripted_actions=[
                _action("call_tool", tool_name="npc_runtime_checker"),
                _action(
                    "expand_scope", expand_task_ids=["task_event_7"]
                ),
                _action(
                    "call_tool", tool_name="npc_static_conflict_checker"
                ),
                _action("finish"),
            ],
            expected_final_status="finished",
            expected_issue_types=[
                "npc_location_mismatch",
                "npc_occupied_by_other_task",
                "potential_npc_conflict",
            ],
            expected_trusted_scope_task_ids=[
                "task_a", "task_b", "task_event_7"
            ],
            trace_enabled=True,
            expected_trace_step_count=4,
            expected_trace_final_status="finished",
        ),
        AgentEvaluationCase(
            case_id="rejected_unauthorized_scope_expansion",
            initial_state=build_dynamic_scope_state(dynamic_tasks),
            tasks=dynamic_tasks,
            runtime_state=case_a_runtime_state(),
            scripted_actions=[
                _action("call_tool", tool_name="npc_runtime_checker"),
                _action("expand_scope", expand_task_ids=["task_c"]),
                _action("finish"),
            ],
            expected_final_status="finished",
            expected_issue_types=[
                "npc_location_mismatch", "npc_occupied_by_other_task"
            ],
            expected_trusted_scope_task_ids=["task_a", "task_b"],
            expected_decision_error_count=1,
            trace_enabled=True,
            expected_trace_step_count=3,
            expected_trace_final_status="finished",
        ),
        AgentEvaluationCase(
            case_id="max_step_exhaustion",
            initial_state=_basic_case_state(basic_tasks),
            tasks=basic_tasks,
            runtime_state=GameRuntimeState(),
            scripted_actions=[
                _action("call_tool", tool_name="unknown_tool")
            ],
            expected_final_status="max_steps_exceeded",
            expected_trusted_scope_task_ids=["task_a"],
            expected_decision_error_count=1,
            max_steps=1,
            trace_enabled=True,
            expected_trace_step_count=1,
            expected_trace_final_status="max_steps_exceeded",
        ),
    ]
