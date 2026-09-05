"""Read-only projections of controller-owned state and optional same-run Trace.

Only current trusted scope IDs may cross the text boundary. Checker messages,
evidence values, NPC IDs, provider/action data and decision-error text never do.
Known issue/checker pairs select fixed report vocabulary; this is presentation,
not checker execution or independent verification of caller-supplied state.
"""

from collections import Counter
from html import escape
from typing import Literal, cast, get_args

from pydantic import BaseModel, ConfigDict, Field

from .models import AgentInvestigationState
from .trace import InMemoryInvestigationTraceRecorder, InvestigationFinalStatus


QAReportStatus = InvestigationFinalStatus | Literal["start", "running", "unknown"]
QAReportIssueType = Literal[
    "dependency_cycle", "missing_dependency", "potential_npc_conflict",
    "missing_npc_runtime_state", "npc_location_mismatch", "npc_state_mismatch",
    "npc_occupied_by_other_task",
]
QAReportCheckerName = Literal[
    "dependency_cycle_checker", "dependency_reference_checker",
    "npc_static_conflict_checker", "npc_runtime_checker",
]
QAReportLimitation = Literal[
    "recorded_findings_only", "static_risk_not_runtime_proof",
    "not_started", "still_running", "clarification_required",
    "human_review_required", "max_steps_exceeded", "unknown_status",
    "unrecognized_findings", "trace_may_be_incomplete",
    "trace_final_status_unavailable", "trace_status_mismatch",
]

_ISSUE_CHECKERS: dict[QAReportIssueType, QAReportCheckerName] = {
    "dependency_cycle": "dependency_cycle_checker",
    "missing_dependency": "dependency_reference_checker",
    "potential_npc_conflict": "npc_static_conflict_checker",
    "missing_npc_runtime_state": "npc_runtime_checker",
    "npc_location_mismatch": "npc_runtime_checker",
    "npc_state_mismatch": "npc_runtime_checker",
    "npc_occupied_by_other_task": "npc_runtime_checker",
}
_ISSUE_LABELS: dict[QAReportIssueType, str] = {
    "dependency_cycle": "Dependency cycle",
    "missing_dependency": "Missing dependency",
    "potential_npc_conflict": "Potential NPC conflict (static risk)",
    "missing_npc_runtime_state": "Missing NPC runtime state",
    "npc_location_mismatch": "NPC location mismatch",
    "npc_state_mismatch": "NPC state mismatch",
    "npc_occupied_by_other_task": "NPC occupied by another task",
}
_STATUS_LIMITATIONS: dict[QAReportStatus, QAReportLimitation] = {
    "start": "not_started",
    "running": "still_running",
    "clarification_required": "clarification_required",
    "human_review_required": "human_review_required",
    "max_steps_exceeded": "max_steps_exceeded",
    "unknown": "unknown_status",
}
_LIMITATION_TEXT: dict[QAReportLimitation, str] = {
    "recorded_findings_only": (
        "Counts reflect recorded findings; full checker coverage is not established."
    ),
    "static_risk_not_runtime_proof": (
        "A potential NPC conflict is a static risk, not proof of a runtime conflict."
    ),
    "not_started": "The investigation has not started.",
    "still_running": "The investigation has not reached a terminal status.",
    "clarification_required": "Clarification is required; the investigation is incomplete.",
    "human_review_required": "Human review is required; the investigation is incomplete.",
    "max_steps_exceeded": "The step limit was reached; the investigation is incomplete.",
    "unknown_status": "The investigation status is unrecognized; completion is unknown.",
    "unrecognized_findings": (
        "Unrecognized issue/checker pairs are counted but their details are omitted."
    ),
    "trace_may_be_incomplete": "Trace is diagnostic and may omit steps.",
    "trace_final_status_unavailable": "The Trace final status is unavailable or unrecognized.",
    "trace_status_mismatch": "The Trace final status differs from the investigation status.",
}
_FINAL_STATUSES = get_args(InvestigationFinalStatus)


class _ReportModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class QAReportFinding(_ReportModel):
    issue_type: QAReportIssueType
    checker_name: QAReportCheckerName
    scoped_task_ids: tuple[str, ...]
    unscoped_task_count: int = Field(ge=0)
    npc_count: int = Field(ge=0)


class QAReportScopeChange(_ReportModel):
    step_number: int
    scope_version_before: int
    scope_version_after: int
    added_task_ids: tuple[str, ...]


class QAReportTraceSummary(_ReportModel):
    step_count: int = Field(ge=0)
    final_status: InvestigationFinalStatus | Literal["unknown"] | None
    scope_changes: tuple[QAReportScopeChange, ...]
    rejected_step_count: int = Field(ge=0)


class QAInvestigationReport(_ReportModel):
    status: QAReportStatus
    scope_task_ids: tuple[str, ...]
    scope_version: int
    issue_count: int = Field(ge=0)
    issue_type_counts: dict[QAReportIssueType, int]
    unrecognized_issue_count: int = Field(ge=0)
    findings: tuple[QAReportFinding, ...]
    decision_error_count: int = Field(ge=0)
    trace: QAReportTraceSummary | None
    limitations: tuple[QAReportLimitation, ...]


def _summarize_trace(
    trace: InMemoryInvestigationTraceRecorder, trusted_scope: set[str]
) -> QAReportTraceSummary:
    changes = []
    for step in trace.steps:
        before = set(step.scope_task_ids_before) & trusted_scope
        after = set(step.scope_task_ids_after) & trusted_scope
        if step.scope_version_before != step.scope_version_after or before != after:
            changes.append(QAReportScopeChange(
                step_number=step.step_number,
                scope_version_before=step.scope_version_before,
                scope_version_after=step.scope_version_after,
                added_task_ids=tuple(sorted(after - before)),
            ))
    final_status = trace.final_status
    if final_status is not None and final_status not in _FINAL_STATUSES:
        final_status = "unknown"
    return QAReportTraceSummary(
        step_count=len(trace.steps),
        final_status=final_status,
        scope_changes=tuple(changes),
        rejected_step_count=sum(step.outcome == "rejected" for step in trace.steps),
    )


def build_qa_report(
    state: AgentInvestigationState,
    trace: InMemoryInvestigationTraceRecorder | None = None,
) -> QAInvestigationReport:
    """Snapshot trusted facts without executing or mutating the investigation.

    Pass controller-owned state and, when available, the recorder from that run.
    No task catalog or current-scope NPC allowlist is present in these models:
    out-of-scope task references and all NPC references are therefore counts only.
    Findings retain their recorded multiplicity, including equal safe projections.
    """
    trusted_scope = set(state.scope_task_ids)
    status = cast(QAReportStatus, state.investigation_status)
    if status not in _FINAL_STATUSES and status not in ("start", "running"):
        status = "unknown"
    findings = []
    for issue in state.issues:
        checker = _ISSUE_CHECKERS.get(issue.issue_type)
        if checker is None or issue.checker_name != checker:
            continue
        task_ids = set(issue.task_ids)
        findings.append(QAReportFinding(
            issue_type=cast(QAReportIssueType, issue.issue_type),
            checker_name=checker,
            scoped_task_ids=tuple(sorted(task_ids & trusted_scope)),
            unscoped_task_count=len(task_ids - trusted_scope),
            npc_count=len(set(issue.npc_ids)),
        ))
    findings.sort(key=lambda finding: (
        finding.issue_type, finding.scoped_task_ids,
        finding.unscoped_task_count, finding.npc_count,
    ))
    counts = dict(sorted(Counter(finding.issue_type for finding in findings).items()))
    unrecognized_count = len(state.issues) - len(findings)
    trace_summary = _summarize_trace(trace, trusted_scope) if trace is not None else None
    limitations: list[QAReportLimitation] = ["recorded_findings_only"]
    if status in _STATUS_LIMITATIONS:
        limitations.append(_STATUS_LIMITATIONS[status])
    if "potential_npc_conflict" in counts:
        limitations.append("static_risk_not_runtime_proof")
    if unrecognized_count:
        limitations.append("unrecognized_findings")
    if trace_summary is not None:
        limitations.append("trace_may_be_incomplete")
        if trace_summary.final_status in (None, "unknown"):
            limitations.append("trace_final_status_unavailable")
        elif trace_summary.final_status != status:
            limitations.append("trace_status_mismatch")
    return QAInvestigationReport(
        status=status,
        scope_task_ids=tuple(sorted(trusted_scope)),
        scope_version=state.scope_version,
        issue_count=len(state.issues),
        issue_type_counts=counts,
        unrecognized_issue_count=unrecognized_count,
        findings=tuple(findings),
        decision_error_count=len(state.decision_errors),
        trace=trace_summary,
        limitations=tuple(limitations),
    )


def _markdown_ids(task_ids: tuple[str, ...]) -> str:
    labels = []
    for task_id in task_ids:
        # Keep trusted labels on one line and inside a code span, including labels
        # containing backticks, HTML, table separators or invisible controls.
        visible = "".join(char if char.isprintable() else ascii(char)[1:-1] for char in task_id)
        label = escape(visible).replace("`", "&#96;").replace("|", "&#124;")
        labels.append(f"`{label}`")
    return ", ".join(labels) or "none"


def render_qa_report_markdown(report: QAInvestigationReport) -> str:
    """Render a report produced by build_qa_report, with stable ordering and LF."""
    lines = [
        "# QA investigation report", "",
        f"Status: `{report.status}`.", "",
        f"Trusted scope (version {report.scope_version}): {_markdown_ids(report.scope_task_ids)}.",
        "",
        f"Findings: {report.issue_count} recorded; {report.unrecognized_issue_count} unrecognized.",
        f"Decision errors: {report.decision_error_count}.", "",
        "## Findings", "",
    ]
    if report.issue_type_counts:
        lines.extend(["Type | Count", "--- | ---"])
        for issue_type, count in sorted(report.issue_type_counts.items()):
            lines.append(f"{_ISSUE_LABELS[issue_type]} | {count}")
        lines.extend(["", "Scoped details (one row per recorded recognized finding):", "",
                      "Type | Tasks in scope | Other task references | NPC references",
                      "--- | --- | --- | ---"])
        for finding in report.findings:
            lines.append(
                f"{_ISSUE_LABELS[finding.issue_type]} | {_markdown_ids(finding.scoped_task_ids)}"
                f" | {finding.unscoped_task_count} | {finding.npc_count}"
            )
    else:
        lines.append("No recognized findings recorded." if report.issue_count else "No findings recorded.")
    lines.extend(["", "## Trace", ""])
    if report.trace is None:
        lines.append("Not provided.")
    else:
        trace = report.trace
        lines.append(
            f"Recorded steps: {trace.step_count}; final status: `{trace.final_status or 'unavailable'}`;"
            f" rejected steps: {trace.rejected_step_count}; scope changes: {len(trace.scope_changes)}."
        )
        if trace.scope_changes:
            lines.append("")
        for change in trace.scope_changes:
            lines.append(
                f"- Step {change.step_number}: scope {change.scope_version_before} ->"
                f" {change.scope_version_after}; added {_markdown_ids(change.added_task_ids)}."
            )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {_LIMITATION_TEXT[limitation]}" for limitation in report.limitations)
    return "\n".join(lines) + "\n"
