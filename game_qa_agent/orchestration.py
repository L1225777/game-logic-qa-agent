from .analysis import refresh_expandable_task_ids
from .models import AgentInvestigationState, GameRuntimeState, NextActionSpec, Task
from .providers import NextActionProvider
from .trace import (
    InvestigationFinalStatus,
    InvestigationStepTrace,
    InvestigationTraceRecorder,
    summarize_investigation_action,
)
from .tools import ToolRegistry, build_tool_inputs


TERMINAL_INVESTIGATION_STATUSES = {
    "finished", "clarification_required", "human_review_required"
}


def reject_agent_action(
    state: AgentInvestigationState, message: str
) -> AgentInvestigationState:
    state.decision_errors.append(message)
    state.investigation_status = "running"
    return state


def execute_action(
    action: NextActionSpec,
    state: AgentInvestigationState,
    tool_registry: ToolRegistry,
    full_task_index: dict[str, Task],
    full_game_runtime_state: GameRuntimeState,
) -> AgentInvestigationState:
    if action.action_type == "call_tool":
        if action.tool_name is None:
            return reject_agent_action(state, "Rejected call_tool because tool_name was missing.")
        if action.tool_args or action.expand_task_ids:
            return reject_agent_action(
                state, "Rejected call_tool because its extra action fields were not empty."
            )
        called_scope_versions = state.called_tool_scope_versions.get(
            action.tool_name, []
        )
        legacy_call_without_version = (
            action.tool_name in state.called_tool_names
            and action.tool_name not in state.called_tool_scope_versions
        )
        if state.scope_version in called_scope_versions or legacy_call_without_version:
            return reject_agent_action(
                state,
                f"Rejected repeated tool call '{action.tool_name}' at scope version "
                f"{state.scope_version}.",
            )
        try:
            tool_function = tool_registry.get_tool(action.tool_name)
        except KeyError:
            return reject_agent_action(state, f"Rejected unknown tool '{action.tool_name}'.")
        tool_inputs = build_tool_inputs(
            action.tool_name, state, full_task_index, full_game_runtime_state
        )
        if action.tool_name not in state.called_tool_names:
            state.called_tool_names.append(action.tool_name)
        state.called_tool_scope_versions.setdefault(action.tool_name, []).append(
            state.scope_version
        )
        for issue in tool_function(**tool_inputs):
            if issue not in state.issues:
                state.issues.append(issue)
        refresh_expandable_task_ids(state, full_task_index)
        state.investigation_status = "running"
    elif action.action_type == "expand_scope":
        if action.tool_name is not None or action.tool_args:
            return reject_agent_action(
                state, "Rejected expand_scope because its tool fields were not empty."
            )
        allowed = set(state.expandable_task_ids)
        requested = set(action.expand_task_ids)
        if not allowed:
            return reject_agent_action(
                state, "Rejected expand_scope because expandable_task_ids is empty."
            )
        if not requested:
            return reject_agent_action(
                state, "Rejected expand_scope because expand_task_ids was empty."
            )
        if not requested.issubset(allowed):
            return reject_agent_action(
                state,
                "Rejected expand_scope because requested task IDs "
                f"{sorted(requested)} are not a subset of expandable_task_ids "
                f"{state.expandable_task_ids}.",
            )
        state.scope_task_ids = sorted(set(state.scope_task_ids).union(requested))
        state.scope_version += 1
        refresh_expandable_task_ids(state, full_task_index)
        state.investigation_status = "running"
    elif action.action_type in {"clarify", "human_review", "finish"}:
        if action.tool_name is not None or action.tool_args or action.expand_task_ids:
            return reject_agent_action(
                state, f"Rejected {action.action_type} because action fields were not empty."
            )
        state.investigation_status = {
            "clarify": "clarification_required",
            "human_review": "human_review_required",
            "finish": "finished",
        }[action.action_type]
    return state


def run_agent_investigation(
    state: AgentInvestigationState,
    tool_registry: ToolRegistry,
    full_task_index: dict[str, Task],
    full_game_runtime_state: GameRuntimeState,
    provider: NextActionProvider,
    max_steps: int = 8,
    trace_recorder: InvestigationTraceRecorder | None = None,
) -> AgentInvestigationState:
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1.")

    for step_index in range(max_steps):
        action = provider.generate_next_action(state)
        trace_before = None
        if trace_recorder is not None:
            try:
                trace_before = {
                    "action": summarize_investigation_action(
                        action, tool_registry.tools
                    ),
                    "scope_version": state.scope_version,
                    "scope_task_ids": list(state.scope_task_ids),
                    "investigation_status": state.investigation_status,
                    "issue_count": len(state.issues),
                    "decision_error_count": len(state.decision_errors),
                }
            except Exception:
                # Trace is passive diagnostic state and must never affect execution.
                trace_before = None
        state = execute_action(
            action, state, tool_registry, full_task_index, full_game_runtime_state
        )
        if trace_recorder is not None and trace_before is not None:
            try:
                new_decision_error_count = (
                    len(state.decision_errors)
                    - trace_before["decision_error_count"]
                )
                if new_decision_error_count:
                    outcome = "rejected"
                elif state.investigation_status in TERMINAL_INVESTIGATION_STATUSES:
                    outcome = "terminal"
                else:
                    outcome = "processed"
                trace_recorder.record_step(InvestigationStepTrace(
                    step_number=step_index + 1,
                    action=trace_before["action"],
                    scope_version_before=trace_before["scope_version"],
                    scope_version_after=state.scope_version,
                    scope_task_ids_before=trace_before["scope_task_ids"],
                    scope_task_ids_after=list(state.scope_task_ids),
                    investigation_status_before=trace_before[
                        "investigation_status"
                    ],
                    investigation_status_after=state.investigation_status,
                    new_issue_types=[
                        issue.issue_type
                        for issue in state.issues[trace_before["issue_count"]:]
                    ],
                    new_decision_error_count=new_decision_error_count,
                    outcome=outcome,
                ))
            except Exception:
                # Recorder and Trace model failures are telemetry-only failures.
                pass
        if state.investigation_status in TERMINAL_INVESTIGATION_STATUSES:
            break
    if state.investigation_status == "running":
        state.investigation_status = "max_steps_exceeded"
    final_status: InvestigationFinalStatus | None = None
    if state.investigation_status == "finished":
        final_status = "finished"
    elif state.investigation_status == "clarification_required":
        final_status = "clarification_required"
    elif state.investigation_status == "human_review_required":
        final_status = "human_review_required"
    elif state.investigation_status == "max_steps_exceeded":
        final_status = "max_steps_exceeded"
    if trace_recorder is not None and final_status is not None:
        try:
            trace_recorder.record_final_status(final_status)
        except Exception:
            pass
    return state
