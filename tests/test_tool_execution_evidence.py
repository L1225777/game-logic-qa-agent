import pytest

from game_qa_agent.analysis import build_task_index
from game_qa_agent.context import build_provider_decision_context
from game_qa_agent.models import GameRuntimeState, Task, ValidationIssue
from game_qa_agent.orchestration import execute_action, run_agent_investigation
from game_qa_agent.report import build_qa_report, render_qa_report_markdown
from game_qa_agent.tools import ToolRegistry, build_default_tool_registry
from game_qa_agent.trace import InMemoryInvestigationTraceRecorder

from conftest import ScriptedProvider, action
from test_action_safety import make_state


def test_empty_success_and_failure_have_distinct_canonical_evidence() -> None:
    task_index = build_task_index([Task(id="task_a")])
    runtime = GameRuntimeState()
    not_run = make_state()
    # Reach running through a real prior Tool call, so global status cannot stand
    # in for the outcome of the next checker.
    execute_action(
        action("call_tool", tool_name="dependency_reference_checker"),
        not_run, build_default_tool_registry(), task_index, runtime,
    )
    empty_success = not_run.model_copy(deep=True)
    failed = not_run.model_copy(deep=True)
    registry = ToolRegistry()
    registry.register_tool("npc_static_conflict_checker", lambda tasks: [])
    execute_action(
        action("call_tool", tool_name="npc_static_conflict_checker"),
        empty_success, registry, task_index, runtime,
    )
    failure = RuntimeError("private-tool-error-marker")

    def failing_checker(tasks):
        raise failure

    registry.register_tool("npc_static_conflict_checker", failing_checker)
    with pytest.raises(RuntimeError) as caught:
        execute_action(
            action("call_tool", tool_name="npc_static_conflict_checker"),
            failed, registry, task_index, runtime,
        )

    assert caught.value is failure
    assert empty_success.issues == failed.issues == []
    assert empty_success.called_tool_scope_versions == failed.called_tool_scope_versions
    assert empty_success.model_dump() != failed.model_dump(), (
        "Canonical state cannot distinguish a successful empty check from a failed check"
    )
    assert not any(
        record.tool_name == "npc_static_conflict_checker" for record in not_run.tool_executions
    )
    for state, status in ((empty_success, "succeeded"), (failed, "failed")):
        assert state.tool_executions[-1].model_dump() == {
            "execution_number": 2, "tool_name": "npc_static_conflict_checker",
            "scope_version": 1, "status": status, "issue_indices": (),
        }
        assert "private-" not in state.model_dump_json()


def test_canonical_findings_retain_their_actual_execution_scope() -> None:
    seed = ValidationIssue(
        issue_type="potential_npc_conflict", message="Expansion evidence.",
        task_ids=["task_a", "task_b"], checker_name="npc_static_conflict_checker",
    )
    finding = ValidationIssue(
        issue_type="potential_npc_conflict", message="Another recorded observation.",
        task_ids=["task_a"], checker_name="private-claimed-checker",
    )

    def investigate(finding_scope):
        def checker(tasks):
            scope_version = 1 if len(tasks) == 1 else 2
            return [seed, finding] if scope_version == finding_scope else [seed]

        registry = ToolRegistry()
        registry.register_tool("npc_static_conflict_checker", checker)
        return run_agent_investigation(
            make_state(), registry, build_task_index([Task(id="task_a"), Task(id="task_b")]),
            GameRuntimeState(), ScriptedProvider([
                action("call_tool", tool_name="npc_static_conflict_checker"),
                action("expand_scope", expand_task_ids=["task_b"]),
                action("call_tool", tool_name="npc_static_conflict_checker"),
                action("finish"),
            ]), max_steps=4,
        )

    early, late = investigate(1), investigate(2)

    assert early.issues == late.issues == [seed, finding]
    assert early.model_dump(exclude={"tool_executions"}) == (
        late.model_dump(exclude={"tool_executions"})
    )
    assert early.model_dump() != late.model_dump(), (
        "Canonical findings lose which trusted execution and scope produced them"
    )
    for state, finding_scope in ((early, 1), (late, 2)):
        assert [record.execution_number for record in state.tool_executions] == [1, 2]
        assert [record.scope_version for record in state.tool_executions] == [1, 2]
        assert all(
            record.tool_name == "npc_static_conflict_checker" for record in state.tool_executions
        )
        assert all(record.status == "succeeded" for record in state.tool_executions)
        assert [
            record.scope_version for record in state.tool_executions if 1 in record.issue_indices
        ] == [finding_scope]
        # One deduplicated seed finding is linked to both actual executions.
        assert all(0 in record.issue_indices for record in state.tool_executions)
        assert all("private-" not in record.model_dump_json() for record in state.tool_executions)


def test_failed_execution_links_partial_findings_and_preserves_exception_behavior() -> None:
    state = make_state()
    issue = ValidationIssue(
        issue_type="potential_npc_conflict", message="private-message-marker",
        task_ids=["task_a"], evidence={"private-evidence-marker": "detail"},
        checker_name="npc_static_conflict_checker",
    )
    failure = RuntimeError("private-tool-error-marker")

    def partial_checker(tasks):
        yield issue
        raise failure

    registry = ToolRegistry()
    registry.register_tool("npc_static_conflict_checker", partial_checker)
    task_index = build_task_index([Task(id="task_a")])
    runtime = GameRuntimeState()
    trace = InMemoryInvestigationTraceRecorder()
    provider = ScriptedProvider([
        action("call_tool", tool_name="npc_static_conflict_checker"), action("finish"),
    ])
    with pytest.raises(RuntimeError) as caught:
        run_agent_investigation(
            state, registry, task_index, runtime, provider, max_steps=2,
            trace_recorder=trace,
        )

    assert caught.value is failure
    assert len(provider.actions) == 1
    assert state.issues == [issue]
    assert state.decision_errors == []
    assert state.tool_executions[0].model_dump() == {
        "execution_number": 1, "tool_name": "npc_static_conflict_checker",
        "scope_version": 1, "status": "failed", "issue_indices": (0,),
    }
    assert "private-" not in state.tool_executions[0].model_dump_json()
    assert trace.steps == []
    assert trace.final_status is None

    before = state.model_dump_json()
    context = build_provider_decision_context(state, registry)
    report = build_qa_report(state, trace)
    assert state.investigation_status == report.status == "running"
    assert "still_running" in report.limitations
    assert "not_started" not in report.limitations
    markdown = render_qa_report_markdown(report)
    assert "The investigation has not started." not in markdown
    assert "The investigation has not reached a terminal status." in markdown
    without_records = state.model_copy(deep=True)
    without_records.tool_executions.clear()
    assert context == build_provider_decision_context(without_records, registry)
    assert report == build_qa_report(without_records, trace)
    assert "private-" not in context.model_dump_json() + report.model_dump_json()
    assert state.model_dump_json() == before

    # Failure does not implicitly authorize another attempt at the same scope.
    execute_action(
        action("call_tool", tool_name="npc_static_conflict_checker"),
        state, registry, task_index, runtime,
    )
    assert state.last_decision_rejection == "tool_already_called"
    assert len(state.tool_executions) == 1
