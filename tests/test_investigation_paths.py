from game_qa_agent.analysis import build_task_index
from game_qa_agent.models import (
    AgentInvestigationState,
    GameRuntimeState,
    ImpactAnalysisResult,
    NPCRequirement,
    NPCRuntimeState,
    Task,
)
from game_qa_agent.orchestration import run_agent_investigation
from game_qa_agent.tools import build_default_tool_registry

from conftest import ScriptedProvider, action


def test_static_risk_branches_to_runtime_evidence_without_becoming_proof() -> None:
    tasks = [
        Task(id="left", npc_requirements=[
            NPCRequirement(npc_id="npc", location="village", state="available")
        ]),
        Task(id="right", npc_requirements=[
            NPCRequirement(npc_id="npc", location="town", state="available")
        ]),
    ]
    state = AgentInvestigationState(
        investigation_goal="Check whether static risk occurs at runtime.",
        impact_analysis=ImpactAnalysisResult(affected_task_ids=["left", "right"]),
        scope_task_ids=["left", "right"],
    )
    runtime = GameRuntimeState(
        task_statuses={"left": "inactive", "right": "inactive"},
        npc_states={"npc": NPCRuntimeState(location="elsewhere", state="available")},
    )
    provider = ScriptedProvider([
        action("call_tool", tool_name="npc_static_conflict_checker"),
        action("call_tool", tool_name="npc_runtime_checker"),
        action("finish"),
    ])

    result = run_agent_investigation(
        state, build_default_tool_registry(), build_task_index(tasks), runtime, provider
    )

    assert result.called_tool_names == [
        "npc_static_conflict_checker", "npc_runtime_checker"
    ]
    assert [issue.issue_type for issue in result.issues] == ["potential_npc_conflict"]
    assert result.investigation_status == "finished"
