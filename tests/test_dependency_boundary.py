from game_qa_agent.analysis import build_task_index
from game_qa_agent.checkers import detect_missing_dependencies
from game_qa_agent.models import Task


def test_dependency_outside_scope_but_in_full_index_is_not_missing() -> None:
    scoped = Task(id="task_b", dependencies=["task_a"])
    full_index = build_task_index([Task(id="task_a"), scoped])

    assert detect_missing_dependencies([scoped], full_index) == []


def test_dependency_absent_from_full_index_is_missing() -> None:
    scoped = Task(id="task_b", dependencies=["missing_task"])

    issues = detect_missing_dependencies([scoped], build_task_index([scoped]))

    assert [issue.issue_type for issue in issues] == ["missing_dependency"]
    assert issues[0].evidence == {"missing_dependency_id": "missing_task"}
