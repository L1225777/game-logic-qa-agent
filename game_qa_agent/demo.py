from dotenv import load_dotenv

from .analysis import build_task_index
from .orchestration import run_agent_investigation
from .providers import DeepSeekProvider
from .scenarios import (
    build_dynamic_scope_state,
    case_a_runtime_state,
    case_b_runtime_state,
    dynamic_scope_tasks,
)
from .tools import build_default_tool_registry


def main() -> None:
    load_dotenv()
    provider = DeepSeekProvider()
    for name, runtime in (("Case A", case_a_runtime_state()),
                          ("Case B", case_b_runtime_state())):
        tasks = dynamic_scope_tasks()
        final_state = run_agent_investigation(
            build_dynamic_scope_state(tasks),
            build_default_tool_registry(),
            build_task_index(tasks),
            runtime,
            provider,
        )
        print(name, final_state.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
