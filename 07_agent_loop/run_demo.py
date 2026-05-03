from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_code.agent_loop import AgentAction, AgentDecision, PlanAndExecute, ScriptedPlanner, rewrite_agent_executor
from agent_code.tool_essence import default_registry


def main() -> None:
    registry = default_registry()
    react = rewrite_agent_executor(
        registry,
        ScriptedPlanner([AgentDecision("calculate first", AgentAction("calculate", {"expression": "(3+5)*2"}))]),
    )
    plan = PlanAndExecute(registry).run([AgentAction("calculate", {"expression": "sqrt(16)"})])
    print(json.dumps({"react": react.run("calculate (3+5)*2"), "plan": plan}, indent=2))


if __name__ == "__main__":
    main()
