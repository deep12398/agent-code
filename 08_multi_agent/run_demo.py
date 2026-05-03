from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_code.multi_agent import CoordinationStormDetector, OrchestratorWorker, Pipeline, demo_agents


def main() -> None:
    agents = demo_agents()
    pipeline = Pipeline([("researcher", agents["researcher"]), ("writer", agents["writer"])])
    orchestrator = OrchestratorWorker(agents)
    messages = orchestrator.run("research and check Agent loop risks")
    print(json.dumps({"pipeline": pipeline.run("RAG"), "orchestrator": messages, "storms": CoordinationStormDetector().detect([])}, indent=2))


if __name__ == "__main__":
    main()
