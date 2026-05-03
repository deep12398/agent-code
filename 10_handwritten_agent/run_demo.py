from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_code.handwritten_agent import HandwrittenAgent


def main() -> None:
    agent = HandwrittenAgent()
    print(json.dumps(agent.run("Please calculate (3+5)*2"), indent=2))


if __name__ == "__main__":
    main()
