from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_code.memory_essence import LayeredMemory, budget_breakdown


def main() -> None:
    memory = LayeredMemory(user_id="u1")
    memory.remember_fact("The user prefers concise Python examples.")
    memory.add_turn("I am learning RAG.", "Start with retrieval before generation.")
    context = memory.build_context("Need Python RAG example")
    print(json.dumps({"context": [message.as_dict() for message in context], "budget": budget_breakdown(context)}, indent=2))


if __name__ == "__main__":
    main()
