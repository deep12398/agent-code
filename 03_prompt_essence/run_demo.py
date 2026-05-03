from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_code.prompt_essence import PromptRegistry, PromptTemplate, classify_ticket_as_tool_call, dumps_strict_json


def main() -> None:
    registry = PromptRegistry()
    registry.add(PromptTemplate("router", "v1", "Route {ticket} to the right team.", ("ticket",)))
    tool_call = classify_ticket_as_tool_call("VPN is blocked and I need access asap")
    schema = {
        "type": "object",
        "properties": {"category": {"type": "string"}, "urgency": {"type": "string"}},
        "required": ["category", "urgency"],
    }
    print(
        json.dumps(
            {
                "prompt": registry.get("router").render(ticket="VPN issue"),
                "tool_call": tool_call,
                "strict_json": dumps_strict_json(tool_call["arguments"], schema),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
