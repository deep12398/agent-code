from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_code.tool_essence import FunctionCallingDemo, MinimalMCPServer, default_registry


def main() -> None:
    registry = default_registry()
    demo = FunctionCallingDemo(registry)
    mcp = MinimalMCPServer(registry)
    print(
        json.dumps(
            {
                "two_round": demo.run("what is (3+5)*2?"),
                "mcp_tools": mcp.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}),
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
