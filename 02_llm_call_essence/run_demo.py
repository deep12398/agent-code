from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_code.llm_call import ChatMessage, LLMRequest, StreamAssembler, build_raw_http_request


def main() -> None:
    request = LLMRequest(
        model="gpt-4.1-mini",
        messages=[ChatMessage("system", "Answer briefly."), ChatMessage("user", "What is an Agent?")],
        stream=True,
    )
    assembler = StreamAssembler()
    assembler.push_event({"choices": [{"delta": {"content": "An agent "}}]})
    assembler.push_event({"choices": [{"delta": {"content": "is a loop."}}], "usage": {"total_tokens": 12}})
    print(json.dumps({"http": build_raw_http_request(request), "stream_text": assembler.text(), "usage": assembler.usage}, indent=2))


if __name__ == "__main__":
    main()
