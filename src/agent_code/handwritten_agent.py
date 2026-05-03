from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from agent_code.agent_loop import AgentAction, AgentDecision, ReactAgent, TerminationGuard
from agent_code.llm_call import ChatMessage, estimate_message_tokens
from agent_code.tool_essence import ToolRegistry, default_registry, extract_math_expression


@dataclass
class RuleBasedPlanner:
    finished: bool = False

    def decide(self, task: str, observations: list[str]) -> AgentDecision:
        if observations:
            self.finished = True
            return AgentDecision("tool result is enough", final_answer=observations[-1])
        lowered = task.lower()
        if any(word in lowered for word in ["weather", "temperature"]):
            city = "Dubai" if "dubai" in lowered else "Shanghai"
            return AgentDecision("need weather tool", AgentAction("get_weather", {"city": city}))
        return AgentDecision("need calculator", AgentAction("calculate", {"expression": extract_math_expression(task)}))


@dataclass
class HandwrittenAgent:
    registry: ToolRegistry = field(default_factory=default_registry)
    system_prompt: str = "You are a small reliable agent. Use tools when needed."
    max_prompt_tokens: int = 500

    def run(self, user_input: str) -> dict[str, Any]:
        messages = [ChatMessage("system", self.system_prompt), ChatMessage("user", user_input)]
        if estimate_message_tokens(messages) > self.max_prompt_tokens:
            return {"status": "stopped", "reason": "prompt_budget"}

        agent = ReactAgent(
            registry=self.registry,
            planner=RuleBasedPlanner(),
            guard=TerminationGuard(max_iterations=3, token_budget=600, repeated_action_limit=2),
        )
        return agent.run(user_input)


@dataclass
class PersistentHandwrittenAgent(HandwrittenAgent):
    state_path: Path = Path(".agent_state.json")

    def run(self, user_input: str) -> dict[str, Any]:
        state = self._load_state()
        state.setdefault("turns", []).append({"role": "user", "content": user_input})
        result = super().run(user_input)
        state["turns"].append({"role": "assistant", "content": result.get("answer", result.get("reason", ""))})
        state.setdefault("traces", []).append(result.get("trace", []))
        self._save_state(state)
        return result

    def stream_answer(self, user_input: str) -> Iterable[str]:
        result = self.run(user_input)
        answer = result.get("answer") or result.get("reason") or ""
        for token in str(answer).split():
            yield token + " "

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {}
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _save_state(self, state: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, ensure_ascii=True, indent=2), encoding="utf-8")
