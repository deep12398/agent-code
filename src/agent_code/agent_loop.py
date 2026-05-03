from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from agent_code.llm_call import estimate_tokens
from agent_code.tool_essence import ToolCall, ToolRegistry


@dataclass(frozen=True)
class AgentAction:
    tool: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class AgentDecision:
    thought: str
    action: AgentAction | None = None
    final_answer: str | None = None


class Planner(Protocol):
    def decide(self, task: str, observations: list[str]) -> AgentDecision:
        ...


@dataclass
class ScriptedPlanner:
    decisions: list[AgentDecision]

    def decide(self, task: str, observations: list[str]) -> AgentDecision:
        if observations and len(observations) >= len(self.decisions):
            return AgentDecision("enough information", final_answer=observations[-1])
        return self.decisions[min(len(observations), len(self.decisions) - 1)]


@dataclass
class TerminationGuard:
    max_iterations: int = 5
    token_budget: int = 1200
    repeated_action_limit: int = 2

    def check(self, trace: list[dict[str, Any]]) -> str | None:
        if len(trace) >= self.max_iterations:
            return "max_iterations"
        total_tokens = estimate_tokens(json.dumps(trace, ensure_ascii=True))
        if total_tokens > self.token_budget:
            return "token_budget"
        if len(trace) >= self.repeated_action_limit:
            recent = trace[-self.repeated_action_limit :]
            signatures = [item.get("action_signature") for item in recent]
            if all(signature == signatures[0] for signature in signatures):
                return "repeated_action"
        return None


@dataclass
class ReactAgent:
    registry: ToolRegistry
    planner: Planner
    guard: TerminationGuard = field(default_factory=TerminationGuard)

    def run(self, task: str) -> dict[str, Any]:
        observations: list[str] = []
        trace: list[dict[str, Any]] = []

        while True:
            if reason := self.guard.check(trace):
                return {"status": "stopped", "reason": reason, "observations": observations, "trace": trace}

            decision = self.planner.decide(task, observations)
            if decision.final_answer is not None:
                return {"status": "finished", "answer": decision.final_answer, "trace": trace}
            if decision.action is None:
                return {"status": "stopped", "reason": "no_action", "trace": trace}

            signature = f"{decision.action.tool}:{json.dumps(decision.action.arguments, sort_keys=True)}"
            try:
                result = self.registry.call(decision.action.tool, decision.action.arguments)
                observation = str(result)
            except Exception as error:  # noqa: BLE001 - observations go back into the loop.
                observation = f"ERROR:{type(error).__name__}:{error}"

            observations.append(observation)
            trace.append(
                {
                    "thought": decision.thought,
                    "action": decision.action.tool,
                    "arguments": decision.action.arguments,
                    "observation": observation,
                    "action_signature": signature,
                }
            )


@dataclass
class PlanAndExecute:
    registry: ToolRegistry

    def run(self, plan: list[AgentAction]) -> dict[str, Any]:
        results = []
        for index, action in enumerate(plan, start=1):
            results.append({"step": index, "tool": action.tool, "result": self.registry.call(action.tool, action.arguments)})
        return {"status": "finished", "steps": results}


@dataclass
class JsonStateStore:
    path: Path

    def save(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(state, ensure_ascii=True, indent=2), encoding="utf-8")

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))


def detect_semantic_loop(calls: list[ToolCall], *, limit: int = 3) -> bool:
    if len(calls) < limit:
        return False
    recent = calls[-limit:]
    first = recent[0]
    return all(call.name == first.name and call.arguments == first.arguments for call in recent)


def rewrite_agent_executor(registry: ToolRegistry, planner: Planner) -> ReactAgent:
    return ReactAgent(
        registry=registry,
        planner=planner,
        guard=TerminationGuard(max_iterations=4, token_budget=900, repeated_action_limit=2),
    )
