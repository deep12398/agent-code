from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

from agent_code.llm_call import estimate_tokens


AgentFn = Callable[[str], str]


@dataclass(frozen=True)
class AgentMessage:
    sender: str
    recipient: str
    content: str


@dataclass
class MessageBus:
    messages: list[AgentMessage] = field(default_factory=list)

    def send(self, sender: str, recipient: str, content: str) -> None:
        self.messages.append(AgentMessage(sender, recipient, content))

    def inbox(self, recipient: str) -> list[AgentMessage]:
        return [message for message in self.messages if message.recipient == recipient]


@dataclass
class SharedState:
    values: dict[str, str] = field(default_factory=dict)

    def write(self, agent: str, key: str, value: str) -> None:
        self.values[f"{agent}.{key}"] = value

    def read_namespace(self, agent: str) -> dict[str, str]:
        prefix = f"{agent}."
        return {key.removeprefix(prefix): value for key, value in self.values.items() if key.startswith(prefix)}


@dataclass
class Pipeline:
    stages: list[tuple[str, AgentFn]]

    def run(self, input_text: str) -> dict[str, str]:
        current = input_text
        trace = {}
        for name, stage in self.stages:
            current = stage(current)
            trace[name] = current
        return trace


@dataclass
class OrchestratorWorker:
    workers: dict[str, AgentFn]

    def route(self, task: str) -> list[str]:
        lowered = task.lower()
        selected = []
        if any(word in lowered for word in ["search", "research", "find"]):
            selected.append("researcher")
        if any(word in lowered for word in ["check", "verify", "risk"]):
            selected.append("reviewer")
        if not selected:
            selected.append(next(iter(self.workers)))
        return selected

    def run(self, task: str) -> dict[str, str]:
        outputs = {name: self.workers[name](task) for name in self.route(task)}
        outputs["orchestrator"] = " | ".join(outputs.values())
        return outputs


@dataclass
class PeerToPeer:
    agents: dict[str, AgentFn]
    max_rounds: int = 2

    def run(self, task: str) -> list[AgentMessage]:
        bus = MessageBus()
        current = task
        names = list(self.agents)
        for round_index in range(self.max_rounds):
            for name in names:
                response = self.agents[name](current)
                bus.send(name, "all", response)
                current = response
            if round_index > 0 and _all_similar([message.content for message in bus.messages[-len(names) :]]):
                break
        return bus.messages


@dataclass
class FactVerifier:
    trusted_facts: set[str]

    def verify(self, claims: list[str]) -> dict[str, bool]:
        normalized = {fact.lower() for fact in self.trusted_facts}
        return {claim: claim.lower() in normalized for claim in claims}


@dataclass
class MultiAgentTokenBudget:
    total_budget: int
    used_by_agent: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def charge(self, agent: str, text: str) -> bool:
        cost = estimate_tokens(text)
        if self.remaining() < cost:
            return False
        self.used_by_agent[agent] += cost
        return True

    def remaining(self) -> int:
        return self.total_budget - sum(self.used_by_agent.values())


@dataclass
class CoordinationStormDetector:
    max_questions_per_agent: int = 2

    def detect(self, messages: list[AgentMessage]) -> list[str]:
        question_counts: dict[str, int] = defaultdict(int)
        for message in messages:
            if "?" in message.content:
                question_counts[message.sender] += 1
        return [agent for agent, count in question_counts.items() if count > self.max_questions_per_agent]


def demo_agents() -> dict[str, AgentFn]:
    return {
        "researcher": lambda task: f"research notes for: {task}",
        "reviewer": lambda task: f"risk check for: {task}",
        "writer": lambda task: f"final draft based on: {task}",
    }


def _all_similar(items: list[str]) -> bool:
    if len(items) <= 1:
        return False
    normalized = {item.lower().strip() for item in items}
    return len(normalized) == 1
