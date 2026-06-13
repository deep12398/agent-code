from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


State = dict[str, object]
Node = Callable[[State], State]
Router = Callable[[State], str]


@dataclass
class StateGraph:
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: dict[str, str] = field(default_factory=dict)
    conditional_edges: dict[str, tuple[Router, dict[str, str]]] = field(default_factory=dict)
    start: str | None = None

    def add_node(self, name: str, node: Node) -> None:
        self.nodes[name] = node

    def add_edge(self, source: str, target: str) -> None:
        if source == "START":
            self.start = target
            return
        self.edges[source] = target

    def add_conditional_edges(self, source: str, router: Router, mapping: dict[str, str]) -> None:
        self.conditional_edges[source] = (router, mapping)

    def compile(self, checkpointer: "SqliteCheckpointer | None" = None) -> "GraphApp":
        if not self.start:
            raise ValueError("Graph needs a START edge")
        return GraphApp(self, checkpointer)


class GraphApp:
    def __init__(self, graph: StateGraph, checkpointer: "SqliteCheckpointer | None" = None):
        self.graph = graph
        self.checkpointer = checkpointer

    def invoke(self, initial_state: State, thread_id: str = "default") -> State:
        state = dict(initial_state)
        current = self.graph.start
        step = 0
        while current and current != "END":
            update = self.graph.nodes[current](state)
            state.update(update)
            if self.checkpointer:
                self.checkpointer.save(thread_id, step, current, state)
            if current in self.graph.conditional_edges:
                router, mapping = self.graph.conditional_edges[current]
                route_key = router(state)
                current = mapping[route_key]
            else:
                current = self.graph.edges.get(current, "END")
            step += 1
            if step > 50:
                raise RuntimeError("graph exceeded 50 steps")
        return state


class SqliteCheckpointer:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS checkpoints (thread_id TEXT, step INTEGER, node TEXT, state TEXT, PRIMARY KEY(thread_id, step))"
            )

    def save(self, thread_id: str, step: int, node: str, state: State) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO checkpoints VALUES (?, ?, ?, ?)",
                (thread_id, step, node, json.dumps(state, ensure_ascii=False, default=str)),
            )

    def history(self, thread_id: str) -> list[tuple[int, str, State]]:
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                "SELECT step, node, state FROM checkpoints WHERE thread_id = ? ORDER BY step",
                (thread_id,),
            ).fetchall()
        return [(step, node, json.loads(state)) for step, node, state in rows]


def intent_graph() -> GraphApp:
    graph = StateGraph()

    def classify(state: State) -> State:
        query = str(state["query"])
        if "天气" in query:
            return {"intent": "weather"}
        if "采购" in query or "报价" in query:
            return {"intent": "quote"}
        return {"intent": "general"}

    def weather(state: State) -> State:
        return {"answer": "今天天气晴朗。"}

    def quote(state: State) -> State:
        return {"answer": "已进入采购报价流程，需要供应商、预算和审批人。"}

    def general(state: State) -> State:
        return {"answer": f"我会处理：{state['query']}"}

    graph.add_node("classify", classify)
    graph.add_node("weather", weather)
    graph.add_node("quote", quote)
    graph.add_node("general", general)
    graph.add_edge("START", "classify")
    graph.add_conditional_edges("classify", lambda state: str(state["intent"]), {
        "weather": "weather",
        "quote": "quote",
        "general": "general",
    })
    graph.add_edge("weather", "END")
    graph.add_edge("quote", "END")
    graph.add_edge("general", "END")
    return graph.compile()

