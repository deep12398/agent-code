from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Callable

from .graph import State, StateGraph


def react_agent(question: str, tools: dict[str, Callable[[str], str]], max_steps: int = 4) -> dict[str, object]:
    trace: list[str] = []
    scratchpad = question
    for step in range(max_steps):
        if "计算" in scratchpad or any(char.isdigit() for char in scratchpad):
            tool_name = "calculator"
        elif "搜索" in scratchpad or "资料" in scratchpad:
            tool_name = "search"
        else:
            answer = f"最终答案：{scratchpad}"
            trace.append(answer)
            return {"answer": answer, "trace": trace}
        result = tools[tool_name](scratchpad)
        trace.append(f"step={step} tool={tool_name} result={result}")
        scratchpad = result
    return {"answer": "达到最大步数，返回当前观察结果。", "trace": trace}


def plan_and_execute(goal: str) -> dict[str, object]:
    plan = ["理解目标", "检索资料", "整理答案"]
    outputs = [f"{item}: {goal}" for item in plan]
    return {"plan": plan, "outputs": outputs, "answer": outputs[-1]}


def orchestrator_worker(task: str) -> dict[str, object]:
    workers = {
        "researcher": f"研究结论：{task} 的背景资料已整理",
        "engineer": f"工程方案：{task} 可拆成数据、流程、监控三层",
        "reviewer": "风险：需要权限、评测和回滚策略",
    }
    return {"task": task, "worker_outputs": workers, "answer": "；".join(workers.values())}


@dataclass
class CrewAgent:
    role: str
    goal: str

    def run(self, task: str) -> str:
        return f"{self.role}围绕“{task}”完成：{self.goal}"


@dataclass
class Crew:
    agents: list[CrewAgent]

    def kickoff(self, task: str) -> dict[str, object]:
        outputs = [agent.run(task) for agent in self.agents]
        return {"task": task, "outputs": outputs, "final": outputs[-1]}


@dataclass
class ADKSession:
    user_id: str
    session_state: dict[str, object] = field(default_factory=dict)
    app_state: dict[str, object] = field(default_factory=dict)
    events: list[dict[str, object]] = field(default_factory=list)

    def emit(self, event_type: str, payload: dict[str, object]) -> None:
        self.events.append({"type": event_type, "payload": payload})


def sequential_workflow(items: list[str]) -> list[str]:
    state = []
    for item in items:
        state.append(f"done:{item}")
    return state


async def parallel_workflow(items: list[str]) -> list[str]:
    async def run(item: str) -> str:
        await asyncio.sleep(0.01)
        return f"done:{item}"

    return await asyncio.gather(*(run(item) for item in items))


def loop_workflow(seed: int, limit: int = 3) -> list[int]:
    values = [seed]
    while values[-1] < limit:
        values.append(values[-1] + 1)
    return values


def delegation_workflow(task: str) -> str:
    if "代码" in task:
        return "delegate:engineer"
    if "调研" in task:
        return "delegate:researcher"
    return "delegate:general"


def json_tool_schema(name: str, description: str, properties: dict[str, str]) -> dict[str, object]:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {key: {"type": value} for key, value in properties.items()},
            "required": list(properties),
        },
    }


def decision_tree(project: dict[str, object]) -> str:
    if project.get("needs_human_approval") or project.get("needs_checkpoint"):
        return "LangGraph"
    if project.get("role_collaboration"):
        return "CrewAI"
    if project.get("google_cloud") or project.get("plugin_chain"):
        return "Google ADK"
    if project.get("long_running_coding"):
        return "Harness"
    return "Plain SDK + small graph"


def compare_frameworks() -> list[dict[str, object]]:
    scenarios = [
        {"name": "审批型客服", "needs_human_approval": True},
        {"name": "内容团队", "role_collaboration": True},
        {"name": "GCP 采购助手", "google_cloud": True},
    ]
    return [{"scenario": item["name"], "recommendation": decision_tree(item)} for item in scenarios]


def virtual_file_system_demo() -> dict[str, str]:
    files: dict[str, str] = {}
    files["todo.md"] = "- research\n- draft\n- review"
    files["report.md"] = "深度研究报告草稿"
    return files


def plugin_chain(input_text: str) -> dict[str, object]:
    stages = [
        ("audit", input_text),
        ("cost", f"{len(input_text)} chars"),
        ("safety", "passed" if "删除全部" not in input_text else "blocked"),
    ]
    return {"stages": stages, "allowed": stages[-1][1] == "passed"}

