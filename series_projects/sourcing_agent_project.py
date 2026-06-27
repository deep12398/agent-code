from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from agent_examples.frameworks import (
    ADKSession,
    Crew,
    CrewAgent,
    compare_frameworks,
    decision_tree,
    delegation_workflow,
    json_tool_schema,
    loop_workflow,
    orchestrator_worker,
    parallel_workflow,
    plan_and_execute,
    plugin_chain,
    react_agent,
    sequential_workflow,
    virtual_file_system_demo,
)
from agent_examples.graph import SqliteCheckpointer, StateGraph, State
from .practice_assets import catalog_summary


@dataclass
class RunRecord:
    """Audit record for each framework stage (becomes a trace span in production)."""

    stage: int
    name: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)


class SourcingAgentProject:
    """Continuous project for chapters 19-33.

    One procurement assistant is re-implemented as we walk through LangGraph,
    CrewAI, Google ADK and DeepAgents.  The business stays the same (帮采购员
    完成"询价 -> 比价 -> 审批 -> 下单"），only the orchestration framework
    changes, so the articles compare frameworks on identical requirements
    instead of unrelated toy demos.
    """

    GOAL = "帮采购员对钢材供应商完成询价、比价并走审批"

    def __init__(self, checkpoint_path: str | None = None) -> None:
        # Reuse the same catalog the RAG/Memory project loaded so the series
        # keeps one coherent procurement domain across all 48 chapters.
        self.catalog = catalog_summary(limit=5)
        self.checkpoint_path = checkpoint_path or ".agent_state_sourcing.db"
        self.run_log: list[RunRecord] = []

    def run_stage(self, chapter: int) -> dict[str, Any]:
        stages = {
            19: self.stage_19_langgraph_intro,
            20: self.stage_20_react_vs_plan,
            21: self.stage_21_checkpoint_hitl_timetravel,
            22: self.stage_22_orchestrator_worker,
            23: self.stage_23_crewai_intro,
            24: self.stage_24_crewai_vs_langgraph,
            25: self.stage_25_adk_intro,
            26: self.stage_26_adk_workflows,
            27: self.stage_27_adk_tools_callbacks,
            28: self.stage_28_adk_skills_memory_artifacts,
            29: self.stage_29_adk_eval_plugins,
            30: self.stage_30_deepagents_harness,
            31: self.stage_31_context_engineering,
            32: self.stage_32_framework_benchmark,
            33: self.stage_33_selection_decision_tree,
        }
        if chapter not in stages:
            raise ValueError(f"chapter {chapter} is outside the supported range 19-33")
        return stages[chapter]()

    def _record(self, stage: int, name: str, summary: str, details: dict[str, Any]) -> dict[str, Any]:
        self.run_log.append(RunRecord(stage, name, summary, details))
        return {
            "project": "sourcing_agent",
            "chapter": stage,
            "stage_name": name,
            "summary": summary,
            "details": details,
            "run_log_size": len(self.run_log),
        }

    # ------------------------------------------------------------------
    # Part 2-A: LangGraph (19-22)
    # ------------------------------------------------------------------
    def _build_sourcing_graph(self, checkpointer: SqliteCheckpointer | None = None):
        graph = StateGraph()

        def classify(state: State) -> State:
            query = str(state["query"])
            if "报价" in query or "询价" in query:
                return {"route": "quote"}
            if "审批" in query:
                return {"route": "approve"}
            return {"route": "general"}

        def quote(state: State) -> State:
            cheapest = min(self.catalog, key=lambda item: item.get("moq") or 0) if self.catalog else {}
            return {"answer": f"已生成询价单，候选 SKU={cheapest.get('id', 'n/a')}", "needs_approval": True}

        def approve(state: State) -> State:
            return {"answer": "审批通过，进入下单环节", "needs_approval": False}

        def general(state: State) -> State:
            return {"answer": f"采购助手处理：{state['query']}", "needs_approval": False}

        graph.add_node("classify", classify)
        graph.add_node("quote", quote)
        graph.add_node("approve", approve)
        graph.add_node("general", general)
        graph.add_edge("START", "classify")
        graph.add_conditional_edges(
            "classify",
            lambda state: str(state["route"]),
            {"quote": "quote", "approve": "approve", "general": "general"},
        )
        graph.add_edge("quote", "END")
        graph.add_edge("approve", "END")
        graph.add_edge("general", "END")
        return graph.compile(checkpointer=checkpointer)

    def stage_19_langgraph_intro(self) -> dict[str, Any]:
        app = self._build_sourcing_graph()
        result = app.invoke({"query": "帮我对 Q235 钢板做一次询价"}, thread_id="ch19")
        return self._record(
            19,
            "langgraph_intro",
            "用状态机把采购助手从一段 prompt 升级成可测试、可路由的图。",
            {
                "goal": self.GOAL,
                "final_state": result,
                "nodes": ["classify", "quote", "approve", "general"],
                "why_state_machine": ["显式控制流", "节点可单测", "条件边可审计"],
            },
        )

    def stage_20_react_vs_plan(self) -> dict[str, Any]:
        tools = {
            "search": lambda text: f"检索到供应商资料：{text[:20]}",
            "calculator": lambda text: "比价完成：供应商B 单价更低",
        }
        react = react_agent("搜索 Q235 钢板供应商并比价", tools, max_steps=4)
        planned = plan_and_execute(self.GOAL)
        return self._record(
            20,
            "react_vs_plan_execute",
            "同一个采购任务，分别用 ReAct（路径不定）和 Plan-and-Execute（路径明确）跑。",
            {
                "react": react,
                "plan_and_execute": planned,
                "selection_rule": "路径不确定用 ReAct；步骤清晰用 Plan-and-Execute；长任务两者混合",
            },
        )

    def stage_21_checkpoint_hitl_timetravel(self) -> dict[str, Any]:
        checkpointer = SqliteCheckpointer(self.checkpoint_path)
        app = self._build_sourcing_graph(checkpointer)
        app.invoke({"query": "帮我对 Q235 钢板做一次询价"}, thread_id="po-2024-001")
        history = checkpointer.history("po-2024-001")
        # HITL: 询价产生 needs_approval=True 时挂起，等待人工
        last_state = history[-1][2] if history else {}
        hitl = {
            "interrupted": bool(last_state.get("needs_approval")),
            "resume_action": "human_approve" if last_state.get("needs_approval") else "auto_continue",
        }
        return self._record(
            21,
            "checkpoint_hitl_timetravel",
            "Checkpoint 持久化每一步状态，HITL 在审批点挂起，Time-Travel 可回放到任意 step。",
            {
                "thread_id": "po-2024-001",
                "checkpoint_steps": [{"step": step, "node": node} for step, node, _ in history],
                "time_travel_state_at_step_0": history[0][2] if history else {},
                "hitl": hitl,
                "dependency": "HITL 能挂起-恢复的前提，是 checkpoint 把状态持久化了",
            },
        )

    def stage_22_orchestrator_worker(self) -> dict[str, Any]:
        result = orchestrator_worker("采购助手：调研钢材市场并产出比价报告")
        return self._record(
            22,
            "orchestrator_worker",
            "用 Orchestrator-Worker 把采购任务拆给 researcher / engineer / reviewer。",
            {
                "orchestration": result,
                "cascade_guards": ["reviewer 质量门控", "结构化输出校验", "最大轮数防无限对话"],
            },
        )

    # ------------------------------------------------------------------
    # Part 2-B: CrewAI (23-24)
    # ------------------------------------------------------------------
    def _sourcing_crew(self) -> Crew:
        return Crew(
            agents=[
                CrewAgent("市场调研员", "整理钢材供应商名单与行情"),
                CrewAgent("比价分析师", "对候选供应商做单价/MOQ/账期比价"),
                CrewAgent("采购主管", "审批并给出下单建议"),
            ]
        )

    def stage_23_crewai_intro(self) -> dict[str, Any]:
        crew = self._sourcing_crew()
        result = crew.kickoff(self.GOAL)
        return self._record(
            23,
            "crewai_intro",
            "用角色驱动的 Crew 重建采购团队：Agent / Task / Crew / Process。",
            {
                "crew_output": result,
                "roles": [agent.role for agent in crew.agents],
                "process": "sequential",
            },
        )

    def stage_24_crewai_vs_langgraph(self) -> dict[str, Any]:
        crew_result = self._sourcing_crew().kickoff(self.GOAL)
        graph_result = self._build_sourcing_graph().invoke({"query": "帮我对 Q235 钢板做一次询价"}, thread_id="ch24")
        return self._record(
            24,
            "crewai_vs_langgraph",
            "同一个采购客服系统，CrewAI（低代码角色）和 LangGraph（显式状态机）两种实现对比。",
            {
                "crewai": {"final": crew_result["final"], "strength": "上手快、角色分工直观"},
                "langgraph": {"final": graph_result.get("answer"), "strength": "控制显式、可中断可恢复"},
                "verdict": "要快用 CrewAI；要强控制和可恢复用 LangGraph",
            },
        )

    # ------------------------------------------------------------------
    # Part 2-C: Google ADK (25-29)
    # ------------------------------------------------------------------
    def stage_25_adk_intro(self) -> dict[str, Any]:
        session = ADKSession(user_id="buyer-1")
        session.session_state["current_po"] = "po-2024-001"
        session.app_state["catalog_size"] = len(self.catalog)
        session.emit("user_message", {"text": self.GOAL})
        session.emit("tool_call", {"name": "search_suppliers"})
        return self._record(
            25,
            "adk_intro",
            "ADK 五件套心智模型：Agent / Tool / Session / State / Event 应用到采购助手。",
            {
                "session_state": session.session_state,
                "app_state": session.app_state,
                "events": session.events,
                "five_pieces": ["Agent", "Tool", "Session", "State", "Event"],
            },
        )

    def stage_26_adk_workflows(self) -> dict[str, Any]:
        steps = ["询价", "比价", "审批", "下单"]
        sequential = sequential_workflow(steps)
        parallel = asyncio.run(parallel_workflow(["供应商A", "供应商B", "供应商C"]))
        loop = loop_workflow(seed=1, limit=3)
        delegation = delegation_workflow("调研钢材行情")
        return self._record(
            26,
            "adk_workflows",
            "ADK 四种编排：Sequential / Parallel / Loop / Delegation 跑采购流程。",
            {
                "sequential": sequential,
                "parallel": parallel,
                "loop": loop,
                "delegation": delegation,
            },
        )

    def stage_27_adk_tools_callbacks(self) -> dict[str, Any]:
        schema = json_tool_schema(
            "get_supplier_quote",
            "获取指定供应商对某 SKU 的报价",
            {"supplier_id": "string", "sku": "string"},
        )
        callbacks_fired: list[str] = []

        def before_tool(name: str) -> None:
            callbacks_fired.append(f"before_tool:{name}")

        def after_tool(name: str) -> None:
            callbacks_fired.append(f"after_tool:{name}")

        before_tool("get_supplier_quote")
        after_tool("get_supplier_quote")
        return self._record(
            27,
            "adk_tools_callbacks",
            "ADK 工具体系：Function Tool 的 JSON Schema + before/after callbacks 钩子。",
            {
                "tool_schema": schema,
                "callbacks_fired": callbacks_fired,
                "agent_as_tool": "比价 Agent 可以作为工具被采购主管 Agent 调用",
            },
        )

    def stage_28_adk_skills_memory_artifacts(self) -> dict[str, Any]:
        session = ADKSession(user_id="buyer-1")
        session.app_state["memory"] = ["买家偏好账期 60 天", "买家主营建材"]
        artifacts = {"比价报告.md": "# 比价报告\n供应商B 单价更低、账期符合偏好"}
        return self._record(
            28,
            "adk_skills_memory_artifacts",
            "ADK 的 Skills / Memory / Artifacts：技能复用、跨会话记忆、产出物落盘。",
            {
                "memory": session.app_state["memory"],
                "artifacts": artifacts,
                "skills": ["quote_skill", "compare_skill", "approval_skill"],
            },
        )

    def stage_29_adk_eval_plugins(self) -> dict[str, Any]:
        eval_cases = [
            {"input": "Q235 钢板询价", "expected_tool": "get_supplier_quote", "passed": True},
            {"input": "随便聊聊", "expected_tool": None, "passed": True},
        ]
        plugins = plugin_chain("帮我对 Q235 钢板做一次询价")
        return self._record(
            29,
            "adk_eval_plugins",
            "ADK Evaluation（轨迹评测）+ Plugins（横切插件链：审计/成本/安全）。",
            {
                "eval_cases": eval_cases,
                "eval_pass_rate": sum(case["passed"] for case in eval_cases) / len(eval_cases),
                "plugin_chain": plugins,
            },
        )

    # ------------------------------------------------------------------
    # Part 2-D: DeepAgents + 横评 + 选型 (30-33)
    # ------------------------------------------------------------------
    def stage_30_deepagents_harness(self) -> dict[str, Any]:
        return self._record(
            30,
            "deepagents_harness",
            "DeepAgents：区分 Harness（现成容器）/ Framework（自己搭）/ Runtime（执行环境）。",
            {
                "three_layers": {
                    "harness": "Claude Code / OpenHands —— 改 prompt 和挂工具",
                    "harness_sdk": "DeepAgents —— 工具+prompt+Middleware 全可换",
                    "framework": "LangGraph / ADK / CrewAI —— 从图到状态自己设计",
                },
                "sourcing_choice": "采购助手用 DeepAgents SDK 当内核，自定义 Middleware 接业务",
            },
        )

    def stage_31_context_engineering(self) -> dict[str, Any]:
        files = virtual_file_system_demo()
        return self._record(
            31,
            "context_engineering",
            "DeepAgents 三大 Context Engineering 模式：TODO 复述 / 虚拟文件系统 / 子 Agent 隔离。",
            {
                "virtual_file_system": files,
                "patterns": ["todo_recitation", "virtual_file_system", "sub_agent_isolation"],
                "applies_to": "任何 Agent 系统都能借鉴这三种长任务上下文管理模式",
            },
        )

    def stage_32_framework_benchmark(self) -> dict[str, Any]:
        return self._record(
            32,
            "framework_benchmark",
            "三框架横评：用同一组采购场景对 LangGraph / CrewAI / ADK 打分。",
            {
                "benchmark": compare_frameworks(),
                "dimensions": ["控制粒度", "上手速度", "可恢复性", "生态绑定"],
            },
        )

    def stage_33_selection_decision_tree(self) -> dict[str, Any]:
        projects = [
            {"name": "审批型采购客服", "needs_human_approval": True},
            {"name": "内容团队协作", "role_collaboration": True},
            {"name": "GCP 采购助手", "google_cloud": True},
            {"name": "长任务代码助手", "long_running_coding": True},
        ]
        recommendations = [
            {"project": project["name"], "recommendation": decision_tree(project)}
            for project in projects
        ]
        return self._record(
            33,
            "selection_decision_tree",
            "从业务特征到框架选择的决策树，给采购助手收口选型逻辑。",
            {
                "recommendations": recommendations,
                "rule": "用 trade-off 论证选型，而不是背特性表",
            },
        )


def run_sourcing_stage(chapter: int) -> dict[str, Any]:
    return SourcingAgentProject().run_stage(chapter)
