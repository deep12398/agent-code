# Series Projects

这里放第 11-50 篇背后的连续项目代码。

- `rag_memory_project.py`：第 11-18 篇，企业知识库 RAG + 长期记忆。
- `sourcing_agent_project.py`：第 19-33 篇，采购助手在 LangGraph / CrewAI / ADK / DeepAgents 思路下的连续演进。
- `llmops_platform_project.py`：第 34-48 篇，LLM Gateway、评测、可观测、安全、成本、性能、后端和平台化。
- `interview_project.py`：第 49-50 篇面试番外，把面试题做成题库 + 玩具级/工业级答案评分器 + 系统设计答题框架。

每个章节目录里的 `demo.py` 只是调用这里的某个阶段：

```bash
python3 19_langgraph_intro/demo.py
```

也可以直接调用统一入口：

```bash
python3 -m series_projects.chapter_runner 19
```

代码默认使用标准库和本仓库内的轻量实现，不依赖外部 API Key。部分设计借鉴了
`/Users/elias/code/practice-and-learning` 里的学习项目结构，尤其是
`langgraph-learn`、`google-adk`、`crewai-learn`、`deepagents-learn` 和
`llamaindex-learn`。

