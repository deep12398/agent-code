# 公众号系列配套代码

本仓库是公开的可运行配套代码。文章正文为付费内容，存放在独立的私有仓库，不在此处。

## 第 02-10 篇（基础篇）

每章一个入口 `run_demo.py`，共享实现集中在 `src/agent_code/`，运行时自动把 `src` 加入路径，无需安装：

```bash
python3 02_llm_call_essence/run_demo.py
python3 -m pytest tests/test_companion_code.py
```

## 第 11-50 篇（连续项目）

这部分按系列组织成四个连续项目，同一套业务随章节演进，而不是每章重置成孤立片段：

- `series_projects/rag_memory_project.py`：第 11-18 篇，企业知识库 RAG + 长期记忆。
- `series_projects/sourcing_agent_project.py`：第 19-33 篇，采购助手在 LangGraph / CrewAI / ADK / DeepAgents 下的连续演进。
- `series_projects/llmops_platform_project.py`：第 34-48 篇，LLM Gateway、评测、可观测、安全、成本、性能、后端与平台化。
- `series_projects/interview_project.py`：第 49-50 篇，面试题库 + 玩具级/工业级答案评分器 + 系统设计答题框架。

每个章节目录保留 `demo.py`，作为该连续项目的阶段入口：

```bash
python3 11_rag_diagnosis/demo.py
python3 34_llm_gateway/demo.py
python3 49_interview_principles/demo.py
```

统一入口也可以直接指定章节（11-50）：

```bash
python3 -m series_projects.chapter_runner 19
```

也可以一次跑完全部示例：

```bash
python3 run_all_demos.py
```

全量测试：

```bash
python3 -m pytest tests/test_all_demos.py
```

这些示例默认不依赖外部 API Key 或在线服务，公共逻辑集中在 `agent_examples/`。

部分结构参考了 `/Users/elias/code/practice-and-learning` 里的已有学习项目：

- `llamaindex-learn`：RAG 数据与采购知识库思路（第 11-18 篇实际读取其 catalog 与 knowledge 文档）。
- `langgraph-learn` / `crewai-learn` / `google-adk` / `deepagents-learn`：第 19-33 篇的框架编排心智模型。
