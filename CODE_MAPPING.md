# 配套代码与已有项目对应关系

这份文档说明本仓库里我新写的代码，和
`/Users/elias/code/practice-and-learning` 里的已有代码/数据之间是什么关系。

> 说明：第 19 篇及之后的连续项目代码（采购 Agent、LLMOps 平台）尚未随文章发布，
> 暂未放到线上。本文档只覆盖当前已公开的 02-18 篇代码。

结论先说清楚：

- **没有把已有项目代码整块搬过来。**
- **实际读取并复用了部分已有数据文件。**
- 当前仓库里的可运行代码，大部分是我为了文章配套重新写的轻量离线版本。

## 当前仓库代码结构

| 当前代码 | 作用 | 来源关系 |
|---|---|---|
| `02_*` 到 `10_*` 的 `run_demo.py` + `src/agent_code/*` | 第 02-10 篇基础篇配套实现 | 我新写 |
| `series_projects/rag_memory_project.py` | 第 11-18 篇连续项目：RAG + Memory | 我新写；实际读取 `llamaindex-learn` 的数据 |
| `series_projects/practice_assets.py` | 从已有项目加载数据资产 | 我新写；真实读取已有数据文件 |
| `series_projects/chapter_runner.py` | 统一章节入口（当前支持 11-18） | 我新写 |
| `agent_examples/*.py` | 离线可运行的轻量基础设施 | 我新写 |
| `11_*` 到 `18_*` 的 `demo.py` | 各文章章节入口 | 我新写；调用 `series_projects.chapter_runner` |
| `tests/test_all_demos.py` | 全量运行验证 | 我新写 |
| `run_all_demos.py` | 一次跑完所有章节 | 我新写 |

## 实际复用的数据

这些是当前代码**真实读取**的已有项目文件。

| 已有文件 | 当前使用位置 | 使用方式 |
|---|---|---|
| `/Users/elias/code/practice-and-learning/llamaindex-learn/data/catalog.json` | `series_projects/practice_assets.py` | 读取采购商品目录，输出到 RAG 阶段结果 |
| `/Users/elias/code/practice-and-learning/llamaindex-learn/data/knowledge/quality_standards.txt` | `series_projects/practice_assets.py` | 转成 RAG 文档，进入第 11-18 篇知识库 |
| `/Users/elias/code/practice-and-learning/llamaindex-learn/data/knowledge/sourcing_guide.txt` | `series_projects/practice_assets.py` | 转成 RAG 文档，进入第 11-18 篇知识库 |
| `/Users/elias/code/practice-and-learning/llamaindex-learn/data/knowledge/steel_knowledge.txt` | `series_projects/practice_assets.py` | 转成 RAG 文档，进入第 11-18 篇知识库 |
| `/Users/elias/code/practice-and-learning/langgraph-learn/phases/03-capstone/data/catalog.json` | `series_projects/practice_assets.py` | 只作为 fallback；如果 `llamaindex-learn` catalog 不存在才读 |

## 章节到代码的对应

| 文章范围 | 当前连续项目 | 实际复用程度 |
|---|---|---|
| 第 02-10 篇 基础篇 | `src/agent_code/*` + 各章 `run_demo.py` | 我新写的离线实现 |
| 第 11-18 篇 RAG / Memory | `series_projects/rag_memory_project.py` | 读取了 `llamaindex-learn` 的 catalog 和 knowledge 文档；Memory 逻辑为我新写 |
