# 公众号系列配套代码

本仓库是公开的可运行配套代码。文章正文为付费内容，存放在独立的私有仓库，不在此处。

## 第 02-10 篇（基础篇）

每章一个入口 `run_demo.py`，共享实现集中在 `src/agent_code/`，运行时自动把 `src` 加入路径，无需安装：

```bash
python3 02_llm_call_essence/run_demo.py
python3 -m pytest tests/test_companion_code.py
```

## 第 11-18 篇（RAG / Memory）

> 第 19 篇及之后的连续项目代码尚未随文章发布，暂未放到线上。

这部分按系列组织成连续项目：

- `series_projects/rag_memory_project.py`：第 11-18 篇，企业知识库 RAG + 长期记忆。

每个章节目录保留 `demo.py`，作为该连续项目的阶段入口：

```bash
python3 11_rag_diagnosis/demo.py
```

统一入口也可以直接指定章节：

```bash
python3 -m series_projects.chapter_runner 11
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

- `llamaindex-learn`：RAG 数据与采购知识库思路。
