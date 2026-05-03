# Industrial AI Agent Companion Code

This repository contains runnable, offline-first companion code for the
`Industrial AI Agent` article series.

The examples intentionally avoid mandatory external services. Every chapter can
run and test without an OpenAI key, database, vector store, or MCP runtime. The
code keeps the same engineering shape as production systems: request building,
message management, token budgeting, memory selection, RAG retrieval, tool
dispatch, agent loops, multi-agent coordination, and 7-layer diagnostics.

## Quick Start

```bash
cd agent-code
python -m unittest discover -s tests
python 02_llm_call_essence/run_demo.py
python 10_handwritten_agent/run_demo.py
```

## Chapter Map

| Article | Directory | What is included |
| --- | --- | --- |
| 02 LLM call essence | `02_llm_call_essence/` | HTTP payload builder, retry wrapper, token estimate, SSE assembler, usage logging |
| 03 Prompt essence | `03_prompt_essence/` | Function-calling classifier, JSON schema validation, CoT cost estimate, versioned prompt templates |
| 04 Memory essence | `04_memory_essence/` | Buffer/window/summary memory, layered memory, vector-like long-term memory, token budget fitting |
| 05 RAG essence | `05_rag_essence/` | Naive RAG, hybrid search, reranking, evaluation pipeline, fine-tune JSONL data builder |
| 06 Tool essence | `06_tool_essence/` | Two-round tool calling, parallel tool calls, minimal MCP server, skill wrapper, idempotency, error translation |
| 07 Agent loop | `07_agent_loop/` | ReAct loop, plan-and-execute, termination guards, repeated-action detection, token budget guard |
| 08 Multi-agent | `08_multi_agent/` | Pipeline, orchestrator-worker, peer-to-peer, shared state vs message bus, fact verification, coordination storm detection |
| 09 Architecture | `09_architecture/` | 7-layer model, fault lookup table, framework mapping, sample system breakdown |
| 10 Handwritten agent | `10_handwritten_agent/` | A compact framework-free agent plus a persisted/streaming-friendly variant |

## Notes

The examples use deterministic mock model behavior so tests stay stable. In a
real integration, replace the mock planner/model adapters with SDK calls while
keeping the guards, state shape, logging, and tool dispatch boundaries.
