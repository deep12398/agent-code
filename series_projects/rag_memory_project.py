from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from agent_examples.memory import Mem0LikeEngine, MemoryStore, build_demo_store, extract_facts
from agent_examples.rag import (
    InMemoryVectorStore,
    answer_from_context,
    bm25_like_search,
    build_chunks,
    context_precision,
    hyde_answer,
    multi_queries,
    query_rewrite,
    reciprocal_rank_fusion,
    rerank,
    sample_corpus,
)
from .practice_assets import catalog_summary, load_practice_knowledge_documents


@dataclass
class RunRecord:
    """A small audit record for each stage run.

    The project intentionally records intermediate decisions. In real systems
    this becomes trace data in Langfuse/LangSmith/OpenTelemetry.
    """

    stage: int
    name: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)


class EnterpriseRagMemoryProject:
    """Continuous project for chapters 11-18.

    The same document corpus flows through diagnosis, query processing,
    chunking, reranking, GraphRAG, and then into a user memory layer.  This
    keeps the RAG and Memory articles connected as one enterprise knowledge
    assistant instead of separate demos.
    """

    def __init__(self) -> None:
        # The first five documents are compact Chinese examples used by the
        # articles.  The practice documents come from llamaindex-learn so this
        # project also exercises the existing procurement knowledge base.
        self.documents = sample_corpus() + load_practice_knowledge_documents()
        self.practice_catalog = catalog_summary()
        self.fixed_chunks = build_chunks(self.documents, "fixed")
        self.structure_chunks = build_chunks(self.documents, "structure")
        self.store = InMemoryVectorStore(self.structure_chunks)
        self.memory_store = build_demo_store()
        self.run_log: list[RunRecord] = []

    def run_stage(self, chapter: int) -> dict[str, Any]:
        stages = {
            11: self.stage_11_diagnosis,
            12: self.stage_12_query_processing,
            13: self.stage_13_chunking,
            14: self.stage_14_rerank_hybrid,
            15: self.stage_15_graph_rag,
            16: self.stage_16_memory_pipeline,
            17: self.stage_17_mem0_engine,
            18: self.stage_18_layered_memory,
        }
        return stages[chapter]()

    def _record(self, stage: int, name: str, summary: str, details: dict[str, Any]) -> dict[str, Any]:
        self.run_log.append(RunRecord(stage, name, summary, details))
        return {
            "project": "enterprise_rag_memory",
            "chapter": stage,
            "stage_name": name,
            "summary": summary,
            "details": details,
            "run_log_size": len(self.run_log),
        }

    def stage_11_diagnosis(self) -> dict[str, Any]:
        query = "我去年入职，今年能休几天假？"
        naive_store = InMemoryVectorStore(self.fixed_chunks)
        naive = naive_store.search(query, k=5)
        advanced = self.store.search(query_rewrite(query), k=5)
        return self._record(
            11,
            "rag_diagnosis",
            "先复现朴素 RAG 的召回混杂，再用同一套知识库进入后续优化。",
            {
                "query": query,
                "naive_sources": [item.metadata["source_id"] for item in naive],
                "advanced_sources": [item.metadata["source_id"] for item in advanced],
                "naive_context_precision": context_precision(naive, "hr_leave"),
                "advanced_context_precision": context_precision(advanced, "hr_leave"),
                "bottlenecks": ["query_document_gap", "chunk_boundary_damage", "pseudo_relevance"],
                "practice_catalog_sample": self.practice_catalog[:2],
            },
        )

    def stage_12_query_processing(self) -> dict[str, Any]:
        query = "我们公司允许远程办公吗？"
        variants = multi_queries(query)
        batches = [self.store.search(variant, k=3) for variant in variants]
        fused = reciprocal_rank_fusion(batches, top_n=3)
        return self._record(
            12,
            "query_processing",
            "在第 11 章同一知识库上加 Query Rewrite / HyDE / Multi-Query。",
            {
                "original_query": query,
                "rewrite": query_rewrite(query),
                "hyde": hyde_answer(query),
                "multi_queries": variants,
                "practice_documents_loaded": [doc.metadata.get("title") for doc in self.documents if doc.id.startswith("practice_")],
                "fused_results": [
                    {"source": item.metadata["source_id"], "score": round(item.score, 4), "text": item.text}
                    for item in fused
                ],
            },
        )

    def stage_13_chunking(self) -> dict[str, Any]:
        strategies = {
            "fixed": build_chunks(self.documents, "fixed"),
            "sentence": build_chunks(self.documents, "sentence"),
            "structure": self.structure_chunks,
        }
        return self._record(
            13,
            "chunking_pipeline",
            "同一批文档通过三种切块策略进入索引，展示结构感知切块如何保留标题路径。",
            {
                name: {
                    "chunk_count": len(chunks),
                    "sample_metadata": chunks[0].metadata,
                    "sample_text": chunks[0].text,
                }
                for name, chunks in strategies.items()
            }
            | {"practice_catalog_sample": self.practice_catalog[:3]},
        )

    def stage_14_rerank_hybrid(self) -> dict[str, Any]:
        query = "这款新手机支持 5G 吗？"
        vector_results = self.store.search(query, k=5)
        keyword_results = bm25_like_search(self.structure_chunks, query, k=5)
        fused = reciprocal_rank_fusion([vector_results, keyword_results], top_n=5)
        ranked = rerank(query, fused, top_n=3)
        return self._record(
            14,
            "hybrid_rerank",
            "在同一索引上加入关键词召回、RRF 融合和 Cross-Encoder 风格重排。",
            {
                "query": query,
                "vector_top": [item.metadata["source_id"] for item in vector_results],
                "keyword_top": [item.metadata["source_id"] for item in keyword_results],
                "reranked_answer": answer_from_context(query, ranked),
                "reranked": [
                    {"source": item.metadata["source_id"], "score": round(item.score, 4)}
                    for item in ranked
                ],
            },
        )

    def stage_15_graph_rag(self) -> dict[str, Any]:
        entity_terms = ["员工", "年假", "远程办公", "经理", "报销", "5G", "电池", "高危工具"]
        graph: dict[str, set[str]] = {}
        for document in self.documents:
            entities = {term for term in entity_terms if term in document.text}
            for entity in entities:
                graph.setdefault(entity, set()).add(document.id)
                graph[entity].update(entities - {entity})
        query_entities = {"员工", "经理", "报销"}
        subgraph = {entity: sorted(graph.get(entity, set())) for entity in query_entities}
        return self._record(
            15,
            "graph_rag",
            "在同一语料上构建实体关系网络，补足向量相似度无法表达的跨文档关系。",
            {
                "query_entities": sorted(query_entities),
                "subgraph": subgraph,
                "serialized_context": " -> ".join(sorted({node for nodes in subgraph.values() for node in nodes})),
            },
        )

    def stage_16_memory_pipeline(self) -> dict[str, Any]:
        fresh_store = MemoryStore()
        operations: list[dict[str, str]] = []
        for utterance in ["我家在上海，刚搬到浦东。", "我是产品经理，喜欢简洁的回答。", "我最近出差去成都。"]:
            for fact in extract_facts("u1", utterance):
                operations.append({"operation": fresh_store.add_or_update(fact), "fact": fact.text})
        retrieved = fresh_store.retrieve("u1", "根据我的背景和偏好回答", k=5)
        return self._record(
            16,
            "memory_pipeline",
            "RAG 项目加入用户长期记忆层：抽取、存储、检索、注入、遗忘。",
            {
                "operations": operations,
                "retrieved_memory": [item.text for item in retrieved],
                "injection_blocks": ["long_term_facts", "short_term_messages", "rag_context"],
            },
        )

    def stage_17_mem0_engine(self) -> dict[str, Any]:
        store = MemoryStore()
        engine = Mem0LikeEngine(store)
        first = engine.decide_and_apply("u1", "我是产品经理，喜欢简洁的回答。")
        duplicate = engine.decide_and_apply("u1", "我是产品经理，喜欢简洁的回答。")
        new_fact = engine.decide_and_apply("u1", "我家在上海，刚搬到浦东。")
        return self._record(
            17,
            "mem0_operation_engine",
            "把长期记忆升级成 Operation Engine，显式区分 ADD / UPDATE / DELETE / NOOP。",
            {
                "first_write": [(op, fact.text) for op, fact in first],
                "duplicate_write": [(op, fact.text) for op, fact in duplicate],
                "new_fact": [(op, fact.text) for op, fact in new_fact],
                "active_facts": [fact.text for fact in store.profile("u1")],
            },
        )

    def stage_18_layered_memory(self) -> dict[str, Any]:
        profile = self.memory_store.retrieve("u1", "用户是谁", fact_type="semantic", k=10)
        episodic = self.memory_store.retrieve("u1", "最近发生了什么", fact_type="episodic", k=3)
        return self._record(
            18,
            "layered_memory",
            "Memory 子模块收束：Profile / Preference / Episodic / Working 分层治理。",
            {
                "layers": {
                    "profile": [item.text for item in profile],
                    "episodic": [item.text for item in episodic],
                    "working": ["当前对话 messages 滑窗"],
                    "preference": ["偏好事实永远优先于普通情节记忆"],
                },
                "async_extraction": ["debounce", "retry", "dead_letter", "confidence_staging"],
                "monitoring": ["memory_hit_rate", "low_confidence_ratio", "delete_latency", "drift_cases"],
            },
        )


def run_rag_memory_stage(chapter: int) -> dict[str, Any]:
    return EnterpriseRagMemoryProject().run_stage(chapter)
