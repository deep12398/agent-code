from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_code.rag_essence import Document, HybridSearch, RAGEvaluator, Reranker


def main() -> None:
    docs = [
        Document("d1", "RAG retrieves relevant context before asking the model to answer."),
        Document("d2", "Function calling lets the model choose a tool and arguments."),
    ]
    hybrid = HybridSearch(docs)
    results = Reranker().rerank("How does RAG answer?", hybrid.search("How does RAG answer?"))
    metrics = RAGEvaluator().evaluate("How does RAG answer?", "RAG retrieves relevant context.", [r.document for r in results])
    print(json.dumps({"results": [r.document.id for r in results], "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
