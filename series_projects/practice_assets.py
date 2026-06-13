from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_examples.rag import Document


PRACTICE_ROOT = Path("/Users/elias/code/practice-and-learning")
LLAMAINDEX_DATA = PRACTICE_ROOT / "llamaindex-learn" / "data"
LANGGRAPH_CAPSTONE = PRACTICE_ROOT / "langgraph-learn" / "phases" / "03-capstone"


def load_json_file(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_practice_catalog(limit: int = 8) -> list[dict[str, Any]]:
    """Load the sourcing catalog used by the existing LlamaIndex/LangGraph projects."""

    path = LLAMAINDEX_DATA / "catalog.json"
    if not path.exists():
        path = LANGGRAPH_CAPSTONE / "data" / "catalog.json"
    catalog = load_json_file(path)
    return catalog[:limit]


def load_practice_knowledge_documents() -> list[Document]:
    """Turn existing sourcing knowledge files into RAG documents.

    This is intentionally lightweight: the article project remains runnable
    without LlamaIndex, while still using the same source material as the
    existing practice project.
    """

    docs: list[Document] = []
    knowledge_dir = LLAMAINDEX_DATA / "knowledge"
    if not knowledge_dir.exists():
        return docs
    for path in sorted(knowledge_dir.glob("*.txt")):
        docs.append(
            Document(
                id=f"practice_{path.stem}",
                text=path.read_text(encoding="utf-8"),
                metadata={"category": "sourcing", "title": path.stem, "source": str(path)},
            )
        )
    return docs


def catalog_summary(limit: int = 5) -> list[dict[str, Any]]:
    return [
        {
            "id": item["id"],
            "name": item["name"],
            "category": item.get("category"),
            "moq": item.get("moq"),
            "lead_time_days": item.get("lead_time_days"),
            "price_range": item.get("price_range"),
        }
        for item in load_practice_catalog(limit)
    ]

