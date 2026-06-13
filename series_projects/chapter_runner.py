from __future__ import annotations

import argparse
from typing import Any

from agent_examples.cli import print_json
from .rag_memory_project import run_rag_memory_stage


def run_chapter(chapter: int) -> dict[str, Any]:
    if 11 <= chapter <= 18:
        return run_rag_memory_stage(chapter)
    raise ValueError(f"chapter {chapter} is outside the supported range 11-18")


def main(chapter: int | None = None) -> None:
    if chapter is None:
        parser = argparse.ArgumentParser(description="Run a chapter stage from the continuous projects.")
        parser.add_argument("chapter", type=int)
        args = parser.parse_args()
        chapter = args.chapter
    print_json(run_chapter(chapter))


if __name__ == "__main__":
    main()
