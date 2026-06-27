from __future__ import annotations

import argparse
from typing import Any

from agent_examples.cli import print_json
from .interview_project import run_interview_stage
from .llmops_platform_project import run_llmops_stage
from .rag_memory_project import run_rag_memory_stage
from .sourcing_agent_project import run_sourcing_stage


def run_chapter(chapter: int) -> dict[str, Any]:
    if 11 <= chapter <= 18:
        return run_rag_memory_stage(chapter)
    if 19 <= chapter <= 33:
        return run_sourcing_stage(chapter)
    if 34 <= chapter <= 48:
        return run_llmops_stage(chapter)
    if 49 <= chapter <= 50:
        return run_interview_stage(chapter)
    raise ValueError(f"chapter {chapter} is outside the supported range 11-50")


def main(chapter: int | None = None) -> None:
    if chapter is None:
        parser = argparse.ArgumentParser(description="Run a chapter stage from the continuous projects.")
        parser.add_argument("chapter", type=int)
        args = parser.parse_args()
        chapter = args.chapter
    print_json(run_chapter(chapter))


if __name__ == "__main__":
    main()
