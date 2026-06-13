from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


DEMO_DIRS = [
    "11_rag_diagnosis",
    "12_query_rewrite",
    "13_chunking",
    "14_rerank_hybrid",
    "15_graphrag",
    "16_memory_pipeline",
    "17_mem0",
    "18_layered_memory",
]


def test_all_chapter_demos_run() -> None:
    for directory in DEMO_DIRS:
        demo = ROOT / directory / "demo.py"
        assert demo.exists(), f"missing {demo}"
        result = subprocess.run(
            [sys.executable, str(demo)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, f"{directory} failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        assert result.stdout.strip(), f"{directory} produced no output"
