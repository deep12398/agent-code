from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
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


def main() -> int:
    for directory in DEMO_DIRS:
        demo = ROOT / directory / "demo.py"
        print(f"\n=== {directory} ===", flush=True)
        result = subprocess.run([sys.executable, str(demo)], cwd=ROOT, text=True)
        if result.returncode != 0:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
