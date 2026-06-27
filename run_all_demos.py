from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
# Single source of truth for the continuous-project chapter demos (11-50).
# tests/test_all_demos.py imports this list so the two never drift apart.
DEMO_DIRS = [
    "11_rag_diagnosis",
    "12_query_rewrite",
    "13_chunking",
    "14_rerank_hybrid",
    "15_graphrag",
    "16_memory_pipeline",
    "17_mem0",
    "18_layered_memory",
    "19_langgraph_intro",
    "20_react_plan_execute",
    "21_checkpoint_hitl_timetravel",
    "22_orchestrator_worker",
    "23_crewai_intro",
    "24_crewai_vs_langgraph",
    "25_adk_intro",
    "26_adk_workflows",
    "27_adk_tools_callbacks",
    "28_adk_skills_memory_artifacts",
    "29_adk_eval_plugins",
    "30_deepagents_harness",
    "31_context_engineering",
    "32_framework_benchmark",
    "33_selection_decision_tree",
    "34_llm_gateway",
    "35_litellm_portkey",
    "36_eval_frameworks",
    "37_eval_pipeline",
    "38_observability_tools",
    "39_opentelemetry",
    "40_prompt_injection",
    "41_rbac_multitenant",
    "42_hitl_guardrails",
    "43_token_cost_compression",
    "44_semantic_cache_routing",
    "45_concurrency_streaming",
    "46_backend_foundation",
    "47_llmops_platform",
    "48_harness_extension",
    "49_interview_principles",
    "50_interview_engineering",
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
