from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
# Single source of truth for the continuous-project chapter demos (11-48).
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
    "27_adk_tools",
    "28_adk_memory",
    "29_adk_eval_plugins",
    "30_deepagents_intro",
    "31_context_engineering",
    "32_three_framework_comparison",
    "33_framework_decision",
    "34_llm_gateway",
    "35_gateway_implementation",
    "36_eval_frameworks",
    "37_eval_pipeline",
    "38_observability_tools",
    "39_otel_self_hosted",
    "40_prompt_injection_defense",
    "41_rbac_jwt",
    "42_output_defense",
    "43_cost_optimization",
    "44_cache_routing",
    "45_performance_optimization",
    "46_backend_foundation",
    "47_full_stack_llmops",
    "48_harness_extension",
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
