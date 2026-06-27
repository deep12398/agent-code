from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from series_projects.chapter_runner import run_chapter
from series_projects.llmops_platform_project import LLMOpsPlatformProject
from series_projects.sourcing_agent_project import SourcingAgentProject


class ChapterRangeTests(unittest.TestCase):
    def test_every_chapter_11_to_48_dispatches(self) -> None:
        for chapter in range(11, 49):
            result = run_chapter(chapter)
            self.assertEqual(result["chapter"], chapter)
            self.assertTrue(result["summary"], f"chapter {chapter} has empty summary")
            self.assertIn("details", result)

    def test_out_of_range_raises(self) -> None:
        with self.assertRaises(ValueError):
            run_chapter(99)
        with self.assertRaises(ValueError):
            SourcingAgentProject().run_stage(34)
        with self.assertRaises(ValueError):
            LLMOpsPlatformProject().run_stage(19)


class SourcingProjectTests(unittest.TestCase):
    def test_checkpoint_records_steps_and_hitl(self) -> None:
        result = SourcingAgentProject().run_stage(21)
        steps = result["details"]["checkpoint_steps"]
        self.assertTrue(steps, "checkpoint should record at least one step")
        # 询价路径会产生 needs_approval=True，HITL 必须挂起
        self.assertTrue(result["details"]["hitl"]["interrupted"])

    def test_langgraph_routes_quote(self) -> None:
        result = SourcingAgentProject().run_stage(19)
        self.assertTrue(result["details"]["final_state"]["needs_approval"])


class LLMOpsProjectTests(unittest.TestCase):
    def test_gateway_second_call_is_cached_and_free(self) -> None:
        result = LLMOpsPlatformProject().run_stage(34)
        self.assertFalse(result["details"]["first_call"]["cached"])
        self.assertTrue(result["details"]["second_call"]["cached"])
        self.assertEqual(result["details"]["second_call"]["cost"], 0.0)

    def test_semantic_cache_collapses_digits(self) -> None:
        details = LLMOpsPlatformProject().run_stage(44)["details"]
        # 仅数字不同的两个 query 应命中同一语义缓存键
        self.assertTrue(details["semantic_cache"]["a_b_hit_same_cache"])

    def test_compression_lowers_cost(self) -> None:
        details = LLMOpsPlatformProject().run_stage(43)["details"]
        self.assertLessEqual(details["compressed_cost"], details["raw_cost"])

    def test_injection_flags_direct_attack(self) -> None:
        details = LLMOpsPlatformProject().run_stage(40)["details"]
        self.assertEqual(details["direct_injection"]["risk"], "high")
        self.assertEqual(details["benign"]["risk"], "low")


if __name__ == "__main__":
    unittest.main()
