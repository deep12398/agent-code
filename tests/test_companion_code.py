from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_code.agent_loop import AgentAction, AgentDecision, ScriptedPlanner, TerminationGuard, rewrite_agent_executor
from agent_code.handwritten_agent import HandwrittenAgent, PersistentHandwrittenAgent
from agent_code.llm_call import (
    ChatMessage,
    LLMRequest,
    StreamAssembler,
    UsageRecord,
    append_usage_record,
    retry_with_backoff,
    sample_next_token,
    trim_messages_to_budget,
)
from agent_code.memory_essence import LayeredMemory, WindowMemory, extract_candidate_facts
from agent_code.multi_agent import CoordinationStormDetector, OrchestratorWorker, demo_agents
from agent_code.prompt_essence import (
    PromptRegistry,
    PromptTemplate,
    classify_ticket_as_tool_call,
    dumps_strict_json,
    parse_natural_language_classifier,
    validate_json_schema,
)
from agent_code.rag_essence import Document, HybridSearch, RAGEvaluator, Reranker, build_finetune_pairs, chunk_text
from agent_code.seven_layer import FRAMEWORK_MAPPING, diagnose
from agent_code.tool_essence import (
    FunctionCallingDemo,
    IdempotencyStore,
    MinimalMCPServer,
    ToolCall,
    default_registry,
    execute_tool_calls_parallel,
    idempotent_refund_demo,
    safe_calculate,
)


class LLMCallTests(unittest.TestCase):
    def test_payload_stream_usage_and_retry(self) -> None:
        request = LLMRequest("model-a", [ChatMessage("user", "hi")], stream=True)
        self.assertTrue(request.payload()["stream_options"]["include_usage"])

        assembler = StreamAssembler()
        assembler.push_event({"choices": [{"delta": {"content": "hel"}}]})
        assembler.push_event({"choices": [{"delta": {"content": "lo"}}], "usage": {"total_tokens": 5}})
        self.assertEqual(assembler.text(), "hello")
        self.assertEqual(assembler.usage, {"total_tokens": 5})

        attempts = {"count": 0}

        def flaky() -> str:
            attempts["count"] += 1
            if attempts["count"] < 2:
                raise TimeoutError("temporary")
            return "ok"

        self.assertEqual(retry_with_backoff(flaky, sleep=lambda _: None), "ok")

    def test_usage_and_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "usage.jsonl"
            append_usage_record(path, UsageRecord("req-1", "m", 3, 4))
            self.assertEqual(json.loads(path.read_text().strip())["total_tokens"], 7)

        messages = [ChatMessage("system", "rules"), ChatMessage("user", "one " * 100), ChatMessage("assistant", "short")]
        trimmed = trim_messages_to_budget(messages, 40)
        self.assertEqual(trimmed[0].role, "system")
        self.assertEqual(sample_next_token({"a": 0.9, "b": 0.1}, temperature=0), "a")


class PromptTests(unittest.TestCase):
    def test_prompt_registry_and_schema(self) -> None:
        registry = PromptRegistry()
        registry.add(PromptTemplate("router", "v1", "Route {ticket}", ("ticket",)))
        self.assertEqual(registry.get("router").render(ticket="vpn"), "Route vpn")

        call = classify_ticket_as_tool_call("urgent vpn problem")
        self.assertEqual(call["arguments"]["category"], "it")
        schema = {
            "type": "object",
            "properties": {"category": {"type": "string"}, "urgency": {"type": "string"}},
            "required": ["category", "urgency"],
        }
        self.assertEqual(validate_json_schema(call["arguments"], schema), [])
        self.assertIn("category", dumps_strict_json(call["arguments"], schema))
        self.assertEqual(parse_natural_language_classifier("category: HR\nurgency: normal")["category"], "hr")


class MemoryTests(unittest.TestCase):
    def test_layered_memory_and_fact_extraction(self) -> None:
        memory = LayeredMemory("u1")
        memory.remember_fact("The user prefers Python examples.")
        memory.add_turn("I need RAG help", "Use retrieval first.")
        context = memory.build_context("Python RAG", max_tokens=120)
        self.assertTrue(any("Python" in message.content for message in context))

        window = WindowMemory(window_size=1)
        window.add("user", "first")
        window.add("assistant", "second")
        self.assertEqual([message.content for message in window.context()], ["second"])
        self.assertEqual(extract_candidate_facts("I prefer short answers. Random sentence."), ["I prefer short answers"])


class RAGTests(unittest.TestCase):
    def test_hybrid_rag_rerank_and_eval(self) -> None:
        docs = [
            Document("rag", "RAG retrieves context before generation.", {"title": "RAG"}),
            Document("tool", "Tools execute external actions.", {"title": "Tools"}),
        ]
        chunks = chunk_text("one two three four five", chunk_size=3, overlap=1)
        self.assertEqual(chunks, ["one two three", "three four five", "five"])

        results = HybridSearch(docs).search("How does RAG retrieve context?")
        self.assertEqual(results[0].document.id, "rag")
        reranked = Reranker().rerank("RAG retrieves context", results)
        metrics = RAGEvaluator().evaluate("RAG context", "RAG retrieves context.", [reranked[0].document])
        self.assertGreater(metrics["faithfulness"], 0.5)
        self.assertEqual(build_finetune_pairs(docs)[0]["document_id"], "rag")


class ToolTests(unittest.TestCase):
    def test_tool_calling_mcp_parallel_and_idempotency(self) -> None:
        registry = default_registry()
        self.assertEqual(safe_calculate("(3+5)*2"), 16.0)

        demo_result = FunctionCallingDemo(registry).run("what is (3+5)*2?")
        self.assertEqual(demo_result["tool_result"], 16.0)

        calls = [ToolCall("a", "calculate", {"expression": "1+1"}), ToolCall("b", "get_weather", {"city": "Dubai"})]
        parallel = execute_tool_calls_parallel(registry, calls)
        self.assertTrue(all(item["ok"] for item in parallel))

        mcp = MinimalMCPServer(registry)
        self.assertIn("tools", mcp.handle({"id": 1, "method": "tools/list"})["result"])

        store = IdempotencyStore()
        args = {"order_id": "o1", "amount": 10, "idempotency_key": "k1"}
        first = idempotent_refund_demo(store, registry, args)
        second = idempotent_refund_demo(store, registry, args)
        self.assertEqual(first, second)


class AgentLoopTests(unittest.TestCase):
    def test_react_agent_finishes_and_guard_stops_repeats(self) -> None:
        registry = default_registry()
        agent = rewrite_agent_executor(
            registry,
            ScriptedPlanner([AgentDecision("math", AgentAction("calculate", {"expression": "2+2"}))]),
        )
        result = agent.run("calculate")
        self.assertEqual(result["status"], "finished")
        self.assertEqual(result["answer"], "4.0")

        guard = TerminationGuard(repeated_action_limit=2)
        trace = [{"action_signature": "x"}, {"action_signature": "x"}]
        self.assertEqual(guard.check(trace), "repeated_action")


class MultiAgentAndArchitectureTests(unittest.TestCase):
    def test_multi_agent_and_architecture_lookup(self) -> None:
        workers = OrchestratorWorker(demo_agents())
        output = workers.run("research and verify this design")
        self.assertIn("orchestrator", output)
        storm = CoordinationStormDetector(max_questions_per_agent=1)
        self.assertEqual(storm.detect([]), [])
        self.assertIn("LangGraph", FRAMEWORK_MAPPING)
        self.assertEqual(diagnose("private data leaked")[0]["layer"], "L6")


class HandwrittenAgentTests(unittest.TestCase):
    def test_handwritten_agent_and_persistence(self) -> None:
        self.assertEqual(HandwrittenAgent().run("calculate (6*7)")["answer"], "42.0")
        with tempfile.TemporaryDirectory() as tmp:
            agent = PersistentHandwrittenAgent(state_path=Path(tmp) / "state.json")
            chunks = list(agent.stream_answer("weather in Dubai"))
            self.assertTrue(chunks)
            self.assertTrue(agent.state_path.exists())


if __name__ == "__main__":
    unittest.main()
