from __future__ import annotations

import ast
import json
import math
import operator
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable


ToolHandler = Callable[[dict[str, Any]], Any]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    requires_approval: bool = False

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        if tool.name in self.tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self.tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self.tools.values()]

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        if name not in self.tools:
            raise KeyError(f"unknown tool: {name}")
        return self.tools[name].handler(arguments)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class IdempotencyStore:
    results: dict[str, Any] = field(default_factory=dict)

    def run_once(self, key: str, operation: Callable[[], Any]) -> Any:
        if key in self.results:
            return self.results[key]
        result = operation()
        self.results[key] = result
        return result


class ErrorTranslator:
    def translate(self, error: Exception) -> dict[str, str]:
        if isinstance(error, TimeoutError):
            return {"type": "temporary", "message": "tool timed out; retry later or ask for a smaller task"}
        if isinstance(error, PermissionError):
            return {"type": "permission", "message": "tool call requires approval or stronger permission"}
        if isinstance(error, KeyError):
            return {"type": "bad_tool", "message": str(error)}
        return {"type": "unknown", "message": str(error)}


def execute_tool_calls_parallel(registry: ToolRegistry, calls: list[ToolCall]) -> list[dict[str, Any]]:
    def run(call: ToolCall) -> dict[str, Any]:
        try:
            return {"tool_call_id": call.id, "ok": True, "result": registry.call(call.name, call.arguments)}
        except Exception as error:  # noqa: BLE001 - translated for the model.
            return {"tool_call_id": call.id, "ok": False, "error": ErrorTranslator().translate(error)}

    with ThreadPoolExecutor(max_workers=max(1, len(calls))) as executor:
        return list(executor.map(run, calls))


class FunctionCallingDemo:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def first_model_turn(self, user_input: str) -> ToolCall:
        lowered = user_input.lower()
        if "weather" in lowered:
            city = "Dubai" if "dubai" in lowered else "Shanghai"
            return ToolCall("call_weather", "get_weather", {"city": city})
        return ToolCall("call_calc", "calculate", {"expression": extract_math_expression(user_input)})

    def second_model_turn(self, user_input: str, tool_result: Any) -> str:
        return f"For '{user_input}', the tool result is: {tool_result}"

    def run(self, user_input: str) -> dict[str, Any]:
        tool_call = self.first_model_turn(user_input)
        tool_result = self.registry.call(tool_call.name, tool_call.arguments)
        return {
            "tool_call": tool_call,
            "tool_result": tool_result,
            "answer": self.second_model_turn(user_input, tool_result),
        }


@dataclass(frozen=True)
class Skill:
    name: str
    instructions: str
    tools: tuple[str, ...]
    resources: dict[str, str] = field(default_factory=dict)

    def build_context(self, registry: ToolRegistry) -> dict[str, Any]:
        return {
            "skill": self.name,
            "instructions": self.instructions,
            "tool_schemas": [registry.tools[name].schema() for name in self.tools],
            "resources": self.resources,
        }


class MinimalMCPServer:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        request_id = request.get("id")
        method = request.get("method")
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": self.registry.schemas()}}
        if method == "tools/call":
            params = request.get("params", {})
            result = self.registry.call(params["name"], params.get("arguments", {}))
            return {"jsonrpc": "2.0", "id": request_id, "result": {"content": result}}
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": "method not found"}}


def default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="get_weather",
            description="Get current mocked weather for a city. Use only for weather questions.",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
            handler=lambda args: f"{args['city']}: 31C, clear",
        )
    )
    registry.register(
        Tool(
            name="calculate",
            description="Evaluate a safe arithmetic expression. Do not use for non-math questions.",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
            handler=lambda args: safe_calculate(args["expression"]),
        )
    )
    registry.register(
        Tool(
            name="create_refund",
            description="Create a refund request. Requires idempotency_key because it has side effects.",
            parameters={
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "amount": {"type": "number"},
                    "idempotency_key": {"type": "string"},
                },
                "required": ["order_id", "amount", "idempotency_key"],
            },
            handler=lambda args: {"refund_id": f"refund_{args['order_id']}", "amount": args["amount"]},
            requires_approval=True,
        )
    )
    return registry


def safe_calculate(expression: str) -> float:
    allowed_names = {"sqrt": math.sqrt, "pow": pow, "abs": abs, "round": round}
    allowed_binary = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
    }
    allowed_unary = {ast.UAdd: operator.pos, ast.USub: operator.neg}

    def eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_binary:
            return allowed_binary[type(node.op)](eval_node(node.left), eval_node(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_unary:
            return allowed_unary[type(node.op)](eval_node(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in allowed_names:
            return float(allowed_names[node.func.id](*(eval_node(arg) for arg in node.args)))
        raise ValueError(f"unsafe expression: {expression}")

    return eval_node(ast.parse(expression, mode="eval"))


def idempotent_refund_demo(store: IdempotencyStore, registry: ToolRegistry, args: dict[str, Any]) -> Any:
    key = args["idempotency_key"]
    return store.run_once(key, lambda: registry.call("create_refund", args | {"created_at": time.time()}))


def extract_math_expression(user_input: str) -> str:
    function_match = re.search(r"\b(?:sqrt|abs|round|pow)\s*\([^)]*\)", user_input, re.IGNORECASE)
    if function_match:
        return function_match.group(0)

    candidates = re.findall(r"[0-9+\-*/().\s]+", user_input)
    expressions = [candidate.strip() for candidate in candidates if any(char.isdigit() for char in candidate)]
    if not expressions:
        return "0"
    return max(expressions, key=lambda item: sum(char.isdigit() or char in "+-*/()" for char in item)).rstrip(".")


def tool_result_message(call: ToolCall, result: Any) -> dict[str, str]:
    return {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, ensure_ascii=True)}
