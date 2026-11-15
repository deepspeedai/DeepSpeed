# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Tools the model can call during multi-turn rollout, and parsers for the calls it emits."""

import asyncio
import concurrent.futures
import json
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch

import deepspeed.comm as dist


class Tool(ABC):
    """Subclasses set ``name``, ``description`` and ``parameters`` (a JSON schema) and implement ``execute``."""

    name: str
    description: str
    parameters: Dict[str, Any]

    def spec(self) -> Dict[str, Any]:
        """The tool in the OpenAI function format chat templates expect."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @abstractmethod
    async def execute(self, arguments: Dict[str, Any]) -> str:
        """Run one call; the result becomes the tool message's content."""


class ToolSet:
    """The tools of a rollout; prompts are rendered with ``specs``, so the model sees exactly these."""

    def __init__(self, tools: Sequence[Tool]):
        self.tools = {}
        for tool in tools:
            if tool.name in self.tools:
                raise ValueError(f"duplicate tool name {tool.name!r}")
            self.tools[tool.name] = tool

    @property
    def specs(self) -> List[Dict[str, Any]]:
        return [tool.spec() for tool in self.tools.values()]

    def get(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)

    async def run(self, call: "ToolCall") -> str:
        """Run one call. Errors are returned as text for the model, so a bad call never fails the rollout."""
        if call.error is not None:
            return f"Error: {call.error}"
        tool = self.get(call.name)
        if tool is None:
            return f"Error: unknown tool {call.name!r}; available tools: {', '.join(self.tools) or 'none'}"
        try:
            return str(await tool.execute(call.arguments))
        except Exception as error:  # noqa: BLE001 - any tool failure is reported to the model
            return f"Error: {type(error).__name__}: {error}"

    async def run_all(self, calls: Sequence["ToolCall"]) -> List[str]:
        """Run one turn's calls concurrently, and return their outputs in call order."""
        return list(await asyncio.gather(*(self.run(call) for call in calls)))

    def __len__(self) -> int:
        return len(self.tools)


class ToolQueue:
    """Runs the tool calls of finished turns in the background while the rollout goes on.

    ``submit`` starts a turn's calls on a background event loop and returns at once. ``pop_ready`` collects the turns
    whose calls have all finished, ``wait`` blocks until one has, and ``drain`` until all have.

    With AutoTP, the rollout passes a host-side ``group`` over the tensor-parallel group and its first rank ``src``:
    only ``src`` runs the tools, and ``pop_ready`` sends their output to the other ranks, so the whole group gets the
    same output at the same step. ``pop_ready`` is then a collective, so the ranks must call it at the same steps.
    """

    def __init__(self, tools: ToolSet, group=None, src=None):
        self.tools = tools
        self.tool_ms = 0.0
        # Trajectory -> (calls, future) while its calls run; the future is None on ranks that take the output from
        # their group's first rank.
        self.in_flight = {}
        self._group = group
        self._src = src
        self._runs_tools = group is None or dist.get_rank() == src
        self._loop = None
        self._thread = None

    def __enter__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="rollout-tools", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc_info):
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._loop.close()
        return False

    @property
    def busy(self):
        return bool(self.in_flight)

    def submit(self, i, calls: Sequence["ToolCall"]) -> None:
        """Start trajectory ``i``'s calls, and return without waiting for them."""
        future = None
        if self._runs_tools:
            future = asyncio.run_coroutine_threadsafe(self._timed(self.tools.run_all(calls)), self._loop)
        self.in_flight[i] = (calls, future)

    @staticmethod
    async def _timed(tool_calls):
        start = time.perf_counter()
        outputs = await tool_calls
        return outputs, (time.perf_counter() - start) * 1000.0

    def pop_ready(self):
        """The trajectories whose calls have all finished, as ``(i, calls, outputs)``; the rest keep running."""
        finished = []
        if self._runs_tools:
            for i in [i for i, (_, future) in self.in_flight.items() if future.done()]:
                outputs, tool_ms = self.in_flight[i][1].result()
                finished.append((i, outputs, tool_ms))
        if self._group is not None:
            finished = self._share(finished)
        ready = []
        for i, outputs, tool_ms in finished:
            calls, _ = self.in_flight.pop(i)
            self.tool_ms += tool_ms
            ready.append((i, calls, outputs))
        return ready

    def _share(self, finished):
        """The first rank's finished calls, on every rank of its group: a size, then the output as JSON bytes.

        Both go over a Gloo group on the host, so they take no device copies and do not queue behind the forward's
        collectives; steps where nothing finished send only the size.
        """
        src = self._src
        payload = json.dumps(finished).encode() if finished else b""
        size = torch.tensor([len(payload)], dtype=torch.long)
        dist.broadcast(size, src, group=self._group)
        if size.item() == 0:
            return []
        if self._runs_tools:
            buffer = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        else:
            buffer = torch.empty(size.item(), dtype=torch.uint8)
        dist.broadcast(buffer, src, group=self._group)
        return json.loads(buffer.numpy().tobytes())

    def drain(self):
        """Block until every running call finishes, and return them all as ``pop_ready`` does."""
        # The other ranks wait in pop_ready, for the first rank's output.
        if self._runs_tools:
            concurrent.futures.wait([future for _, future in self.in_flight.values()])
        return self.pop_ready()

    def wait(self):
        """Block until one of the running turns' calls have all finished."""
        # The other ranks wait in pop_ready, for the first rank's output.
        if not self._runs_tools:
            return
        concurrent.futures.wait([future for _, future in self.in_flight.values()],
                                return_when=concurrent.futures.FIRST_COMPLETED)


@dataclass
class ToolCall:
    """One call parsed from model output; ``error`` is set when it could not be parsed."""

    name: Optional[str]
    arguments: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


_TOOL_CALL_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def parse_hermes_tool_calls(text: str) -> List[ToolCall]:
    """Parse ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>`` blocks (Hermes, Qwen2.5, Qwen3).

    A block that cannot be parsed yields a ``ToolCall`` with ``error`` set, so the model learns what went wrong.
    """
    calls = []
    for match in _TOOL_CALL_BLOCK.finditer(text):
        try:
            payload = json.loads(match.group(1).strip())
        except json.JSONDecodeError as error:
            calls.append(ToolCall(name=None, error=f"the tool call is not valid JSON ({error.msg})"))
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
            calls.append(ToolCall(name=None, error='a tool call must be a JSON object with a string "name"'))
            continue

        name = payload["name"]
        arguments = payload.get("arguments", {})
        # Some models emit the arguments as a JSON-encoded string.
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as error:
                calls.append(ToolCall(name=name, error=f"the tool arguments are not valid JSON ({error.msg})"))
                continue
        if not isinstance(arguments, dict):
            calls.append(ToolCall(name=name, error='"arguments" must be a JSON object'))
            continue
        calls.append(ToolCall(name=name, arguments=arguments))

    # An opening tag without a closing one means the call was cut off.
    if text.count("<tool_call>") > text.count("</tool_call>"):
        calls.append(ToolCall(name=None, error="the <tool_call> block is not closed with </tool_call>"))
    return calls


_FUNCTION_BLOCK = re.compile(r"<function=([^>\n]*)>(.*?)</function>", re.DOTALL)
_PARAMETER_BLOCK = re.compile(r"<parameter=([^>\n]*)>(.*?)</parameter>", re.DOTALL)
_STRING_TYPES = ("string", "str", "text", "char", "varchar", "enum")


def _parameter_types(tools: Optional[ToolSet], name: str) -> Dict[str, str]:
    """The JSON schema type of each parameter of tool ``name``; empty if the tool is unknown."""
    tool = tools.get(name) if tools is not None else None
    if tool is None:
        return {}
    types = {}
    for key, schema in tool.parameters.get("properties", {}).items():
        param_type = schema.get("type", "string")
        # A nullable parameter lists its type next to "null", e.g. ["integer", "null"].
        if isinstance(param_type, list):
            param_type = next((t for t in param_type if t != "null"), "string")
        types[key] = param_type
    return types


def _convert_parameter(value: str, param_type: Optional[str]) -> Any:
    # ``null`` becomes ``None`` for any type, as in SGLang's Qwen3-Coder parser, so rollouts parse calls as serving does.
    if value.lower() == "null":
        return None
    if param_type is None or param_type in _STRING_TYPES:
        return value
    try:
        if param_type.startswith("int"):
            return int(value)
        if param_type.startswith("num") or param_type.startswith("float"):
            number = float(value)
            looks_integral = "." not in value and "e" not in value.lower()
            return int(number) if looks_integral and number.is_integer() else number
        if param_type in ("boolean", "bool"):
            if value.lower() not in ("true", "false"):
                return value
            return value.lower() == "true"
        # The template writes objects, arrays and any other type as JSON.
        return json.loads(value)
    except (ValueError, OverflowError):
        return value


def parse_qwen3_coder_tool_calls(text: str, tools: Optional[ToolSet] = None) -> List[ToolCall]:
    """Parse ``<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>`` blocks.

    This is the Qwen3-Coder and Qwen3.5 format. Values are converted to the types in ``tools``' parameter schemas;
    unparsable blocks yield a ``ToolCall`` with ``error`` set.
    """
    calls = []
    for match in _TOOL_CALL_BLOCK.finditer(text):
        block = match.group(1)
        functions = list(_FUNCTION_BLOCK.finditer(block))
        if not functions:
            if "<function=" in block:
                error = "the <function=...> block is not closed with </function>"
            else:
                error = "a tool call must contain a <function=NAME> block"
            calls.append(ToolCall(name=None, error=error))
            continue
        for function in functions:
            name = function.group(1).strip()
            body = function.group(2)
            if body.count("<parameter=") > body.count("</parameter>"):
                calls.append(ToolCall(name=name, error="a <parameter=...> block is not closed with </parameter>"))
                continue
            param_types = _parameter_types(tools, name)
            arguments = {}
            for parameter in _PARAMETER_BLOCK.finditer(body):
                key = parameter.group(1).strip()
                value = parameter.group(2)
                # The template puts each value on lines of its own.
                if value.startswith("\n"):
                    value = value[1:]
                if value.endswith("\n"):
                    value = value[:-1]
                arguments[key] = _convert_parameter(value, param_types.get(key))
            calls.append(ToolCall(name=name, arguments=arguments))

    # An opening tag without a closing one means the call was cut off.
    if text.count("<tool_call>") > text.count("</tool_call>"):
        calls.append(ToolCall(name=None, error="the <tool_call> block is not closed with </tool_call>"))
    return calls


# Tool-call formats by name, for ``RolloutConfig.tool_call_parser``.
TOOL_CALL_PARSERS = {
    "hermes": lambda text, tools: parse_hermes_tool_calls(text),
    "qwen3_coder": parse_qwen3_coder_tool_calls,
}
