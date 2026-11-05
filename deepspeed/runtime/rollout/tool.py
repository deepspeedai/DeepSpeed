# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Tools the model can call during multi-turn rollout.

Users subclass ``Tool``: its spec tells the model what the tool does (it is
passed to ``tokenizer.apply_chat_template(tools=...)``) and ``execute`` runs
a call. ``ToolSet`` groups the tools of a rollout, and a parser from
``TOOL_CALL_PARSERS`` reads the calls a model emits, in the format its chat
template teaches.
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


class Tool(ABC):
    """A tool the model can call.

    Subclasses set ``name``, ``description`` and ``parameters`` (a JSON
    schema of the arguments) and implement ``execute``.
    """

    name: str
    description: str
    parameters: Dict[str, Any]

    def spec(self) -> Dict[str, Any]:
        """The tool in OpenAI function format, as chat templates expect it."""
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
        """Run one call. The returned string becomes the tool message's content."""


class ToolSet:
    """The tools available in a rollout, looked up by name.

    Prompts must be rendered with ``specs`` so the model is shown exactly the
    tools the rollout can run.
    """

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
        """Execute one call and return the text the model reads next.

        Parse errors, unknown tools and exceptions are reported as text, so
        one bad call never fails the rollout.
        """
        if call.error is not None:
            return f"Error: {call.error}"
        tool = self.get(call.name)
        if tool is None:
            return f"Error: unknown tool {call.name!r}; available tools: {', '.join(self.tools) or 'none'}"
        try:
            return str(await tool.execute(call.arguments))
        except Exception as error:  # noqa: BLE001 - any tool failure is reported to the model
            return f"Error: {type(error).__name__}: {error}"

    def __len__(self) -> int:
        return len(self.tools)


@dataclass
class ToolCall:
    """One call parsed from model output; ``error`` is set when it could not be parsed."""

    name: Optional[str]
    arguments: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


_TOOL_CALL_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def parse_hermes_tool_calls(text: str) -> List[ToolCall]:
    """Parse ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>`` blocks.

    This is the format the Hermes, Qwen2.5 and Qwen3 chat templates teach. A
    block that cannot be parsed still yields a ``ToolCall`` with ``error``
    set, so the model can be told what went wrong instead of the call
    silently disappearing.
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
    """Convert a parameter's text to its schema type; the text is kept when conversion fails.

    Mirrors SGLang's ``qwen3_coder`` parser. Parameters missing from the
    schema stay strings, so a string argument such as code is never
    reinterpreted as a number.
    """
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
            return value.lower() == "true"
        # object, array and anything else: the template wrote it as JSON.
        return json.loads(value)
    except (ValueError, OverflowError):
        return value


def parse_qwen3_coder_tool_calls(text: str, tools: Optional[ToolSet] = None) -> List[ToolCall]:
    """Parse ``<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>`` blocks.

    This is the format the Qwen3-Coder and Qwen3.5 chat templates teach (SGLang
    calls its parser ``qwen3_coder``). The template wraps every value in one
    leading and one trailing newline, which are removed. Values are converted
    to the types in the tool's parameter schema, so ``tools`` should be the
    rollout's tools. As with ``parse_hermes_tool_calls``, a block that cannot
    be parsed yields a ``ToolCall`` with ``error`` set.
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


# Tool-call formats by name, as in SGLang's --tool-call-parser.
TOOL_CALL_PARSERS = {
    "hermes": lambda text, tools: parse_hermes_tool_calls(text),
    "qwen3_coder": parse_qwen3_coder_tool_calls,
}
