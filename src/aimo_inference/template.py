from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Mapping
from typing import Sequence


@dataclass(frozen=True)
class AIMOChatMessage:

    role: str
    content: str

    def as_dict(self) -> dict[str, str]:

        return {
            "role": self.role,
            "content": self.content,
        }


class AIMOChatTemplate:

    allowed_roles = {"system", "user", "assistant", "tool", "environment"}

    def render(
        self,
        messages: Sequence[AIMOChatMessage | Mapping[str, str]],
        add_generation_prompt: bool = False,
    ) -> str:

        normalized_messages = [
            self._normalize_message(message)
            for message in messages
        ]

        if not normalized_messages:
            raise ValueError("At least one chat message is required.")

        rendered_messages = [
            self._render_message(message)
            for message in normalized_messages
        ]

        if add_generation_prompt:
            rendered_messages.append("<|im_start|>assistant\n")

        return "".join(rendered_messages)

    def _normalize_message(
        self,
        message: AIMOChatMessage | Mapping[str, str],
    ) -> AIMOChatMessage:

        if isinstance(message, AIMOChatMessage):
            normalized_message = message
        else:
            normalized_message = AIMOChatMessage(
                role=str(message.get("role", "")),
                content=str(message.get("content", "")),
            )

        if normalized_message.role not in self.allowed_roles:
            raise ValueError(f"Invalid chat role: {normalized_message.role}")

        if not normalized_message.content.strip():
            raise ValueError("Chat message content cannot be empty.")

        return normalized_message

    def _render_message(self, message: AIMOChatMessage) -> str:

        content = self._escape_content(message.content)

        return f"<|im_start|>{message.role}\n{content}<|im_end|>\n"

    def _escape_content(self, content: str) -> str:

        return (
            content.replace("<|im_start|>", "<|escaped_im_start|>")
            .replace("<|im_end|>", "<|escaped_im_end|>")
        )


@dataclass(frozen=True)
class AIMOChatMLToolCall:

    name: str
    code: str


class AIMOOLMoChatMLToolTemplate:

    function_call_pattern = re.compile(r"<function_calls>(.*?)</function_calls>", re.DOTALL)

    def render_initial_prompt(
        self,
        messages: Sequence[AIMOChatMessage],
        tool_prompt: str,
    ) -> str:

        normalized_messages = [
            self._normalize_message(message)
            for message in messages
        ]
        rendered_messages = []

        for message in normalized_messages:
            if message.role == "system":
                rendered_messages.append(
                    self._render_system_message(
                        content=message.content,
                        tool_prompt=tool_prompt,
                    )
                )
            elif message.role == "user":
                rendered_messages.append(self._render_basic_message(message))
            else:
                rendered_messages.append(self._render_basic_message(message))

        rendered_messages.append(self.assistant_generation_prompt())

        return "".join(rendered_messages)

    def assistant_generation_prompt(self) -> str:

        return "<|im_start|>assistant\n<think>"

    def render_environment_turn(self, content: str) -> str:

        return f"<|im_start|>environment\n{self._escape_content(content)}<|im_end|>\n"

    def parse_tool_calls(self, text: str) -> list[AIMOChatMLToolCall]:

        calls: list[AIMOChatMLToolCall] = []

        for match in self.function_call_pattern.finditer(text):
            call_text = match.group(1).strip()
            json_calls = self._parse_json_calls(call_text)

            if json_calls:
                calls.extend(json_calls)

                continue

            for line in call_text.splitlines():
                stripped_line = line.strip()

                if not stripped_line:
                    continue

                call = self._parse_python_call(stripped_line)

                if call is not None:
                    calls.append(call)

        return calls

    def _parse_json_calls(self, text: str) -> list[AIMOChatMLToolCall]:

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return []

        items = payload if isinstance(payload, list) else [payload]
        calls = []

        for item in items:
            if not isinstance(item, dict):
                continue

            name = str(item.get("name") or item.get("function") or "")
            arguments = item.get("arguments", {})

            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {
                        "code": arguments,
                    }

            if name != "python" or not isinstance(arguments, dict):
                continue

            code = str(arguments.get("code", "")).strip()

            if code:
                calls.append(
                    AIMOChatMLToolCall(
                        name="python",
                        code=code,
                    )
                )

        return calls

    def function_schema(self, tool_prompt: str) -> str:

        return json.dumps(
            [
                {
                    "name": "python",
                    "description": tool_prompt,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to execute.",
                            },
                        },
                        "required": [
                            "code",
                        ],
                    },
                },
            ],
            ensure_ascii=False,
            separators=(
                ",",
                ":",
            ),
        )

    def strip_tool_calls(self, text: str) -> str:

        return self.function_call_pattern.sub("", text).strip()

    def _parse_python_call(self, text: str) -> AIMOChatMLToolCall | None:

        try:
            expression = ast.parse(text, mode="eval").body
        except SyntaxError:
            return None

        if not isinstance(expression, ast.Call):
            return None

        if not isinstance(expression.func, ast.Name) or expression.func.id != "python":
            return None

        code = self._code_from_call(expression)

        if not code:
            return None

        return AIMOChatMLToolCall(
            name="python",
            code=code,
        )

    def _code_from_call(self, expression: ast.Call) -> str:

        for keyword in expression.keywords:
            if keyword.arg == "code":
                try:
                    value = ast.literal_eval(keyword.value)
                except (ValueError, TypeError):
                    return ""

                return str(value).strip()

        if expression.args:
            try:
                value = ast.literal_eval(expression.args[0])
            except (ValueError, TypeError):
                return ""

            return str(value).strip()

        return ""

    def _render_system_message(
        self,
        content: str,
        tool_prompt: str,
    ) -> str:

        return (
            "<|im_start|>system\n"
            f"{self._escape_content(content)} "
            "You are provided with function signatures within <functions></functions> "
            "XML tags. You may call Python to assist with the problem. "
            "Output Python calls only inside <function_calls></function_calls> "
            "XML tags using exactly python(code=\"...\"). "
            f"<functions>{self.function_schema(tool_prompt)}</functions><|im_end|>\n"
        )

    def _render_basic_message(self, message: AIMOChatMessage) -> str:

        return f"<|im_start|>{message.role}\n{self._escape_content(message.content)}<|im_end|>\n"

    def _normalize_message(self, message: AIMOChatMessage) -> AIMOChatMessage:

        if message.role not in AIMOChatTemplate.allowed_roles:
            raise ValueError(f"Invalid chat role: {message.role}")

        if not message.content.strip():
            raise ValueError("Chat message content cannot be empty.")

        return message

    def _escape_content(self, content: str) -> str:

        return (
            content.replace("<|im_start|>", "<|escaped_im_start|>")
            .replace("<|im_end|>", "<|escaped_im_end|>")
        )


@dataclass(frozen=True)
class AIMOHarmonyRenderedPrompt:

    conversation: object
    prompt_token_ids: list[int]
    stop_token_ids: list[int]
    input_tokens: int


class AIMOHarmonyTemplate:

    def __init__(self, tool_prompt: str) -> None:

        self.tool_prompt = tool_prompt
        self._encoding = None

    @property
    def encoding(self) -> object:

        if self._encoding is None:
            try:
                from openai_harmony import HarmonyEncodingName
                from openai_harmony import load_harmony_encoding

                self._encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
            except Exception as error:
                raise RuntimeError(
                    "openai_harmony is required for GPT-OSS Harmony mode."
                ) from error

        return self._encoding

    def conversation_from_chat_messages(
        self,
        messages: Sequence[AIMOChatMessage],
        tool: object,
    ) -> object:

        try:
            from openai_harmony import Conversation
            from openai_harmony import Message
            from openai_harmony import ReasoningEffort
            from openai_harmony import Role
            from openai_harmony import SystemContent
        except Exception as error:
            raise RuntimeError(
                "openai_harmony is required for GPT-OSS Harmony mode."
            ) from error

        system_prompt = self._system_prompt(messages)
        system_content = (
            SystemContent.new()
            .with_model_identity(system_prompt)
            .with_reasoning_effort(ReasoningEffort.HIGH)
            .with_tools(tool.harmony_tool_config())
        )
        harmony_messages = [
            Message.from_role_and_content(Role.SYSTEM, system_content),
        ]

        for message in messages:
            if message.role == "system":
                continue

            harmony_messages.append(
                Message.from_role_and_content(
                    self._role_from_text(message.role),
                    message.content,
                )
            )

        return Conversation.from_messages(harmony_messages)

    def render_for_completion(
        self,
        messages: Sequence[AIMOChatMessage],
        tool: object,
    ) -> AIMOHarmonyRenderedPrompt:

        try:
            from openai_harmony import Role
        except Exception as error:
            raise RuntimeError(
                "openai_harmony is required for GPT-OSS Harmony mode."
            ) from error

        conversation = self.conversation_from_chat_messages(
            messages=messages,
            tool=tool,
        )
        prompt_token_ids = self.encoding.render_conversation_for_completion(
            conversation,
            Role.ASSISTANT,
        )
        stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()

        return AIMOHarmonyRenderedPrompt(
            conversation=conversation,
            prompt_token_ids=prompt_token_ids,
            stop_token_ids=stop_token_ids,
            input_tokens=len(prompt_token_ids),
        )

    def parse_completion(self, token_ids: list[int]) -> list[object]:

        try:
            from openai_harmony import Role
        except Exception as error:
            raise RuntimeError(
                "openai_harmony is required for GPT-OSS Harmony mode."
            ) from error

        if not token_ids:
            return []

        return self.encoding.parse_messages_from_completion_tokens(
            token_ids,
            Role.ASSISTANT,
            strict=False,
        )

    def text_from_message(self, message: object) -> str:

        content = getattr(message, "content", [])

        if not content:
            return ""

        return str(getattr(content[0], "text", ""))

    def text_from_messages(self, messages: list[object]) -> str:

        return "\n".join(
            text
            for text in [
                self.text_from_message(message)
                for message in messages
            ]
            if text
        ).strip()

    def is_final_message(self, message: object) -> bool:

        return getattr(message, "channel", None) == "final"

    def is_python_tool_call(self, message: object) -> bool:

        return getattr(message, "recipient", None) == "python"

    def encode_text(self, text: str, allow_special: bool = False) -> list[int]:

        if allow_special:
            return self.encoding.encode(text, allowed_special="all")

        return self.encoding.encode(text, disallowed_special=())

    def _system_prompt(self, messages: Sequence[AIMOChatMessage]) -> str:

        system_prompts = [
            message.content
            for message in messages
            if message.role == "system"
        ]

        if system_prompts:
            return "\n\n".join(system_prompts)

        raise ValueError("Harmony messages require a system prompt.")

    def _role_from_text(self, role: str) -> object:

        try:
            from openai_harmony import Role
        except Exception as error:
            raise RuntimeError(
                "openai_harmony is required for GPT-OSS Harmony mode."
            ) from error

        if role == "developer":
            return getattr(Role, "DEVELOPER", Role.USER)

        role_map = {
            "assistant": Role.ASSISTANT,
            "tool": Role.TOOL,
            "user": Role.USER,
        }

        return role_map.get(role, Role.USER)
