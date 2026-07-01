from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol
from typing import Sequence

from aimo_inference.client import AIMOCompletionGeneration
from aimo_inference.config import AIMOConfig
from aimo_inference.sandbox import AIMOSandbox
from aimo_inference.template import AIMOChatMessage
from aimo_inference.template import AIMOHarmonyTemplate
from aimo_inference.tools import AIMOPythonTool


class AIMOCompletionClient(Protocol):

    def complete_token_ids(
        self,
        prompt_token_ids: list[int],
        max_tokens: int,
        stop_token_ids: list[int] | None = None,
        seed: int | None = None,
    ) -> AIMOCompletionGeneration:

        ...


@dataclass(frozen=True)
class AIMOHarmonyRunResult:

    text: str
    input_tokens: int
    output_tokens: int
    finish_reason: str
    latency_seconds: float
    python_calls: int
    python_errors: int
    tool_tokens: int
    timeout_events: int
    entropy: float
    raw: dict[str, object]


class AIMOHarmonyToolLoop:

    def __init__(
        self,
        config: AIMOConfig,
        client: AIMOCompletionClient,
        sandbox: AIMOSandbox,
        template: AIMOHarmonyTemplate | None = None,
    ) -> None:

        self.config = config
        self.client = client
        self.sandbox = sandbox
        self.template = template or AIMOHarmonyTemplate(tool_prompt=config.harmony_tool_prompt)

    def run(
        self,
        messages: Sequence[AIMOChatMessage],
        max_tokens: int,
        seed: int | None = None,
    ) -> AIMOHarmonyRunResult:

        tool = AIMOPythonTool(
            sandbox=self.sandbox,
            tool_prompt=self.config.harmony_tool_prompt,
        )
        rendered_prompt = self.template.render_for_completion(
            messages=messages,
            tool=tool,
        )
        conversation = rendered_prompt.conversation
        input_tokens = rendered_prompt.input_tokens
        output_tokens = 0
        tool_tokens = 0
        python_calls = 0
        python_errors = 0
        timeout_events = 0
        latency_seconds = 0.0
        entropy_values: list[float] = []
        final_text = ""
        finish_reason = ""
        raw_generations: list[dict[str, object]] = []

        for _ in range(self.config.max_python_calls + 1):
            prompt_token_ids = self.template.encoding.render_conversation_for_completion(
                conversation,
                self._assistant_role(),
            )
            available_tokens = max(
                0,
                self.config.available_generation_tokens(len(prompt_token_ids)),
            )
            requested_max_tokens = min(max_tokens, available_tokens)

            if requested_max_tokens <= 0:
                finish_reason = "context_exhausted"
                break

            generation = self.client.complete_token_ids(
                prompt_token_ids=prompt_token_ids,
                max_tokens=requested_max_tokens,
                stop_token_ids=rendered_prompt.stop_token_ids,
                seed=seed,
            )
            raw_generations.append(generation.raw)
            output_tokens += generation.output_tokens or len(generation.token_ids)
            latency_seconds += generation.latency_seconds
            finish_reason = generation.finish_reason

            if generation.entropy is not None:
                entropy_values.append(generation.entropy)

            new_messages = self._parse_generation(generation=generation)

            if not new_messages:
                final_text = generation.text.strip()
                break

            conversation.messages.extend(new_messages)
            last_message = new_messages[-1]

            if self.template.is_final_message(last_message):
                final_text = self.template.text_from_message(last_message).strip()
                break

            if self.template.is_python_tool_call(last_message):
                python_calls += 1
                response_message = tool.harmony_tool_response(last_message)
                response_text = self.template.text_from_message(response_message)
                tool_tokens += len(self.template.encode_text(response_text))

                if "timed out" in response_text.casefold():
                    timeout_events += 1

                if "error" in response_text.casefold() or "traceback" in response_text.casefold():
                    python_errors += 1

                conversation.messages.append(response_message)
                continue

            final_text = self.template.text_from_message(last_message).strip()
            break

        if not final_text:
            final_text = self.template.text_from_messages(conversation.messages)

        return AIMOHarmonyRunResult(
            text=final_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
            latency_seconds=latency_seconds,
            python_calls=python_calls,
            python_errors=python_errors,
            tool_tokens=tool_tokens,
            timeout_events=timeout_events,
            entropy=self._mean_entropy(entropy_values),
            raw={
                "generations": raw_generations,
                "prompt_hash": self._prompt_hash(messages),
            },
        )

    def _parse_generation(self, generation: AIMOCompletionGeneration) -> list[object]:

        if generation.token_ids:
            return self.template.parse_completion(generation.token_ids)

        encoded_text = self.template.encode_text(
            generation.text,
            allow_special=True,
        )

        if not encoded_text:
            return []

        return self.template.parse_completion(encoded_text)

    def _assistant_role(self) -> object:

        try:
            from openai_harmony import Role

            return Role.ASSISTANT
        except Exception as error:
            raise RuntimeError(
                "openai_harmony is required for GPT-OSS Harmony mode."
            ) from error

    def _mean_entropy(self, entropy_values: list[float]) -> float:

        if not entropy_values:
            return float("inf")

        return sum(entropy_values) / len(entropy_values)

    def _prompt_hash(self, messages: Sequence[AIMOChatMessage]) -> str:

        payload = "\n".join(
            f"{message.role}:{message.content}"
            for message in messages
        )

        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
