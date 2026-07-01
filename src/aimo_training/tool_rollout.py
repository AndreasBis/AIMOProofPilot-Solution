from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from aimo_inference.client import AIMOInferenceClient
from aimo_inference.config import AIMOConfig
from aimo_inference.page_count import strip_code_blocks_and_tool_transcripts
from aimo_inference.prompts import AIMOPromptBuilder
from aimo_inference.sandbox import AIMOSandbox
from aimo_inference.template import AIMOChatTemplate
from aimo_inference.template import AIMOOLMoChatMLToolTemplate


@dataclass(frozen=True)
class AIMOToolRolloutResult:

    prompt: str
    prompt_ids: list[int]
    completion: str
    completion_ids: list[int]
    token_logprobs: list[float]
    env_mask: list[int]
    proof_text: str
    input_tokens: int | None
    output_tokens: int | None
    tool_tokens: int
    finish_reason: str
    python_calls: int
    python_errors: int
    timeout_events: int
    latency_seconds: float
    raw_generations: list[dict[str, Any]]


class AIMOToolRolloutEngine:

    python_block_pattern = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)

    def __init__(
        self,
        config: AIMOConfig,
        client: AIMOInferenceClient,
        sandbox: AIMOSandbox,
        prompt_builder: AIMOPromptBuilder | None = None,
        chat_template: AIMOChatTemplate | None = None,
    ) -> None:

        self.config = config
        self.client = client
        self.sandbox = sandbox
        self.prompt_builder = prompt_builder or AIMOPromptBuilder()
        self.chat_template = chat_template or AIMOChatTemplate()
        self.olmo_tool_template = AIMOOLMoChatMLToolTemplate()
        self._tokenizer: Any | None = None

    def run_problem(
        self,
        problem_text: str,
        seed: int | None = None,
    ) -> AIMOToolRolloutResult:

        messages = self.prompt_builder.build_first_pass_messages(
            problem_text=problem_text,
            enable_tools=self.config.enable_tools,
        )
        prompt = self._render_prompt(messages)
        prompt_ids = self._encode_text(prompt)
        completion_ids: list[int] = []
        token_logprobs: list[float] = []
        env_mask: list[int] = []
        model_text_parts: list[str] = []
        raw_generations: list[dict[str, Any]] = []
        output_tokens = 0
        tool_tokens = 0
        python_calls = 0
        python_errors = 0
        timeout_events = 0
        latency_seconds = 0.0
        finish_reason = ""

        for _ in range(self.config.max_python_calls + 1):
            generation = self.client.complete_token_ids(
                prompt_token_ids=prompt_ids + completion_ids,
                max_tokens=self.config.max_tokens_for_pass(self.config.max_new_tokens),
                seed=seed,
            )
            raw_generations.append(generation.raw)
            latency_seconds += generation.latency_seconds
            finish_reason = generation.finish_reason

            if not generation.token_ids:
                finish_reason = "missing_token_ids"
                fallback_token_ids = self._encode_text("No proof was produced.")
                completion_ids.extend(fallback_token_ids)
                token_logprobs.extend([
                    0.0
                    for _ in fallback_token_ids
                ])
                env_mask.extend([
                    0
                    for _ in fallback_token_ids
                ])
                output_tokens += len(fallback_token_ids)
                break

            if len(generation.token_logprobs) != len(generation.token_ids):
                completion_ids.extend(generation.token_ids)
                token_logprobs.extend([
                    0.0
                    for _ in generation.token_ids
                ])
                env_mask.extend([
                    0
                    for _ in generation.token_ids
                ])
                output_tokens += generation.output_tokens or len(generation.token_ids)
                finish_reason = "missing_token_logprobs"
                break

            completion_ids.extend(generation.token_ids)
            token_logprobs.extend(generation.token_logprobs)
            env_mask.extend([
                1
                for _ in generation.token_ids
            ])
            output_tokens += generation.output_tokens or len(generation.token_ids)
            model_text_parts.append(generation.text)

            tool_context = self._run_tool_calls(generation.text)

            if not tool_context["token_ids"]:
                break

            python_calls += tool_context["python_calls"]
            python_errors += tool_context["python_errors"]
            timeout_events += tool_context["timeout_events"]
            completion_ids.extend(tool_context["token_ids"])
            token_logprobs.extend([
                0.0
                for _ in tool_context["token_ids"]
            ])
            env_mask.extend([
                0
                for _ in tool_context["token_ids"]
            ])
            tool_tokens += len(tool_context["token_ids"])

            if finish_reason in {"length", "max_tokens", "content_filter"}:
                break

        completion = self._strip_tool_calls("".join(model_text_parts).strip())
        proof_text = strip_code_blocks_and_tool_transcripts(
            completion or "No proof was produced.",
            preserve_blank_lines=True,
        )

        return AIMOToolRolloutResult(
            prompt=prompt,
            prompt_ids=prompt_ids,
            completion=completion or "No proof was produced.",
            completion_ids=completion_ids,
            token_logprobs=token_logprobs,
            env_mask=env_mask,
            proof_text=proof_text,
            input_tokens=len(prompt_ids),
            output_tokens=output_tokens,
            tool_tokens=tool_tokens,
            finish_reason=finish_reason,
            python_calls=python_calls,
            python_errors=python_errors,
            timeout_events=timeout_events,
            latency_seconds=latency_seconds,
            raw_generations=raw_generations,
        )

    def _python_blocks(self, text: str) -> list[str]:

        if not self.config.enable_tools:
            return []

        return [
            match.group(1).strip()
            for match in self.python_block_pattern.finditer(text)
            if match.group(1).strip()
        ]

    def _render_prompt(self, messages: list[object]) -> str:

        if self.config.tool_protocol == "olmo_chatml":
            return self.olmo_tool_template.render_initial_prompt(
                messages=messages,
                tool_prompt=self.config.harmony_tool_prompt,
            )

        return self.chat_template.render(
            messages=messages,
            add_generation_prompt=True,
        )

    def _run_tool_calls(self, text: str) -> dict[str, int | list[int]]:

        if self.config.tool_protocol == "olmo_chatml":
            return self._run_olmo_tool_calls(text)

        return self._run_markdown_tool_calls(text)

    def _run_olmo_tool_calls(self, text: str) -> dict[str, int | list[int]]:

        tool_calls = self.olmo_tool_template.parse_tool_calls(text)
        output_parts = []
        python_calls = 0
        python_errors = 0
        timeout_events = 0

        for tool_call in tool_calls:
            if python_calls >= self.config.max_python_calls:
                break

            python_calls += 1
            execution = self.sandbox.execute(tool_call.code)
            output_parts.append(execution.to_tool_payload())

            if execution.timed_out:
                timeout_events += 1

            if not execution.success:
                python_errors += 1

        if not output_parts:
            return self._empty_tool_context()

        tool_text = (
            self.olmo_tool_template.render_environment_turn("\n\n".join(output_parts))
            + self.olmo_tool_template.assistant_generation_prompt()
        )

        return {
            "token_ids": self._encode_text(tool_text),
            "python_calls": python_calls,
            "python_errors": python_errors,
            "timeout_events": timeout_events,
        }

    def _run_markdown_tool_calls(self, text: str) -> dict[str, int | list[int]]:

        code_blocks = self._python_blocks(text)
        tool_token_ids: list[int] = []
        python_calls = 0
        python_errors = 0
        timeout_events = 0

        for code_block in code_blocks:
            if python_calls >= self.config.max_python_calls:
                break

            python_calls += 1
            execution = self.sandbox.execute(code_block)

            if execution.timed_out:
                timeout_events += 1

            if not execution.success:
                python_errors += 1

            tool_text = f"\nTool output:\n{execution.to_tool_payload()}\n"
            tool_token_ids.extend(self._encode_text(tool_text))

        if not tool_token_ids:
            return self._empty_tool_context()

        return {
            "token_ids": tool_token_ids,
            "python_calls": python_calls,
            "python_errors": python_errors,
            "timeout_events": timeout_events,
        }

    def _empty_tool_context(self) -> dict[str, int | list[int]]:

        return {
            "token_ids": [],
            "python_calls": 0,
            "python_errors": 0,
            "timeout_events": 0,
        }

    def _strip_tool_calls(self, text: str) -> str:

        if self.config.tool_protocol == "olmo_chatml":
            return self.olmo_tool_template.strip_tool_calls(text)

        return text

    def _encode_text(self, text: str) -> list[int]:

        tokenizer = self._load_tokenizer()

        return [
            int(token_id)
            for token_id in tokenizer.encode(text, add_special_tokens=False)
        ]

    def _load_tokenizer(self) -> Any:

        if self._tokenizer is not None:
            return self._tokenizer

        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        return self._tokenizer
