from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Sequence

from aimo_inference.config import AIMOConfig
from aimo_inference.template import AIMOChatMessage
from aimo_inference.template import AIMOChatTemplate


@dataclass(frozen=True)
class AIMOGeneration:

    text: str
    input_tokens: int | None
    output_tokens: int | None
    finish_reason: str
    latency_seconds: float
    raw: dict[str, Any]


@dataclass(frozen=True)
class AIMOCompletionGeneration:

    text: str
    token_ids: list[int]
    input_tokens: int | None
    output_tokens: int | None
    finish_reason: str
    latency_seconds: float
    entropy: float | None
    raw: dict[str, Any]
    token_logprobs: list[float] = field(default_factory=list)


class AIMOInferenceClient:

    def __init__(self, config: AIMOConfig) -> None:

        self.config = config
        self.chat_template = AIMOChatTemplate()
        self._tokenizer: Any | None = None
        self._tokenizer_load_failed = False

    def generate(
        self,
        messages: Sequence[AIMOChatMessage],
        max_tokens: int,
    ) -> AIMOGeneration:

        stored_messages = list(messages)
        resolved_max_tokens = self._resolve_max_tokens(
            messages=stored_messages,
            max_tokens=max_tokens,
        )
        payload = {
            "model": self.config.resolved_generation_model_name,
            "messages": [
                message.as_dict()
                for message in stored_messages
            ],
            **self.config.sampling_payload(max_tokens=resolved_max_tokens),
        }
        started_at = time.monotonic()
        response_payload = self._post_json(payload=payload)

        return self._normalize_response(
            response_payload=response_payload,
            latency_seconds=time.monotonic() - started_at,
        )

    def complete_token_ids(
        self,
        prompt_token_ids: list[int],
        max_tokens: int,
        stop_token_ids: list[int] | None = None,
        seed: int | None = None,
    ) -> AIMOCompletionGeneration:

        resolved_max_tokens = self._resolve_completion_max_tokens(
            prompt_token_count=len(prompt_token_ids),
            max_tokens=max_tokens,
        )
        payload: dict[str, Any] = {
            "model": self.config.resolved_generation_model_name,
            "prompt": prompt_token_ids,
            **self.config.sampling_payload(max_tokens=resolved_max_tokens),
            "return_token_ids": True,
        }
        payload.pop("top_logprobs", None)
        payload["logprobs"] = max(0, self.config.top_logprobs)

        if stop_token_ids:
            payload["stop_token_ids"] = stop_token_ids

        if seed is not None:
            payload["seed"] = seed

        started_at = time.monotonic()
        response_payload = self._post_json(
            payload=payload,
            endpoint="/completions",
        )

        return self._normalize_completion_response(
            response_payload=response_payload,
            prompt_token_count=len(prompt_token_ids),
            latency_seconds=time.monotonic() - started_at,
        )

    def _post_json(
        self,
        payload: dict[str, Any],
        endpoint: str = "/chat/completions",
    ) -> dict[str, Any]:

        request_data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.config.resolved_api_base}{endpoint}",
            data=request_data,
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_error = ""

        for attempt_index in range(3):
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=self.config.request_timeout_seconds,
                ) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                last_error = error.read().decode("utf-8", errors="replace")
            except (TimeoutError, urllib.error.URLError) as error:
                last_error = str(error)

            if attempt_index < 2:
                time.sleep(0.5 * (attempt_index + 1))

        raise RuntimeError(f"vLLM request failed: {last_error}")

    def _normalize_response(
        self,
        response_payload: dict[str, Any],
        latency_seconds: float,
    ) -> AIMOGeneration:

        choices = response_payload.get("choices", [])

        if not choices:
            raise RuntimeError("vLLM response did not include choices.")

        first_choice = choices[0]
        message = first_choice.get("message", {})
        usage = response_payload.get("usage", {})

        return AIMOGeneration(
            text=str(message.get("content", "")),
            input_tokens=self._optional_int(usage.get("prompt_tokens")),
            output_tokens=self._optional_int(usage.get("completion_tokens")),
            finish_reason=str(first_choice.get("finish_reason", "")),
            latency_seconds=latency_seconds,
            raw=response_payload,
        )

    def _normalize_completion_response(
        self,
        response_payload: dict[str, Any],
        prompt_token_count: int,
        latency_seconds: float,
    ) -> AIMOCompletionGeneration:

        choices = response_payload.get("choices", [])

        if not choices:
            raise RuntimeError("vLLM completion response did not include choices.")

        first_choice = choices[0]
        usage = response_payload.get("usage", {})
        text = str(first_choice.get("text", ""))
        token_ids = self._completion_token_ids(first_choice)
        token_logprobs = self._completion_token_logprobs(first_choice)
        output_tokens = self._optional_int(usage.get("completion_tokens"))

        if output_tokens is None and token_ids:
            output_tokens = len(token_ids)

        return AIMOCompletionGeneration(
            text=text,
            token_ids=token_ids,
            token_logprobs=token_logprobs,
            input_tokens=self._optional_int(usage.get("prompt_tokens")) or prompt_token_count,
            output_tokens=output_tokens,
            finish_reason=str(first_choice.get("finish_reason", "")),
            latency_seconds=latency_seconds,
            entropy=self._completion_entropy(first_choice),
            raw=response_payload,
        )

    def _completion_token_ids(self, choice: dict[str, Any]) -> list[int]:

        token_ids = choice.get("token_ids")

        if token_ids is None:
            token_ids = choice.get("tokens")

        if token_ids is None:
            return []

        try:
            return [
                int(token_id)
                for token_id in token_ids
            ]
        except (TypeError, ValueError):
            return []

    def _completion_token_logprobs(self, choice: dict[str, Any]) -> list[float]:

        logprobs = choice.get("logprobs", {})

        if not isinstance(logprobs, dict):
            return []

        token_logprobs = logprobs.get("token_logprobs")

        if token_logprobs is not None:
            return self._float_list(token_logprobs)

        content_logprobs = logprobs.get("content")

        if isinstance(content_logprobs, list):
            return self._float_list([
                item.get("logprob")
                for item in content_logprobs
                if isinstance(item, dict)
            ])

        return []

    def _completion_entropy(self, choice: dict[str, Any]) -> float | None:

        logprobs = choice.get("logprobs", {})

        if not isinstance(logprobs, dict):
            return None

        top_logprobs = logprobs.get("top_logprobs")

        if not isinstance(top_logprobs, list):
            return None

        entropy_values = [
            self._top_logprobs_entropy(item)
            for item in top_logprobs
            if isinstance(item, dict)
        ]
        entropy_values = [
            value
            for value in entropy_values
            if value is not None
        ]

        if not entropy_values:
            return None

        return sum(entropy_values) / len(entropy_values)

    def _top_logprobs_entropy(self, item: dict[str, Any]) -> float | None:

        entropy = 0.0
        probability_count = 0

        for raw_logprob in item.values():
            if isinstance(raw_logprob, dict):
                raw_logprob = raw_logprob.get("logprob")

            try:
                probability = math.exp(float(raw_logprob))
            except (TypeError, ValueError, OverflowError):
                continue

            if probability <= 0:
                continue

            entropy -= probability * math.log2(probability)
            probability_count += 1

        if probability_count == 0:
            return None

        return entropy

    def _optional_int(self, value: object) -> int | None:

        if value is None:
            return None

        return int(value)

    def _resolve_max_tokens(
        self,
        messages: Sequence[AIMOChatMessage],
        max_tokens: int,
    ) -> int:

        requested_max_tokens = (
            max_tokens
            if max_tokens > 0
            else self.config.max_tokens_for_pass(max_tokens)
        )
        prompt_tokens = self._count_prompt_tokens(messages)
        available_tokens = self.config.available_generation_tokens(prompt_tokens)

        if available_tokens <= 0:
            raise ValueError(
                f"Prompt uses {prompt_tokens} tokens, "
                f"exceeding max_model_len={self.config.max_model_len}."
            )

        return min(requested_max_tokens, available_tokens)

    def _resolve_completion_max_tokens(
        self,
        prompt_token_count: int,
        max_tokens: int,
    ) -> int:

        requested_max_tokens = (
            max_tokens
            if max_tokens > 0
            else self.config.max_tokens_for_pass(max_tokens)
        )
        available_tokens = self.config.available_generation_tokens(prompt_token_count)

        if available_tokens <= 0:
            raise ValueError(
                f"Prompt uses {prompt_token_count} tokens, "
                f"exceeding max_model_len={self.config.max_model_len}."
            )

        return min(requested_max_tokens, available_tokens)

    def _count_prompt_tokens(
        self,
        messages: Sequence[AIMOChatMessage],
    ) -> int:

        tokenizer = self._load_tokenizer()
        message_payload = [
            message.as_dict()
            for message in messages
        ]

        if tokenizer is not None:
            try:
                tokenized_prompt = tokenizer.apply_chat_template(
                    message_payload,
                    tokenize=True,
                    add_generation_prompt=True,
                )

                return self._tokenized_length(tokenized_prompt)
            except Exception:
                rendered_prompt = self.chat_template.render(
                    messages,
                    add_generation_prompt=True,
                )

                return len(tokenizer.encode(rendered_prompt, add_special_tokens=False))

        rendered_prompt = self.chat_template.render(
            messages,
            add_generation_prompt=True,
        )

        return max(1, len(rendered_prompt.encode("utf-8")) // 4)

    def _load_tokenizer(self) -> Any | None:

        if self._tokenizer is not None:
            return self._tokenizer

        if self._tokenizer_load_failed:
            return None

        if not self.config.model_path.exists():
            self._tokenizer_load_failed = True

            return None

        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.config.model_path),
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception:
            self._tokenizer_load_failed = True

            return None

        return self._tokenizer

    def _tokenized_length(self, tokenized_prompt: object) -> int:

        if isinstance(tokenized_prompt, dict):
            input_ids = tokenized_prompt.get("input_ids", [])

            return len(input_ids)

        return len(tokenized_prompt)

    def _float_list(self, values: object) -> list[float]:

        try:
            return [
                float(value)
                for value in values
            ]
        except (TypeError, ValueError):
            return []
