from __future__ import annotations

import math
import urllib.error
from pathlib import Path
from typing import Any
from typing import Callable

import pytest

import aimo_inference.client as client_module
from aimo_inference.client import AIMOInferenceClient
from aimo_inference.config import AIMOConfig
from aimo_inference.template import AIMOChatMessage
from conftest import FakeHTTPServer


def chat_response(text: str = "Proof.") -> dict[str, Any]:

    return {
        "choices": [
            {
                "message": {
                    "content": text,
                },
                "finish_reason": "stop",
            },
        ],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 7,
        },
    }


def completion_response() -> dict[str, Any]:

    return {
        "choices": [
            {
                "text": "Final.",
                "token_ids": [
                    1,
                    2,
                ],
                "finish_reason": "stop",
                "logprobs": {
                    "top_logprobs": [
                        {
                            "a": math.log(0.75),
                            "b": math.log(0.25),
                        },
                    ],
                },
            },
        ],
        "usage": {
            "prompt_tokens": 5,
        },
    }


def test_chat_completion_request_body(
    tmp_path: Path,
    http_server_factory: Callable[[list[Any]], FakeHTTPServer],
) -> None:

    with http_server_factory([chat_response("Proof A.")]) as server:
        config = AIMOConfig(
            api_base=server.api_base,
            model_path=tmp_path / "missing-model",
            served_model_name="fake-model",
            request_timeout_seconds=2.0,
        )
        client = AIMOInferenceClient(config=config)

        result = client.generate(
            messages=[
                AIMOChatMessage(
                    role="system",
                    content="System.",
                ),
                AIMOChatMessage(
                    role="user",
                    content="Problem.",
                ),
            ],
            max_tokens=25,
        )

    assert result.text == "Proof A."
    assert result.input_tokens == 11
    assert result.output_tokens == 7
    assert server.requests[0].path == "/v1/chat/completions"
    assert server.requests[0].payload["model"] == "fake-model"
    assert server.requests[0].payload["max_tokens"] == 25
    assert server.requests[0].payload["messages"][1] == {
        "role": "user",
        "content": "Problem.",
    }


def test_retries_on_transient_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    http_server_factory: Callable[[list[Any]], FakeHTTPServer],
) -> None:

    monkeypatch.setattr(client_module.time, "sleep", lambda seconds: None)

    with http_server_factory([
        (
            500,
            "temporary",
        ),
        chat_response("Recovered."),
    ]) as server:
        config = AIMOConfig(
            api_base=server.api_base,
            model_path=tmp_path / "missing-model",
            request_timeout_seconds=2.0,
        )
        client = AIMOInferenceClient(config=config)

        result = client.generate(
            messages=[
                AIMOChatMessage(
                    role="user",
                    content="Problem.",
                ),
            ],
            max_tokens=8,
        )

    assert result.text == "Recovered."
    assert len(server.requests) == 2


def test_timeout_error_formatting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    def raise_timeout(*args: object, **kwargs: object) -> object:

        raise urllib.error.URLError("timed out")

    monkeypatch.setattr(client_module.urllib.request, "urlopen", raise_timeout)
    monkeypatch.setattr(client_module.time, "sleep", lambda seconds: None)
    client = AIMOInferenceClient(
        config=AIMOConfig(
            api_base="http://127.0.0.1:9/v1",
            model_path=tmp_path / "missing-model",
            request_timeout_seconds=0.01,
        )
    )

    with pytest.raises(RuntimeError, match="vLLM request failed"):
        client.generate(
            messages=[
                AIMOChatMessage(
                    role="user",
                    content="Problem.",
                ),
            ],
            max_tokens=1,
        )


def test_response_normalization_rejects_missing_choices() -> None:

    client = AIMOInferenceClient(config=AIMOConfig())

    with pytest.raises(RuntimeError, match="choices"):
        client._normalize_response(
            response_payload={},
            latency_seconds=0.0,
        )


def test_token_count_fallback_and_context_overflow(tmp_path: Path) -> None:

    client = AIMOInferenceClient(
        config=AIMOConfig(
            model_path=tmp_path / "missing-model",
            max_model_len=8,
        )
    )
    messages = [
        AIMOChatMessage(
            role="user",
            content="A" * 100,
        ),
    ]

    assert client._count_prompt_tokens(messages) > 0

    with pytest.raises(ValueError, match="Prompt uses"):
        client._resolve_max_tokens(
            messages=messages,
            max_tokens=10,
        )


def test_completion_body_and_token_count_fallback(
    tmp_path: Path,
    http_server_factory: Callable[[list[Any]], FakeHTTPServer],
) -> None:

    with http_server_factory([completion_response()]) as server:
        client = AIMOInferenceClient(
            config=AIMOConfig(
                api_base=server.api_base,
                model_path=tmp_path / "missing-model",
                top_logprobs=2,
            )
        )

        result = client.complete_token_ids(
            prompt_token_ids=[
                9,
                10,
            ],
            max_tokens=4,
            stop_token_ids=[
                99,
            ],
            seed=123,
        )

    payload = server.requests[0].payload

    assert server.requests[0].path == "/v1/completions"
    assert payload["prompt"] == [
        9,
        10,
    ]
    assert payload["logprobs"] == 2
    assert "top_logprobs" not in payload
    assert payload["stop_token_ids"] == [
        99,
    ]
    assert payload["seed"] == 123
    assert result.input_tokens == 5
    assert result.output_tokens == 2
    assert result.entropy is not None


def test_completion_body_requests_selected_logprobs_without_top_alternatives(
    tmp_path: Path,
    http_server_factory: Callable[[list[Any]], FakeHTTPServer],
) -> None:

    with http_server_factory([completion_response()]) as server:
        client = AIMOInferenceClient(
            config=AIMOConfig(
                api_base=server.api_base,
                model_path=tmp_path / "missing-model",
                top_logprobs=0,
            )
        )

        client.complete_token_ids(
            prompt_token_ids=[
                9,
                10,
            ],
            max_tokens=4,
        )

    payload = server.requests[0].payload

    assert payload["logprobs"] == 0
    assert "top_logprobs" not in payload
