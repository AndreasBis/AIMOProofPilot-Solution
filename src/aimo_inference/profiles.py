from __future__ import annotations

import copy
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AIMOModelProfile:

    name: str
    model_family: str
    model_path: Path
    served_model_name: str
    template_format: str
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    top_logprobs: int
    max_model_len: int
    kv_cache_dtype: str
    load_format: str
    moe_backend: str
    max_num_batched_tokens: int
    performance_mode: str
    tensor_parallel_size: int
    gpu_memory_utilization: float
    enable_expert_parallel: bool
    enable_prefix_caching: bool
    enable_chunked_prefill: bool
    async_scheduling: bool
    environment: dict[str, str] = field(default_factory=dict)
    extra_server_arguments: dict[str, str | int | float | bool] = field(default_factory=dict)

    def as_config_overrides(self) -> dict[str, Any]:

        return {
            "model_profile": self.name,
            "model_path": copy.deepcopy(self.model_path),
            "served_model_name": self.served_model_name,
            "template_format": self.template_format,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "top_logprobs": self.top_logprobs,
            "max_logprobs": self.top_logprobs,
            "max_model_len": self.max_model_len,
            "kv_cache_dtype": self.kv_cache_dtype,
            "load_format": self.load_format,
            "moe_backend": self.moe_backend,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "performance_mode": self.performance_mode,
            "tensor_parallel_size": self.tensor_parallel_size,
            "num_gpus": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enable_expert_parallel": self.enable_expert_parallel,
            "enable_prefix_caching": self.enable_prefix_caching,
            "enable_chunked_prefill": self.enable_chunked_prefill,
            "async_scheduling": self.async_scheduling,
            "extra_server_arguments": copy.deepcopy(self.extra_server_arguments),
        }


MODEL_PROFILES = {
    "contestant": AIMOModelProfile(
        name="contestant",
        model_family="OLMo-3.1-32B-Think",
        model_path=Path("models/contestant"),
        served_model_name="OLMo-3.1-32B-Think",
        template_format="chatml",
        temperature=0.6,
        top_p=0.95,
        top_k=-1,
        min_p=0.0,
        top_logprobs=0,
        max_model_len=65536,
        kv_cache_dtype="auto",
        load_format="auto",
        moe_backend="",
        max_num_batched_tokens=0,
        performance_mode="throughput",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.98,
        enable_expert_parallel=False,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        async_scheduling=True,
        environment={},
        extra_server_arguments={},
    ),
    "judge": AIMOModelProfile(
        name="judge",
        model_family="GPT-OSS-120B",
        model_path=Path("models/judge"),
        served_model_name="GPT-OSS-120B",
        template_format="harmony",
        temperature=1.0,
        top_p=0.0,
        top_k=-1,
        min_p=0.02,
        top_logprobs=0,
        max_model_len=65536,
        kv_cache_dtype="auto",
        load_format="auto",
        moe_backend="marlin",
        max_num_batched_tokens=0,
        performance_mode="throughput",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.98,
        enable_expert_parallel=True,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        async_scheduling=True,
        environment={
            "VLLM_USE_V2_MODEL_RUNNER": "1",
        },
        extra_server_arguments={},
    ),
}


def profile_names() -> tuple[str, ...]:

    return tuple(sorted(MODEL_PROFILES))


def resolve_model_profile(name: str) -> AIMOModelProfile:

    normalized_name = name.strip().casefold()

    if normalized_name not in MODEL_PROFILES:
        allowed_names = ", ".join(profile_names())

        raise ValueError(f"Unknown model profile: {name}. Expected one of: {allowed_names}.")

    return MODEL_PROFILES[normalized_name]


def default_profile_for_inference_mode(inference_mode: str) -> str:

    if inference_mode in {"judge", "aimo3_answer"}:
        return "judge"

    return "contestant"
