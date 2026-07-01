from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from pathlib import Path
from typing import Any

from aimo_inference.context import available_output_tokens
from aimo_inference.defaults import DEFAULT_DUMMY_TEST
from aimo_inference.defaults import DUMMY_MODEL_PATH
from aimo_inference.defaults import DUMMY_SERVED_MODEL_NAME
from aimo_inference.defaults import DUMMY_TENSOR_PARALLEL_SIZE
from aimo_inference.prompts import TOOL_PROMPT
from aimo_inference.profiles import default_profile_for_inference_mode
from aimo_inference.profiles import resolve_model_profile


RUNTIME_MODES = {
    "colab",
    "kaggle",
    "singularity",
}

INFERENCE_MODES = {
    "proof",
    "judge",
    "aimo3_answer",
}

MAX_GENERATION_CONTEXT_TOKENS = 65536

DEFAULT_PAGE_TEMPLATE = (
    "\\documentclass[12pt]{article}\n"
    "\\usepackage[a4paper,margin=1in]{geometry}\n"
    "\\usepackage{amsmath,amssymb,amsthm}\n"
    "\\setlength{\\parindent}{0pt}\n"
    "\\setlength{\\parskip}{0.6em}\n"
    "\\begin{document}\n"
    "<solution>\n"
    "\\end{document}\n"
)


def build_default_compilation_config() -> dict[str, Any]:

    return {
        "cudagraph_num_of_warmups": 1,
        "cudagraph_capture_sizes": [
            1,
            2,
            4,
            8,
            16,
            24,
            32,
            48,
            64,
            96,
            128,
            256,
            512,
            1024,
        ],
        "cudagraph_specialize_lora": False,
        "max_cudagraph_capture_size": 1024,
    }


def build_default_attention_config() -> dict[str, Any]:

    return {
        "backend": "FLASH_ATTN",
        "flash_attn_version": 3,
    }


@dataclass(frozen=True)
class AIMOConfig:

    model_path: Path = Path("models/contestant")
    input_csv: Path = Path("/content/input/input.csv")
    output_csv: Path = Path("/content/output/output.csv")
    logdir: Path = Path("/content/logs")
    eval_dataset_path: Path = Path("/content/input/mathnet_eval_00000.parquet")
    mode: str = "colab"
    inference_mode: str = "proof"
    model_profile: str = "contestant"
    template_format: str = "chatml"
    contestant_model_path: Path = Path("models/contestant")
    judge_model_path: Path = Path("models/judge")
    dummy_test: bool = False
    dummy_model_path: Path = DUMMY_MODEL_PATH
    host: str = "127.0.0.1"
    port: int = 8000
    contestant_port: int = 8001
    judge_port: int = 8000
    api_base: str = ""
    judge_api_base: str = ""
    served_model_name: str = "aimo-proof-model"
    judge_served_model_name: str = "GPT-OSS-120B"
    lora_adapter_path: Path | None = None
    lora_served_model_name: str = "aimo-proof-adapter"
    tensor_parallel_size: int = 1
    num_gpus: int = 1
    gpu_memory_utilization: float = 0.98
    dtype: str = "auto"
    kv_cache_dtype: str = "auto"
    load_format: str = "auto"
    moe_backend: str = ""
    max_model_len: int = 65536
    max_new_tokens: int = 0
    first_pass_max_tokens: int = 0
    second_pass_max_tokens: int = 0
    third_pass_max_tokens: int = 0
    judge_max_tokens: int = 0
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = -1
    top_logprobs: int = 0
    max_logprobs: int = 0
    min_p: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    max_num_seqs: int = 1
    max_num_batched_tokens: int = 0
    max_running_problems: int = 1
    group_size: int = 16
    active_problem_count: int = 6
    sandbox_count: int = 96
    stream_interval: int = 128
    request_timeout_seconds: float = 3300.0
    server_start_timeout_seconds: float = 900.0
    problem_timeout_seconds: float = 2700.0
    tool_timeout_seconds: float = 10.0
    max_python_calls: int = 64
    global_rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    master_addr: str = "127.0.0.1"
    master_port: str = "29500"
    cuda_visible_devices: str = ""
    launch_server: bool = True
    reuse_server: bool = False
    enable_tools: bool = True
    enable_judge: bool = False
    use_jupyter_sandbox: bool = False
    tool_protocol: str = "olmo_chatml"
    write_intermediate_outputs: bool = True
    sample_eval_problems: bool = False
    eval_sample_size: int = 6
    eval_sample_seed: int = 42
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    async_scheduling: bool = True
    enable_expert_parallel: bool = False
    disable_log_stats: bool = True
    compilation_config: dict[str, Any] = field(default_factory=build_default_compilation_config)
    attention_config: dict[str, Any] = field(default_factory=build_default_attention_config)
    extra_server_arguments: dict[str, Any] = field(default_factory=dict)
    performance_mode: str = "throughput"
    page_count_method: str = "latex"
    page_template: str = DEFAULT_PAGE_TEMPLATE
    latex_command: str = "pdflatex"
    pdfinfo_command: str = "pdfinfo"
    page_count_timeout_seconds: int = 20
    harmony_tool_prompt: str = TOOL_PROMPT
    seed: int = 42
    tiktoken_encodings_base: str = ""

    @classmethod
    def default_for_mode(cls, mode: str) -> AIMOConfig:

        if mode == "singularity":
            return cls(
                model_path=Path("/weights"),
                input_csv=Path("/input/input.csv"),
                output_csv=Path("/output/output.csv"),
                logdir=Path("/logs"),
                mode="singularity",
                tensor_parallel_size=8,
                num_gpus=8,
                gpu_memory_utilization=0.98,
                max_num_seqs=256,
                max_running_problems=96,
                sandbox_count=96,
                problem_timeout_seconds=2700.0,
            )

        return cls(
            mode=mode,
            tensor_parallel_size=1,
            num_gpus=1,
            gpu_memory_utilization=0.98,
            max_num_seqs=1,
            sample_eval_problems=True,
        )

    @classmethod
    def from_environment(cls, mode: str | None = None) -> AIMOConfig:

        selected_mode = mode or os.environ.get("AIMO_MODE", "colab")
        runtime_mode, inference_mode = cls._split_mode_argument(
            mode_value=selected_mode,
            default_runtime_mode=os.environ.get("AIMO_RUNTIME_MODE", "colab"),
            default_inference_mode=os.environ.get("AIMO_INFERENCE_MODE", "proof"),
        )
        profile_name = os.environ.get(
            "AIMO_MODEL_PROFILE",
            default_profile_for_inference_mode(inference_mode),
        )
        config = cls.default_for_mode(runtime_mode).with_profile_defaults(
            profile_name=profile_name,
            keep_runtime_model_path=runtime_mode == "singularity" and profile_name == "contestant",
        ).with_inference_mode_defaults(inference_mode).with_runtime_topology(runtime_mode)

        config = replace(
            config,
            mode=runtime_mode,
            inference_mode=cls._validate_inference_mode(
                cls._environment_str("AIMO_INFERENCE_MODE", inference_mode)
            ),
            model_profile=cls._environment_str("AIMO_MODEL_PROFILE", config.model_profile),
            template_format=cls._environment_str("AIMO_TEMPLATE_FORMAT", config.template_format),
            model_path=cls._environment_path("AIMO_MODEL_PATH", config.model_path),
            contestant_model_path=cls._environment_path(
                "AIMO_CONTESTANT_MODEL_PATH",
                config.contestant_model_path,
            ),
            judge_model_path=cls._environment_path(
                "AIMO_JUDGE_MODEL_PATH",
                config.judge_model_path,
            ),
            dummy_test=cls._environment_bool("AIMO_DUMMY_TEST", DEFAULT_DUMMY_TEST),
            dummy_model_path=cls._environment_path(
                "AIMO_DUMMY_MODEL_PATH",
                config.dummy_model_path,
            ),
            input_csv=cls._environment_path("AIMO_INPUT_CSV", config.input_csv),
            output_csv=cls._environment_path("AIMO_OUTPUT_CSV", config.output_csv),
            logdir=cls._environment_path("AIMO_LOGDIR", config.logdir),
            eval_dataset_path=cls._environment_path(
                "AIMO_EVAL_DATASET_PATH",
                config.eval_dataset_path,
            ),
            host=cls._environment_str("AIMO_HOST", config.host),
            port=cls._environment_int("AIMO_PORT", config.port),
            contestant_port=cls._environment_int("AIMO_CONTESTANT_PORT", config.contestant_port),
            judge_port=cls._environment_int("AIMO_JUDGE_PORT", config.judge_port),
            api_base=cls._environment_str("AIMO_API_BASE", config.api_base),
            judge_api_base=cls._environment_str("AIMO_JUDGE_API_BASE", config.judge_api_base),
            served_model_name=cls._environment_str(
                "AIMO_SERVED_MODEL_NAME",
                config.served_model_name,
            ),
            judge_served_model_name=cls._environment_str(
                "AIMO_JUDGE_SERVED_MODEL_NAME",
                config.judge_served_model_name,
            ),
            lora_adapter_path=cls._environment_optional_path(
                "AIMO_LORA_ADAPTER_PATH",
                config.lora_adapter_path,
            ),
            lora_served_model_name=cls._environment_str(
                "AIMO_LORA_SERVED_MODEL_NAME",
                config.lora_served_model_name,
            ),
            tensor_parallel_size=cls._environment_int(
                "AIMO_TENSOR_PARALLEL_SIZE",
                cls._environment_int("AIMO_NUM_GPUS", config.tensor_parallel_size),
            ),
            num_gpus=cls._environment_int(
                "AIMO_NUM_GPUS",
                cls._environment_int("AIMO_TENSOR_PARALLEL_SIZE", config.num_gpus),
            ),
            gpu_memory_utilization=cls._environment_float(
                "AIMO_GPU_MEMORY_UTILIZATION",
                config.gpu_memory_utilization,
            ),
            dtype=cls._environment_str("AIMO_DTYPE", config.dtype),
            kv_cache_dtype=cls._environment_str("AIMO_KV_CACHE_DTYPE", config.kv_cache_dtype),
            load_format=cls._environment_str("AIMO_LOAD_FORMAT", config.load_format),
            moe_backend=cls._environment_str("AIMO_MOE_BACKEND", config.moe_backend),
            max_model_len=cls._environment_int("AIMO_MAX_MODEL_LEN", config.max_model_len),
            max_new_tokens=cls._environment_int("AIMO_MAX_NEW_TOKENS", config.max_new_tokens),
            first_pass_max_tokens=cls._environment_int(
                "AIMO_FIRST_PASS_MAX_TOKENS",
                config.first_pass_max_tokens,
            ),
            second_pass_max_tokens=cls._environment_int(
                "AIMO_SECOND_PASS_MAX_TOKENS",
                config.second_pass_max_tokens,
            ),
            third_pass_max_tokens=cls._environment_int(
                "AIMO_THIRD_PASS_MAX_TOKENS",
                config.third_pass_max_tokens,
            ),
            judge_max_tokens=cls._environment_int("AIMO_JUDGE_MAX_TOKENS", config.judge_max_tokens),
            temperature=cls._environment_float("AIMO_TEMPERATURE", config.temperature),
            top_p=cls._environment_float("AIMO_TOP_P", config.top_p),
            top_k=cls._environment_int("AIMO_TOP_K", config.top_k),
            top_logprobs=cls._environment_int("AIMO_TOP_LOGPROBS", config.top_logprobs),
            max_logprobs=cls._environment_int("AIMO_MAX_LOGPROBS", config.max_logprobs),
            min_p=cls._environment_float("AIMO_MIN_P", config.min_p),
            presence_penalty=cls._environment_float(
                "AIMO_PRESENCE_PENALTY",
                config.presence_penalty,
            ),
            repetition_penalty=cls._environment_float(
                "AIMO_REPETITION_PENALTY",
                config.repetition_penalty,
            ),
            max_num_seqs=cls._environment_int("AIMO_MAX_NUM_SEQS", config.max_num_seqs),
            max_num_batched_tokens=cls._environment_int(
                "AIMO_MAX_NUM_BATCHED_TOKENS",
                config.max_num_batched_tokens,
            ),
            max_running_problems=cls._environment_int(
                "AIMO_MAX_RUNNING_PROBLEMS",
                config.max_running_problems,
            ),
            group_size=cls._environment_int("AIMO_GROUP_SIZE", config.group_size),
            active_problem_count=cls._environment_int(
                "AIMO_ACTIVE_PROBLEM_COUNT",
                config.active_problem_count,
            ),
            sandbox_count=cls._environment_int("AIMO_SANDBOX_COUNT", config.sandbox_count),
            stream_interval=cls._environment_int("AIMO_STREAM_INTERVAL", config.stream_interval),
            request_timeout_seconds=cls._environment_float(
                "AIMO_REQUEST_TIMEOUT_SECONDS",
                config.request_timeout_seconds,
            ),
            server_start_timeout_seconds=cls._environment_float(
                "AIMO_SERVER_START_TIMEOUT_SECONDS",
                config.server_start_timeout_seconds,
            ),
            problem_timeout_seconds=cls._environment_float(
                "AIMO_PROBLEM_TIMEOUT_SECONDS",
                config.problem_timeout_seconds,
            ),
            tool_timeout_seconds=cls._environment_float(
                "AIMO_TOOL_TIMEOUT_SECONDS",
                config.tool_timeout_seconds,
            ),
            max_python_calls=cls._environment_int(
                "AIMO_MAX_PYTHON_CALLS",
                config.max_python_calls,
            ),
            global_rank=cls._environment_int_any(
                [
                    "GLOBAL_RANK",
                    "AIMO_GLOBAL_RANK",
                ],
                config.global_rank,
            ),
            local_rank=cls._environment_int_any(
                [
                    "LOCAL_RANK",
                    "AIMO_LOCAL_RANK",
                ],
                config.local_rank,
            ),
            world_size=cls._environment_int_any(
                [
                    "WORLD_SIZE",
                    "AIMO_WORLD_SIZE",
                ],
                config.world_size,
            ),
            master_addr=cls._environment_str_any(
                [
                    "MASTER_ADDR",
                    "AIMO_MASTER_ADDR",
                ],
                config.master_addr,
            ),
            master_port=cls._environment_str_any(
                [
                    "MASTER_PORT",
                    "AIMO_MASTER_PORT",
                ],
                config.master_port,
            ),
            cuda_visible_devices=cls._environment_str(
                "CUDA_VISIBLE_DEVICES",
                config.cuda_visible_devices,
            ),
            launch_server=cls._environment_bool("AIMO_LAUNCH_SERVER", config.launch_server),
            reuse_server=cls._environment_bool("AIMO_REUSE_SERVER", config.reuse_server),
            enable_tools=cls._environment_bool("AIMO_ENABLE_TOOLS", config.enable_tools),
            enable_judge=cls._environment_bool("AIMO_ENABLE_JUDGE", config.enable_judge),
            use_jupyter_sandbox=cls._environment_bool(
                "AIMO_USE_JUPYTER_SANDBOX",
                config.use_jupyter_sandbox,
            ),
            tool_protocol=cls._environment_str("AIMO_TOOL_PROTOCOL", config.tool_protocol),
            write_intermediate_outputs=cls._environment_bool(
                "AIMO_WRITE_INTERMEDIATE_OUTPUTS",
                config.write_intermediate_outputs,
            ),
            sample_eval_problems=cls._environment_bool(
                "AIMO_SAMPLE_EVAL_PROBLEMS",
                config.sample_eval_problems,
            ),
            eval_sample_size=cls._environment_int(
                "AIMO_EVAL_SAMPLE_SIZE",
                config.eval_sample_size,
            ),
            eval_sample_seed=cls._environment_int(
                "AIMO_EVAL_SAMPLE_SEED",
                config.eval_sample_seed,
            ),
            enable_prefix_caching=cls._environment_bool(
                "AIMO_ENABLE_PREFIX_CACHING",
                config.enable_prefix_caching,
            ),
            enable_chunked_prefill=cls._environment_bool(
                "AIMO_ENABLE_CHUNKED_PREFILL",
                config.enable_chunked_prefill,
            ),
            async_scheduling=cls._environment_bool(
                "AIMO_ASYNC_SCHEDULING",
                config.async_scheduling,
            ),
            enable_expert_parallel=cls._environment_bool(
                "AIMO_ENABLE_EXPERT_PARALLEL",
                config.enable_expert_parallel,
            ),
            disable_log_stats=cls._environment_bool(
                "AIMO_DISABLE_LOG_STATS",
                config.disable_log_stats,
            ),
            compilation_config=cls._environment_dict(
                "AIMO_COMPILATION_CONFIG",
                config.compilation_config,
            ),
            attention_config=cls._environment_dict(
                "AIMO_ATTENTION_CONFIG",
                config.attention_config,
            ),
            extra_server_arguments=cls._environment_dict(
                "AIMO_EXTRA_SERVER_ARGUMENTS",
                config.extra_server_arguments,
            ),
            performance_mode=cls._environment_str(
                "AIMO_PERFORMANCE_MODE",
                config.performance_mode,
            ),
            page_count_method=cls._environment_str(
                "AIMO_PAGE_COUNT_METHOD",
                config.page_count_method,
            ),
            page_template=cls._environment_str("AIMO_PAGE_TEMPLATE", config.page_template),
            latex_command=cls._environment_str("AIMO_LATEX_COMMAND", config.latex_command),
            pdfinfo_command=cls._environment_str("AIMO_PDFINFO_COMMAND", config.pdfinfo_command),
            page_count_timeout_seconds=cls._environment_int(
                "AIMO_PAGE_COUNT_TIMEOUT_SECONDS",
                config.page_count_timeout_seconds,
            ),
            harmony_tool_prompt=cls._environment_str(
                "AIMO_HARMONY_TOOL_PROMPT",
                config.harmony_tool_prompt,
            ),
            seed=cls._environment_int("AIMO_SEED", config.seed),
            tiktoken_encodings_base=cls._environment_str(
                "TIKTOKEN_ENCODINGS_BASE",
                config.tiktoken_encodings_base,
            ),
        )

        return config.with_dummy_test_defaults()

    @classmethod
    def from_cli_args(cls, argv: list[str] | None = None) -> AIMOConfig:

        mode_parser = argparse.ArgumentParser(add_help=False)
        mode_parser.add_argument("--mode", choices=tuple(sorted(RUNTIME_MODES | INFERENCE_MODES)))
        mode_parser.add_argument("--inference_mode", choices=tuple(sorted(INFERENCE_MODES)))
        mode_parser.add_argument("--model_profile")
        known_args, _ = mode_parser.parse_known_args(argv)
        mode = known_args.mode or os.environ.get("AIMO_MODE", "colab")
        config = cls.from_environment(mode=mode)

        if known_args.inference_mode:
            profile_name = known_args.model_profile or default_profile_for_inference_mode(
                known_args.inference_mode
            )
            config = config.with_profile_defaults(
                profile_name=profile_name,
            ).with_inference_mode_defaults(
                known_args.inference_mode,
            ).with_runtime_topology(config.mode)

        if known_args.model_profile:
            config = config.with_profile_defaults(
                profile_name=known_args.model_profile,
            ).with_runtime_topology(config.mode)

        parser = cls._build_parser(config)
        args = parser.parse_args(argv)
        runtime_mode, inference_mode = cls._split_mode_argument(
            mode_value=args.mode,
            default_runtime_mode=config.mode,
            default_inference_mode=config.inference_mode,
        )

        config = cls(
            model_path=args.model_path,
            contestant_model_path=args.contestant_model_path,
            judge_model_path=args.judge_model_path,
            dummy_test=args.dummy_test,
            dummy_model_path=args.dummy_model_path,
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            logdir=args.logdir,
            eval_dataset_path=args.eval_dataset_path,
            mode=runtime_mode,
            inference_mode=args.inference_mode or inference_mode,
            model_profile=args.model_profile,
            template_format=args.template_format,
            host=args.host,
            port=args.port,
            contestant_port=args.contestant_port,
            judge_port=args.judge_port,
            api_base=args.api_base,
            judge_api_base=args.judge_api_base,
            served_model_name=args.served_model_name,
            judge_served_model_name=args.judge_served_model_name,
            lora_adapter_path=args.lora_adapter_path,
            lora_served_model_name=args.lora_served_model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            num_gpus=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
            kv_cache_dtype=args.kv_cache_dtype,
            load_format=args.load_format,
            moe_backend=args.moe_backend,
            max_model_len=args.max_model_len,
            max_new_tokens=args.max_new_tokens,
            first_pass_max_tokens=args.first_pass_max_tokens,
            second_pass_max_tokens=args.second_pass_max_tokens,
            third_pass_max_tokens=args.third_pass_max_tokens,
            judge_max_tokens=args.judge_max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            top_logprobs=args.top_logprobs,
            max_logprobs=args.max_logprobs,
            min_p=args.min_p,
            presence_penalty=args.presence_penalty,
            repetition_penalty=args.repetition_penalty,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_running_problems=args.max_running_problems,
            group_size=args.group_size,
            active_problem_count=args.active_problem_count,
            sandbox_count=args.sandbox_count,
            stream_interval=args.stream_interval,
            request_timeout_seconds=args.request_timeout_seconds,
            server_start_timeout_seconds=args.server_start_timeout_seconds,
            problem_timeout_seconds=args.problem_timeout_seconds,
            tool_timeout_seconds=args.tool_timeout_seconds,
            max_python_calls=args.max_python_calls,
            global_rank=args.global_rank,
            local_rank=args.local_rank,
            world_size=args.world_size,
            master_addr=args.master_addr,
            master_port=args.master_port,
            cuda_visible_devices=args.cuda_visible_devices,
            launch_server=args.launch_server,
            reuse_server=args.reuse_server,
            enable_tools=args.enable_tools,
            enable_judge=args.enable_judge,
            use_jupyter_sandbox=args.use_jupyter_sandbox,
            tool_protocol=args.tool_protocol,
            write_intermediate_outputs=args.write_intermediate_outputs,
            sample_eval_problems=args.sample_eval_problems,
            eval_sample_size=args.eval_sample_size,
            eval_sample_seed=args.eval_sample_seed,
            enable_prefix_caching=args.enable_prefix_caching,
            enable_chunked_prefill=args.enable_chunked_prefill,
            async_scheduling=args.async_scheduling,
            enable_expert_parallel=args.enable_expert_parallel,
            disable_log_stats=args.disable_log_stats,
            compilation_config=args.compilation_config,
            attention_config=args.attention_config,
            extra_server_arguments=args.extra_server_arguments,
            performance_mode=args.performance_mode,
            page_count_method=args.page_count_method,
            page_template=args.page_template,
            latex_command=args.latex_command,
            pdfinfo_command=args.pdfinfo_command,
            page_count_timeout_seconds=args.page_count_timeout_seconds,
            harmony_tool_prompt=args.harmony_tool_prompt,
            seed=args.seed,
            tiktoken_encodings_base=args.tiktoken_encodings_base,
        )

        return config.with_dummy_test_defaults()

    @property
    def resolved_api_base(self) -> str:

        if self.api_base:
            return self.api_base.rstrip("/")

        return f"http://{self.host}:{self.port}/v1"

    @property
    def resolved_generation_model_name(self) -> str:

        if self.lora_adapter_path is not None:
            return self.lora_served_model_name

        return self.served_model_name

    @property
    def health_url(self) -> str:

        return f"http://{self.host}:{self.port}/health"

    def sampling_payload(self, max_tokens: int) -> dict[str, float | int | bool]:

        payload: dict[str, float | int | bool] = {
            "max_tokens": max_tokens,
            "temperature": self.temperature,
        }

        if self.top_p > 0:
            payload["top_p"] = self.top_p

        if self.top_k > 0:
            payload["top_k"] = self.top_k

        if self.top_logprobs > 0:
            payload["logprobs"] = True
            payload["top_logprobs"] = self.top_logprobs

        if self.min_p > 0:
            payload["min_p"] = self.min_p

        if self.presence_penalty != 0.0:
            payload["presence_penalty"] = self.presence_penalty

        if self.repetition_penalty != 1.0:
            payload["repetition_penalty"] = self.repetition_penalty

        return payload

    def max_tokens_for_pass(self, configured_max_tokens: int) -> int:

        if configured_max_tokens <= 0 and self.max_new_tokens <= 0:
            return MAX_GENERATION_CONTEXT_TOKENS

        requested_max_tokens = configured_max_tokens

        if requested_max_tokens <= 0:
            requested_max_tokens = self.max_new_tokens

        return min(requested_max_tokens, MAX_GENERATION_CONTEXT_TOKENS)

    def available_generation_tokens(self, input_tokens: int) -> int:

        context_token_limit = min(self.max_model_len, MAX_GENERATION_CONTEXT_TOKENS)

        return available_output_tokens(
            max_model_len=context_token_limit,
            prompt_tokens=input_tokens,
        )

    @property
    def sequential_refinement_enabled(self) -> bool:

        return (
            self.inference_mode == "proof"
            and self.mode == "kaggle"
            and self.tensor_parallel_size == 1
            and self.num_gpus == 1
            and not self.enable_judge
        )

    def with_overrides(self, **overrides: object) -> AIMOConfig:

        return replace(self, **overrides).with_dummy_test_defaults()

    @property
    def resolved_dummy_model_path(self) -> Path:

        if self.dummy_model_path != DUMMY_MODEL_PATH:
            return self.dummy_model_path

        default_model_paths = {
            Path("models/contestant"),
            Path("models/judge"),
        }

        for candidate_path in [
            self.model_path,
            self.contestant_model_path,
            self.judge_model_path,
        ]:
            if candidate_path in default_model_paths and self.model_path not in default_model_paths:
                continue

            if candidate_path.name in {
                "contestant",
                "judge",
            }:
                return candidate_path.parent / "dummy"

        if self.model_path != Path("models/contestant"):
            return self.model_path

        return self.dummy_model_path

    def with_dummy_test_defaults(self) -> AIMOConfig:

        if not self.dummy_test:
            return self

        dummy_model_path = self.resolved_dummy_model_path
        tool_protocol = self.tool_protocol

        if tool_protocol == "harmony":
            tool_protocol = "olmo_chatml"

        return replace(
            self,
            model_path=dummy_model_path,
            contestant_model_path=dummy_model_path,
            judge_model_path=dummy_model_path,
            dummy_model_path=dummy_model_path,
            served_model_name=DUMMY_SERVED_MODEL_NAME,
            judge_served_model_name=DUMMY_SERVED_MODEL_NAME,
            tensor_parallel_size=DUMMY_TENSOR_PARALLEL_SIZE,
            num_gpus=DUMMY_TENSOR_PARALLEL_SIZE,
            template_format="chatml",
            moe_backend="",
            enable_expert_parallel=False,
            tool_protocol=tool_protocol,
        )

    def with_profile_defaults(
        self,
        profile_name: str,
        keep_runtime_model_path: bool = False,
    ) -> AIMOConfig:

        profile = resolve_model_profile(profile_name)
        overrides = profile.as_config_overrides()

        if keep_runtime_model_path:
            overrides.pop("model_path", None)

        if profile.name == "judge":
            overrides["judge_model_path"] = profile.model_path
            overrides["judge_served_model_name"] = profile.served_model_name
        else:
            overrides["contestant_model_path"] = profile.model_path

        return replace(self, **overrides).with_dummy_test_defaults()

    def with_inference_mode_defaults(self, inference_mode: str) -> AIMOConfig:

        if inference_mode == "aimo3_answer":
            return replace(
                self,
                inference_mode=inference_mode,
                model_profile="judge",
                template_format="harmony",
                top_logprobs=0,
                max_logprobs=0,
                kv_cache_dtype="fp8_e4m3",
                moe_backend="marlin",
                enable_expert_parallel=True,
                gpu_memory_utilization=0.98,
                tensor_parallel_size=1,
                num_gpus=1,
                max_num_seqs=1,
                max_model_len=MAX_GENERATION_CONTEXT_TOKENS,
                max_num_batched_tokens=2048,
            ).with_dummy_test_defaults()

        if inference_mode == "judge":
            return replace(
                self,
                inference_mode=inference_mode,
                model_profile="judge",
                template_format="harmony",
            ).with_dummy_test_defaults()

        return replace(
            self,
            inference_mode=inference_mode,
        ).with_dummy_test_defaults()

    def with_runtime_topology(self, runtime_mode: str) -> AIMOConfig:

        if runtime_mode == "singularity":
            return replace(
                self,
                mode=runtime_mode,
                tensor_parallel_size=8,
                num_gpus=8,
                gpu_memory_utilization=0.98,
            ).with_dummy_test_defaults()

        return replace(
            self,
            mode=runtime_mode,
            tensor_parallel_size=1,
            num_gpus=1,
            gpu_memory_utilization=0.98,
        ).with_dummy_test_defaults()

    @classmethod
    def _build_parser(cls, config: AIMOConfig) -> argparse.ArgumentParser:

        parser = argparse.ArgumentParser()
        parser.add_argument("--model_path", type=Path, required=True)
        parser.add_argument("--input_csv", type=Path, required=True)
        parser.add_argument("--output_csv", type=Path, required=True)
        parser.add_argument("--logdir", type=Path, required=True)
        parser.add_argument("--eval_dataset_path", type=Path, default=config.eval_dataset_path)
        parser.add_argument(
            "--mode",
            choices=tuple(sorted(RUNTIME_MODES | INFERENCE_MODES)),
            default=config.inference_mode,
        )
        parser.add_argument(
            "--inference_mode",
            choices=tuple(sorted(INFERENCE_MODES)),
            default=config.inference_mode,
        )
        parser.add_argument("--model_profile", default=config.model_profile)
        parser.add_argument(
            "--template_format",
            choices=("chatml", "harmony"),
            default=config.template_format,
        )
        parser.add_argument(
            "--contestant_model_path",
            type=Path,
            default=config.contestant_model_path,
        )
        parser.add_argument("--judge_model_path", type=Path, default=config.judge_model_path)
        parser.add_argument("--dummy_test", type=cls._parse_bool, default=config.dummy_test)
        parser.add_argument("--dummy_model_path", type=Path, default=config.dummy_model_path)
        parser.add_argument("--host", default=config.host)
        parser.add_argument("--port", type=int, default=config.port)
        parser.add_argument("--contestant_port", type=int, default=config.contestant_port)
        parser.add_argument("--judge_port", type=int, default=config.judge_port)
        parser.add_argument("--api_base", default=config.api_base)
        parser.add_argument("--judge_api_base", default=config.judge_api_base)
        parser.add_argument("--served_model_name", default=config.served_model_name)
        parser.add_argument("--judge_served_model_name", default=config.judge_served_model_name)
        parser.add_argument("--lora_adapter_path", type=Path, default=config.lora_adapter_path)
        parser.add_argument("--lora_served_model_name", default=config.lora_served_model_name)
        parser.add_argument(
            "--tensor_parallel_size",
            "--num_gpus",
            dest="tensor_parallel_size",
            type=int,
            default=config.tensor_parallel_size,
        )
        parser.add_argument(
            "--gpu_memory_utilization",
            type=float,
            default=config.gpu_memory_utilization,
        )
        parser.add_argument("--dtype", default=config.dtype)
        parser.add_argument("--kv_cache_dtype", default=config.kv_cache_dtype)
        parser.add_argument("--load_format", default=config.load_format)
        parser.add_argument("--moe_backend", default=config.moe_backend)
        parser.add_argument(
            "--max_model_len",
            "--num_ctx",
            dest="max_model_len",
            type=int,
            default=config.max_model_len,
        )
        parser.add_argument("--max_new_tokens", type=int, default=config.max_new_tokens)
        parser.add_argument(
            "--first_pass_max_tokens",
            type=int,
            default=config.first_pass_max_tokens,
        )
        parser.add_argument(
            "--second_pass_max_tokens",
            type=int,
            default=config.second_pass_max_tokens,
        )
        parser.add_argument(
            "--third_pass_max_tokens",
            type=int,
            default=config.third_pass_max_tokens,
        )
        parser.add_argument("--judge_max_tokens", type=int, default=config.judge_max_tokens)
        parser.add_argument("--temperature", type=float, default=config.temperature)
        parser.add_argument("--top_p", type=float, default=config.top_p)
        parser.add_argument("--top_k", type=int, default=config.top_k)
        parser.add_argument(
            "--top_logprobs",
            "--top-logprobs",
            dest="top_logprobs",
            type=int,
            default=config.top_logprobs,
        )
        parser.add_argument("--max_logprobs", type=int, default=config.max_logprobs)
        parser.add_argument("--min_p", type=float, default=config.min_p)
        parser.add_argument("--presence_penalty", type=float, default=config.presence_penalty)
        parser.add_argument("--repetition_penalty", type=float, default=config.repetition_penalty)
        parser.add_argument("--max_num_seqs", type=int, default=config.max_num_seqs)
        parser.add_argument(
            "--max_num_batched_tokens",
            type=int,
            default=config.max_num_batched_tokens,
        )
        parser.add_argument("--max_running_problems", type=int, default=config.max_running_problems)
        parser.add_argument("--group_size", type=int, default=config.group_size)
        parser.add_argument("--active_problem_count", type=int, default=config.active_problem_count)
        parser.add_argument("--sandbox_count", type=int, default=config.sandbox_count)
        parser.add_argument("--stream_interval", type=int, default=config.stream_interval)
        parser.add_argument(
            "--request_timeout_seconds",
            type=float,
            default=config.request_timeout_seconds,
        )
        parser.add_argument(
            "--server_start_timeout_seconds",
            type=float,
            default=config.server_start_timeout_seconds,
        )
        parser.add_argument(
            "--problem_timeout_seconds",
            type=float,
            default=config.problem_timeout_seconds,
        )
        parser.add_argument(
            "--tool_timeout_seconds",
            type=float,
            default=config.tool_timeout_seconds,
        )
        parser.add_argument("--max_python_calls", type=int, default=config.max_python_calls)
        parser.add_argument("--global_rank", type=int, default=config.global_rank)
        parser.add_argument("--local_rank", type=int, default=config.local_rank)
        parser.add_argument("--world_size", type=int, default=config.world_size)
        parser.add_argument("--master_addr", default=config.master_addr)
        parser.add_argument("--master_port", default=config.master_port)
        parser.add_argument("--cuda_visible_devices", default=config.cuda_visible_devices)
        parser.add_argument("--launch_server", type=cls._parse_bool, default=config.launch_server)
        parser.add_argument("--reuse_server", type=cls._parse_bool, default=config.reuse_server)
        parser.add_argument("--enable_tools", type=cls._parse_bool, default=config.enable_tools)
        parser.add_argument("--enable_judge", type=cls._parse_bool, default=config.enable_judge)
        parser.add_argument(
            "--use_jupyter_sandbox",
            type=cls._parse_bool,
            default=config.use_jupyter_sandbox,
        )
        parser.add_argument(
            "--tool_protocol",
            choices=("olmo_chatml", "markdown_code", "harmony"),
            default=config.tool_protocol,
        )
        parser.add_argument(
            "--write_intermediate_outputs",
            type=cls._parse_bool,
            default=config.write_intermediate_outputs,
        )
        parser.add_argument(
            "--sample_eval_problems",
            type=cls._parse_bool,
            default=config.sample_eval_problems,
        )
        parser.add_argument("--eval_sample_size", type=int, default=config.eval_sample_size)
        parser.add_argument("--eval_sample_seed", type=int, default=config.eval_sample_seed)
        parser.add_argument(
            "--enable_prefix_caching",
            type=cls._parse_bool,
            default=config.enable_prefix_caching,
        )
        parser.add_argument(
            "--enable_chunked_prefill",
            type=cls._parse_bool,
            default=config.enable_chunked_prefill,
        )
        parser.add_argument(
            "--async_scheduling",
            type=cls._parse_bool,
            default=config.async_scheduling,
        )
        parser.add_argument(
            "--enable_expert_parallel",
            type=cls._parse_bool,
            default=config.enable_expert_parallel,
        )
        parser.add_argument(
            "--disable_log_stats",
            type=cls._parse_bool,
            default=config.disable_log_stats,
        )
        parser.add_argument(
            "--compilation_config",
            type=cls._parse_dict,
            default=config.compilation_config,
        )
        parser.add_argument(
            "--attention_config",
            type=cls._parse_dict,
            default=config.attention_config,
        )
        parser.add_argument(
            "--extra_server_arguments",
            type=cls._parse_dict,
            default=config.extra_server_arguments,
        )
        parser.add_argument("--performance_mode", default=config.performance_mode)
        parser.add_argument("--page_count_method", default=config.page_count_method)
        parser.add_argument("--page_template", default=config.page_template)
        parser.add_argument("--latex_command", default=config.latex_command)
        parser.add_argument("--pdfinfo_command", default=config.pdfinfo_command)
        parser.add_argument(
            "--page_count_timeout_seconds",
            type=int,
            default=config.page_count_timeout_seconds,
        )
        parser.add_argument("--harmony_tool_prompt", default=config.harmony_tool_prompt)
        parser.add_argument("--seed", type=int, default=config.seed)
        parser.add_argument("--tiktoken_encodings_base", default=config.tiktoken_encodings_base)

        return parser

    @classmethod
    def _split_mode_argument(
        cls,
        mode_value: str,
        default_runtime_mode: str,
        default_inference_mode: str,
    ) -> tuple[str, str]:

        normalized_mode = mode_value.strip()

        if normalized_mode in RUNTIME_MODES:
            return normalized_mode, cls._validate_inference_mode(default_inference_mode)

        if normalized_mode in INFERENCE_MODES:
            return cls._validate_runtime_mode(default_runtime_mode), normalized_mode

        expected_modes = ", ".join(sorted(RUNTIME_MODES | INFERENCE_MODES))

        raise ValueError(f"Invalid mode: {mode_value}. Expected one of: {expected_modes}.")

    @staticmethod
    def _validate_runtime_mode(mode: str) -> str:

        if mode not in RUNTIME_MODES:
            expected_modes = ", ".join(sorted(RUNTIME_MODES))

            raise ValueError(f"Invalid runtime mode: {mode}. Expected one of: {expected_modes}.")

        return mode

    @staticmethod
    def _validate_inference_mode(mode: str) -> str:

        if mode not in INFERENCE_MODES:
            expected_modes = ", ".join(sorted(INFERENCE_MODES))

            raise ValueError(f"Invalid inference mode: {mode}. Expected one of: {expected_modes}.")

        return mode

    @staticmethod
    def _parse_bool(value: str | bool) -> bool:

        if isinstance(value, bool):
            return value

        normalized_value = value.strip().lower()

        if normalized_value in {"1", "true", "yes", "y", "on"}:
            return True

        if normalized_value in {"0", "false", "no", "n", "off"}:
            return False

        raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

    @staticmethod
    def _parse_dict(value: str | dict[str, Any]) -> dict[str, Any]:

        if isinstance(value, dict):
            return copy.deepcopy(value)

        stripped_value = value.strip()

        if not stripped_value:
            return {}

        try:
            parsed_value = json.loads(stripped_value)
        except json.JSONDecodeError as error:
            raise argparse.ArgumentTypeError(f"Invalid JSON dictionary: {value}") from error

        if not isinstance(parsed_value, dict):
            raise argparse.ArgumentTypeError(f"Expected JSON dictionary: {value}")

        return parsed_value

    @staticmethod
    def _environment_path(name: str, default: Path) -> Path:

        value = os.environ.get(name)

        if value is None:
            return copy.deepcopy(default)

        return Path(value)

    @staticmethod
    def _environment_optional_path(name: str, default: Path | None) -> Path | None:

        value = os.environ.get(name)

        if value is None:
            return copy.deepcopy(default)

        stripped_value = value.strip()

        if not stripped_value:
            return None

        return Path(stripped_value)

    @staticmethod
    def _environment_str(name: str, default: str) -> str:

        return os.environ.get(name, default)

    @staticmethod
    def _environment_str_any(names: list[str], default: str) -> str:

        for name in names:
            value = os.environ.get(name)

            if value is not None:
                return value

        return default

    @staticmethod
    def _environment_int(name: str, default: int) -> int:

        value = os.environ.get(name)

        if value is None:
            return default

        return int(value)

    @staticmethod
    def _environment_int_any(names: list[str], default: int) -> int:

        for name in names:
            value = os.environ.get(name)

            if value is not None:
                return int(value)

        return default

    @staticmethod
    def _environment_float(name: str, default: float) -> float:

        value = os.environ.get(name)

        if value is None:
            return default

        return float(value)

    @classmethod
    def _environment_bool(cls, name: str, default: bool) -> bool:

        value = os.environ.get(name)

        if value is None:
            return default

        return cls._parse_bool(value)

    @classmethod
    def _environment_dict(cls, name: str, default: dict[str, Any]) -> dict[str, Any]:

        value = os.environ.get(name)

        if value is None:
            return copy.deepcopy(default)

        return cls._parse_dict(value)


CFG = AIMOConfig
