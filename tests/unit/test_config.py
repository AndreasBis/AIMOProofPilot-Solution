from __future__ import annotations

import json
import os
import socket
from types import SimpleNamespace
from pathlib import Path

import pytest

from aimo_inference.config import AIMOConfig
from aimo_inference.config import MAX_GENERATION_CONTEXT_TOKENS
from aimo_inference.defaults import DEFAULT_DUMMY_TEST
from aimo_inference.server import AIMOInferenceServer
from aimo_inference.server import AIMOServicePreflight


def test_default_colab_and_singularity_configs() -> None:

    colab_config = AIMOConfig.default_for_mode("colab")
    singularity_config = AIMOConfig.default_for_mode("singularity")

    assert colab_config.mode == "colab"
    assert colab_config.tensor_parallel_size == 1
    assert colab_config.num_gpus == 1
    assert colab_config.sample_eval_problems is True
    assert singularity_config.mode == "singularity"
    assert singularity_config.model_path == Path("/weights")
    assert singularity_config.input_csv == Path("/input/input.csv")
    assert singularity_config.output_csv == Path("/output/output.csv")
    assert singularity_config.tensor_parallel_size == 8
    assert singularity_config.num_gpus == 8
    assert singularity_config.sandbox_count == 96


def test_dummy_test_defaults_to_current_release_value() -> None:

    assert DEFAULT_DUMMY_TEST is True


def test_dummy_test_environment_can_disable_release_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    monkeypatch.setenv("AIMO_DUMMY_TEST", "false")

    config = AIMOConfig.from_environment()

    assert config.dummy_test is False


def test_environment_overrides_are_parsed(monkeypatch: pytest.MonkeyPatch) -> None:

    monkeypatch.setenv("AIMO_MODE", "singularity")
    monkeypatch.setenv("AIMO_INPUT_CSV", "/tmp/problems.csv")
    monkeypatch.setenv("AIMO_OUTPUT_CSV", "/tmp/predictions.csv")
    monkeypatch.setenv("AIMO_LOGDIR", "/tmp/logs")
    monkeypatch.setenv("AIMO_LAUNCH_SERVER", "false")
    monkeypatch.setenv("AIMO_ENABLE_TOOLS", "0")
    monkeypatch.setenv("AIMO_COMPILATION_CONFIG", "{\"capture\": 1}")
    monkeypatch.setenv("AIMO_ATTENTION_CONFIG", "{\"backend\": \"TEST\"}")
    monkeypatch.setenv("GLOBAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "4")

    config = AIMOConfig.from_environment()

    assert config.mode == "singularity"
    assert config.input_csv == Path("/tmp/problems.csv")
    assert config.output_csv == Path("/tmp/predictions.csv")
    assert config.logdir == Path("/tmp/logs")
    assert config.launch_server is False
    assert config.enable_tools is False
    assert config.compilation_config == {
        "capture": 1,
    }
    assert config.attention_config == {
        "backend": "TEST",
    }
    assert config.global_rank == 1
    assert config.world_size == 4


def test_default_attention_config_uses_current_vllm_keys() -> None:

    config = AIMOConfig()

    assert config.attention_config == {
        "backend": "FLASH_ATTN",
        "flash_attn_version": 3,
    }


def test_cli_overrides_are_parsed(tmp_path: Path) -> None:

    config = AIMOConfig.from_cli_args([
        "--mode",
        "colab",
        "--inference_mode",
        "proof",
        "--model_path",
        str(tmp_path / "model"),
        "--input_csv",
        str(tmp_path / "input.csv"),
        "--output_csv",
        str(tmp_path / "output.csv"),
        "--logdir",
        str(tmp_path / "logs"),
        "--dummy_test",
        "false",
        "--launch_server",
        "false",
        "--enable_tools",
        "no",
        "--compilation_config",
        "{\"max_capture\": 2}",
        "--top_logprobs",
        "0",
        "--max_new_tokens",
        "99",
    ])

    assert config.model_path == tmp_path / "model"
    assert config.input_csv == tmp_path / "input.csv"
    assert config.output_csv == tmp_path / "output.csv"
    assert config.logdir == tmp_path / "logs"
    assert config.launch_server is False
    assert config.enable_tools is False
    assert config.compilation_config == {
        "max_capture": 2,
    }
    assert config.top_logprobs == 0
    assert config.max_new_tokens == 99


def test_dummy_test_cli_points_all_models_to_dummy_checkpoint(tmp_path: Path) -> None:

    config = AIMOConfig.from_cli_args([
        "--mode",
        "judge",
        "--model_path",
        str(tmp_path / "models" / "contestant"),
        "--judge_model_path",
        str(tmp_path / "models" / "judge"),
        "--input_csv",
        str(tmp_path / "input.csv"),
        "--output_csv",
        str(tmp_path / "output.csv"),
        "--logdir",
        str(tmp_path / "logs"),
        "--dummy_test",
        "true",
    ])
    expected_dummy_path = tmp_path / "models" / "dummy"

    assert config.dummy_test is True
    assert config.model_path == expected_dummy_path
    assert config.contestant_model_path == expected_dummy_path
    assert config.judge_model_path == expected_dummy_path
    assert config.dummy_model_path == expected_dummy_path
    assert config.served_model_name == "SmolLM-3B"
    assert config.judge_served_model_name == "SmolLM-3B"
    assert config.tensor_parallel_size == 2
    assert config.num_gpus == 2
    assert config.template_format == "chatml"
    assert config.tool_protocol == "olmo_chatml"
    assert config.moe_backend == ""
    assert config.enable_expert_parallel is False


def test_boolean_and_json_dictionary_parsing() -> None:

    assert AIMOConfig._parse_bool("on") is True
    assert AIMOConfig._parse_bool("OFF") is False
    assert AIMOConfig._parse_dict("") == {}
    assert AIMOConfig._parse_dict("{\"a\": 1}") == {
        "a": 1,
    }

    with pytest.raises(Exception):
        AIMOConfig._parse_bool("maybe")

    with pytest.raises(Exception):
        AIMOConfig._parse_dict("[1, 2]")


def test_sampling_payload_handles_top_logprobs_zero_and_positive() -> None:

    zero_logprob_config = AIMOConfig(top_logprobs=0)
    positive_logprob_config = AIMOConfig(
        top_p=0.0,
        top_logprobs=3,
        min_p=0.02,
        presence_penalty=0.1,
        repetition_penalty=1.05,
    )

    assert zero_logprob_config.sampling_payload(max_tokens=12) == {
        "max_tokens": 12,
        "temperature": 0.6,
        "top_p": 0.95,
    }
    assert positive_logprob_config.sampling_payload(max_tokens=12) == {
        "max_tokens": 12,
        "temperature": 0.6,
        "logprobs": True,
        "top_logprobs": 3,
        "min_p": 0.02,
        "presence_penalty": 0.1,
        "repetition_penalty": 1.05,
    }


def test_server_command_keeps_max_logprobs_zero(tmp_path: Path) -> None:

    config = AIMOConfig(
        model_path=tmp_path / "missing-model",
        top_logprobs=0,
        max_logprobs=0,
    )
    command = AIMOInferenceServer(config=config).build_command()
    max_logprobs_index = command.index("--max-logprobs")

    assert command[max_logprobs_index + 1] == "0"


def test_judge_server_command_sets_marlin_moe_backend(tmp_path: Path) -> None:

    config = AIMOConfig(
        model_path=tmp_path / "missing-judge-model",
        model_profile="judge",
        served_model_name="GPT-OSS-120B",
        moe_backend="marlin",
    )
    command = AIMOInferenceServer(config=config).build_command()
    moe_backend_index = command.index("--moe-backend")

    assert command[moe_backend_index + 1] == "marlin"
    assert "--enable-expert-parallel" in command


def test_contestant_server_command_does_not_enable_expert_parallel(tmp_path: Path) -> None:

    config = AIMOConfig(
        model_path=tmp_path / "missing-contestant-model",
        model_profile="contestant",
        served_model_name="OLMo-3.1-32B-Think",
    )
    command = AIMOInferenceServer(config=config).build_command()

    assert "--enable-expert-parallel" not in command


def test_vllm_exit_message_includes_log_tails_and_health_url(tmp_path: Path) -> None:

    config = AIMOConfig(
        model_path=tmp_path / "model",
        logdir=tmp_path / "logs",
        port=8123,
    )
    server = AIMOInferenceServer(config=config)
    server.stdout_path.parent.mkdir(parents=True, exist_ok=True)
    server.stdout_path.write_text("stdout tail\n", encoding="utf-8")
    server.stderr_path.write_text("stderr tail\n", encoding="utf-8")
    server.process = SimpleNamespace(returncode=17)

    message = server.build_process_exit_message()

    assert "return_code=17" in message
    assert "stdout tail" in message
    assert "stderr tail" in message
    assert "http://127.0.0.1:8123/health" in message
    assert str(server.command_path) in message


def test_server_self_probe_uses_loopback_when_binding_all_interfaces(tmp_path: Path) -> None:

    server = AIMOInferenceServer(AIMOConfig(
        host="0.0.0.0",
        port=8124,
        logdir=tmp_path / "logs",
    ))

    assert server.config.health_url == "http://0.0.0.0:8124/health"
    assert server.local_health_url() == "http://127.0.0.1:8124/health"


def test_vllm_exit_message_handles_missing_log_files(tmp_path: Path) -> None:

    server = AIMOInferenceServer(AIMOConfig(logdir=tmp_path / "logs"))
    server.process = SimpleNamespace(returncode=2)

    message = server.build_process_exit_message()

    assert "return_code=2" in message
    assert "missing log file" in message


def test_service_preflight_writes_artifact_for_valid_model(tmp_path: Path) -> None:

    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}\n", encoding="utf-8")
    (model_path / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (model_path / "model.safetensors").write_text("weights\n", encoding="utf-8")
    config = AIMOConfig(
        model_path=model_path,
        logdir=tmp_path / "logs",
        port=free_port(),
        tensor_parallel_size=8,
    )

    payload = AIMOServicePreflight(
        config=config,
        role="contestant",
        rank=1,
        detected_gpu_count=8,
    ).run()

    assert payload["passed"] is True
    assert payload["rank"] == 1
    assert payload["preflight_scope"] == "filesystem_ports_gpus_temporary_paths"
    assert (tmp_path / "logs" / "service_preflight.json").exists()


def test_service_preflight_fails_before_launch_for_missing_model(tmp_path: Path) -> None:

    config = AIMOConfig(
        model_path=tmp_path / "missing-model",
        logdir=tmp_path / "logs",
        port=free_port(),
        tensor_parallel_size=8,
    )

    with pytest.raises(RuntimeError, match="model path does not exist"):
        AIMOServicePreflight(
            config=config,
            role="contestant",
            rank=1,
            detected_gpu_count=8,
        ).run()

    payload = json.loads((tmp_path / "logs" / "service_preflight.json").read_text(
        encoding="utf-8",
    ))

    assert payload["passed"] is False
    assert "model path does not exist" in "; ".join(payload["failures"])


def test_service_preflight_rejects_incomplete_tokenizer_metadata(tmp_path: Path) -> None:

    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}\n", encoding="utf-8")
    (model_path / "special_tokens_map.json").write_text("{}\n", encoding="utf-8")
    (model_path / "model.safetensors").write_text("weights\n", encoding="utf-8")
    config = AIMOConfig(
        model_path=model_path,
        logdir=tmp_path / "logs",
        port=free_port(),
        tensor_parallel_size=8,
    )

    with pytest.raises(RuntimeError, match="tokenizer metadata is missing"):
        AIMOServicePreflight(
            config=config,
            role="contestant",
            rank=1,
            detected_gpu_count=8,
        ).run()


def test_service_preflight_rejects_unreadable_model_files(tmp_path: Path) -> None:

    model_path = tmp_path / "model"
    model_path.mkdir()
    config_path = model_path / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    (model_path / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (model_path / "model.safetensors").write_text("weights\n", encoding="utf-8")
    config_path.chmod(0)

    if os.access(config_path, os.R_OK):
        pytest.skip("unreadable-file permissions are not enforced on this platform")

    config = AIMOConfig(
        model_path=model_path,
        logdir=tmp_path / "logs",
        port=free_port(),
        tensor_parallel_size=8,
    )

    with pytest.raises(RuntimeError, match="model files are not readable"):
        AIMOServicePreflight(
            config=config,
            role="contestant",
            rank=1,
            detected_gpu_count=8,
        ).run()


def test_generation_token_resolution_caps_context() -> None:

    config = AIMOConfig(
        max_model_len=81920,
        max_new_tokens=4000,
    )

    assert config.max_tokens_for_pass(configured_max_tokens=0) == 4000
    assert config.max_tokens_for_pass(configured_max_tokens=90000) == (
        MAX_GENERATION_CONTEXT_TOKENS
    )
    assert config.available_generation_tokens(input_tokens=4096) == 61440


def test_profile_defaults_for_olmo_and_gpt_oss() -> None:

    contestant_config = AIMOConfig().with_profile_defaults("contestant")
    judge_config = AIMOConfig().with_profile_defaults("judge")
    answer_config = AIMOConfig().with_inference_mode_defaults("aimo3_answer")

    assert contestant_config.model_profile == "contestant"
    assert contestant_config.served_model_name == "OLMo-3.1-32B-Think"
    assert contestant_config.template_format == "chatml"
    assert contestant_config.kv_cache_dtype == "auto"
    assert judge_config.model_profile == "judge"
    assert judge_config.served_model_name == "GPT-OSS-120B"
    assert judge_config.judge_served_model_name == "GPT-OSS-120B"
    assert judge_config.template_format == "harmony"
    assert judge_config.min_p == 0.02
    assert judge_config.kv_cache_dtype == "auto"
    assert judge_config.moe_backend == "marlin"
    assert judge_config.enable_expert_parallel is True
    assert answer_config.inference_mode == "aimo3_answer"
    assert answer_config.template_format == "harmony"
    assert answer_config.top_logprobs == 0
    assert answer_config.kv_cache_dtype == "fp8_e4m3"


def free_port() -> int:

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(("127.0.0.1", 0))

        return int(server_socket.getsockname()[1])


def test_grpo_runtime_defaults() -> None:

    config = AIMOConfig()

    assert config.group_size == 16
    assert config.active_problem_count == 6
    assert config.sandbox_count == 96
