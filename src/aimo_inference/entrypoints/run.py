from __future__ import annotations

import contextlib
import os
from dataclasses import asdict
from pathlib import Path

from aimo_inference.answer import AIMOAnswerEngine
from aimo_inference.client import AIMOInferenceClient
from aimo_inference.config import AIMOConfig
from aimo_inference.io import AIMOInferenceIO
from aimo_inference.io import AIMOProblemResult
from aimo_inference.io import file_sha256
from aimo_inference.judge import AIMOJudgeEngine
from aimo_inference.judge import AIMOProofJudge
from aimo_inference.refinement import AIMORefinementEngine
from aimo_inference.scheduler import AIMOScheduler
from aimo_inference.server import AIMOInferenceServer


def main(argv: list[str] | None = None) -> int:

    config = AIMOConfig.from_cli_args(argv)
    run(config=config)

    return 0


def run(config: AIMOConfig) -> None:

    io_manager = AIMOInferenceIO(config=config)
    records = io_manager.read_records()
    client = AIMOInferenceClient(config=config)
    engine = build_engine(config=config, client=client)
    server_context = build_server_context(config=config)

    with server_context:
        scheduler = AIMOScheduler(
            config=config,
            engine=engine,
        )
        results, summary = scheduler.run(records)

    output_path = write_results(
        io_manager=io_manager,
        config=config,
        results=results,
    )
    io_manager.write_problem_logs(results)
    io_manager.write_run_metadata({
        "config": serializable_config(config),
        "summary": asdict(summary),
        "environment": runtime_environment_summary(),
        "resolved_paths": resolved_paths(config=config, output_path=output_path),
        "output_sha256": file_sha256(output_path),
    })


def build_engine(
    config: AIMOConfig,
    client: AIMOInferenceClient,
) -> object:

    if config.inference_mode == "aimo3_answer":
        return AIMOAnswerEngine(
            config=config,
            client=client,
        )

    if config.inference_mode == "judge":
        judge = AIMOProofJudge(
            config=config,
            client=client,
        )

        return AIMOJudgeEngine(
            config=config,
            judge=judge,
        )

    return AIMORefinementEngine(
        config=config,
        client=client,
        judge=build_optional_judge(config=config),
    )


def build_optional_judge(config: AIMOConfig) -> AIMOProofJudge | None:

    if not config.enable_judge:
        return None

    judge_config = config.with_profile_defaults("judge").with_runtime_topology(
        config.mode
    ).with_overrides(
        model_path=config.judge_model_path,
        port=config.judge_port,
        api_base=config.judge_api_base,
        served_model_name=config.judge_served_model_name,
        launch_server=False,
        reuse_server=True,
    )
    judge_client = AIMOInferenceClient(config=judge_config)

    return AIMOProofJudge(
        config=judge_config,
        client=judge_client,
    )


def build_server_context(config: AIMOConfig) -> contextlib.AbstractContextManager[object]:

    if config.launch_server or config.reuse_server:
        return AIMOInferenceServer(config=config)

    return contextlib.nullcontext()


def write_results(
    io_manager: AIMOInferenceIO,
    config: AIMOConfig,
    results: list[AIMOProblemResult],
) -> Path:

    if config.inference_mode == "aimo3_answer":
        return io_manager.write_answers(results)

    if config.inference_mode == "judge":
        return io_manager.write_judge_results(results)

    return io_manager.write_predictions(results)


def serializable_config(config: AIMOConfig) -> dict[str, object]:

    payload = asdict(config)

    return {
        key: serializable_config_value(key=key, value=value)
        for key, value in payload.items()
    }


def serializable_config_value(key: str, value: object) -> object:

    if value is None:
        return None

    if key.endswith("_path") or key.endswith("_csv") or key == "logdir":
        return str(value)

    return value


def runtime_environment_summary() -> dict[str, str]:

    environment_names = [
        "GLOBAL_RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "CUDA_VISIBLE_DEVICES",
        "AIMO_MODE",
        "AIMO_RUNTIME_MODE",
        "AIMO_INFERENCE_MODE",
        "AIMO_MODEL_PROFILE",
        "AIMO_NUM_GPUS",
        "AIMO_TENSOR_PARALLEL_SIZE",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "TORCH_HOME",
        "TRITON_CACHE_DIR",
        "VLLM_CACHE_ROOT",
    ]

    return {
        name: os.environ[name]
        for name in environment_names
        if name in os.environ
    }


def resolved_paths(config: AIMOConfig, output_path: Path) -> dict[str, str]:

    return {
        "model_path": str(config.model_path),
        "contestant_model_path": str(config.contestant_model_path),
        "judge_model_path": str(config.judge_model_path),
        "input_csv": str(config.input_csv),
        "output_csv": str(output_path),
        "requested_output_csv": str(config.output_csv),
        "logdir": str(config.logdir),
        "dataset_manifest_path": str(dataset_manifest_path(config=config)),
        "vllm_command_path": str(config.logdir / "vllm_command.json"),
        "vllm_stdout_path": str(config.logdir / "vllm_stdout.log"),
        "vllm_stderr_path": str(config.logdir / "vllm_stderr.log"),
    }


def dataset_manifest_path(config: AIMOConfig) -> Path:

    candidates = [
        config.input_csv.parent / "manifest.json",
        config.eval_dataset_path.parent / "manifest.json",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


if __name__ == "__main__":
    raise SystemExit(main())
