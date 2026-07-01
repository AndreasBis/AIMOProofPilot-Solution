# AIMOProofPilot 2026-06-11 Training Job

This is a Fields/NII train-variant package for a 3-node online GRPO training run.

The package is transferred as a Google Drive zip because S3 transfer is not available for this submission. The zip should contain this README at the package root.

## Package Contents

```text
README.md
upload.py
requirements_container.txt
aimo-proof_train_20260611.sif
aimo-proof_train_20260611.def
dataset/aimo_proof_train.parquet
models/contestant/
models/judge/
```

The Singularity image contains code and dependencies only. Model weights, tokenizer files, chat templates, and the training dataset are external package files.

Required contestant model files are under:

```text
models/contestant/
```

Required judge model files are under:

```text
models/judge/
```

Both model directories must include `config.json`, tokenizer metadata, chat template files when provided by the model, and all `.safetensors` weight shards.

For dummy pipeline checks, set `--dummy_test true` and replace the production model directories with:

```text
models/dummy/
```

In that mode, both contestant and judge paths resolve to the SmolLM-3B checkpoint under `models/dummy/`. The judge uses ChatML instead of GPT-OSS Harmony, and the production Harmony judge path remains unchanged when dummy mode is off.

## Compute Target

The intended run uses:

```text
3 nodes
8 H200 GPUs per node
24 H200 GPUs total
```

Run one top-level container process per node. Do not launch one top-level service process per GPU. vLLM uses the 8 GPUs on each service node internally through tensor parallelism.

Current package topology:

```text
rank 0: GPT-OSS-120B judge vLLM service, tensor_parallel_size=8, port 8000
rank 1: OLMo-3.1-32B-Think contestant rollout vLLM service, tensor_parallel_size=8, port 8001
rank 2: online controller and trainer
```

The judge and contestant are independent single-node vLLM services. They are not one multi-node vLLM replica.

## Required Environment

The cluster wrapper should set:

```text
GLOBAL_RANK
WORLD_SIZE
MASTER_ADDR
MASTER_PORT
```

For this package:

```text
WORLD_SIZE=3
GLOBAL_RANK=0 on the judge node
GLOBAL_RANK=1 on the contestant rollout node
GLOBAL_RANK=2 on the controller/trainer node
```

The trainer process may also use:

```text
LOCAL_RANK
CUDA_VISIBLE_DEVICES
```

## Command

Replace `<rank0_host>,<rank1_host>,<rank2_host>` with the actual hostnames assigned to ranks 0, 1, and 2.

```bash
apptainer run --nv aimo-proof_train_20260611.sif \
    --model_path /shared/models/contestant \
    --judge_model_path /shared/models/judge \
    --dataset_path /shared/dataset/aimo_proof_train.parquet \
    --output_path /shared/output/adapters \
    --logdir /shared/output/logs \
    --online true \
    --problems_per_update 64 \
    --group_size 16 \
    --active_problem_count 6 \
    --judge_node_rank 0 \
    --rollout_node_ranks 1 \
    --trainer_node_rank 2 \
    --node_hostnames <rank0_host>,<rank1_host>,<rank2_host> \
    --online_control_dir /shared/output/logs/online_control \
    --adapter_reload_timeout_seconds 900 \
    --allow_base_rollouts true
```

For the dummy package, keep the same topology and add:

```text
--dummy_test true
--dummy_model_path /shared/models/dummy
```

If an initial LoRA adapter is supplied, remove `--allow_base_rollouts true` and add:

```text
--initial_adapter_path /shared/optional_initial_adapter
```

## Service Details

The contestant rollout service uses:

```text
model: OLMo-3.1-32B-Think
port: 8001
tensor_parallel_size: 8
tool protocol: OLMo ChatML Python tools
```

The judge service uses:

```text
model: GPT-OSS-120B
port: 8000
tensor_parallel_size: 8
moe backend: marlin
expert parallelism: enabled
tool protocol: Harmony Python tools
```

The controller waits for both services and fails fast if either service writes a failure marker.

## Diagnostics

On failure, the package writes diagnostic artifacts under `/shared/output/logs` and `/shared/output/adapters/failure_artifacts`.

Important files include:

```text
/shared/output/logs/failure_report.json
/shared/output/logs/failure_report.txt
/shared/output/logs/failure_traceback.txt
/shared/output/logs/phase_events.jsonl
/shared/output/logs/online_events.jsonl
/shared/output/logs/online_control/judge_rank_0_failed.json
/shared/output/logs/online_control/contestant_rank_1_failed.json
/shared/output/logs/online_servers/rank_0_judge/service_preflight.json
/shared/output/logs/online_servers/rank_0_judge/vllm_stdout.log
/shared/output/logs/online_servers/rank_0_judge/vllm_stderr.log
/shared/output/logs/online_servers/rank_0_judge/vllm_command.json
/shared/output/logs/online_servers/rank_1_contestant/service_preflight.json
/shared/output/logs/online_servers/rank_1_contestant/vllm_stdout.log
/shared/output/logs/online_servers/rank_1_contestant/vllm_stderr.log
/shared/output/logs/online_servers/rank_1_contestant/vllm_command.json
```

The service preflight checks model paths, tokenizer files, weight files, log writability, port availability, visible GPU count, `/tmp`, and `/dev/shm` before launching vLLM. Failure markers include `launch_stage` and `service_preflight_path` so preflight failures are distinguishable from vLLM process failures.

## Expected Outputs

Successful training writes adapter chunks under:

```text
/shared/output/adapters/chunk_00000/adapter_model.safetensors
/shared/output/adapters/chunk_00000/adapter_config.json
/shared/output/adapters/chunk_00001/adapter_model.safetensors
/shared/output/adapters/chunk_00001/adapter_config.json
```

The controller writes:

```text
/shared/output/logs/online_training_manifest.json
/shared/output/logs/online_control/adapter_state.json
/shared/output/logs/online_control/judge_rank_0_ready.json
/shared/output/logs/online_control/contestant_rank_1_ready.json
/shared/output/logs/online_chunks/chunk_00000/grpo_groups.jsonl
```

## Return Artifacts

If S3 return upload is unavailable, return the full `/shared/output` directory by the agreed alternate transfer method.

If an S3 return URL is provided later, `upload.py` supports:

```bash
python upload.py \
    --source_dir /shared/output \
    --s3_url "$FIELDS_UPLOAD_S3_URL"
```

The upload helper inventories logs, adapters, failure reports, service preflight reports, vLLM logs, and command files before uploading.

## Pre-Run Checks

Before launching the training job, verify:

```text
aimo-proof_train_20260611.sif exists
aimo-proof_train_20260611.def exists
apptainer test aimo-proof_train_20260611.sif passes
dataset/aimo_proof_train.parquet exists
models/contestant/config.json exists
models/contestant/tokenizer.json or tokenizer_config.json exists
models/contestant/*.safetensors exists
models/judge/config.json exists
models/judge/tokenizer.json or tokenizer_config.json exists
models/judge/*.safetensors exists
```

For dummy checks, replace the contestant and judge model checks with:

```text
models/dummy/config.json exists
models/dummy/tokenizer.json or tokenizer_config.json exists
models/dummy/*.safetensors exists
```
