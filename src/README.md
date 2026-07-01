# Source Code Status

The source code under `src/` is an **incomplete** engineering implementation and is **not** yet suitable for independent production or competition usage. It should be read as a structured prototype of the intended system rather than as a turnkey package. Further debugging is required across runtime integration, large-model service launch, tool execution reliability, online GRPO orchestration, adapter handoff, and end-to-end container execution.

The high-level concept is a modular offline proof-generation stack:

- `aimo_data` builds deterministic MathNet-derived proof and judge datasets with held-out evaluation rows, training rows, page-count metadata, and reproducibility manifests.
- `aimo_inference` provides the proof-generation runtime: model profiles, vLLM/OpenAI-compatible clients, prompt templates, page-count checks, optional Python sandbox tooling, judge calls, result writing, and per-problem logs.
- `aimo_training` encodes the intended online GRPO loop: rollout generation, judge-based reward scoring, durable complete-group queues, LoRA adapter training, adapter-state publication, and rollout-service reload coordination.
- `execution` packages result directories for upload and records inventories, archive metadata, and receipts.
- `app` and `containers` wrap the same source package for Fields/NII-style Singularity execution without embedding model weights inside the image.

The design target was one coherent code path that could support local tests, Fields/NII training jobs, and Kaggle-style inference notebooks. The current repository preserves that design and its test coverage, but the maintained final inference artifact is the Kaggle notebook path rather than a fully validated `src` deployment.

# AIMO Data

The `aimo_data` package builds deterministic MathNet-derived datasets for local Proof Pilot development. It reads parquet shards from `data/data/all` by default and writes all artifacts under `output/data`.

Run it from the repository root with `PYTHONPATH=src`:

```bash
PYTHONPATH=src python -m aimo_data.build_dataset \
    --source_dir data/data/all \
    --output_dir output/data \
    --eval_size 16 \
    --train_size 4096 \
    --seed 42 \
    --exclude_images true \
    --language_filter any \
    --page_count_method latex
```

Required outputs are:

- `aimo_proof_eval.parquet`: 16 held-out problem metadata rows without reference solutions.
- `aimo_proof_eval_input.csv`: runtime-safe input with exactly `id,problem`.
- `aimo_proof_eval_reference.parquet`: 16 held-out references and metadata rows for local judging only.
- `aimo_proof_train.parquet`: 4096 deterministic topic-stratified proof training rows excluding held-out eval ids.
- `aimo_judge_train.parquet`: 4096 judge/reward rows matching the proof training split.
- `manifest.json`: source paths, filter counts, excluded counts, eval ids, selected train size, selected train topic counts, arguments, row counts, output hashes, and creation timestamp.

By default the proof builder keeps `proof only` and `proof and answer` rows, drops rows without reference solutions, excludes image-bearing rows, preserves non-English rows, reserves 16 held-out eval problems, and keeps 4096 deterministic topic-stratified train problems. Use `--language_filter English` for English-only smoke datasets, `--exclude_images false` only when image-dependent rows are acceptable, and `--train_size 0` only when a full eligible train split is intentionally needed.

Reference page counts are recorded with the requested method and fallback order: `latex`, `sanitized_latex`, `line_count`, then `word_count`. If a LaTeX executable is unavailable or rendering fails, the builder falls back automatically and records the method actually used.

Optional answer-mode rows can be emitted with `--write_answer_dataset true`. That writes `aimo_answer_train.parquet` with `id,problem,answer` only when `final_answer` parses to an integer in the configured answer range.

# AIMO Inference

The `aimo_inference` package runs three offline inference modes:

- `proof`: uses the contestant profile, defaults to `models/contestant`, runs the three proof passes `solve`, `audit_repair`, and `finalize`, and writes `id,prediction`.
- `judge`: uses the GPT-OSS judge profile, extracts the last boxed grade from `0`, `1`, `6`, or `7`, and writes `id,grade`.
- `aimo3_answer`: uses the GPT-OSS Harmony path with Python tools, runs one sequence per problem, extracts an integer from `0` through `7`, and writes `id,answer`.

Run the Fields-compatible entry point from the repository root with `PYTHONPATH=src`:

```bash
PYTHONPATH=src python -m aimo_inference.entrypoints.run \
    --mode proof \
    --model_path models/contestant \
    --input_csv output/data/aimo_proof_eval_input.csv \
    --output_csv output/inference/predictions.csv \
    --logdir output/inference/logs
```

The `/app/run.py` wrapper accepts the same options for container submissions. Optional controls include `--num_ctx`, `--max_new_tokens`, `--temperature`, `--top_p`, `--top_k`, `--min_p`, `--num_gpus`, `--judge_model_path`, `--dummy_test`, `--dummy_model_path`, `--max_python_calls`, `--problem_timeout_seconds`, `--contestant_port`, `--judge_port`, `--group_size`, `--active_problem_count`, `--sandbox_count`, `--kv_cache_dtype`, `--page_count_method`, and `--page_template`.

`colab` mode uses tensor parallel size `1`. `singularity` mode uses tensor parallel size `8`, so the contestant OLMo server is intended for the first 8-H200 node, the GPT-OSS judge server for the second 8-H200 node, and training/gradient updates for the third node.

Set `--dummy_test true` to resolve both contestant and judge paths to `models/dummy` or to the path supplied by `--dummy_model_path`. Dummy mode uses SmolLM-3B for both services and switches the judge template from GPT-OSS Harmony to ChatML.

All modes write per-problem logs and `run_metadata.json` under `--logdir`. Proof logs include pass tokens, finish reasons, latency, Python-call counts, prompt hashes, model profile metadata, and page-count reward metadata.

# AIMO Training

The `aimo_training` package trains an unmerged Rank-Stabilized LoRA adapter for the OLMo contestant model from complete GRPO groups. A group contains one problem, sixteen rollout proofs by default, GPT-OSS boxed grades, context rewards, rendered page-count rewards, scalar rewards, token counts, and tool statistics.

The intended three-node workflow is:

- Node 0 runs the OLMo rollout vLLM endpoint and writes complete groups to a durable `grpo_groups.jsonl` queue.
- Node 1 runs the GPT-OSS judge vLLM endpoint and returns boxed grades from `0`, `1`, `6`, or `7`.
- Node 2 runs `/app/train.py`, consumes only complete groups, and writes the LoRA adapter plus logs and manifests.

The online GRPO path consumes the 4096-problem train split in 128 updates at 64 problems per update. Each update rolls out 16 completions per problem, writes 1024 rollout samples to the queue, trains one adapter chunk, publishes that adapter state, waits for rollout servers to reload it, and records the adapter update index, adapter hash, and adapter path on subsequent rollout samples.

For coordinator scripts, `aimo_training.rollout.AIMORolloutCoordinator` connects to the external rollout and judge endpoints, uses one sandbox per rollout sequence, scores rewards, and appends complete groups through `AIMODurableGroupQueue`.

Run the Fields-compatible train entry point from the repository root with `PYTHONPATH=src`:

```bash
PYTHONPATH=src python -m aimo_training.entrypoints.train \
    --model_path models/contestant \
    --dataset_path output/data/aimo_proof_train.parquet \
    --output_path output/training/adapter \
    --logdir output/training/logs
```

The `/app/train.py` wrapper accepts the same required arguments. Optional controls include `--num_gpus`, `--learning_rate`, `--num_train_epochs`, `--per_device_batch_size`, `--gradient_accumulation_steps`, `--lora_rank`, `--lora_alpha`, `--max_model_len`, `--group_size`, `--judge_model_path`, `--dummy_test`, `--dummy_model_path`, `--judge_port`, `--rollout_temperature`, `--rollout_top_p`, `--rollout_min_p`, `--max_python_calls`, `--reward_weights_json`, `--active_problem_count`, `--sandbox_count`, `--kv_cache_dtype`, `--page_count_method`, `--page_template`, `--importance_sampling_level`, and `--kl_beta`.

By default the trainer expects complete groups at `LOGDIR/grpo_groups.jsonl`. You can also pass a `.jsonl` file directly as `--dataset_path`, or a directory containing `grpo_groups.jsonl`. Raw MathNet parquet or CSV input is used for manifests and problem metadata, but gradient updates require complete GRPO groups so that no problem group is split across optimizer steps.

The scalar reward is:

```text
reward = judge_grade + context_reward + solution_page_reward
```

The page reward uses the same canonical LaTeX rendering path as inference, with deterministic fallbacks through sanitized LaTeX, line count, and word count. Judge parse failures are assigned grade `0` and logged separately.

Each run writes `adapter_model.safetensors` and `adapter_config.json` under `--output_path`. Logs under `--logdir` include training arguments, reward configuration, source dataset manifest, tokenizer reference manifest, per-step reward summaries, full rollout training table, sample generated proofs, judge parse-failure rows, checkpoint hashes, and final evaluation summary.

Training sequences include tool-output tokens so the model receives the same transcript that the rollout policy saw, but the loss mask excludes Python/tool-output tokens and computes policy loss only on model-generated tokens. Rollout requests keep top-logprob alternatives disabled for CPU efficiency while retaining selected-token logprobs required by GRPO ratios.

Detailed logs include `gradient_update_reward_summaries.jsonl`, `gradient_update_reward_samples.jsonl`, `online_events.jsonl`, rollout failure reports, skipped group reports when present, and adapter version metadata for each rollout/training handoff.

# Execution

The repository has three execution targets:

- Kaggle notebook inference keeps competition paths and dependencies separate from the Fields code. AIMO3 answer mode writes `id,answer`; Proof Pilot mode writes `id,prediction`.
- Fields/NII run packages use `/app/run.py` inside a Singularity image and keep model weights outside the SIF.
- Fields/NII train packages use `/app/train.py` inside a Singularity image and keep base weights, tokenizer metadata, datasets, and generated adapters outside the SIF.

The Singularity definition files are:

- `containers/aimo-proof_run_20260611.def`
- `containers/aimo-proof_train_20260611.def`

The run image entry point accepts `--model_path`, `--input_csv`, `--output_csv`, and `--logdir`, plus inference controls such as `--num_ctx`, `--max_new_tokens`, `--temperature`, `--top_p`, `--top_k`, `--min_p`, and `--num_gpus`. For multi-process run jobs, set `GLOBAL_RANK` and `WORLD_SIZE`; rows are sharded by `order_index % WORLD_SIZE == GLOBAL_RANK`. Each rank writes its own CSV under `LOGDIR/rank_outputs` unless an external merge step is provided.

The train image entry point accepts `--model_path`, `--dataset_path`, `--output_path`, and `--logdir`, plus `--num_gpus`, `--learning_rate`, `--num_train_epochs`, `--per_device_batch_size`, and `--gradient_accumulation_steps`. Multi-node training reads `GLOBAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`, `LOCAL_RANK`, and `CUDA_VISIBLE_DEVICES`. The train container runscript uses `torchrun` for `train_update` when `AIMO_TRAIN_NPROC_PER_NODE` or `AIMO_TRAIN_NNODES` requests distributed execution.

Both container definitions install `requirements_container.txt`, copy `src` and the relevant `/app` entry point, configure cache paths under `/tmp/aimo-cache` unless the host provides writable cache locations, install LaTeX and PDF page-count dependencies, and run a lightweight import test in `%test`.

Package result directories should contain outputs, logs, run metadata, manifests, failure reports when present, and training artifacts such as `adapter_model.safetensors` and `training_table.jsonl`. Upload a completed result directory with:

```bash
PYTHONPATH=src python -m execution.upload \
    --source_dir output/inference \
    --s3_url "$AIMO_UPLOAD_S3_URL"
```

If `--s3_url` is omitted, `execution.upload` reads `AIMO_UPLOAD_S3_URL`, `AIMO_S3_URL`, `FIELDS_UPLOAD_S3_URL`, `FIELDS_S3_URL`, `PRESIGNED_S3_URL`, or `S3_URL`.

The helper writes `upload_manifest.json`, creates a `.tar.gz` archive beside the source directory, uploads it to the presigned S3 URL, and writes `upload_receipt.json`.

# Tests

The test suite is organized by scope:

- `tests/unit`: focused tests for config parsing, templates, IO, scheduling, sandbox behavior, client normalization, refinement, judging, page counting, data building, training queues, rewards, and artifact writers.
- `tests/integration`: fake-service smoke tests for proof inference, Harmony tool loops, data-to-inference flow, judge-assisted proof metadata, mocked training dry runs, and interleaved GRPO scheduling.
- `tests/contract`: Fields/NII command-shape tests for `/app/run.py` and `/app/train.py` behavior through the Python entry points.
- `tests/fixtures`: tiny CSV and JSON payloads for proof input, alternative problem columns, fake OpenAI-compatible responses, fake Harmony tool calls, fake judge responses, and MathNet-like rows.

Run tests from the repository root:

```bash
pytest
```

Run only a scope:

```bash
pytest tests/unit
pytest tests/integration
pytest tests/contract
```

The default tests do not require model weights or internet access. GPU-dependent and model-dependent paths are exercised through fake HTTP servers, fake generation clients, mocked trainers, and tiny fixtures. Parquet-dependent tests create tiny local parquet files and skip when `pyarrow` is unavailable. Harmony tests use a fake `openai_harmony` module unless the test is specifically checking package availability.

One refinement contract test is marked `xfail` while sequential repair/finalize prompting still reuses the first-pass prompt. It records the intended behavior from `sandbox/TESTS_PLAN.md`: repair and finalize prompts should receive the previous generated solution.
