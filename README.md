# AIMOProofPilot

The final package version is `2026.6.25`.

## Repository Map

```text
AIMOProofPilot/
|-- .gitignore
|-- README.md
|-- CHANGELOG.md
|-- pyproject.toml
|-- requirements_container.txt
|-- requirements_local.txt
|-- app/
|   |-- README.md
|   |-- run.py
|   |-- train.py
|   `-- upload.py
|-- artifacts/
|   `-- README.md
|-- containers/
|   |-- README.md
|   |-- aimo-proof_run_20260611.def
|   `-- aimo-proof_train_20260611.def
|-- data/
|   |-- .gitattributes
|   |-- README.md
|   |-- __pycache__/
|   |-- assets/
|   |-- data/
|   |   |-- all/
|   |   `-- <competition_or_country>/
|   `-- _mathpix_cache/
|-- docs/
|   |-- SETUP_CONTAINER.md
|   `-- SETUP_LOCAL.md
|-- kaggle/
|   |-- README.md
|   |-- aimoproofpilot-submission.ipynb
|   `-- aimoproofpilot-utils.ipynb
|-- notebook_outputs/
|   |-- README.md
|   |-- aimo_notebook_status_1.json
|   |-- aimo_notebook_status_2.json
|   |-- aimo_notebook_status_3.json
|   |-- aimo_notebook_status_4.json
|   |-- aimo_notebook_status_5.json
|   |-- aimo_notebook_status_6.json
|   |-- aimo_proof_outputs_1.txt
|   |-- aimo_proof_outputs_2.txt
|   |-- aimo_proof_outputs_3.txt
|   |-- aimo_proof_outputs_4.txt
|   |-- aimo_proof_outputs_5.txt
|   `-- aimo_proof_outputs_6.txt
|-- notebooks/
|   |-- README.md
|   |-- submission_v1.ipynb
|   |-- submission_v2.ipynb
|   |-- submission_v3.ipynb
|   |-- submission_v4.ipynb
|   |-- submission_v5.ipynb
|   `-- submission_v6.ipynb
|-- output/
|   |-- .data_page_count_cache.json
|   `-- data/
|       |-- manifest.json
|       |-- aimo_proof_eval.parquet
|       |-- aimo_proof_eval_input.csv
|       |-- aimo_proof_eval_reference.parquet
|       |-- aimo_proof_train.parquet
|       `-- aimo_judge_train.parquet
|-- reports/
|   |-- README.md
|   |-- stage_1.tex
|   `-- stage_2.tex
|-- src/
|   |-- README.md
|   |-- aimo_data/
|   |-- aimo_inference/
|   |-- aimo_training/
|   `-- execution/
|-- tests/
|   |-- README.md
|   |-- conftest.py
|   |-- fixtures/
|   |-- unit/
|   |-- integration/
|   `-- contract/
|-- utils/
|   |-- README.md
|   |-- colab_hf_to_kaggle_olmo_upload.ipynb
|   |-- zip_training_drive_folder.ipynb
|   |-- upload_dummy_to_drive.ipynb
|   |-- upload_judge_to_drive.ipynb
|   |-- upload_contestant_to_drive.ipynb
|   `-- rclone_google_drive_oauth.md
`-- venv-local/
```

## What Is Included

`src/` contains the maintained Python code:

- `aimo_data`: deterministic dataset construction from MathNet-style parquet shards.
- `aimo_inference`: OpenAI-compatible vLLM inference clients, prompt templates, proof refinement, judging, page counting, Python sandbox execution, and runtime entry points.
- `aimo_training`: GRPO rollout orchestration, reward scoring, queueing, online adapter handoff, LoRA training, and training artifacts.
- `execution`: result inventory and upload packaging helpers.

`kaggle/` contains the Kaggle-facing notebooks. `aimoproofpilot-submission.ipynb` is the final submission notebook. `aimoproofpilot-utils.ipynb` builds the wheel cache expected by the submission environment.

`notebooks/` contains six development iterations before the final Kaggle notebook. The README in that directory explains the concrete differences from `submission_v1.ipynb` through `submission_v6.ipynb`.

`notebook_outputs/` contains captured proof outputs and run-status JSON from the notebook iterations. Its README summarizes the JSON metadata in tables so the raw JSON files are not the first thing a reader has to inspect.

`utils/` contains Colab and Google Drive utility notebooks for moving large model artifacts, creating a Kaggle Dataset mirror of OLMo, uploading model folders to Drive, and streaming a large Google Drive package ZIP through rclone.

`containers/` and `app/` contain the Fields/NII packaging surface. The Singularity/Apptainer definitions build runtime images that call `/app/run.py` or `/app/train.py`, while the model weights and generated artifacts remain outside the image.

`output/data/` contains the derived proof and judge datasets used by the local pipeline. `data/` contains the source MathNet-style dataset tree and supporting assets.

`reports/` contains the AIMO Proof Pilot staged report fragments. `stage_1.tex` narrates the full experimental process, including model selection, dataset selection, RLAIF engineering, dead ends, and human interventions. `stage_2.tex` records the final linear Kaggle submission pipeline and links to the final Kaggle artifacts. `reports/README.md` shows how to render the fragments locally.

## Tests

The remaining test suite is scoped to the Python modules under `src/`.

```bash
PYTHONPATH=src venv-local/bin/python -m pytest -q -p no:cacheprovider tests
```

The suite covers data building, inference clients, prompt templates, page counting, sandbox behavior, judging, refinement, rollout scheduling, GRPO queueing, online training handoff, upload inventory, and command-shape contracts for the Python entry points.

## Documentation Index

- `CHANGELOG.md`: dated project versions.
- `reports/README.md`: local rendering commands for the staged report fragments.
- `reports/stage_1.tex`: stage 1 full-process report.
- `reports/stage_2.tex`: stage 2 final-pipeline report.
- `src/README.md`: detailed source-package usage.
- `tests/README.md`: test-scope notes.
- `kaggle/README.md`: Kaggle notebook roles.
- `notebooks/README.md`: notebook iteration history.
- `notebook_outputs/README.md`: captured output and status summaries.
- `utils/README.md`: Colab, Google Drive, rclone, and Kaggle Dataset transfer utilities.
- `containers/README.md`: Singularity/Apptainer recipe notes.
- `artifacts/README.md`: image-build artifact workflow.
- `docs/SETUP_LOCAL.md`: local setup notes.
- `docs/SETUP_CONTAINER.md`: container setup notes.

## Notes

The repository is organized so that the source pipeline, notebook artifacts, Kaggle-specific notebooks, and utility transfer notebooks are separate. The final report can describe the modeling and engineering decisions without requiring readers to infer project structure from raw notebooks or captured logs.
