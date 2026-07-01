# Local Setup

`venv-local` is the local environment.

## Create Virtual Environment

```bash
python3 -m venv venv-local
source venv-local/bin/activate
```

## Upgrade Packaging Tools

```bash
python -m pip install --upgrade pip setuptools wheel
```

## Install Hugging Face Hub and pyarrow

```bash
python -m pip install huggingface_hub click typer pyarrow pytest
```

## Freeze Requirements

```bash
pip freeze > requirements_local.txt
```

## Download MathNet Dataset

```bash
cd data
hf download ShadenA/MathNet --repo-type dataset --local-dir .
rm -rf .cache
```

## Download OLMo-3.1-32B-Think

```bash
cd models/contestant
hf download allenai/Olmo-3.1-32B-Think --repo-type model --local-dir .
rm -rf .cache
```

## Download GPT-OSS-120B

```bash
cd models/judge
hf download openai/gpt-oss-120b --repo-type model --local-dir . --exclude "metal/**" --exclude "original/**"
rm -rf .cache
```

## Download SmolLM-3B Dummy Checkpoint

```bash
cd models/dummy
hf download HuggingFaceTB/SmolLM3-3B --repo-type model --local-dir .
rm -rf .cache
```
