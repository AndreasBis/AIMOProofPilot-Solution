# Container Setup

`venv-container` is the local Python environment used to assemble and test the package set that later goes into the Singularity image. The Singularity definition files install from `requirements_container.txt`; the local venv is built explicitly, then frozen.

## Install System Utilities

On Fedora, install the host tools needed by local smoke tests, package builds, and page-counting checks:

```bash
sudo dnf install -y gcc gcc-c++ make ca-certificates curl git python3-devel poppler-utils texlive-scheme-basic texlive-collection-latexrecommended texlive-collection-fontsrecommended
```

This command is for the Fedora host only. The Singularity definition files use `apt-get` because their base image is Debian/Ubuntu-family.

`texlive-*` provides `pdflatex` for rendered page counting. `poppler-utils` provides `pdfinfo`.

## Create Virtual Environment

```bash
python3 -m venv venv-container
source venv-container/bin/activate
```

## Upgrade Packaging Tools

```bash
python -m pip install --upgrade pip setuptools wheel
```

## Install Runtime Requirements

Install the runtime Python stack explicitly. Do not use `requirements_container.txt` as the hand-built venv install command; it is the frozen record produced after this environment is resolved.

```bash
python -m pip install --no-cache-dir "vllm>=0.19.0" "numpy<=2.3.5" trl transformers peft datasets pyarrow openai-harmony safetensors jupyter-client sympy networkx mpmath z3-solver pytest
```

Keep NumPy below `2.4` while `mistral-common==1.11.2` is present, because that package requires `numpy<2.4`. `peft` is required for standard LoRA adapters. `jupyter-client` is required only when the persistent Jupyter sandbox path is enabled, but it should still be present in the container environment.

## Download Tiktoken Encodings

GPT-OSS-120B Harmony mode requires the `o200k_base.tiktoken` and `cl100k_base.tiktoken` files when running offline. OLMo ChatML-only paths do not need this setting. Keep the files under `venv-container` while developing locally, then copy or bind them into the runtime image.

```bash
mkdir -p venv-container/tiktoken_encodings

wget -O venv-container/tiktoken_encodings/o200k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"

wget -O venv-container/tiktoken_encodings/cl100k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
```

For local host runs, set `TIKTOKEN_ENCODINGS_BASE` to the `venv-container` directory. For baked Singularity images, set it to the path used inside the image, such as `/app/tiktoken_encodings`.

```bash
export TIKTOKEN_ENCODINGS_BASE="$PWD/venv-container/tiktoken_encodings"
```

## Freeze Requirements

After the explicit install succeeds, freeze the resolved environment. This file is used by the Singularity definition files and as an audit record.

```bash
pip freeze > requirements_container.txt
```

## Verify CUDA 13.0 Backend

Run these checks from the activated `venv-container`. The CUDA assertion should match the target container base image and NII runtime.

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
python -c "import torch; assert torch.version.cuda == \"13.0\", torch.version.cuda"
python -c "import vllm; print(vllm.__version__)"
```
