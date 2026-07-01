from __future__ import annotations

import math
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AIMOLoRAConfig:

    rank: int
    alpha: int
    target_module_suffixes: list[str]
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    use_rslora: bool = True

    def as_dict(self) -> dict[str, Any]:

        return asdict(self)


def inject_lora_adapters(model: object, config: AIMOLoRAConfig) -> list[str]:

    import torch

    replaced_modules: list[str] = []

    for module_name, parent, child_name, child in iter_named_children(model):
        if not isinstance(child, torch.nn.Linear):
            continue

        if not should_target_module(module_name, config.target_module_suffixes):
            continue

        setattr(
            parent,
            child_name,
            AIMOLoRALinear(
                base_layer=child,
                rank=config.rank,
                alpha=config.alpha,
                use_rslora=config.use_rslora,
            ),
        )
        replaced_modules.append(module_name)

    if not replaced_modules:
        raise ValueError("No target modules matched for LoRA injection.")

    return replaced_modules


def iter_named_children(model: object) -> list[tuple[str, object, str, object]]:

    modules: list[tuple[str, object, str, object]] = []

    for parent_name, parent in model.named_modules():
        for child_name, child in parent.named_children():
            module_name = f"{parent_name}.{child_name}" if parent_name else child_name
            modules.append((module_name, parent, child_name, child))

    return modules


def should_target_module(module_name: str, suffixes: list[str]) -> bool:

    return any(
        module_name.endswith(suffix)
        or f".{suffix}." in module_name
        for suffix in suffixes
    )


class AIMOLoRALinear:

    def __new__(
        cls,
        base_layer: object,
        rank: int,
        alpha: int,
        use_rslora: bool,
    ) -> object:

        import torch

        class _AIMOLoRALinear(torch.nn.Module):

            def __init__(self) -> None:

                super().__init__()
                self.base_layer = base_layer
                self.rank = rank
                self.alpha = alpha
                self.scaling = alpha / math.sqrt(rank) if use_rslora else alpha / rank
                self.lora_down = torch.nn.Linear(base_layer.in_features, rank, bias=False)
                self.lora_up = torch.nn.Linear(rank, base_layer.out_features, bias=False)
                torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
                torch.nn.init.zeros_(self.lora_up.weight)
                self.lora_down.to(
                    device=base_layer.weight.device,
                    dtype=base_layer.weight.dtype,
                )
                self.lora_up.to(
                    device=base_layer.weight.device,
                    dtype=base_layer.weight.dtype,
                )

                for parameter in self.base_layer.parameters():
                    parameter.requires_grad = False

            def forward(self, input_tensor: object) -> object:

                return self.base_layer(input_tensor) + self.lora_up(
                    self.lora_down(input_tensor)
                ) * self.scaling

        return _AIMOLoRALinear()


def lora_state_dict(model: object) -> dict[str, object]:

    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if ".lora_" in name
    }


def mark_only_lora_trainable(model: object) -> None:

    for name, parameter in model.named_parameters():
        parameter.requires_grad = ".lora_" in name


def load_lora_adapter(model: object, adapter_path: Path) -> None:

    from safetensors.torch import load_file

    adapter_state = load_file(str(adapter_path))
    parameters = dict(model.named_parameters())
    missing_names = []

    for name, tensor in adapter_state.items():
        parameter = parameters.get(name)

        if parameter is None:
            missing_names.append(name)
            continue

        parameter.data.copy_(tensor.to(
            device=parameter.device,
            dtype=parameter.dtype,
        ))

    if missing_names:
        missing_text = ", ".join(missing_names[:5])

        raise ValueError(f"Adapter contains parameters not present in model: {missing_text}")


def save_lora_adapter(
    model: object,
    adapter_path: Path,
    config_path: Path,
    config: AIMOLoRAConfig,
    replaced_modules: list[str],
) -> None:

    from safetensors.torch import save_file

    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(lora_state_dict(model), str(adapter_path))
    config_payload = config.as_dict()
    config_payload["replaced_modules"] = replaced_modules
    config_payload["adapter_path"] = str(adapter_path)
    config_path.write_text(
        json_dumps(config_payload),
        encoding="utf-8",
    )


def json_dumps(payload: dict[str, Any]) -> str:

    import json

    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
