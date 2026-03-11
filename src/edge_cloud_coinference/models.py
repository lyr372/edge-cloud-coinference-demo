from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ModelProfile:
    name: str
    params_billion: float
    side: str


def parse_qwen_params(model_name: str) -> float:
    """Extract parameter size (in billions) from model name.

    Supports names like `Qwen2.5-1.5B-Instruct`, `Qwen2.5-72B`, etc.
    """
    if "-" not in model_name or "B" not in model_name:
        raise ValueError(f"Unsupported Qwen model name: {model_name}")

    segment = model_name.split("-")[1]
    size_text = segment.replace("B", "")
    try:
        return float(size_text)
    except ValueError as exc:
        raise ValueError(f"Cannot parse parameter size from: {model_name}") from exc


def build_profile(model_name: str, side: str) -> ModelProfile:
    return ModelProfile(name=model_name, params_billion=parse_qwen_params(model_name), side=side)
