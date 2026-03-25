from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .models import ModelProfile


@dataclass(slots=True)
class ExecutorOutput:
    text: str
    confidence: float
    latency_ms: int
    first_token_ms: int
    metadata: dict[str, Any] | None = None


class BaseExecutor:
    def __init__(
        self,
        profile: ModelProfile,
        max_tokens: int,
        runtime: str = "onnxruntime",
        quantization: str = "fp16",
        distillation_enabled: bool = False,
    ) -> None:
        self.profile = profile
        self.max_tokens = max_tokens
        self.runtime = runtime
        self.quantization = quantization
        self.distillation_enabled = distillation_enabled

    def _simulate_confidence(self, prompt: str) -> float:
        base = min(0.95, 0.45 + math.log(self.profile.params_billion + 1) / 2)
        complexity_penalty = min(0.22, len(prompt) / 2500)
        if self.distillation_enabled and self.profile.side == "edge":
            base += 0.02
        return max(0.2, round(base - complexity_penalty, 3))

    def _quant_speedup_factor(self) -> float:
        return {"int4": 0.68, "int8": 0.78, "fp16": 1.0}.get(self.quantization, 1.0)

    def _runtime_factor(self) -> float:
        if self.profile.side != "edge":
            return 1.0
        runtime = "onnxruntime" if self.runtime == "onnx" else self.runtime
        return 0.86 if runtime == "mnn" else 1.0

    def _simulate_latency(self, prompt: str) -> int:
        token_factor = min(self.max_tokens, max(16, len(prompt) // 3))
        hardware_factor = 26 if self.profile.side == "edge" else 18
        network_penalty = 0 if self.profile.side == "edge" else 65
        raw = token_factor * hardware_factor / (self.profile.params_billion + 0.8) + network_penalty
        return int(raw * self._quant_speedup_factor() * self._runtime_factor())

    def _simulate_first_token_latency(self, prompt: str) -> int:
        base = 18 if self.profile.side == "edge" else 95
        return int(base + min(45, len(prompt) / 12))

    def infer(self, prompt: str) -> ExecutorOutput:
        confidence = self._simulate_confidence(prompt)
        latency = self._simulate_latency(prompt)
        first_token_ms = self._simulate_first_token_latency(prompt)
        text = (
            f"[{self.profile.name}@{self.profile.side}:{self.runtime}] "
            f"processed {min(self.max_tokens, len(prompt.split()))} tokens: {prompt[:140]}"
        )
        return ExecutorOutput(text=text, confidence=confidence, latency_ms=latency, first_token_ms=first_token_ms)


class EdgeExecutor(BaseExecutor):
    def _normalize_runtime(self) -> str:
        return "onnxruntime" if self.runtime == "onnx" else self.runtime

    def _provider(self) -> str:
        runtime = self._normalize_runtime()
        if runtime == "mnn":
            return "mnn_interpreter"
        return "onnxruntime_cpu"

    def infer(self, prompt: str) -> ExecutorOutput:
        runtime = self._normalize_runtime()
        if runtime not in {"onnxruntime", "mnn"}:
            raise ValueError(f"Unsupported edge runtime: {self.runtime}")

        out = super().infer(prompt)
        out.text = (
            f"[edge-{runtime}:{self.profile.name}] "
            f"provider={self._provider()} tokens={min(self.max_tokens, len(prompt.split()))} "
            f"quant={self.quantization} distill={self.distillation_enabled}"
        )
        out.metadata = {
            "edge_runtime": runtime,
            "edge_provider": self._provider(),
            "edge_quantization": self.quantization,
            "edge_distillation": str(self.distillation_enabled).lower(),
        }
        return out


class CloudExecutor(BaseExecutor):
    def __init__(
        self,
        profile: ModelProfile,
        max_tokens: int,
        runtime: str = "cloud-runtime",
        quantization: str = "fp16",
        distillation_enabled: bool = False,
        execution_mode: str = "simulated",
        api_base_url: str = "",
        api_model: str = "",
        full_model_runtime: str = "vllm",
    ) -> None:
        super().__init__(profile, max_tokens, runtime, quantization, distillation_enabled)
        self.execution_mode = execution_mode
        self.api_base_url = api_base_url
        self.api_model = api_model
        self.full_model_runtime = full_model_runtime

    def infer(self, prompt: str) -> ExecutorOutput:
        out = super().infer(prompt)
        metadata: dict[str, Any] = {"cloud_execution_mode": self.execution_mode}
        if self.execution_mode == "api":
            out.text = (
                f"[cloud-api reserved:{self.api_model}] "
                f"POST {self.api_base_url} -> prompt_size={len(prompt)}"
            )
            out.latency_ms = int(out.latency_ms * 1.15)
            metadata.update(
                {
                    "cloud_api_reserved": "true",
                    "cloud_api_model": self.api_model,
                    "cloud_api_base_url": self.api_base_url,
                }
            )
        elif self.execution_mode == "full_model":
            out.text = (
                f"[cloud-full-model reserved:{self.profile.name}] "
                f"runtime={self.full_model_runtime}; prompt_size={len(prompt)}"
            )
            out.latency_ms = int(out.latency_ms * 1.3)
            metadata.update(
                {
                    "cloud_full_model_reserved": "true",
                    "cloud_full_model_runtime": self.full_model_runtime,
                }
            )
        out.metadata = metadata
        return out
