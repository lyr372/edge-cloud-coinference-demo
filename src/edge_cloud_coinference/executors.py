from __future__ import annotations

import math
from dataclasses import dataclass

from .models import ModelProfile


@dataclass(slots=True)
class ExecutorOutput:
    text: str
    confidence: float
    latency_ms: int
    first_token_ms: int


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
        return 0.86 if self.runtime == "mnn" else 1.0

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
    pass


class CloudExecutor(BaseExecutor):
    pass
