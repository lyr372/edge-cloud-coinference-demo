from __future__ import annotations

import math
from dataclasses import dataclass

from .models import ModelProfile


@dataclass(slots=True)
class ExecutorOutput:
    text: str
    confidence: float
    latency_ms: int


class BaseExecutor:
    def __init__(self, profile: ModelProfile, max_tokens: int) -> None:
        self.profile = profile
        self.max_tokens = max_tokens

    def _simulate_confidence(self, prompt: str) -> float:
        base = min(0.95, 0.45 + math.log(self.profile.params_billion + 1) / 2)
        complexity_penalty = min(0.2, len(prompt) / 3000)
        return max(0.2, round(base - complexity_penalty, 3))

    def _simulate_latency(self, prompt: str) -> int:
        token_factor = min(self.max_tokens, max(16, len(prompt) // 3))
        # Edge is faster on short text; cloud has network overhead but better throughput on long text.
        hardware_factor = 26 if self.profile.side == "edge" else 18
        network_penalty = 0 if self.profile.side == "edge" else 65
        return int(token_factor * hardware_factor / (self.profile.params_billion + 0.8) + network_penalty)

    def infer(self, prompt: str) -> ExecutorOutput:
        confidence = self._simulate_confidence(prompt)
        latency = self._simulate_latency(prompt)
        text = (
            f"[{self.profile.name}@{self.profile.side}] "
            f"processed {min(self.max_tokens, len(prompt.split()))} tokens: {prompt[:140]}"
        )
        return ExecutorOutput(text=text, confidence=confidence, latency_ms=latency)


class EdgeExecutor(BaseExecutor):
    pass


class CloudExecutor(BaseExecutor):
    pass
