from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CollabMode = Literal["token", "task"]


@dataclass(slots=True)
class DeploymentConfig:
    """Static deployment configuration shared by all requests."""

    edge_model: str = "Qwen2.5-1.5B-Instruct"
    cloud_model: str = "Qwen2.5-7B-Instruct"
    collaboration_mode: CollabMode = "token"
    edge_max_tokens: int = 128
    cloud_max_tokens: int = 512


@dataclass(slots=True)
class RuntimeConfig:
    """Runtime knobs to control policy and latency behavior."""

    confidence_threshold: float = 0.72
    token_batch_size: int = 16
    cloud_retry: int = 1
