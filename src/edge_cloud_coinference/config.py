from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CollabMode = Literal["token", "task"]
EdgeRuntime = Literal["mnn", "onnxruntime", "onnx"]
CloudExecutionMode = Literal["simulated", "api", "full_model"]


@dataclass(slots=True)
class DeploymentConfig:
    """Static deployment configuration shared by all requests."""

    edge_model: str = "Qwen2.5-1.5B-Instruct"
    cloud_model: str = "Qwen2.5-7B-Instruct"
    collaboration_mode: CollabMode = "token"
    edge_max_tokens: int = 128
    cloud_max_tokens: int = 512
    edge_runtime: EdgeRuntime = "onnxruntime"
    quantization: str = "int8"
    distillation_enabled: bool = True
    cloud_execution_mode: CloudExecutionMode = "simulated"
    cloud_api_base_url: str = "https://api.example.com/v1/chat/completions"
    cloud_api_key_env: str = "EDGE_COINFER_CLOUD_API_KEY"
    cloud_api_model: str = "qwen-plus"
    cloud_full_model_runtime: str = "vllm"
    tee_enabled: bool = True
    tee_type: str = "intel_tdx"
    kv_cache_path: str = ".cache/kv_cache.json"


@dataclass(slots=True)
class RuntimeConfig:
    """Runtime knobs to control policy and latency behavior."""

    confidence_threshold: float = 0.72
    token_batch_size: int = 16
    cloud_retry: int = 1
    short_prompt_threshold: int = 120
    low_complexity_threshold: float = 0.42
    cache_hot_query_threshold: int = 2
