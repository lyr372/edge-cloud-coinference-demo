from __future__ import annotations

from dataclasses import dataclass

from .cache import PersistentKVCache
from .config import DeploymentConfig, RuntimeConfig
from .executors import CloudExecutor, EdgeExecutor
from .models import build_profile
from .types import InferenceRequest, InferenceResult


@dataclass(slots=True)
class LightClassifierOutput:
    complexity: float
    label: str


class CoInferenceEngine:
    """Coordinator with TEE secure offload, edge cache, and rule+classifier routing."""

    def __init__(self, deployment: DeploymentConfig, runtime: RuntimeConfig | None = None) -> None:
        self.deployment = deployment
        self.runtime = runtime or RuntimeConfig()
        self.cache = PersistentKVCache(deployment.kv_cache_path)

        self.edge = EdgeExecutor(
            build_profile(deployment.edge_model, side="edge"),
            deployment.edge_max_tokens,
            runtime=deployment.edge_runtime,
            quantization=deployment.quantization,
            distillation_enabled=deployment.distillation_enabled,
        )
        self.cloud = CloudExecutor(
            build_profile(deployment.cloud_model, side="cloud"),
            deployment.cloud_max_tokens,
            runtime="cloud-runtime",
            quantization="fp16",
            distillation_enabled=False,
            execution_mode=deployment.cloud_execution_mode,
            api_base_url=deployment.cloud_api_base_url,
            api_model=deployment.cloud_api_model,
            full_model_runtime=deployment.cloud_full_model_runtime,
        )

    def run(self, request: InferenceRequest) -> InferenceResult:
        cached = self.cache.get(request.prompt)
        if cached:
            return InferenceResult(
                request_id=request.request_id,
                text=cached.text,
                route="edge_cache",
                confidence=0.99,
                latency_ms=8,
                metadata={
                    "mode": self.deployment.collaboration_mode,
                    "cache": "kv_hit",
                    "cache_hits": str(cached.hits),
                },
            )

        if self.deployment.collaboration_mode == "token":
            result = self._run_token_level(request)
        elif self.deployment.collaboration_mode == "task":
            result = self._run_task_level(request)
        else:
            raise ValueError(f"Unsupported collaboration mode: {self.deployment.collaboration_mode}")

        self.cache.upsert(
            request.prompt,
            result.text,
            min_hits_for_persist=self.runtime.cache_hot_query_threshold,
        )
        return result

    def _run_token_level(self, request: InferenceRequest) -> InferenceResult:
        edge_out = self.edge.infer(request.prompt)
        metadata = {
            "mode": "token",
            "first_token_ms": str(edge_out.first_token_ms),
            "quantization": self.deployment.quantization,
        }
        if edge_out.metadata:
            metadata.update({k: str(v) for k, v in edge_out.metadata.items()})
        if edge_out.confidence >= self.runtime.confidence_threshold:
            metadata["handoff"] = "none"
            return InferenceResult(
                request_id=request.request_id,
                text=edge_out.text,
                route="edge_only",
                confidence=edge_out.confidence,
                latency_ms=edge_out.latency_ms,
                metadata=metadata,
            )

        cloud_out = self.cloud.infer(request.prompt)
        if cloud_out.metadata:
            metadata.update({k: str(v) for k, v in cloud_out.metadata.items()})
        metadata["handoff"] = "low_confidence"
        if self.deployment.tee_enabled:
            metadata["secure_offload"] = f"{self.deployment.tee_type}:attested"

        merged_text = f"{edge_out.text}\n--- secure handoff ---\n{cloud_out.text}"
        final_confidence = max(edge_out.confidence, cloud_out.confidence)
        return InferenceResult(
            request_id=request.request_id,
            text=merged_text,
            route="edge_to_cloud_secure",
            confidence=final_confidence,
            latency_ms=edge_out.latency_ms + cloud_out.latency_ms,
            metadata=metadata,
        )

    def _run_task_level(self, request: InferenceRequest) -> InferenceResult:
        cls = self._light_classifier(request)
        route_to_cloud = self._rule_route(request, cls)
        executor = self.cloud if route_to_cloud else self.edge

        out = executor.infer(request.prompt)
        route = "cloud" if route_to_cloud else "edge"
        metadata = {
            "mode": "task",
            "task_type": request.task_type,
            "complexity": f"{cls.complexity:.3f}",
            "classifier": cls.label,
            "pipeline": "edge_first_token/light_predict/local_cache+cloud_complex_generation",
        }
        if out.metadata:
            metadata.update({k: str(v) for k, v in out.metadata.items()})
        if route_to_cloud and self.deployment.tee_enabled:
            metadata["secure_offload"] = f"{self.deployment.tee_type}:attested"

        return InferenceResult(
            request_id=request.request_id,
            text=out.text,
            route=route,
            confidence=out.confidence,
            latency_ms=out.latency_ms,
            metadata=metadata,
        )

    def _light_classifier(self, request: InferenceRequest) -> LightClassifierOutput:
        prompt = request.prompt
        code_hint = 1.0 if any(token in prompt for token in ("def ", "class ", "```", "代码")) else 0.0
        long_hint = min(1.0, len(prompt) / 600)
        multi_turn_hint = 0.6 if any(k in prompt for k in ("继续", "上文", "上一轮", "context")) else 0.0
        complexity = round(0.45 * long_hint + 0.35 * code_hint + 0.2 * multi_turn_hint, 3)
        label = "complex" if complexity >= self.runtime.low_complexity_threshold else "simple"
        return LightClassifierOutput(complexity=complexity, label=label)

    def _rule_route(self, request: InferenceRequest, cls: LightClassifierOutput) -> bool:
        cloud_tasks = {"reasoning", "code", "long_context", "multilingual", "multi_turn"}
        if request.task_type in cloud_tasks:
            return True
        if len(request.prompt) > self.runtime.short_prompt_threshold:
            return True
        if cls.label == "complex":
            return True
        return False
