from __future__ import annotations

from .config import DeploymentConfig, RuntimeConfig
from .executors import CloudExecutor, EdgeExecutor
from .models import build_profile
from .types import InferenceRequest, InferenceResult


class CoInferenceEngine:
    """Unified coordinator supporting token-level and task-level collaboration."""

    def __init__(self, deployment: DeploymentConfig, runtime: RuntimeConfig | None = None) -> None:
        self.deployment = deployment
        self.runtime = runtime or RuntimeConfig()

        self.edge = EdgeExecutor(build_profile(deployment.edge_model, side="edge"), deployment.edge_max_tokens)
        self.cloud = CloudExecutor(
            build_profile(deployment.cloud_model, side="cloud"), deployment.cloud_max_tokens
        )

    def run(self, request: InferenceRequest) -> InferenceResult:
        if self.deployment.collaboration_mode == "token":
            return self._run_token_level(request)
        if self.deployment.collaboration_mode == "task":
            return self._run_task_level(request)
        raise ValueError(f"Unsupported collaboration mode: {self.deployment.collaboration_mode}")

    def _run_token_level(self, request: InferenceRequest) -> InferenceResult:
        edge_out = self.edge.infer(request.prompt)
        if edge_out.confidence >= self.runtime.confidence_threshold:
            return InferenceResult(
                request_id=request.request_id,
                text=edge_out.text,
                route="edge_only",
                confidence=edge_out.confidence,
                latency_ms=edge_out.latency_ms,
                metadata={"mode": "token", "handoff": "none"},
            )

        cloud_out = self.cloud.infer(request.prompt)
        merged_text = f"{edge_out.text}\n--- handoff ---\n{cloud_out.text}"
        final_confidence = max(edge_out.confidence, cloud_out.confidence)
        return InferenceResult(
            request_id=request.request_id,
            text=merged_text,
            route="edge_to_cloud",
            confidence=final_confidence,
            latency_ms=edge_out.latency_ms + cloud_out.latency_ms,
            metadata={"mode": "token", "handoff": "low_confidence"},
        )

    def _run_task_level(self, request: InferenceRequest) -> InferenceResult:
        cloud_tasks = {"reasoning", "code", "long_context", "multilingual"}
        route_to_cloud = request.task_type in cloud_tasks or len(request.prompt) > 220
        executor = self.cloud if route_to_cloud else self.edge

        out = executor.infer(request.prompt)
        return InferenceResult(
            request_id=request.request_id,
            text=out.text,
            route="cloud" if route_to_cloud else "edge",
            confidence=out.confidence,
            latency_ms=out.latency_ms,
            metadata={"mode": "task", "task_type": request.task_type},
        )
