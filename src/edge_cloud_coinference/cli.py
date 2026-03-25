from __future__ import annotations

import argparse
import json

from .config import DeploymentConfig, RuntimeConfig
from .coordinator import CoInferenceEngine
from .types import InferenceRequest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen 端云协同推理 Demo")
    parser.add_argument("--prompt", required=True, help="输入内容")
    parser.add_argument("--task-type", default="general", help="任务类型，如 reasoning/code/general")
    parser.add_argument("--mode", default="token", choices=["token", "task"], help="协同模式")
    parser.add_argument("--edge-model", default="Qwen2.5-1.5B-Instruct")
    parser.add_argument("--cloud-model", default="Qwen2.5-7B-Instruct")
    parser.add_argument("--confidence-threshold", type=float, default=0.72)
    parser.add_argument("--edge-runtime", default="onnxruntime", choices=["mnn", "onnxruntime", "onnx"])
    parser.add_argument("--quantization", default="int8", choices=["int4", "int8", "fp16"])
    parser.add_argument("--cloud-exec-mode", default="simulated", choices=["simulated", "api", "full_model"])
    parser.add_argument("--cloud-api-base-url", default="https://api.example.com/v1/chat/completions")
    parser.add_argument("--cloud-api-model", default="qwen-plus")
    parser.add_argument("--cloud-full-runtime", default="vllm")
    parser.add_argument("--disable-distillation", action="store_true")
    parser.add_argument("--disable-tee", action="store_true")
    parser.add_argument("--session-id", default="demo-session")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    deployment = DeploymentConfig(
        edge_model=args.edge_model,
        cloud_model=args.cloud_model,
        collaboration_mode=args.mode,
        edge_runtime=args.edge_runtime,
        quantization=args.quantization,
        distillation_enabled=not args.disable_distillation,
        cloud_execution_mode=args.cloud_exec_mode,
        cloud_api_base_url=args.cloud_api_base_url,
        cloud_api_model=args.cloud_api_model,
        cloud_full_model_runtime=args.cloud_full_runtime,
        tee_enabled=not args.disable_tee,
    )
    runtime = RuntimeConfig(confidence_threshold=args.confidence_threshold)

    engine = CoInferenceEngine(deployment, runtime)
    result = engine.run(
        InferenceRequest(
            request_id="demo-request",
            prompt=args.prompt,
            task_type=args.task_type,
            session_id=args.session_id,
        )
    )

    print(
        json.dumps(
            {
                "request_id": result.request_id,
                "route": result.route,
                "confidence": result.confidence,
                "latency_ms": result.latency_ms,
                "metadata": result.metadata,
                "text": result.text,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
