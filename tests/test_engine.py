import tempfile
import unittest
from pathlib import Path

from edge_cloud_coinference.config import DeploymentConfig, RuntimeConfig
from edge_cloud_coinference.coordinator import CoInferenceEngine
from edge_cloud_coinference.types import InferenceRequest


class CoInferenceEngineTests(unittest.TestCase):
    def _build_engine(self, mode: str = "token") -> CoInferenceEngine:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        cache_path = str(Path(tmp_dir.name) / "kv_cache.json")
        deployment = DeploymentConfig(
            edge_model="Qwen2.5-1.5B-Instruct",
            cloud_model="Qwen2.5-14B-Instruct",
            collaboration_mode=mode,
            kv_cache_path=cache_path,
        )
        runtime = RuntimeConfig(confidence_threshold=0.95, cache_hot_query_threshold=1)
        return CoInferenceEngine(deployment, runtime)

    def _build_engine_with_cloud_mode(self, cloud_execution_mode: str) -> CoInferenceEngine:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        cache_path = str(Path(tmp_dir.name) / "kv_cache.json")
        deployment = DeploymentConfig(
            edge_model="Qwen2.5-1.5B-Instruct",
            cloud_model="Qwen2.5-14B-Instruct",
            collaboration_mode="task",
            cloud_execution_mode=cloud_execution_mode,
            kv_cache_path=cache_path,
        )
        runtime = RuntimeConfig(confidence_threshold=0.95, cache_hot_query_threshold=1)
        return CoInferenceEngine(deployment, runtime)

    def test_token_mode_secure_fallback_to_cloud_when_low_confidence(self) -> None:
        engine = self._build_engine(mode="token")

        result = engine.run(
            InferenceRequest(
                request_id="r1",
                prompt="请详细分析该系统在复杂负载下的调度行为并给出优化建议",
                task_type="general",
            )
        )

        self.assertEqual(result.route, "edge_to_cloud_secure")
        self.assertEqual(result.metadata["secure_offload"], "intel_tdx:attested")

    def test_task_mode_rule_and_classifier_route_complex_to_cloud(self) -> None:
        engine = self._build_engine(mode="task")

        result = engine.run(
            InferenceRequest(
                request_id="r2",
                prompt="请基于以下上下文继续完成多轮推理，并给出 Python 代码实现。",
                task_type="general",
            )
        )

        self.assertEqual(result.route, "cloud")
        self.assertEqual(result.metadata["classifier"], "complex")

    def test_edge_cache_hit_for_hot_query(self) -> None:
        engine = self._build_engine(mode="task")
        req = InferenceRequest(request_id="r3", prompt="天气怎么样", task_type="general")

        first = engine.run(req)
        second = engine.run(req)

        self.assertNotEqual(first.route, "edge_cache")
        self.assertEqual(second.route, "edge_cache")
        self.assertEqual(second.metadata["cache"], "kv_hit")

    def test_cloud_api_mode_reserves_api_fields(self) -> None:
        engine = self._build_engine_with_cloud_mode(cloud_execution_mode="api")
        result = engine.run(
            InferenceRequest(
                request_id="r4",
                prompt="请基于上文继续完成代码并解释",
                task_type="reasoning",
            )
        )
        self.assertEqual(result.route, "cloud")
        self.assertEqual(result.metadata["cloud_execution_mode"], "api")
        self.assertEqual(result.metadata["cloud_api_reserved"], "true")

    def test_cloud_full_model_mode_reserves_runtime_fields(self) -> None:
        engine = self._build_engine_with_cloud_mode(cloud_execution_mode="full_model")
        result = engine.run(
            InferenceRequest(
                request_id="r5",
                prompt="请基于上文继续完成代码并解释",
                task_type="reasoning",
            )
        )
        self.assertEqual(result.route, "cloud")
        self.assertEqual(result.metadata["cloud_execution_mode"], "full_model")
        self.assertEqual(result.metadata["cloud_full_model_reserved"], "true")

    def test_edge_executor_reports_runtime_provider_metadata(self) -> None:
        engine = self._build_engine(mode="task")
        result = engine.run(
            InferenceRequest(
                request_id="r6",
                prompt="你好",
                task_type="general",
            )
        )
        self.assertEqual(result.route, "edge")
        self.assertEqual(result.metadata["edge_runtime"], "onnxruntime")
        self.assertEqual(result.metadata["edge_provider"], "onnxruntime_cpu")


if __name__ == "__main__":
    unittest.main()
