import unittest

from edge_cloud_coinference.config import DeploymentConfig, RuntimeConfig
from edge_cloud_coinference.coordinator import CoInferenceEngine
from edge_cloud_coinference.types import InferenceRequest


class CoInferenceEngineTests(unittest.TestCase):
    def test_token_mode_edge_only_when_confident(self) -> None:
        deployment = DeploymentConfig(
            edge_model="Qwen2.5-3B-Instruct",
            cloud_model="Qwen2.5-7B-Instruct",
            collaboration_mode="token",
        )
        runtime = RuntimeConfig(confidence_threshold=0.5)
        engine = CoInferenceEngine(deployment, runtime)

        result = engine.run(InferenceRequest(request_id="r1", prompt="简短问题", task_type="general"))

        self.assertEqual(result.route, "edge_only")
        self.assertEqual(result.metadata["mode"], "token")

    def test_token_mode_fallback_to_cloud_when_low_confidence(self) -> None:
        deployment = DeploymentConfig(
            edge_model="Qwen2.5-0.5B-Instruct",
            cloud_model="Qwen2.5-14B-Instruct",
            collaboration_mode="token",
        )
        runtime = RuntimeConfig(confidence_threshold=0.9)
        engine = CoInferenceEngine(deployment, runtime)

        result = engine.run(
            InferenceRequest(request_id="r2", prompt="请详细分析该系统在复杂负载下的调度行为并给出优化建议", task_type="general")
        )

        self.assertEqual(result.route, "edge_to_cloud")
        self.assertIn("handoff", result.metadata)

    def test_task_mode_routes_reasoning_to_cloud(self) -> None:
        deployment = DeploymentConfig(collaboration_mode="task")
        engine = CoInferenceEngine(deployment)

        result = engine.run(
            InferenceRequest(request_id="r3", prompt="证明该算法的收敛性", task_type="reasoning")
        )

        self.assertEqual(result.route, "cloud")
        self.assertEqual(result.metadata["mode"], "task")


if __name__ == "__main__":
    unittest.main()
