from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class InferenceRequest:
    request_id: str
    prompt: str
    task_type: str = "general"
    session_id: str = "default-session"


@dataclass(slots=True)
class InferenceResult:
    request_id: str
    text: str
    route: str
    confidence: float
    latency_ms: int
    metadata: dict[str, str] = field(default_factory=dict)
