from dataclasses import dataclass
from typing import Literal

@dataclass
class TraceStep:
    step_id: int
    type: Literal["thought", "tool_call", "tool_result", "output"]
    content: str
    tool_name: str | None
    tool_args: dict | None
    tool_result: str | None
    timestamp: float

@dataclass
class AgentTrace:
    task_id: str
    input: str
    steps: list[TraceStep]
    final_output: str
    actual_state: dict
    metadata: dict

@dataclass
class EvalTestCase:
    task_id: str
    input: str
    expected_tool_calls: list[dict]
    expected_final_state: dict
    expected_output_contains: list[str]
    max_steps: int
    timeout_seconds: float

@dataclass
class MetricResult:
    metric_name: str
    score: float
    passed: bool
    details: str
