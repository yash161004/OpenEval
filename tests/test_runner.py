import json
import pytest
from pathlib import Path
from openeval.models import AgentTrace, EvalTestCase, TraceStep, MetricResult
from openeval.metrics import BaseMetric
from openeval.runner import run_eval, run_suite

class DummyMetric(BaseMetric):
    name = "Dummy"
    description = "Dummy metric"
    
    def score(self, trace: AgentTrace, test_case: EvalTestCase) -> MetricResult:
        return MetricResult(self.name, 1.0, True, "Passed dummy")
        
    def explain(self, result: MetricResult) -> str:
        return "Explained"

class ErrorMetric(BaseMetric):
    name = "Error"
    description = "Raises error"
    
    def score(self, trace: AgentTrace, test_case: EvalTestCase) -> MetricResult:
        raise ValueError("Boom")
        
    def explain(self, result: MetricResult) -> str:
        return "Explained"

def test_run_eval_happy_path():
    tc = EvalTestCase("t1", "", [], {}, [], 10, 10.0)
    trace = AgentTrace("t1", "", [], "", {}, {})
    
    result = run_eval(trace, tc, [DummyMetric()])
    assert result["task_id"] == "t1"
    assert "Dummy" in result["metrics"]
    assert result["metrics"]["Dummy"].score == 1.0

def test_run_eval_exception_handling():
    tc = EvalTestCase("t1", "", [], {}, [], 10, 10.0)
    trace = AgentTrace("t1", "", [], "", {}, {})
    
    result = run_eval(trace, tc, [ErrorMetric(), DummyMetric()])
    assert result["task_id"] == "t1"
    
    # ErrorMetric should not crash the run, should return 0.0
    err_res = result["metrics"]["Error"]
    assert err_res.score == 0.0
    assert err_res.passed is False
    assert "Metric raised an error: Boom" in err_res.details
    
    # DummyMetric should still run
    assert result["metrics"]["Dummy"].score == 1.0

def test_run_suite_success(tmp_path: Path):
    suite_dir = tmp_path / "suite"
    traces_dir = suite_dir / "traces"
    testcases_dir = suite_dir / "testcases"
    
    traces_dir.mkdir(parents=True)
    testcases_dir.mkdir(parents=True)
    
    tc_data = {
        "task_id": "task_1",
        "input": "hello",
        "expected_tool_calls": [],
        "expected_final_state": {},
        "expected_output_contains": [],
        "max_steps": 10,
        "timeout_seconds": 10.0
    }
    with open(testcases_dir / "task_1.json", "w") as f:
        json.dump(tc_data, f)
        
    trace_data = {
        "task_id": "task_1",
        "input": "hello",
        "steps": [],
        "final_output": "done",
        "actual_state": {},
        "metadata": {}
    }
    with open(traces_dir / "task_1.json", "w") as f:
        json.dump(trace_data, f)
        
    results = run_suite(suite_dir, [DummyMetric()])
    assert "task_1" in results
    task_res = results["task_1"]
    assert task_res["task_id"] == "task_1"
    assert task_res["metrics"]["Dummy"].score == 1.0

def test_run_suite_missing_dirs(tmp_path: Path):
    with pytest.raises(ValueError, match="must contain 'traces' and 'testcases'"):
        run_suite(tmp_path, [])

def test_run_suite_missing_trace(tmp_path: Path):
    suite_dir = tmp_path / "suite2"
    (suite_dir / "traces").mkdir(parents=True)
    tc_dir = suite_dir / "testcases"
    tc_dir.mkdir(parents=True)
    
    with open(tc_dir / "task_orphan.json", "w") as f:
        json.dump({"task_id": "task_orphan", "input": ""}, f)
        
    results = run_suite(suite_dir, [])
    assert "task_orphan" in results
    assert results["task_orphan"]["task_id"] == "task_orphan"
    assert "error" in results["task_orphan"]
    assert "Missing matching trace file: task_orphan.json" in results["task_orphan"]["error"]
