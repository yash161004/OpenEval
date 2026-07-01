import json
from pathlib import Path
from openeval.models import AgentTrace, EvalTestCase, TraceStep
from openeval.metrics import BaseMetric, MetricResult

def run_eval(trace: AgentTrace, test_case: EvalTestCase, metrics: list[BaseMetric]) -> dict:
    results = {}
    for metric in metrics:
        try:
            results[metric.name] = metric.score(trace, test_case)
        except Exception as e:
            results[metric.name] = MetricResult(
                metric_name=metric.name,
                score=0.0,
                passed=False,
                details=f"Metric raised an error: {e}"
            )
            
    return {
        "task_id": trace.task_id,
        "metrics": results
    }

def run_suite(suite_dir: Path, metrics: list[BaseMetric]) -> dict[str, dict]:
    """
    Expects suite_dir to contain:
      - traces/ (with {task_id}.json)
      - testcases/ (with {task_id}.json)
    """
    traces_dir = suite_dir / "traces"
    testcases_dir = suite_dir / "testcases"
    
    if not traces_dir.exists() or not testcases_dir.exists():
        raise ValueError("Suite directory must contain 'traces' and 'testcases' subdirectories.")
        
    suite_results = {}
    
    for tc_file in testcases_dir.glob("*.json"):
        task_id = tc_file.stem
        trace_file = traces_dir / f"{task_id}.json"
        
        if not trace_file.exists():
            suite_results[task_id] = {
                "task_id": task_id,
                "error": f"Missing matching trace file: {trace_file.name}",
                "metrics": {}
            }
            continue
            
        try:
            with open(tc_file, "r", encoding="utf-8") as f:
                tc_data = json.load(f)
                
            with open(trace_file, "r", encoding="utf-8") as f:
                trace_data = json.load(f)
                
            test_case = EvalTestCase(
                task_id=tc_data["task_id"],
                input=tc_data["input"],
                expected_tool_calls=tc_data.get("expected_tool_calls", []),
                expected_final_state=tc_data.get("expected_final_state", {}),
                expected_output_contains=tc_data.get("expected_output_contains", []),
                max_steps=tc_data.get("max_steps", 10),
                timeout_seconds=tc_data.get("timeout_seconds", 10.0)
            )
            
            steps = []
            for step_data in trace_data.get("steps", []):
                steps.append(TraceStep(
                    step_id=step_data["step_id"],
                    type=step_data["type"],
                    content=step_data["content"],
                    tool_name=step_data.get("tool_name"),
                    tool_args=step_data.get("tool_args"),
                    tool_result=step_data.get("tool_result"),
                    timestamp=step_data["timestamp"]
                ))
                
            trace = AgentTrace(
                task_id=trace_data["task_id"],
                input=trace_data["input"],
                steps=steps,
                final_output=trace_data.get("final_output", ""),
                actual_state=trace_data.get("actual_state", {}),
                metadata=trace_data.get("metadata", {})
            )
        except (KeyError, json.JSONDecodeError, TypeError) as e:
            suite_results[task_id] = {
                "task_id": task_id,
                "error": f"Failed to parse task files: {e}",
                "metrics": {}
            }
            continue
        
        suite_results[task_id] = run_eval(trace, test_case, metrics)
        
    return suite_results
