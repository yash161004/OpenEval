import typer
import json
from pathlib import Path
from typing import Optional
from openeval.metrics import (
    ToolSelectionAccuracy,
    ArgumentCorrectness,
    StepEfficiency,
    GoalCompletionRate
)
from openeval.models import AgentTrace, EvalTestCase, TraceStep
from openeval.runner import run_eval, run_suite
from openeval.report import format_report

app = typer.Typer(help="OpenEval: CI/CD-native agent evaluation.")

def _get_metrics():
    return [
        ToolSelectionAccuracy(),
        ArgumentCorrectness(),
        StepEfficiency(),
        GoalCompletionRate()
    ]

@app.command()
def run(
    trace: Optional[Path] = typer.Option(None, help="Path to trace JSON"),
    testcase: Optional[Path] = typer.Option(None, help="Path to test case JSON"),
    suite: Optional[Path] = typer.Option(None, help="Path to test suite directory"),
    output: Optional[Path] = typer.Option(None, help="Output directory for results")
):
    """
    Run evaluation on a trace or test suite.
    """
    if trace and testcase:
        if not trace.exists():
            typer.echo(f"Error: Trace file not found: {trace}", err=True)
            raise typer.Exit(code=1)
        if not testcase.exists():
            typer.echo(f"Error: Test case file not found: {testcase}", err=True)
            raise typer.Exit(code=1)
            
        try:
            with open(testcase, "r", encoding="utf-8") as f:
                tc_data = json.load(f)
            with open(trace, "r", encoding="utf-8") as f:
                trace_data = json.load(f)
                
            tc = EvalTestCase(
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
                
            tr = AgentTrace(
                task_id=trace_data["task_id"],
                input=trace_data["input"],
                steps=steps,
                final_output=trace_data.get("final_output", ""),
                actual_state=trace_data.get("actual_state", {}),
                metadata=trace_data.get("metadata", {})
            )
        except (KeyError, json.JSONDecodeError, TypeError) as e:
            typer.echo(f"Error: Failed to parse task files: {e}", err=True)
            raise typer.Exit(code=1)
        
        result = run_eval(tr, tc, _get_metrics())
        typer.echo(json.dumps(result, indent=2, default=lambda x: x.__dict__))
        
        all_passed = all(m.passed for m in result["metrics"].values())
        if not all_passed:
            raise typer.Exit(code=1)
            
    elif suite and output:
        output.mkdir(parents=True, exist_ok=True)
        results = run_suite(suite, _get_metrics())
        
        suite_passed = True
        
        for task_id, task_result in results.items():
            out_file = output / f"{task_id}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(task_result, f, indent=2, default=lambda x: x.__dict__)
                
            if "error" in task_result:
                suite_passed = False
            else:
                if not all(m.passed for m in task_result["metrics"].values()):
                    suite_passed = False
                    
        typer.echo(f"Saved {len(results)} result files to {output}")
        if not suite_passed:
            raise typer.Exit(code=1)
            
    else:
        typer.echo("Invalid arguments. Provide either --trace and --testcase OR --suite and --output", err=True)
        raise typer.Exit(code=1)

@app.command()
def report(
    input: Path = typer.Option(..., help="Path to results directory"),
    format: str = typer.Option("markdown", help="Format of the report (markdown or json)")
):
    """
    Generate a report from evaluation results.
    """
    if not input.exists() or not input.is_dir():
        typer.echo(f"Error: Input directory not found: {input}", err=True)
        raise typer.Exit(code=1)
        
    run_id = input.name
    test_cases = []
    
    for file_path in input.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                test_cases.append(data)
        except Exception as e:
            typer.echo(f"Warning: Failed to load {file_path}: {e}", err=True)
            
    results = {
        "run_id": run_id,
        "test_cases": test_cases
    }
    
    # We must explicitly map format_type due to the parameter name in format_report
    typer.echo(format_report(results, format_type=format))

if __name__ == "__main__":
    app()
