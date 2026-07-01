from typing import Literal, Dict, Any, Optional
import json

def format_report(results: Dict[str, Any], *, timestamp: Optional[str] = None) -> str:
    """Pure function: structured results -> markdown string. No I/O inside.
    Deviation note: Computes aggregate passed_count/pass_rate internally to avoid 
    cluttering the CLI layer with percentage math, but output remains strictly deterministic.
    """

    run_id = results.get("run_id", "results")
    test_cases = sorted(results.get("test_cases", []), key=lambda x: x.get("task_id", ""))
    
    total = len(test_cases)
    passed_count = 0
    
    processed_cases = []
    has_failures = False
    
    for tc in test_cases:
        task_id = tc.get("task_id", "unknown")
        error = tc.get("error")
        metrics = tc.get("metrics", {})
        
        if error:
            status = "FAIL"
            has_failures = True
            tool_sel = arg_corr = step_eff = goal_comp = "-"
        else:
            all_passed = all(m.get("passed", False) for m in metrics.values())
            status = "PASS" if all_passed else "FAIL"
            if not all_passed:
                has_failures = True
            else:
                passed_count += 1
                
            def get_score(name: str) -> str:
                if name in metrics:
                    return f"{metrics[name].get('score', 0.0):.2f}"
                return "-"
                
            tool_sel = get_score("Tool Selection Accuracy")
            arg_corr = get_score("Argument Correctness")
            step_eff = get_score("Step Efficiency")
            goal_comp = get_score("Goal Completion Rate")
            
        processed_cases.append({
            "task_id": task_id,
            "status": status,
            "tool_sel": tool_sel,
            "arg_corr": arg_corr,
            "step_eff": step_eff,
            "goal_comp": goal_comp,
            "error": error,
            "metrics": metrics
        })

    failed_count = total - passed_count
    pass_rate = (passed_count / total * 100) if total > 0 else 0.0

    lines = []
    lines.append(f"# OpenEval Report \u2014 {run_id}")
    if timestamp:
        lines.append(f"Generated at: {timestamp}")
    lines.append("")
    lines.append("## Summary")
    lines.append("| Total | Passed | Failed | Pass Rate |")
    lines.append("|-------|--------|--------|-----------|")
    lines.append(f"| {total} | {passed_count} | {failed_count} | {pass_rate:.1f}% |")
    lines.append("")
    lines.append("## Results")
    lines.append("| Task ID | Status | Tool Sel | Arg Corr | Step Eff | Goal Comp |")
    lines.append("|---------|--------|----------|----------|----------|-----------|")
    
    for tc in processed_cases:
        lines.append(f"| {tc['task_id']} | {tc['status']} | {tc['tool_sel']} | {tc['arg_corr']} | {tc['step_eff']} | {tc['goal_comp']} |")
        
    if has_failures:
        lines.append("")
        lines.append("## Failures")
        for tc in processed_cases:
            if tc["status"] == "FAIL":
                lines.append("")
                lines.append(f"### {tc['task_id']}")
                if tc["error"]:
                    lines.append(f"- **Error:** {tc['error']}")
                else:
                    for m_name in sorted(tc["metrics"].keys()):
                        m = tc["metrics"][m_name]
                        if not m.get("passed", False):
                            score = m.get("score", 0.0)
                            details = m.get("details", "")
                            lines.append(f"- **{m_name}:** {score:.2f}")
                            lines.append(f"  - *Reason:* {details}")
                            
    return "\n".join(lines)
