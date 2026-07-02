from langchain_core.tracers.schemas import Run
from openeval.models import AgentTrace, TraceStep

def from_langchain_run(run: Run, task_id: str = "langchain_task", input_text: str = "") -> AgentTrace:
    """
    Convert a LangChain Run object into an OpenEval AgentTrace.
    Flattens the nested run tree into a linear sequence of tool_call/tool_result steps. 
    LIMITATION: parent/child tool relationships are not preserved structurally 
    (only call order is retained), and non-tool runs (sub-chains, intermediate 
    LLM calls) between tool boundaries are not captured in the resulting trace.
    """
    steps: list[TraceStep] = []
    
    def process_run(r: Run):
        is_tool = r.run_type == "tool"
        
        if is_tool:
            steps.append(
                TraceStep(
                    step_id=len(steps) + 1,
                    type="tool_call",
                    content=f"Calling tool: {r.name}",
                    tool_name=r.name,
                    tool_args=r.inputs,
                    tool_result=None,
                    timestamp=r.start_time.timestamp() if r.start_time else 0.0,
                    error=None
                )
            )
            
        if r.child_runs:
            for child in r.child_runs:
                process_run(child)
                
        if is_tool:
            error = r.error
            output = None
            if r.outputs:
                output = r.outputs.get("output", str(r.outputs)) if isinstance(r.outputs, dict) else str(r.outputs)
            
            steps.append(
                TraceStep(
                    step_id=len(steps) + 1,
                    type="tool_result",
                    content="Tool execution finished",
                    tool_name=r.name,
                    tool_args=None,
                    tool_result=output,
                    timestamp=r.end_time.timestamp() if r.end_time else 0.0,
                    error=error
                )
            )
            
    process_run(run)
    
    # Extract final output
    final_output = ""
    if run.outputs:
        if isinstance(run.outputs, dict):
            if "output" in run.outputs:
                final_output = run.outputs["output"]
            else:
                final_output = next(iter(run.outputs.values())) if run.outputs else ""
        else:
            final_output = str(run.outputs)
            
    if not final_output and run.error:
        final_output = f"Error: {run.error}"

    input_val = input_text
    if not input_val and run.inputs:
        if isinstance(run.inputs, dict):
            if "input" in run.inputs:
                input_val = run.inputs["input"]
            else:
                input_val = str(run.inputs)
        else:
            input_val = str(run.inputs)
            
    return AgentTrace(
        task_id=task_id,
        input=input_val,
        steps=steps,
        final_output=str(final_output),
        actual_state={},
        metadata={"run_id": str(run.id)}
    )
