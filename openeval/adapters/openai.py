import json
from openeval.models import AgentTrace, TraceStep

def from_openai_messages(messages: list[dict], task_id: str = "openai_task", final_output: str = "") -> AgentTrace:
    """
    Convert a list of OpenAI chat completion messages into an OpenEval AgentTrace.
    """
    steps: list[TraceStep] = []
    
    pending_tool_calls: dict[str, TraceStep] = {}
    
    input_text = ""
    for msg in messages:
        if msg.get("role") == "user" and msg.get("content") and not input_text:
            input_text = msg["content"]
            
    for msg in messages:
        role = msg.get("role")
        
        if role == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tc_id = tc.get("id")
                function = tc.get("function", {})
                name = function.get("name")
                arguments_str = function.get("arguments", "")
                
                tool_args = None
                error = None
                
                if arguments_str:
                    try:
                        tool_args = json.loads(arguments_str)
                    except json.JSONDecodeError as e:
                        tool_args = {"raw": arguments_str}
                        error = str(e)
                else:
                    tool_args = {}
                    
                step = TraceStep(
                    step_id=len(steps) + 1,
                    type="tool_call",
                    content=f"Calling tool: {name}",
                    tool_name=name,
                    tool_args=tool_args,
                    tool_result=None,
                    timestamp=0.0,
                    error=error
                )
                steps.append(step)
                
                if tc_id:
                    pending_tool_calls[tc_id] = step
                    
        elif role == "tool":
            tc_id = msg.get("tool_call_id")
            content = msg.get("content")
            
            matching_step = pending_tool_calls.get(tc_id)
            tool_name = matching_step.tool_name if matching_step else None
            
            step = TraceStep(
                step_id=len(steps) + 1,
                type="tool_result",
                content="Tool execution finished",
                tool_name=tool_name,
                tool_args=None,
                tool_result=str(content) if content is not None else None,
                timestamp=0.0,
                error=None
            )
            steps.append(step)
            
            if tc_id in pending_tool_calls:
                del pending_tool_calls[tc_id]
                
    if not final_output:
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content") and not msg.get("tool_calls"):
                final_output = str(msg["content"])
                break
                
    return AgentTrace(
        task_id=task_id,
        input=str(input_text),
        steps=steps,
        final_output=str(final_output),
        actual_state={},
        metadata={}
    )
