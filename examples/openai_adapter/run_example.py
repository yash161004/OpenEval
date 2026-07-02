from openeval.adapters.openai import from_openai_messages
from openeval.metrics import ToolSelectionAccuracy, ArgumentCorrectness, StepEfficiency
from openeval.models import EvalTestCase
import json

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Calculate the length of 'hello world'"},
    {"role": "assistant", "content": None, "tool_calls": [
        {
            "id": "call_xyz123",
            "type": "function",
            "function": {
                "name": "calculate_length",
                "arguments": '{"input": "hello world"}'
            }
        }
    ]},
    {"role": "tool", "tool_call_id": "call_xyz123", "content": "11"},
    {"role": "assistant", "content": "The length of 'hello world' is 11."}
]

trace = from_openai_messages(messages, task_id="openai_task_1")

test_case = EvalTestCase(
    task_id="openai_task_1",
    input="Calculate the length of 'hello world'",
    expected_tool_calls=[{"tool": "calculate_length", "args": {"input": "hello world"}}],
    expected_final_state={},
    expected_output_contains=["11"],
    max_steps=1,
    timeout_seconds=10.0
)

print("==============================================")
print("1. Converting OpenAI Messages to OpenEval Trace")
print("==============================================")
print(f"Extracted {len(trace.steps)} steps.")
for step in trace.steps:
    print(f"  Step {step.step_id} [{step.type}]: {step.content}")
    if step.type == "tool_call":
        print(f"    Args: {step.tool_args}")
    elif step.type == "tool_result":
        print(f"    Result: {step.tool_result}")

print("\n==============================================")
print("2. Running OpenEval Metrics")
print("==============================================")

metrics = [ToolSelectionAccuracy(), ArgumentCorrectness(), StepEfficiency()]
for metric in metrics:
    result = metric.score(trace, test_case)
    status = "PASS" if result.passed else "FAIL"
    print(f"[{status}] {result.metric_name}: Score {result.score:.2f} - {result.details}")
