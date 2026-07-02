# Phase 1.2 — Raw OpenAI Tool-Calling Adapter for OpenEval

---

## What Is This? (The Big Picture)

**OpenEval** is a grading system for AI agents. Instead of judging what an agent *says*, it judges what an agent *does* — which tools it called, with what arguments, in what order, and whether they succeeded.

**The Problem:**  
Many developers do not use heavy frameworks like LangChain; instead, they interact with the raw OpenAI API natively. A typical raw OpenAI trace is just a chronological list of messages (roles like `user`, `assistant` with `tool_calls`, and `tool` with results). OpenEval needs a way to seamlessly parse these raw message lists into its standard `AgentTrace` format without the user having to write boilerplate mapping code.

**What Phase 1.2 Built:**  
An **OpenAI Tool-Calling Adapter** — a translation layer that accepts a raw OpenAI chat completions message list and safely extracts it into a flat sequence of `AgentTrace` steps. It gracefully handles malformed JSON arguments, pairs tool results robustly using `tool_call_id`, and requires zero network calls or API keys to function.

---

## Files Overview

| File | Status | Purpose |
|---|---|---|
| `openeval/adapters/__init__.py` | Modified | Exported the new `from_openai_messages` adapter |
| `openeval/adapters/openai.py` | Created | The core adapter logic for parsing OpenAI messages |
| `tests/test_openai_adapter.py` | Created | Automated tests for the adapter (including out-of-order results) |
| `examples/openai_adapter/run_example.py` | Created | End-to-end working example with static fixture data |
| `README.md` | Modified | Added OpenAI Tool Calling usage section |

---

## File-by-File Explanation

---

### 1. `openeval/adapters/__init__.py` — Modified

**Why touched:** To make the new adapter easily importable right alongside the LangChain one.

**What changed:**
```python
from .langchain import from_langchain_run
from .openai import from_openai_messages

__all__ = ["from_langchain_run", "from_openai_messages"]
```

---

### 2. `openeval/adapters/openai.py` — Created (Core File)

**Why created:** This is the core logic that traverses an OpenAI message array and populates an OpenEval `AgentTrace`.

**What it does:**
- Sweeps through the list of dictionaries representing OpenAI messages.
- Looks for `assistant` messages containing `tool_calls`. When found, loops over each call and generates a `tool_call` step.
- Gracefully parses `function.arguments` (a JSON string). If it hits a `json.JSONDecodeError` (which is common with real LLM hallucinations), it catches it, stores the raw string in `tool_args={"raw": ...}`, and logs the exception in the `TraceStep.error` field.
- Looks for `tool` role messages and pairs them with their corresponding `tool_calls` by ID, generating a `tool_result` step.

**Key design choice:** The adapter strictly pairs tool calls and results using the `tool_call_id`, mapping them correctly even if the `tool` messages arrive entirely out of chronological order.

**Known Limitation:** This adapter flattens the message list. Only the first `user` message is extracted as the task `input`, and the final `assistant` message (without tool calls) is used as the `final_output`. Any intermediate non-tool conversational turns are not structurally captured.

---

### 3. `tests/test_openai_adapter.py` — Created

**Why created:** To prove the adapter gracefully handles both happy paths and edge cases inherent to real LLM usage.

**Tests written:**

| Test | What it proves |
|---|---|
| `test_clean_multi_tool_call` | 2 sequential tools inside a single assistant message → 4 flat steps in correct order |
| `test_dangling_tool_call` | An assistant requested a tool call, but the tool result message never arrived |
| `test_malformed_json_arguments` | The LLM returned broken JSON; it doesn't crash, but properly populates the `error` field |
| `test_empty_message_list` | An empty array produces an empty trace safely |
| `test_out_of_order_tool_results` | Proves that if tool results arrive out of order, they still resolve and pair to the correct tool call by ID |

**Result: 38 total tests in the suite, 38 passed.**

---

### 4. `examples/openai_adapter/run_example.py` — Created

**Why created:** Proves the adapter works end-to-end with OpenEval metrics without requiring any live OpenAI API calls.

**What it does step by step:**
1. Defines a hardcoded list of OpenAI messages simulating a tool execution (calculating string length).
2. Converts the message list via `from_openai_messages()`.
3. Defines an `EvalTestCase` expecting the tool `calculate_length` to be called with `{"input": "hello world"}`.
4. Scores the resulting `AgentTrace` against `ToolSelectionAccuracy`, `ArgumentCorrectness`, and `StepEfficiency`.

**Terminal output produced:**
```
==============================================
1. Converting OpenAI Messages to OpenEval Trace
==============================================
Extracted 2 steps.
  Step 1 [tool_call]: Calling tool: calculate_length
    Args: {'input': 'hello world'}
  Step 2 [tool_result]: Tool execution finished
    Result: 11

==============================================
2. Running OpenEval Metrics
==============================================
[PASS] Tool Selection Accuracy: Score 1.00 - Correctly called 1 of 1 expected tools.
[PASS] Argument Correctness: Score 1.00 - Correctly matched 1 of 1 expected arguments.
[PASS] Step Efficiency: Score 1.00 - Took 1 steps vs optimal 1 steps.
```

---

### 5. `README.md` — Modified

**What was added:** A new "OpenAI Tool Calling" section below the LangChain section, providing a clear code snippet and explicitly detailing the adapter's limitations so users know exactly what is and isn't captured.

---

## Summary

Phase 1.2 is complete. OpenEval can now seamlessly accept traces from the two most dominant segments of the ecosystem: LangChain users and direct OpenAI API consumers. Next up: **Anthropic tool-calling adapter (Phase 1.3)**.

---

## Demo Practical Example

Here is a full, practical code snippet a developer could write to evaluate their raw OpenAI tool-calling logic using OpenEval:

```python
from openeval.adapters.openai import from_openai_messages
from openeval.metrics import ToolSelectionAccuracy
from openeval.models import EvalTestCase
from openai import OpenAI

client = OpenAI()

# 1. Define your test case
test_case = EvalTestCase(
    task_id="weather_task_1",
    input="What is the weather in Tokyo?",
    expected_tool_calls=[{"tool": "get_weather", "args": {"location": "Tokyo"}}],
    expected_final_state={},
    expected_output_contains=["sunny"],
    max_steps=1,
    timeout_seconds=10.0
)

# 2. Run your OpenAI chat completion
messages = [{"role": "user", "content": test_case.input}]
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=[{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
)
messages.append(response.choices[0].message.model_dump(exclude_unset=True))
# ... (app developer executes the tool and appends the 'tool' result message) ...

# 3. Convert to OpenEval Trace
trace = from_openai_messages(messages)

# 4. Grade the execution
metric = ToolSelectionAccuracy()
result = metric.score(trace, test_case)

print(f"[{'PASS' if result.passed else 'FAIL'}] {result.metric_name}: {result.score * 100}%")
print(f"Details: {result.details}")
```
