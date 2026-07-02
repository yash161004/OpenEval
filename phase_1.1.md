# Phase 1.1 — LangChain Trace Adapter for OpenEval

---

## What Is This? (The Big Picture)

**OpenEval** is a grading system for AI agents. Instead of judging what an agent *says*, it judges what an agent *does* — which tools it called, with what arguments, in what order, and whether they succeeded.

To grade an agent, OpenEval needs its action history in a specific clean format called `AgentTrace` — a simple, flat, ordered list of steps.

**The Problem:**  
Developers build real agents using frameworks like **LangChain**. When a LangChain agent runs, it produces a massive, deeply nested execution tree called a `Run` object — full of internal LLM thoughts, sub-chains, retries, metadata — that OpenEval cannot understand.

**What Phase 1.1 Built:**  
A **LangChain Adapter** — a translation layer that takes a LangChain `Run` object and converts it into a clean `AgentTrace` that OpenEval can score. Zero LLM calls, zero API keys, pure deterministic logic.

---

## Files Overview

| File | Status | Purpose |
|---|---|---|
| `openeval/models.py` | Modified | Added `error` field to `TraceStep` schema |
| `openeval/adapters/__init__.py` | Created | Makes `adapters/` a Python package |
| `openeval/adapters/langchain.py` | Created | The core adapter logic |
| `tests/test_langchain_adapter.py` | Created | Automated tests for the adapter |
| `examples/langchain_adapter/run_example.py` | Created | End-to-end working example |
| `README.md` | Modified | Added LangChain usage section |

---

## File-by-File Explanation

---

### 1. `openeval/models.py` — Modified

**Why touched:** LangChain captures whether a tool crashed with an `error` string. The original `TraceStep` schema had no field to store this. Without adding it, tool errors would be silently swallowed.

**What changed:**
```python
@dataclass
class TraceStep:
    step_id: int
    type: Literal["thought", "tool_call", "tool_result", "output"]
    content: str
    tool_name: str | None
    tool_args: dict | None
    tool_result: str | None
    timestamp: float
    error: str | None = None   # <-- NEW field added
```

`error: str | None = None` is appended with a default of `None` so all existing code that creates `TraceStep` objects without this field continues to work without any changes.

---

### 2. `openeval/adapters/__init__.py` — Created

**Why created:** Python requires an `__init__.py` file to treat a folder as an importable package.

**What it does:**
```python
from .langchain import from_langchain_run

__all__ = ["from_langchain_run"]
```

This lets developers write `from openeval.adapters import from_langchain_run` anywhere.

---

### 3. `openeval/adapters/langchain.py` — Created (Core File)

**Why created:** This is the adapter itself — the entire point of Phase 1.1.

**What it does:** Takes a LangChain `Run` object (a nested tree) and returns an `AgentTrace` (a flat list).

**Full code with explanation:**

```python
from langchain_core.tracers.schemas import Run
from openeval.models import AgentTrace, TraceStep

def from_langchain_run(run: Run, task_id: str = "langchain_task", input_text: str = "") -> AgentTrace:
    """
    Flattens the nested run tree into a linear sequence of tool_call/tool_result steps.
    LIMITATION: parent/child tool relationships are not preserved structurally
    (only call order is retained), and non-tool runs (sub-chains, intermediate
    LLM calls) between tool boundaries are not captured in the resulting trace.
    """
    steps: list[TraceStep] = []

    def process_run(r: Run):
        # Only tool runs are relevant. Thoughts, chains, LLM calls are ignored.
        is_tool = r.run_type == "tool"

        if is_tool:
            # First: record the tool CALL (what was called, with what args)
            steps.append(TraceStep(
                step_id=len(steps) + 1,
                type="tool_call",
                content=f"Calling tool: {r.name}",
                tool_name=r.name,
                tool_args=r.inputs,          # The exact args passed in
                tool_result=None,
                timestamp=r.start_time.timestamp() if r.start_time else 0.0,
                error=None
            ))

        # Recurse into children BEFORE closing this tool result,
        # so nested tools are interleaved in the correct chronological order
        if r.child_runs:
            for child in r.child_runs:
                process_run(child)

        if is_tool:
            # Then: record the tool RESULT (what it returned, or the error)
            output = None
            if r.outputs:
                if isinstance(r.outputs, dict):
                    if "output" in r.outputs:
                        output = r.outputs["output"]
                    else:
                        output = next(iter(r.outputs.values()))
                else:
                    output = str(r.outputs)

            steps.append(TraceStep(
                step_id=len(steps) + 1,
                type="tool_result",
                content="Tool execution finished",
                tool_name=r.name,
                tool_args=None,
                tool_result=output,           # What the tool returned
                timestamp=r.end_time.timestamp() if r.end_time else 0.0,
                error=r.error                 # None on success, error string on crash
            ))

    process_run(run)

    # Safely extract final output (no eager evaluation of fallback)
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

    # Safely extract agent input
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
```

> **Key design choice:** The recursion walks into child runs *between* emitting the parent's `tool_call` and `tool_result`. This means nested tool calls produce a chronologically ordered interleaved sequence.

> **Known Limitation:** `TraceStep` has no `parent_id` field. The flat list does not structurally encode which child tool was spawned by which parent tool — only the order is preserved. Non-tool nodes (chains, LLM calls) are completely discarded.

---

### 4. `tests/test_langchain_adapter.py` — Created

**Why created:** To prove the adapter is correct across every scenario OpenEval will encounter. All tests are deterministic — no API keys, no network, no randomness.

**Tests written:**

| Test | What it proves |
|---|---|
| `test_clean_multi_tool_run` | 2 sequential tools → 4 flat steps in correct order, final output correct |
| `test_tool_error_run` | A crashing tool → `error` field on `tool_result` step is populated, not swallowed |
| `test_empty_no_tool_calls_run` | An agent that never calls any tools → 0 steps, final output still correct |
| `test_nested_run_handling` | `parent_tool → inner_chain → child_tool` → child_tool appears between parent's call/result; inner_chain (non-tool) is discarded |

**Result: 33 total tests in the suite, 33 passed.**

---

### 5. `examples/langchain_adapter/run_example.py` — Created

**Why created:** Proves the adapter works end-to-end with real LangChain execution and real OpenEval metrics. Run standalone with `python run_example.py`.

**What it does step by step:**
1. Creates a `FakeListLLM` with scripted responses — no OpenAI key needed
2. Attaches a real `@tool`-decorated function (`calculate_length`)
3. Builds a LangChain chain (`prompt | llm | calculate_length`)
4. Uses `collect_runs()` to capture the live `Run` object during execution
5. Passes it through `from_langchain_run()` and prints the extracted steps
6. Scores the resulting `AgentTrace` against `ToolSelectionAccuracy`, `ArgumentCorrectness`, and `StepEfficiency`

**Terminal output produced:**
```
==============================================
1. Setting up LangChain chain with FakeListLLM
==============================================

==============================================
2. Executing LangChain chain
==============================================
Chain Final Output: 11

==============================================
3. Converting LangChain Run to OpenEval Trace
==============================================
Extracted 2 steps.
  Step 1 [tool_call]: Calling tool: calculate_length
    Args: {'input': 'hello world'}
  Step 2 [tool_result]: Tool execution finished
    Result: 11

==============================================
4. Running OpenEval Metrics
==============================================
[PASS] Tool Selection Accuracy: Score 1.00 - Correctly called 1 of 1 expected tools.
[PASS] Argument Correctness: Score 1.00 - Correctly matched 1 of 1 expected arguments.
[PASS] Step Efficiency: Score 1.00 - Took 1 steps vs optimal 1 steps.
```

---

### 6. `README.md` — Modified

**What was added:** A new "LangChain Adapter" section at the bottom with:
- A 3-line code snippet showing how to use the adapter
- A plain-English limitation note for first-time readers

---

## Summary

Phase 1.1 is complete. OpenEval can now accept real LangChain agent runs directly — the first of four planned real-world adapters. Next up: **OpenAI raw tool-calling adapter (Phase 1.2)**.

---

## Demo Practical Example

Here is a full, practical code snippet a developer could write to evaluate their LangChain agent using OpenEval:

```python
from openeval.adapters.langchain import from_langchain_run
from openeval.metrics import ToolSelectionAccuracy
from openeval.models import EvalTestCase
from langchain_core.tracers.context import collect_runs

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

# 2. Run your LangChain Agent
with collect_runs() as cb:
    # 'agent' is your pre-configured LangChain agent
    agent.invoke({"input": test_case.input})
    run_tree = cb.traced_runs[0]

# 3. Convert to OpenEval Trace
trace = from_langchain_run(run_tree)

# 4. Grade the execution
metric = ToolSelectionAccuracy()
result = metric.score(trace, test_case)

print(f"[{'PASS' if result.passed else 'FAIL'}] {result.metric_name}: {result.score * 100}%")
print(f"Details: {result.details}")
```
