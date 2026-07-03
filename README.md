# OpenEval

[![PyPI - Version](https://img.shields.io/pypi/v/openeval-core.svg)](https://pypi.org/project/openeval-core)
[![Python - Versions](https://img.shields.io/pypi/pyversions/openeval-core.svg)](https://pypi.org/project/openeval-core)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/yash161004/OpenEval/actions/workflows/ci.yml/badge.svg)](https://github.com/yash161004/OpenEval/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/yash161004/OpenEval/branch/main/graph/badge.svg)](https://codecov.io/gh/yash161004/OpenEval)

Every existing tool evaluates what the agent said. OpenEval evaluates what the agent did.

## Installation

```bash
pip install openeval-core
```

*Note: While the package is installed as `openeval-core`, the CLI command to run it is `openeval`.*

## Usage

Run a single test case using the included example:
```bash
openeval run --trace examples/simple_agent/trace.json --testcase examples/simple_agent/testcase.json
```

Run a test suite:
```bash
openeval run --suite tests/ --output results/
```

Generate a report:
```bash
openeval report --input results/ --format markdown
```

*Note: The `report` command will exit with a non-zero status code (1) if any JSON file in the input directory is malformed or fails to load, guaranteeing pipeline failures on corrupted evals.*

## Quickstart

Get your first deterministic evaluation running in under 2 minutes. OpenEval is completely framework-agnostic.

### Tier 1: The Core Engine (No Framework Required)

The core engine requires zero dependencies on external frameworks or LLM SDKs. You just feed it plain Python data.

**1. Install the package**
```bash
pip install openeval-core
```

**2. Score a trace deterministically**
Create a file named `hello_eval.py` and run it:
```python
from openeval.metrics import ToolSelectionAccuracy
from openeval.models import AgentTrace, EvalTestCase, TraceStep

# 1. Define what the agent was supposed to do
test_case = EvalTestCase(
    task_id="quickstart-1",
    input="Search for the weather in Tokyo",
    expected_tool_calls=[{"tool": "search", "args": {"query": "weather in Tokyo"}}],
    expected_final_state={},
    expected_output_contains=[],
    max_steps=5,
    timeout_seconds=10.0
)

# 2. Provide the raw trace of what the agent actually did
trace = AgentTrace(
    task_id="quickstart-1",
    input="Search for the weather in Tokyo",
    steps=[
        TraceStep(
            step_id=1, 
            type="tool_call", 
            content="", 
            tool_name="search", 
            tool_args={"query": "weather in Tokyo"}, 
            tool_result="85 degrees and sunny", 
            timestamp=0.0
        )
    ],
    final_output="The weather in Tokyo is 85 degrees and sunny.",
    actual_state={},
    metadata={}
)

# 3. Score it deterministically (no LLM required)
metric = ToolSelectionAccuracy()
result = metric.score(trace, test_case)

print(f"Metric: {result.metric_name}")
print(f"Score: {result.score} (Passed: {result.passed})")
print(f"Details: {result.details}")
```

**Expected Output:**
```
Metric: Tool Selection Accuracy
Score: 1.0 (Passed: True)
Details: Correctly called 1 of 1 expected tools.
```

### Tier 2: Using OpenEval with LangChain

If your traces come from LangChain, you can use our built-in adapter to automatically convert LangChain runs into OpenEval traces.

**1. Install with LangChain support**
```bash
pip install "openeval-core[langchain]"
```

**2. Convert and Score**
*(See the `examples/` directory in our GitHub repository for full, runnable agent scripts using this adapter.)*

```python
from openeval.adapters.langchain import from_langchain_run
from langchain_core.tracers.context import collect_runs

with collect_runs() as cb:
    # Run your langchain agent
    agent.invoke({"input": "task input"})
    
# Convert the run to an OpenEval AgentTrace
trace = from_langchain_run(cb.traced_runs[0])

# Score it exactly like Tier 1
# metric.score(trace, test_case)
```

### OpenAI Tool Calling

OpenEval can automatically convert raw OpenAI chat completions message lists into `AgentTrace` objects.

*LIMITATION: This adapter flattens the message list. Only the first `user` message is used as the input, and the final `assistant` message without tool calls is used as the output. Any intermediate non-tool conversational turns are not captured structurally.*

```python
from openeval.adapters.openai import from_openai_messages

messages = [
    # ... your OpenAI messages list
]

# Convert the messages
trace = from_openai_messages(messages)
```

### GitHub Action

OpenEval provides a GitHub Action to seamlessly gate PRs based on your agents' performance.

```yaml
steps:
  - uses: actions/checkout@v4
  - name: Grade Agent Execution
    uses: organization/openeval-core@v1
    with:
      suite: path/to/your/test_suite_dir
      fail-under: '0.8'  # Fail the step if average metric score is < 80%
```
