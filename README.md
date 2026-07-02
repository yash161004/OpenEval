# OpenEval

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

Get your first evaluation running in under 5 minutes using a real LangChain agent.

**1. Install the package**
*(Note: Because the newest adapters are not yet published to PyPI, please install from source for now. A future update will simplify this to a single `pip install openeval-core` once shipped.)*
```bash
git clone <repo>
cd <repo>
pip install -e .
```
*(Note: the CLI command is `openeval`)*

**2. Run the Quickstart Agent**
Run the included example agent. It uses a mock LLM so you don't need any API keys. It will execute a simple tool and save its trace to disk.
```bash
python examples/quickstart/quickstart_agent.py
```

**3. Grade the Execution**
Run the OpenEval CLI to score the agent's behavior against our predefined test case:
```bash
openeval run --trace examples/quickstart/trace.json --testcase examples/quickstart/testcase.json
```

**Expected Output:**
```json
{
  "task_id": "quickstart",
  "metrics": {
    "Tool Selection Accuracy": {
      "metric_name": "Tool Selection Accuracy",
      "score": 1.0,
      "passed": true,
      "details": "Correctly called 1 of 1 expected tools."
    },
    "Argument Correctness": {
      "metric_name": "Argument Correctness",
      "score": 1.0,
      "passed": true,
      "details": "Correctly matched 1 of 1 expected arguments."
    },
    "Step Efficiency": {
      "metric_name": "Step Efficiency",
      "score": 1.0,
      "passed": true,
      "details": "Took 1 steps vs optimal 1 steps."
    },
    "Goal Completion Rate": {
      "metric_name": "Goal Completion Rate",
      "score": 1.0,
      "passed": true,
      "details": "No expected state to match."
    }
  }
}
```

## Next Steps
- See the **LangChain Adapter** or **OpenAI Tool Calling** sections below to connect your real app.
- Check out the **GitHub Action** to add CI/CD gating to your repository.

### LangChain Adapter

OpenEval can automatically convert LangChain Agent executions into `AgentTrace` objects.

*LIMITATION: This adapter flattens the run tree. Parent/child tool relationships are not preserved structurally (only call order is retained), and non-tool runs (sub-chains, intermediate LLM calls) between tool boundaries are not captured.*

```python
from openeval.adapters.langchain import from_langchain_run
from langchain_core.tracers.context import collect_runs

with collect_runs() as cb:
    # Run your langchain agent
    agent.invoke({"input": "task input"})
    
# Convert the run
trace = from_langchain_run(cb.traced_runs[0])
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
