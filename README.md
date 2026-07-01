# OpenEval

Every existing tool evaluates what the agent said. OpenEval evaluates what the agent did.

## Installation

```bash
pip install -e .
```

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
