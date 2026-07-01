# OpenEval

Every existing tool evaluates what the agent said. OpenEval evaluates what the agent did.

## Installation

```bash
pip install -e .
```

## Usage

Run a single test case:
```bash
openeval run --trace trace.json --testcase test.json
```

Run a test suite:
```bash
openeval run --suite tests/ --output results/
```

Generate a report:
```bash
openeval report --input results/ --format markdown
```
