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
