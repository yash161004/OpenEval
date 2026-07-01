# OpenEval: Project Status & Onboarding

Welcome to the project! This document outlines exactly where the OpenEval codebase stands, the core architecture decisions we've locked in, and what tasks are teed up next.

## 1. What We Have Done (Current Status)

We have built a fully functional, deterministic, CI/CD-native agent evaluation core. 
Currently, the codebase is stable, and **all 23 tests are passing**.

### Completed Components:
*   **Data Models (`openeval/models.py`):** 
    *   Defined `AgentTrace` (what the agent actually did) and `EvalTestCase` (what the agent was expected to do). 
    *   Both are strict JSON schemas. We extract state at scoring time rather than hardcoding provider-specific formats.
*   **Deterministic Metrics (`openeval/metrics.py`):**
    *   `ToolSelectionAccuracy`: Did the agent use the correct tools?
    *   `ArgumentCorrectness`: Did the agent pass the right arguments to those tools?
    *   `StepEfficiency`: Did the agent complete the task in the optimal number of steps?
    *   `GoalCompletionRate`: Did the final state match the expected state?
*   **Runner & CLI (`openeval/runner.py`, `openeval/cli.py`):**
    *   Implemented `openeval run --trace <file> --testcase <file>` for single runs.
    *   Implemented `openeval run --suite <dir> --output <dir>` for batch processing.
    *   Includes bulletproof error handling (won't crash the suite if a JSON file is missing or malformed).
*   **Infrastructure:**
    *   Configured `pyproject.toml` with `hatchling`. `pip install -e .` works locally.
    *   GitHub Actions CI is live (`.github/workflows/ci.yml`) and runs `pytest` on all PRs.
    *   Created our first working example in `examples/simple_agent/`.

## 2. What's Broken / Needs to be Done

Nothing is fundamentally "broken" or crashing, but we have two critical missing pieces of functionality to hit v0.1 completion:

1.  **Reporting is a Stub:** Running `openeval report` currently crashes intentionally with a `NotImplementedError`. The CLI can generate raw JSON result files, but it cannot yet compile them into human-readable Markdown summaries.
2.  **Missing Complex Example:** We have an empty directory at `examples/multi_step/`. We need a robust example of an agent performing multiple sequential steps so users have a template for complex evaluations.

## 3. Next Steps (Task Assignments)

To avoid merge conflicts and step on each other's toes, we are dividing the work based on context isolation.

### Your Tasks (The Collaborator)
You are taking on two isolated tasks that require zero dependencies on the rest of the moving parts:

1.  **Implement `report.py`:**
    *   You will write the `format_report(results: dict, format: Literal["markdown", "json"]) -> str` function.
    *   **Goal:** Take the raw dictionary output from a suite run and format it cleanly into Markdown (or raw JSON).
2.  **Create the `multi_step` Example:**
    *   Navigate to `examples/multi_step/`.
    *   Write a valid `trace.json` and `testcase.json` pair that passes our 4 core metrics. 
    *   **Goal:** Force you to read `models.py` and understand the JSON schema hands-on before you write the reporting logic.

### Core Rules for Collaboration
*   **No LLM-as-a-judge:** Our metrics are strictly deterministic (math and exact matching). Keep it that way.
*   **PRs Only:** Branch protection is enabled. Do not push directly to `main`. Create a branch, do your work, and open a Pull Request so we can review the GitHub Actions CI results before merging.
