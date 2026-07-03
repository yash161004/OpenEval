# Changelog

All notable changes to OpenEval are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2] - 2026-07-03

### Added
- **Project Polish:** Added `LICENSE`, `CONTRIBUTING.md`, GitHub issue templates, README badges, and configured Codecov integration.

### Fixed
- **Dependencies:** Added `langchain-core` and `openai` as proper optional dependencies (`openeval-core[langchain,openai]`). Previously, these were undeclared transitive dependencies, which caused test collection to crash in a clean CI environment and meant the adapter tests were never actually validated during Phase 1.
- **Metrics:** Clarified the `ArgumentCorrectness` failure message when no expected tools are called to explicitly state it is treated as a failure.
- **CI:** Fixed an issue where the `publish.yml` workflow was not triggering automatically on tag pushes because it was listening for `release: [published]` instead of `push: tags`.

## [0.1.1] - 2026-07-02

### Fixed
- **CLI:** `openeval report` now exits with a non-zero status code (exit code 1) when encountering corrupted or malformed evaluation result JSON files. Previously, it silently logged a warning to standard error and exited with 0, which could mask data loss and cause false-positive passes in CI/CD environments.
- Threshold-scoring algorithm corrected to hard-fail on task-level errors rather than diluting them into averages.

## [0.1.0] - Initial Release

### Added
- Four deterministic evaluation metrics: `ToolSelectionAccuracy`, `ArgumentCorrectness`, `StepEfficiency`, `GoalCompletionRate`.
- `runner.py`, `cli.py`, and `report.py` (pure Markdown formatter, zero computation at render time).
- GitHub Actions CI (`ci.yml`) and PyPI publish (`publish.yml`) workflows.
- LangChain and OpenAI adapters for converting agent traces into OpenEval's trace format.
- GitHub Action for running OpenEval in CI.
- Example scripts using `FakeListLLM` for deterministic trace-conversion testing.
