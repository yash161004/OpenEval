# Contributing to OpenEval

Thanks for considering a contribution. OpenEval is small and opinionated on
purpose — read this before opening a PR so your work doesn't get bounced.

## Ground rules

1. **No LLM-judge dependencies in core.** The entire value proposition of
   OpenEval's v0.1 metrics (`ToolSelectionAccuracy`, `ArgumentCorrectness`,
   `StepEfficiency`, `GoalCompletionRate`) is that they are deterministic and
   require no API calls. PRs that introduce an LLM call into the core scoring
   path will be rejected, regardless of how useful the feature is. If you
   want LLM-judge-based scoring, propose it as a clearly separate, opt-in
   module — not a change to core.
2. **No scope expansion without discussion.** Hosted dashboards, new metric
   categories, and framework integrations beyond what's already adopted
   (LangChain, OpenAI) should be raised as an issue first, not shipped as a
   surprise PR.
3. **Tests are not optional.** Every new metric or CLI behavior needs
   corresponding tests. CLI changes should use `typer.testing.CliRunner` with
   `tmp_path` fixtures, consistent with the existing test suite.
4. **Exit codes matter.** This is a CI/CD tool. If your change touches
   `report.py` or `runner.py`, verify the exit code is correct for both
   success and failure paths — silent success on a failure path is treated
   as a bug, not a style nit.

## Before you open a PR

- Run the full test suite locally and include the raw output in your PR
  description. Summaries without raw `pytest` output will not be accepted
  as evidence that something passes.
- Run coverage (`pytest --cov`) if you touched scoring logic. Don't claim a
  coverage number without the report to back it up.
- If you're changing scoring semantics (e.g. how a metric handles a missing
  field, extra kwargs, or partial matches), document the new edge-case
  behavior explicitly in the PR — don't leave it implicit in code.

## Workflow

1. Open an issue describing the bug or feature before writing code, unless
   it's a trivial fix (typo, docs, obvious bug with an obvious fix).
2. Fork, branch, implement, test.
3. Open a PR against `main`. CI (`ci.yml`) must pass before review.
4. Keep PRs scoped to one concern. A PR that fixes a bug and adds a feature
   will be asked to split.

## What NOT to contribute right now

- Hosted platform / dashboard features — explicitly out of scope until
  OpenEval has real adopted users.
- New adapters beyond LangChain/OpenAI, unless you're also committing to
  maintain them.
- Planning or roadmap documents committed into the repo itself — open a
  GitHub Discussion or issue instead.

## Questions

Open an issue with the `question` label, or start a Discussion if the repo
has Discussions enabled.
