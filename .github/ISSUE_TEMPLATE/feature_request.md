---
name: Feature request
about: Propose a change or addition to OpenEval
title: "[FEATURE] "
labels: enhancement
assignees: ''
---

**What problem does this solve?**
Describe the gap in your evaluation workflow this would address. Be specific
about a real use case, not a hypothetical one.

**Proposed solution**
What would you like OpenEval to do? If this touches scoring semantics,
describe the exact edge-case behavior (e.g. how it handles missing fields,
partial matches, extra data) — not just the happy path.

**Does this require an LLM judge or external API call?**
OpenEval's core metrics are deterministic and API-free by design. If your
proposal requires an LLM call, explain why it can't be deterministic, and
note that it will likely need to live in an opt-in module rather than core.

**Alternatives considered**
Any other approaches you thought about and why you didn't propose them.

**Additional context**
Anything else relevant — links, examples from other tools, etc.
