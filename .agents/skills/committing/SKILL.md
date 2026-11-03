---
name: committing
description: Use when creating git commits in Megatron-Energon. Covers Conventional Commit format, scopes, sign-off, and checking commits locally.
---

<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Committing

The authoritative rules are in [CONTRIBUTING.md](../../../CONTRIBUTING.md#commit-messages) and
[Signing Your Work](../../../CONTRIBUTING.md#signing-your-work). Summary:

## Format

```
<type>(<scope>): <description>

[optional body]
```

- `type`: `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `build`, `ci`, `style`, `chore`,
  `revert`.
- `scope` (optional): affected area, e.g. `loader`, `sampler`, `recipe`, `flavors`, `wrappers`,
  `task_encoder`, `cache`, `av`, `cli`.
- `description`: imperative mood, lowercase start, no trailing period.
- Breaking changes: add `!` after the scope (`refactor(recipe)!: ...`) and explain in the body.
- Pull requests are merged without squashing: every commit must follow the format.

## Rules

- Always sign off: `git commit -s`. The human you work for signs off and is responsible.
- Never add `Co-Authored-By` (or similar) trailers for AI assistants.
- One concern per commit. Run `just fix` and `just check` before committing.
- Before finishing, run `just check-commits` to validate sign-off and format of the branch.
- Never push unless the human explicitly asks.
