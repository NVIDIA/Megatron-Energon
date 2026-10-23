---
name: opening-a-pr
description: Use when preparing or opening a pull request or issue for Megatron-Energon. Covers the PR template, the human-review checkbox, linked issues for external contributions, and what agents may and may not do on GitHub.
---

<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Opening a pull request

Policy lives in [CONTRIBUTING.md](../../../CONTRIBUTING.md); the template is
`.github/pull_request_template.md`.

## Agent rules

- Only push, open a PR or issue, or comment on GitHub when the human explicitly asks.
- Open PRs as **drafts** and fill in every section of the template.
- Leave the "I, a human, have personally reviewed every line" checkbox **unticked**. Only the human
  may tick it; CI fails until they do.
- Never post review replies or comments on the human's behalf unless asked.

## Before opening

1. `just fix`, `just check` and `just test` pass.
2. `just check-commits` passes (Conventional Commits, `Signed-off-by` of the commit author).
3. User-facing changes have docs in `docs/source`.
4. The branch is focused on a single concern.
5. No code was copied from other repositories or projects; everything is original and
   contributable under the project's BSD-3-Clause license.

## PR content

- **What does this PR do?** What and why, briefly.
- **Linked issue:** required for external contributors (`Fixes #123`). Issues must describe a real
  problem encountered while using Energon, not automated or speculative findings.
- **Testing:** exact commands that were run and their results.
- Tick only the checklist items that are true.
