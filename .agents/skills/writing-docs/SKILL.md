---
name: writing-docs
description: Use when adding or changing user-facing behavior or documentation in Megatron-Energon, or when creating new .py, .sh or .md files that need license headers.
---

<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Documentation and license headers

## Docs

- User documentation lives in `docs/source` (Sphinx, MyST markdown). Every user-facing change
  (new API, option, CLI command, behavior change) needs matching documentation there.
- Find the existing page that covers the area and extend it before adding a new page; new pages
  must be linked from a `toctree`.
- Match the tone, heading structure and code-block style of the surrounding pages.
- Update docstrings of public API you change; they feed the API docs.

## License headers

Every new `.py`, `.sh` and `.md` file needs the NVIDIA license header.

- `just fix` adds it automatically in `src/` and `tests/`.
- Elsewhere, copy it from an existing file of the same type, e.g. `src/megatron/energon/__init__.py`
  for Python and the top of `AGENTS.md` for Markdown.
- Markdown files that need front matter (such as `SKILL.md`) put the header right after it.

Run `just fix` and `just check` when done.
