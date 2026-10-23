<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

## Megatron-Energon OSS Contribution Rules

Megatron Energon is a general-purpose PyTorch data loader for large-scale, multi-modal training.


#### Human Responsibility and AI Tools

You may use AI tools (coding agents, assistants, code generators) to help with your contribution.
However:

- **A human must personally review every line** of every pull request, issue and comment before it
  is submitted. You must understand the change and be able to answer review questions yourself,
  without relaying them to an AI.
- **You are responsible for the contribution**, regardless of how it was produced. The
  `Signed-off-by` line on your commits (see [Signing Your Work](#signing-your-work)) is your
  statement of that responsibility.
- Do not add `Co-Authored-By` (or similar) trailers for AI tools. Only humans author and sign off
  commits in this repository.
- Pull requests and issues must confirm the human review in their template. We may close
  contributions without review when they lack this confirmation or look like unreviewed generated
  output.

Agents working in this repository additionally follow [AGENTS.md](AGENTS.md).


#### External Contributions

Contributions from outside the core team are welcome when they solve a real-world problem:

- **Open an issue first** that describes a problem you actually encountered while using Energon:
  what you were loading or training, what went wrong, and how to reproduce it.
- **Every external pull request must link such an issue.** Pull requests without one may be closed
  without review.
- We do not accept contributions whose main purpose is volume, such as findings from automated or
  AI-driven bug hunting, static analysis or fuzzing without a real-world impact, typo-only or
  style-only changes, or speculative "hardening". Report a security vulnerability through
  [NVIDIA's security process](https://www.nvidia.com/en-us/security/) instead of a public issue.
- For larger changes (new features, architectural changes), discuss the design in the issue before
  writing code.


#### Coding Guidelines

- All source code contributions must strictly adhere to ruff linter and formatter as configured in `pyproject.toml`.
    - You can apply fixes with `just fix` and check them by running `just check`.

- All source code files must include the NVIDIA license header as you can see in `src/megatron/energon/__init__.py`.

- In addition, please follow the existing conventions in the relevant file, submodule, module, and project when you add new code or when you extend/fix existing functionality.

- Avoid introducing unnecessary complexity into existing code so that maintainability and readability are preserved.

- Try to keep pull requests (PRs) as concise as possible:
  - Avoid committing commented-out code.
  - Do not touch code outside the scope of the PR, including formatting or renames.
  - Wherever possible, each PR should address a single concern. If there are several otherwise-unrelated things that should be fixed to reach a desired endpoint, our recommendation is to open several PRs and indicate the dependencies in the description. The more complex the changes are in a single PR, the more time it will take to review those changes.

- All components must contain accompanying documentation in `docs/source`.

- Make sure that you can contribute your work to open source (no license and/or patent conflict is introduced by your code). You will need to [`sign`](#signing-your-work) your commit.


#### Testing

- Ensure that all tests pass prior to submitting your code (`just test`).
- We prefer end-to-end tests over many small unit tests: create a small real dataset in a temporary
  directory, load it through the public API, and check the samples that come out, including
  save/restore of the loader state where relevant. See `tests/test_dataset.py` for examples.
- Extend existing test files where possible instead of creating new ones.


#### Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/). Pull requests are
merged without squashing, so every commit ends up in the history and must follow the format:

```
<type>(<scope>): <description>

[optional body]

Signed-off-by: Your Name <your@email.com>
```

- **type** is one of:

  | Type       | Use for                                                       |
  | ---------- | ------------------------------------------------------------- |
  | `feat`     | A new feature or user-facing capability                       |
  | `fix`      | A bug fix                                                     |
  | `perf`     | A performance improvement without behavior change             |
  | `refactor` | A code change that neither fixes a bug nor adds a feature     |
  | `docs`     | Documentation only                                            |
  | `test`     | Adding or changing tests only                                 |
  | `build`    | Build system, packaging, dependencies, `justfile`             |
  | `ci`       | CI configuration (`.github/`)                                 |
  | `style`    | Formatting only, no code change                               |
  | `chore`    | Maintenance that fits none of the above                       |
  | `revert`   | Reverting a previous commit                                   |

- **scope** (optional) names the affected area, usually a module or package, e.g. `loader`,
  `sampler`, `recipe`, `flavors`, `wrappers`, `task_encoder`, `cache`, `av`, `cli`.
- **description** is a short summary in the imperative mood ("add", not "added"), starting
  lowercase, without a trailing period.

Examples from this repository:

```
fix(loader): close dataset and reader resources deterministically
perf(sampler): use Feistel-network permutations for slice shuffling
build(just): check Ruff formatting in just check
```

CI checks every commit of a pull request for the format and the sign-off. Merge commits, e.g. from
GitHub's "Update branch" button, are exempt. Run `just check-commits` to check your branch locally.


#### Signing Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

  * The sign-off must be made by the human author of the commit, with the same email address as the commit author. It also states that you take responsibility for the contribution, including any parts produced with AI tools.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "feat(loader): add cool feature"
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO:

  ```
    Developer Certificate of Origin
    Version 1.1
    
    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129
    
    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.


    Developer's Certificate of Origin 1.1
    
    By making a contribution to this project, I certify that:
    
    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or
    
    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or
    
    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.
    
    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
  ```
