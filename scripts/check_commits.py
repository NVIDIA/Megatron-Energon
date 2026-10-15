# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
"""Checks that commits are signed off (DCO) and follow Conventional Commits.

Usage: python scripts/check_commits.py <revision range>, e.g. origin/develop..HEAD
"""

import re
import subprocess
import sys

COMMIT_TYPES = (
    "feat",
    "fix",
    "perf",
    "refactor",
    "docs",
    "test",
    "build",
    "ci",
    "chore",
    "style",
    "revert",
)

SUBJECT_RE = re.compile(rf"^(?:{'|'.join(COMMIT_TYPES)})(?:\([a-z0-9_./-]+\))?!?: \S")
EMAIL_RE = re.compile(r"<([^>]+)>")

# Separators that cannot appear in commit metadata
FIELD_SEP = "\x1f"
RECORD_SEP = "\x1e"


def list_commits(rev_range: str) -> list[tuple[str, str, str, str]]:
    """Returns (sha, author email, sign-off trailers, raw message) for all non-merge commits in the
    range. Git parses the trailers, so sign-offs quoted in the message body are not included."""
    out = subprocess.run(
        [
            "git",
            "log",
            "--no-merges",
            f"--format=%H{FIELD_SEP}%ae{FIELD_SEP}%(trailers:key=Signed-off-by,valueonly)"
            f"{FIELD_SEP}%B{RECORD_SEP}",
            rev_range,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    commits = []
    for record in out.split(RECORD_SEP):
        record = record.strip("\n")
        if not record:
            continue
        sha, email, signoffs, message = record.split(FIELD_SEP, 3)
        commits.append((sha, email, signoffs, message))
    return commits


def check_commit(email: str, signoffs: str, message: str) -> list[str]:
    """Returns the list of problems found in a single commit."""
    problems = []
    subject = message.split("\n", 1)[0]
    if not SUBJECT_RE.match(subject):
        problems.append(
            f"subject {subject!r} is not a Conventional Commit "
            f"('<type>(<scope>): <description>', type one of: {', '.join(COMMIT_TYPES)})"
        )
    signoff_emails = [m.lower() for m in EMAIL_RE.findall(signoffs)]
    if email.lower() not in signoff_emails:
        problems.append(
            f"missing 'Signed-off-by' trailer for the author <{email}> (commit with 'git commit -s')"
        )
    return problems


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    commits = list_commits(sys.argv[1])
    failed = 0
    for sha, email, signoffs, message in commits:
        problems = check_commit(email, signoffs, message)
        if problems:
            failed += 1
            print(f"{sha[:12]}: {message.split(chr(10), 1)[0]}")
            for problem in problems:
                print(f"  - {problem}")
    if failed:
        print(
            f"\n{failed} of {len(commits)} commit(s) failed. See CONTRIBUTING.md#commit-messages. "
            "Fix them with 'git rebase -i' / 'git commit --amend -s' and force-push."
        )
        return 1
    print(f"All {len(commits)} commit(s) are signed off and follow Conventional Commits.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
