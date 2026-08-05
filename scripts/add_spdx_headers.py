#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Add ``# SPDX-License-Identifier: AGPL-3.0-or-later`` headers to source files.

Walks the repository for .py and .sh files and inserts the SPDX header
after any shebang and any PEP 263 encoding declaration, before everything
else (including module docstrings).

- Idempotent: files that already contain an SPDX-License-Identifier line
  are left untouched.
- Files that carry a pre-existing licence boilerplate comment block right
  where the header would go (e.g. an inherited GPL notice) have that block
  removed and replaced by the single SPDX line, rather than stacking the
  new header on top of the old one.

Usage:
    python scripts/add_spdx_headers.py --dry-run
    python scripts/add_spdx_headers.py
"""
import argparse
import difflib
import re
import sys
from pathlib import Path

SPDX_LINE = "# SPDX-License-Identifier: AGPL-3.0-or-later"

SPDX_HEADER_RE = re.compile(r"^\s*#.*SPDX-License-Identifier\s*:")
SHEBANG_RE = re.compile(r"^#!")
CODING_RE = re.compile(r"^[ \t\f]*#.*coding[:=][ \t]*[-_.a-zA-Z0-9]+")

LICENSE_BLOCK_RE = re.compile(
    r"GNU (GENERAL|AFFERO|LESSER)?\s*PUBLIC LICENSE"
    r"|Free Software Foundation"
    r"|WITHOUT ANY WARRANTY"
    r"|Redistribute it and/or modify",
    re.IGNORECASE,
)

EXCLUDE_DIR_NAMES = {
    ".git", "__pycache__", ".venv", "venv", "build", "dist",
    ".ipynb_checkpoints", "node_modules",
}


def is_excluded(path: Path) -> bool:
    return any(
        part in EXCLUDE_DIR_NAMES or part.endswith(".egg-info")
        for part in path.parts
    )


def iter_source_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in (".py", ".sh"):
            continue
        if is_excluded(path):
            continue
        yield path


def find_insert_point(lines):
    """Return the index after any shebang and PEP 263 coding line."""
    idx = 0
    if lines and SHEBANG_RE.match(lines[0]):
        idx = 1
    if idx < len(lines) and CODING_RE.match(lines[idx]):
        idx += 1
    return idx


def find_license_block(lines, start):
    """If a licence boilerplate comment block starts at `start`, return its end index."""
    end = start
    while end < len(lines) and (lines[end].startswith("#") or lines[end].strip() == ""):
        end += 1
    if end == start:
        return None
    block_text = "".join(lines[start:end])
    if LICENSE_BLOCK_RE.search(block_text):
        return end
    return None


def build_new_lines(lines):
    """Return (new_lines, action, removed_block) for a file's original lines.

    action is one of "skip", "add", "replace".
    """
    # Only the first few lines count as "already has a header" — a match deep
    # in a docstring or in this very script's own source text must not count.
    if any(SPDX_HEADER_RE.match(l) for l in lines[:5]):
        return lines, "skip", None

    idx = find_insert_point(lines)
    block_end = find_license_block(lines, idx)

    if block_end is not None:
        removed_block = lines[idx:block_end]
        remainder = lines[block_end:]
        action = "replace"
    else:
        removed_block = None
        remainder = lines[idx:]
        action = "add"

    header = [SPDX_LINE + "\n"]
    if remainder and remainder[0].strip() != "":
        header.append("\n")

    new_lines = lines[:idx] + header + remainder
    return new_lines, action, removed_block


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root to scan (default: .)")
    parser.add_argument("--dry-run", action="store_true", help="Report planned changes without writing")
    args = parser.parse_args()

    root = Path(args.root).resolve()

    counts = {"add": 0, "replace": 0, "skip": 0}
    added_files = []
    replaced = []  # (path, removed_block, diff)

    for path in iter_source_files(root):
        original_text = path.read_text(encoding="utf-8")
        original_lines = original_text.splitlines(keepends=True)

        new_lines, action, removed_block = build_new_lines(original_lines)
        counts[action] += 1

        if action == "skip":
            continue

        rel = path.relative_to(root)

        if action == "add":
            added_files.append(rel)
        else:
            diff = "".join(difflib.unified_diff(
                original_lines, new_lines,
                fromfile=f"a/{rel}", tofile=f"b/{rel}",
            ))
            replaced.append((rel, removed_block, diff))

        if not args.dry_run:
            path.write_text("".join(new_lines), encoding="utf-8")

    mode = "DRY RUN" if args.dry_run else "APPLIED"
    print(f"=== {mode} ===")
    print(f"Header added:      {counts['add']}")
    print(f"Header replaced:   {counts['replace']}")
    print(f"Skipped (has SPDX):{counts['skip']}")
    print()

    if added_files:
        print(f"-- Files with header added ({len(added_files)}) --")
        for rel in added_files:
            print(f"  {rel}")
        print()

    if replaced:
        print(f"-- Files with pre-existing licence block replaced ({len(replaced)}) --")
        for rel, removed_block, diff in replaced:
            print(f"\n### {rel}")
            print("Removed block:")
            print("".join(f"    {l}" for l in removed_block))
            print("Diff:")
            print(diff)
    else:
        print("-- No pre-existing licence boilerplate blocks found to replace --")


if __name__ == "__main__":
    main()
