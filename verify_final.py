"""Verify the paper-first repository surface without running experiments."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUIRED = [
    "README.md",
    "STATUS.md",
    "CLAIM_EVIDENCE.md",
    "BRANCH_AUDIT.md",
    "SOURCE_AUDIT.md",
    "ENVIRONMENT.md",
    "AUTHOR_THANK_YOU.md",
    "CITATION.cff",
    "claims.json",
    "EVIDENCE_MANIFEST.json",
    "paper_2604.14669v1.pdf",
    "paper_2604.14669v2.pdf",
    "source/arxiv/2604.14669v1.tar",
    "source/arxiv/2604.14669v2.tar",
    "pyproject.toml",
    "uv.lock",
    "zo_eos/operators.py",
    "zo_eos/formulas.py",
    "zo_eos/verify_theory.py",
    "zo_eos/nn_train.py",
]
EXPECTED_BRANCHES = {
    "main",
    "baseline/theory-reproduction",
    "experiment/cnn-eos",
}
CANONICAL_NAME = "MachineLearning-Nerd"
CANONICAL_EMAIL = "MachineLearning-Nerd@users.noreply.github.com"


def fail(message: str) -> None:
    raise SystemExit(f"VERIFY_FINAL_FAIL: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def main() -> int:
    missing = [path for path in REQUIRED if not (ROOT / path).is_file()]
    if missing:
        fail("missing required files: " + ", ".join(missing))

    claims = json.loads((ROOT / "claims.json").read_text())
    claim_rows = claims.get("claims", [])
    if [row.get("id") for row in claim_rows] != [f"C{i}" for i in range(1, 7)]:
        fail("claims.json must contain C1 through C6 in order")
    for row in claim_rows:
        if not row.get("implementation"):
            fail(f"{row.get('id')} has no implementation path")
        if not row.get("evidence"):
            fail(f"{row.get('id')} has no evidence path")
        for path in row["evidence"]:
            if not (ROOT / path).is_file():
                fail(f"claim evidence is missing: {path}")

    manifest = json.loads((ROOT / "EVIDENCE_MANIFEST.json").read_text())
    checked = 0
    for entry in manifest.get("artifacts", []):
        path = ROOT / entry["path"]
        if not path.is_file():
            fail(f"manifest artifact is missing: {entry['path']}")
        actual = sha256(path)
        if actual != entry["sha256"]:
            fail(f"hash mismatch for {entry['path']}: {actual}")
        checked += 1

    branches = {
        line.strip()
        for line in git(
            "for-each-ref", "--format=%(refname:short)", "refs/heads"
        ).splitlines()
        if line.strip()
    }
    if branches != EXPECTED_BRANCHES:
        fail(f"local branches differ: expected {sorted(EXPECTED_BRANCHES)}, got {sorted(branches)}")

    stale = sorted(
        branch for branch in branches if branch == "master" or branch.startswith("orx/")
    )
    if stale:
        fail("stale branches remain: " + ", ".join(stale))

    author_rows = git("log", "--all", "--format=%an%x09%ae").splitlines()
    committer_rows = git("log", "--all", "--format=%cn%x09%ce").splitlines()
    canonical = f"{CANONICAL_NAME}\t{CANONICAL_EMAIL}"
    noncanonical = sorted(
        {
            identity
            for identity in author_rows + committer_rows
            if identity and identity != canonical
        }
    )
    if noncanonical:
        fail("non-canonical commit identities: " + " | ".join(noncanonical))

    messages = git("log", "--all", "--format=%B").splitlines()
    if any(line.lower().startswith("co-authored-by:") for line in messages):
        fail("co-author trailer found")

    summary = json.loads(
        (ROOT / "audit/faithful-theory/theory_summary.json").read_text()
    )
    if summary.get("verdict") != "VERIFIED" or summary.get("n_problems") != 34:
        fail("local theory summary does not report 34 verified problems")

    print(
        "VERIFY_FINAL_PASS: "
        f"{len(REQUIRED)} required files, {len(claim_rows)} claims, "
        f"{checked} manifest artifacts, {len(branches)} local branches, "
        "canonical author and committer identities"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
