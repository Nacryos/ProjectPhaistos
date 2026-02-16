"""Helpers for pinned third-party repositories."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from repro.paths import ROOT, THIRD_PARTY
from repro.utils import read_json


@dataclass(frozen=True)
class RepoLock:
    name: str
    url: str
    commit: str
    commit_date: str
    required: bool


def lock_path() -> Path:
    return THIRD_PARTY / "LOCK.json"


def load_lock() -> Dict:
    return read_json(lock_path())


def repo_path(name: str) -> Path:
    return THIRD_PARTY / name


def current_commit(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def validate_lock(strict: bool = True) -> List[str]:
    lock = load_lock()
    errors: List[str] = []
    for repo in lock.get("repos", []):
        name = repo["name"]
        target = repo_path(name)
        if not target.exists():
            msg = f"Missing repo: {name} at {target}"
            if repo.get("required", False) or strict:
                errors.append(msg)
            continue
        sha = current_commit(target)
        if sha != repo["commit"]:
            errors.append(f"Repo {name} at {sha} does not match lock {repo['commit']}")
    return errors
