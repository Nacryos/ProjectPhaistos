"""Wrapper for running xib-based extraction/decipher jobs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

from repro.paths import ARTIFACTS, ROOT, THIRD_PARTY
from repro.utils import run_subprocess, write_json


def _pythonpath() -> str:
    entries = [
        str((THIRD_PARTY / "xib").resolve()),
        str((THIRD_PARTY / "dev_misc").resolve()),
    ]
    return ":".join(entries)


def run_xib_extract(
    data_path: Path,
    vocab_path: Path,
    log_dir: Optional[Path] = None,
    extra_args: Optional[List[str]] = None,
    dry_run: bool = False,
) -> Dict[str, object]:
    """Run xib extract task without patching upstream core logic."""
    log_dir = log_dir or (ARTIFACTS / "runs" / "xib_extract")
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "xib.main",
        "--task",
        "extract",
        "--data_path",
        str(data_path),
        "--vocab_path",
        str(vocab_path),
        "--log_dir",
        str(log_dir),
        "--num_steps",
        "20",
        "--eval_interval",
        "10",
        "--check_interval",
        "5",
        "--input_format",
        "ipa",
    ]
    if extra_args:
        cmd.extend(extra_args)

    log_path = log_dir / "run.log"
    if dry_run:
        payload = {
            "status": "dry_run",
            "cmd": cmd,
            "log_path": str(log_path),
        }
        write_json(log_dir / "metrics.json", payload)
        return payload

    proc = run_subprocess(
        cmd,
        cwd=ROOT,
        env={"PYTHONPATH": _pythonpath()},
        log_path=log_path,
        check=False,
    )

    payload = {
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "log_path": str(log_path),
        "cmd": cmd,
    }
    write_json(log_dir / "metrics.json", payload)
    return payload
