"""Wrapper for running NeuroDecipher baseline."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

from repro.paths import ARTIFACTS, ROOT, THIRD_PARTY
from repro.utils import run_subprocess, write_json


def _pythonpath() -> str:
    entries = [
        str((THIRD_PARTY / "NeuroDecipher").resolve()),
        str((THIRD_PARTY / "neuro_arglib").resolve()),
        str((THIRD_PARTY / "neuro_dev_misc").resolve()),
        # Fallback locations.
        str((THIRD_PARTY / "arglib").resolve()),
        str((THIRD_PARTY / "dev_misc").resolve()),
    ]
    return ":".join(entries)


def run_neurodecipher(
    cfg: str = "UgaHebSmallNoSpe",
    log_dir: Optional[Path] = None,
    overrides: Optional[Dict[str, str]] = None,
    dry_run: bool = False,
) -> Dict[str, object]:
    log_dir = log_dir or (ARTIFACTS / "runs" / "neurodecipher")
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [
        sys.executable,
        "-m",
        "nd.main",
        "--config",
        cfg,
    ]
    if overrides:
        for k, v in overrides.items():
            cmd.extend([f"--{k}", str(v)])

    log_path = log_dir / "run.log"
    if dry_run:
        result = {
            "status": "dry_run",
            "cmd": cmd,
            "log_path": str(log_path),
        }
        write_json(log_dir / "metrics.json", result)
        return result

    proc = run_subprocess(
        cmd,
        cwd=THIRD_PARTY / "NeuroDecipher",
        env={"PYTHONPATH": _pythonpath()},
        log_path=log_path,
        check=False,
    )

    run_text = proc.stdout + "\n" + proc.stderr
    pattern = re.compile(r"\b\d+/\d+=(0?\.\d+|1(?:\.0+)?)\b")
    found = [float(x) for x in pattern.findall(run_text)]
    best = max(found) if found else None
    log_dir_match = re.search(r"'log_dir': '([^']+)'", run_text)
    inferred_log_dir = log_dir_match.group(1) if log_dir_match else None

    result = {
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "scores": found,
        "best_score": best,
        "nd_log_dir": inferred_log_dir,
        "cfg": cfg,
        "log_path": str(log_path),
        "cmd": cmd,
    }
    write_json(log_dir / "metrics.json", result)
    return result
