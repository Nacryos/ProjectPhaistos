"""Fast smoke test for DP code paths and training step."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch

from repro.model import PhoneticPriorConfig, PhoneticPriorModel, train_one_step
from repro.paths import ARTIFACTS
from repro.utils import set_global_seeds, utc_now_iso, write_json


def run_smoke(steps: int = 5, seed: int = 1234) -> Path:
    set_global_seeds(seed)

    inscriptions = [
        "þammuhsaminhaidau",
        "gards",
        "wulfs",
        "sunus",
        "hausjan",
    ]
    known_vocab = ["xaið", "raið", "braið", "gard", "wulf", "sunu", "hausjan"]

    chars_lost = sorted(set("".join(inscriptions)))
    chars_known = sorted(set("".join(known_vocab)))

    cfg = PhoneticPriorConfig(
        temperature=0.2,
        alpha=3.5,
        lambda_cov=1.0,
        lambda_loss=1.0,
        min_span=3,
        max_span=6,
        embedding_dim=16,
        lr=0.2,
        seed=seed,
    )
    model = PhoneticPriorModel(chars_lost, chars_known, config=cfg)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)

    history = []
    for step in range(1, steps + 1):
        out = train_one_step(model, optimizer, inscriptions, known_vocab)
        history.append(
            {
                "step": step,
                "objective": out.objective,
                "quality": out.quality,
                "omega_cov": out.omega_cov,
                "omega_loss": out.omega_loss,
                "num_sequences": out.num_sequences,
            }
        )

    result = {
        "status": "ok",
        "created_at": utc_now_iso(),
        "seed": seed,
        "steps": steps,
        "history": history,
    }

    out_path = ARTIFACTS / "runs" / "smoke_test.json"
    write_json(out_path, result)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a quick smoke test for DP/training code paths.")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    out = run_smoke(steps=args.steps, seed=args.seed)
    print(f"Smoke test written to {out}")


if __name__ == "__main__":
    main()
