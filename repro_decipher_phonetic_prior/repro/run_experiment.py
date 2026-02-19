"""CLI entrypoint for paper-style experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

from repro.eval.common import MissingDataError, write_json
from repro.eval.gothic import run_gothic
from repro.eval.iberian_closeness import run_iberian_closeness
from repro.eval.iberian_names import run_iberian_names
from repro.eval.ugaritic import run_ugaritic
from repro.paths import ROOT


DEFAULT_CONFIGS = {
    "gothic": ROOT / "configs" / "gothic.yaml",
    "ugaritic": ROOT / "configs" / "ugaritic.yaml",
    "iberian-names": ROOT / "configs" / "iberian.yaml",
    "iberian-closeness": ROOT / "configs" / "iberian.yaml",
}


def _split_variants(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None or raw.strip() == "":
        return None
    return [x.strip() for x in raw.split(",") if x.strip()]


def _resolve_config(experiment: str, config_override: Optional[str]) -> Path:
    if config_override:
        path = Path(config_override)
    else:
        path = DEFAULT_CONFIGS[experiment]
    if not path.is_absolute():
        path = ROOT / path
    return path


def _common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="Path to YAML config.")
    parser.add_argument("--variants", help="Comma-separated variants (e.g. base,full).")
    parser.add_argument("--restarts", type=int, default=5)
    parser.add_argument("--seed-base", type=int, default=1234)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="Use 1 restart and short runs on <=50 queries.")
    parser.add_argument("--output-root", default="outputs")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TACL 2021 paper-style experiments.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_g = sub.add_parser("gothic", help="Run Gothic paper experiment (P@10, table 2/4 style).")
    _common_args(p_g)

    p_u = sub.add_parser("ugaritic", help="Run Ugaritic paper experiment (P@1, table 3 style).")
    _common_args(p_u)

    p_in = sub.add_parser("iberian-names", help="Run Iberian personal-names experiment (Figure 4a style).")
    _common_args(p_in)

    p_ic = sub.add_parser("iberian-closeness", help="Run Gothic/Ugaritic/Iberian closeness analysis (Figure 4b/c/d style).")
    _common_args(p_ic)

    p_all = sub.add_parser("all", help="Run all paper experiments.")
    _common_args(p_all)

    args = parser.parse_args()

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        if args.cmd == "gothic":
            payload = run_gothic(
                config_path=_resolve_config("gothic", args.config),
                output_root=output_root,
                variants=_split_variants(args.variants),
                restarts=args.restarts,
                seed_base=args.seed_base,
                max_queries=args.max_queries,
                smoke=args.smoke,
            )
            print(f"Wrote Gothic outputs under {output_root / 'gothic'}")
            write_json(output_root / "gothic" / "_run_invocation.json", payload)
            return

        if args.cmd == "ugaritic":
            payload = run_ugaritic(
                config_path=_resolve_config("ugaritic", args.config),
                output_root=output_root,
                variants=_split_variants(args.variants),
                restarts=args.restarts,
                seed_base=args.seed_base,
                max_queries=args.max_queries,
                smoke=args.smoke,
            )
            print(f"Wrote Ugaritic outputs under {output_root / 'ugaritic'}")
            write_json(output_root / "ugaritic" / "_run_invocation.json", payload)
            return

        if args.cmd == "iberian-names":
            payload = run_iberian_names(
                config_path=_resolve_config("iberian-names", args.config),
                output_root=output_root,
                variants=_split_variants(args.variants),
                restarts=args.restarts,
                seed_base=args.seed_base,
                max_queries=args.max_queries,
                smoke=args.smoke,
            )
            print(f"Wrote Iberian names outputs under {output_root / 'iberian_names'}")
            write_json(output_root / "iberian_names" / "_run_invocation.json", payload)
            return

        if args.cmd == "iberian-closeness":
            payload = run_iberian_closeness(
                config_path=_resolve_config("iberian-closeness", args.config),
                output_root=output_root,
                variants=_split_variants(args.variants),
                restarts=args.restarts,
                seed_base=args.seed_base,
                max_queries=args.max_queries,
                smoke=args.smoke,
            )
            print(f"Wrote closeness outputs under {output_root / 'iberian_closeness'}")
            write_json(output_root / "iberian_closeness" / "_run_invocation.json", payload)
            return

        if args.cmd == "all":
            gothic_variants = _split_variants(args.variants) or ["base", "partial", "full"]
            payload_g = run_gothic(
                config_path=_resolve_config("gothic", args.config),
                output_root=output_root,
                variants=gothic_variants,
                restarts=args.restarts,
                seed_base=args.seed_base,
                max_queries=args.max_queries,
                smoke=args.smoke,
            )
            payload_u = run_ugaritic(
                config_path=_resolve_config("ugaritic", None),
                output_root=output_root,
                variants=["base", "full"],
                restarts=args.restarts,
                seed_base=args.seed_base,
                max_queries=args.max_queries,
                smoke=args.smoke,
            )
            payload_in = run_iberian_names(
                config_path=_resolve_config("iberian-names", None),
                output_root=output_root,
                variants=["base", "full"],
                restarts=args.restarts,
                seed_base=args.seed_base,
                max_queries=args.max_queries,
                smoke=args.smoke,
            )
            payload_ic = run_iberian_closeness(
                config_path=_resolve_config("iberian-closeness", None),
                output_root=output_root,
                variants=["base", "full"],
                restarts=args.restarts,
                seed_base=args.seed_base,
                max_queries=args.max_queries,
                smoke=args.smoke,
            )
            write_json(
                output_root / "paper_run_summary.json",
                {
                    "gothic": payload_g,
                    "ugaritic": payload_u,
                    "iberian_names": payload_in,
                    "iberian_closeness": payload_ic,
                },
            )
            print(f"Wrote full paper run outputs under {output_root}")
            return

    except MissingDataError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
