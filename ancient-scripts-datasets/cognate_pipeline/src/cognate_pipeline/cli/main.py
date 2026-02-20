"""Main Typer application with 6 subcommands."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="cognate-pipeline",
    help="Cross-linguistic cognate detection pipeline with provenance tracking.",
    no_args_is_help=True,
)


@app.command()
def ingest_sources(
    config: str = typer.Option(..., "--config", "-c", help="Path to config YAML"),
    output_dir: str = typer.Option(
        None, "--output-dir", "-o", help="Override staging output directory"
    ),
) -> None:
    """Ingest lexical sources into staging JSONL files."""
    from .ingest_cmd import run_ingest

    run_ingest(config, output_dir)


@app.command()
def normalise_ipa(
    config: str = typer.Option(..., "--config", "-c", help="Path to config YAML"),
) -> None:
    """Normalise IPA and compute sound classes for staged lexemes."""
    from .normalise_cmd import run_normalise

    run_normalise(config)


@app.command()
def detect_cognates(
    config: str = typer.Option(..., "--config", "-c", help="Path to config YAML"),
    method: str = typer.Option(
        None, "--method", "-m", help="Override cognate detection method"
    ),
) -> None:
    """Detect cognate candidates and cluster into cognate sets."""
    from .detect_cmd import run_detect

    run_detect(config, method)


@app.command()
def load_db(
    config: str = typer.Option(..., "--config", "-c", help="Path to config YAML"),
) -> None:
    """Load staged data into PostgreSQL."""
    from .load_cmd import run_load

    run_load(config)


@app.command()
def export_cldf(
    config: str = typer.Option(..., "--config", "-c", help="Path to config YAML"),
) -> None:
    """Export database contents as CLDF Wordlist."""
    from .export_cldf_cmd import run_export_cldf

    run_export_cldf(config)


@app.command()
def export_jsonld(
    config: str = typer.Option(..., "--config", "-c", help="Path to config YAML"),
) -> None:
    """Export cognate links as JSON-LD."""
    from .export_jsonld_cmd import run_export_jsonld

    run_export_jsonld(config)


if __name__ == "__main__":
    app()
