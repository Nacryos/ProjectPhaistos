"""Export database contents as CLDF Wordlist."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session, sessionmaker

from cognate_pipeline.config.schema import ExportConfig
from cognate_pipeline.db.schema import (
    CognateSetMember,
    CognateSetTable,
    Language,
    Lexeme,
    Source,
)

logger = logging.getLogger(__name__)


class CldfExporter:
    """Exports database contents as a CLDF Wordlist dataset."""

    def __init__(self, engine: Engine, config: ExportConfig) -> None:
        self.engine = engine
        self.config = config
        self._session_factory = sessionmaker(bind=engine)

    def export(self, output_dir: Path) -> None:
        """Export CLDF files to the given directory."""
        try:
            from pycldf import Wordlist
        except ImportError:
            # Fallback: write CSV manually
            self._export_csv_fallback(output_dir)
            return

        ds = Wordlist.in_dir(output_dir)
        ds.add_component("LanguageTable")
        ds.add_component("ParameterTable")
        ds.add_component("CognateTable")

        with self._session_factory() as session:
            # Languages
            languages = list(session.execute(select(Language)).scalars())
            for lang in languages:
                ds.add_language(
                    ID=lang.glottocode,
                    Name=lang.name,
                    Glottocode=lang.glottocode,
                    ISO639P3code=lang.iso639_3 or "",
                )

            # Lexemes -> Forms + Parameters
            concepts_seen: set[str] = set()
            lexemes = list(session.execute(select(Lexeme)).scalars())
            for lex in lexemes:
                if lex.concept_id and lex.concept_id not in concepts_seen:
                    ds.add_concept(ID=lex.concept_id, Name=lex.concept_id)
                    concepts_seen.add(lex.concept_id)

                lang = session.get(Language, lex.language_id)
                segments = list(lex.phonetic_canonical) if lex.phonetic_canonical else []
                ds.add_form(
                    ID=str(lex.id),
                    Language_ID=lang.glottocode if lang else "",
                    Parameter_ID=lex.concept_id,
                    Form=lex.lemma,
                    Segments=segments,
                )

            # Cognate sets
            cog_sets = list(session.execute(select(CognateSetTable)).scalars())
            for cs in cog_sets:
                members = list(
                    session.execute(
                        select(CognateSetMember).where(
                            CognateSetMember.cognate_set_id == cs.id
                        )
                    ).scalars()
                )
                for member in members:
                    ds.add_cognate(
                        ID=f"cog_{cs.id}_{member.lexeme_id}",
                        Form_ID=str(member.lexeme_id),
                        Cognateset_ID=str(cs.id),
                    )

        ds.write()
        logger.info("CLDF export complete: %s", output_dir)

    def _export_csv_fallback(self, output_dir: Path) -> None:
        """Export as plain CSV when pycldf is not available."""
        import csv

        output_dir.mkdir(parents=True, exist_ok=True)

        with self._session_factory() as session:
            # Languages
            languages = list(session.execute(select(Language)).scalars())
            with (output_dir / "languages.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "Name", "Glottocode", "ISO639P3code"])
                for lang in languages:
                    writer.writerow([lang.glottocode, lang.name, lang.glottocode, lang.iso639_3 or ""])

            # Forms
            lexemes = list(session.execute(select(Lexeme)).scalars())
            concepts: set[str] = set()
            with (output_dir / "forms.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "Language_ID", "Parameter_ID", "Form", "Segments"])
                for lex in lexemes:
                    lang = session.get(Language, lex.language_id)
                    segments = " ".join(list(lex.phonetic_canonical)) if lex.phonetic_canonical else ""
                    writer.writerow([
                        lex.id,
                        lang.glottocode if lang else "",
                        lex.concept_id,
                        lex.lemma,
                        segments,
                    ])
                    if lex.concept_id:
                        concepts.add(lex.concept_id)

            # Parameters
            with (output_dir / "parameters.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "Name"])
                for c in sorted(concepts):
                    writer.writerow([c, c])

            # Cognates
            cog_sets = list(session.execute(select(CognateSetTable)).scalars())
            with (output_dir / "cognates.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "Form_ID", "Cognateset_ID"])
                for cs in cog_sets:
                    members = list(
                        session.execute(
                            select(CognateSetMember).where(
                                CognateSetMember.cognate_set_id == cs.id
                            )
                        ).scalars()
                    )
                    for member in members:
                        writer.writerow([f"cog_{cs.id}_{member.lexeme_id}", member.lexeme_id, cs.id])

        logger.info("CSV fallback export complete: %s", output_dir)
