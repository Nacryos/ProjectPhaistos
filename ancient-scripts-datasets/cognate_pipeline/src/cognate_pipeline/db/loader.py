"""Batch data loader for PostgreSQL."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import Engine, select, text
from sqlalchemy.orm import Session, sessionmaker

from cognate_pipeline.db.schema import (
    Base,
    CognateLink as CognateLinkTable,
    CognateSetMember,
    CognateSetTable,
    Language,
    Lexeme,
    Source,
)
from cognate_pipeline.utils.batching import batched

logger = logging.getLogger(__name__)


class BatchLoader:
    """Loads staged data into PostgreSQL in batches."""

    def __init__(self, engine: Engine, batch_size: int = 5000) -> None:
        self.engine = engine
        self.batch_size = batch_size
        self._session_factory = sessionmaker(bind=engine)
        # Ensure tables exist
        Base.metadata.create_all(engine)

    def _session(self) -> Session:
        return self._session_factory()

    def load_sources(self, registry: dict[str, Any]) -> None:
        """Load source metadata from license registry dict."""
        with self._session() as session:
            for source_data in registry.get("sources", {}).values():
                existing = session.execute(
                    select(Source).where(Source.source_name == source_data["source_name"])
                ).scalar_one_or_none()
                if existing is None:
                    session.add(Source(
                        source_name=source_data["source_name"],
                        license=source_data.get("license", "unknown"),
                        license_url=source_data.get("license_url", ""),
                        citation_bibtex=source_data.get("citation_bibtex", ""),
                    ))
            session.commit()

    def load_lexemes(self, records: list[dict[str, Any]]) -> None:
        """Load normalised lexeme records into the database."""
        with self._session() as session:
            # Pre-load language and source caches
            lang_cache: dict[str, int] = {}
            source_cache: dict[str, int] = {}

            for batch in batched(records, self.batch_size):
                for rec in batch:
                    # Ensure language exists
                    lang_id_str = rec.get("language_id", "")
                    glottocode = rec.get("glottocode", "") or lang_id_str
                    if glottocode not in lang_cache:
                        lang = session.execute(
                            select(Language).where(Language.glottocode == glottocode)
                        ).scalar_one_or_none()
                        if lang is None:
                            lang = Language(glottocode=glottocode, name=lang_id_str)
                            session.add(lang)
                            session.flush()
                        lang_cache[glottocode] = lang.id

                    # Ensure source exists
                    source_name = rec.get("source_name", "unknown")
                    if source_name not in source_cache:
                        src = session.execute(
                            select(Source).where(Source.source_name == source_name)
                        ).scalar_one_or_none()
                        if src is None:
                            src = Source(source_name=source_name)
                            session.add(src)
                            session.flush()
                        source_cache[source_name] = src.id

                    session.add(Lexeme(
                        external_id=rec.get("id", ""),
                        language_id=lang_cache[glottocode],
                        source_id=source_cache[source_name],
                        concept_id=rec.get("concept_id", ""),
                        lemma=rec.get("form", ""),
                        orthography=rec.get("form", ""),
                        phonetic_raw=rec.get("phonetic_raw", rec.get("ipa_raw", "")),
                        phonetic_canonical=rec.get("phonetic_canonical", rec.get("ipa_canonical", "")),
                        transcription_type=rec.get("transcription_type", "unknown"),
                        sound_class=rec.get("sound_class", ""),
                        confidence=rec.get("confidence", 1.0),
                        provenance=rec.get("provenance"),
                    ))
                session.commit()
                logger.debug("Committed batch of %d lexemes", len(batch))

    def load_cognate_links(self, links: list[dict[str, Any]]) -> None:
        """Load cognate link records.

        Note: This requires lexemes to be loaded first, as it needs to
        resolve external IDs to database IDs.
        """
        with self._session() as session:
            # Build external_id -> db_id mapping
            id_map: dict[str, int] = {}
            for lex in session.execute(select(Lexeme)).scalars():
                id_map[lex.external_id] = lex.id

            # Get default source
            default_source = session.execute(select(Source)).scalars().first()
            source_id = default_source.id if default_source else 1

            for batch in batched(links, self.batch_size):
                for link_data in batch:
                    db_id_a = id_map.get(link_data["lexeme_id_a"])
                    db_id_b = id_map.get(link_data["lexeme_id_b"])
                    if db_id_a is None or db_id_b is None:
                        continue
                    # Ensure ordering
                    if db_id_a > db_id_b:
                        db_id_a, db_id_b = db_id_b, db_id_a
                    session.add(CognateLinkTable(
                        lexeme_id_a=db_id_a,
                        lexeme_id_b=db_id_b,
                        source_id=source_id,
                        concept_id=link_data.get("concept_id", ""),
                        relationship_type=link_data.get("relationship_type", "cognate_candidate"),
                        score=link_data.get("score", 0.0),
                        method=link_data.get("method", ""),
                        threshold_used=link_data.get("threshold_used", 0.0),
                        evidence=link_data.get("evidence"),
                    ))
                session.commit()

    def load_cognate_sets(self, sets_data: list[dict[str, Any]]) -> None:
        """Load cognate set records."""
        with self._session() as session:
            # Build external_id -> db_id mapping
            id_map: dict[str, int] = {}
            for lex in session.execute(select(Lexeme)).scalars():
                id_map[lex.external_id] = lex.id

            for batch in batched(sets_data, self.batch_size):
                for set_data in batch:
                    cs = CognateSetTable(
                        external_id=set_data.get("id", ""),
                        concept_id=set_data.get("concept_id", ""),
                        method=set_data.get("method", ""),
                        quality=set_data.get("quality"),
                    )
                    session.add(cs)
                    session.flush()

                    for member_data in set_data.get("members", []):
                        db_id = id_map.get(member_data["lexeme_id"])
                        if db_id is None:
                            continue
                        session.add(CognateSetMember(
                            cognate_set_id=cs.id,
                            lexeme_id=db_id,
                            role=member_data.get("role", "member"),
                        ))
                session.commit()
