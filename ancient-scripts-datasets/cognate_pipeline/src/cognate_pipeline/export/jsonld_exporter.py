"""Export cognate links as JSON-LD."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import orjson
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session, sessionmaker

from cognate_pipeline.config.schema import ExportConfig
from cognate_pipeline.db.schema import (
    CognateLink,
    CognateSetMember,
    CognateSetTable,
    Language,
    Lexeme,
)

logger = logging.getLogger(__name__)

_CONTEXT = {
    "@context": {
        "ontolex": "http://www.w3.org/ns/lemon/ontolex#",
        "lexinfo": "http://www.lexinfo.net/ontology/3.0/lexinfo#",
        "glottolog": "https://glottolog.org/resource/languoid/id/",
        "concepticon": "https://concepticon.clld.org/parameters/",
        "cognate": "http://example.org/cognate#",
        "lexeme": "cognate:lexeme",
        "form": "ontolex:writtenRep",
        "language": "lexinfo:language",
        "glottocode": "glottolog:",
        "concept": "concepticon:",
        "ipa": "ontolex:phoneticRep",
        "soundClass": "cognate:soundClass",
        "cognateLink": "cognate:CognateLink",
        "score": "cognate:score",
        "method": "cognate:method",
        "evidence": "cognate:evidence",
    }
}


class JsonLdExporter:
    """Exports cognate links as JSON-LD."""

    def __init__(self, engine: Engine, config: ExportConfig) -> None:
        self.engine = engine
        self.config = config
        self._session_factory = sessionmaker(bind=engine)

    def export(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        with self._session_factory() as session:
            # Build lexeme nodes
            lexeme_nodes = self._build_lexeme_nodes(session)

            # Build cognate link edges
            link_edges = self._build_link_edges(session)

            # Build cognate sets
            set_nodes = self._build_set_nodes(session)

        doc: dict[str, Any] = {
            **_CONTEXT,
            "@graph": {
                "lexemes": lexeme_nodes,
                "cognateLinks": link_edges,
                "cognateSets": set_nodes,
            },
        }

        out_path = output_dir / "cognates.jsonld"
        out_path.write_bytes(orjson.dumps(doc, option=orjson.OPT_INDENT_2))
        logger.info("JSON-LD export written to %s", out_path)

    def _build_lexeme_nodes(self, session: Session) -> list[dict]:
        nodes = []
        for lex in session.execute(select(Lexeme)).scalars():
            lang = session.get(Language, lex.language_id)
            node: dict[str, Any] = {
                "@id": f"lexeme:{lex.id}",
                "@type": "lexeme",
                "externalId": lex.external_id,
                "form": lex.lemma,
                "language": lang.glottocode if lang else "",
                "concept": lex.concept_id,
                "phoneticCanonical": lex.phonetic_canonical,
                "transcriptionType": lex.transcription_type,
                "soundClass": lex.sound_class,
                "confidence": lex.confidence,
            }
            if self.config.include_provenance and lex.provenance:
                node["provenance"] = lex.provenance
            nodes.append(node)
        return nodes

    def _build_link_edges(self, session: Session) -> list[dict]:
        edges = []
        for link in session.execute(select(CognateLink)).scalars():
            edge: dict[str, Any] = {
                "@type": "cognateLink",
                "lexemeA": f"lexeme:{link.lexeme_id_a}",
                "lexemeB": f"lexeme:{link.lexeme_id_b}",
                "concept": link.concept_id,
                "relationshipType": link.relationship_type,
                "score": link.score,
                "method": link.method,
            }
            if link.evidence:
                edge["evidence"] = link.evidence
            edges.append(edge)
        return edges

    def _build_set_nodes(self, session: Session) -> list[dict]:
        sets = []
        for cs in session.execute(select(CognateSetTable)).scalars():
            members = list(
                session.execute(
                    select(CognateSetMember).where(
                        CognateSetMember.cognate_set_id == cs.id
                    )
                ).scalars()
            )
            sets.append({
                "@id": f"cognateSet:{cs.id}",
                "concept": cs.concept_id,
                "method": cs.method,
                "members": [
                    {"lexeme": f"lexeme:{m.lexeme_id}", "role": m.role}
                    for m in members
                ],
                "quality": cs.quality,
            })
        return sets
