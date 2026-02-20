"""SQLAlchemy 2.0 ORM models for the cognate pipeline database.

8 tables:
  - language
  - source
  - lexeme
  - name_entity
  - name_form
  - cognate_link
  - cognate_set
  - cognate_set_member
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from geoalchemy2 import Geometry


class Base(DeclarativeBase):
    pass


class Language(Base):
    __tablename__ = "language"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    glottocode: Mapped[str] = mapped_column(String(8), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(256), nullable=False, default="")
    iso639_3: Mapped[str | None] = mapped_column(String(3), nullable=True)
    family_glottocode: Mapped[str | None] = mapped_column(String(8), nullable=True)
    classification_path = mapped_column(ARRAY(Text), nullable=True)
    location = mapped_column(Geometry("POINT", srid=4326), nullable=True)
    metadata_ = mapped_column("metadata", JSONB, nullable=True)

    lexemes: Mapped[list[Lexeme]] = relationship(back_populates="language")
    name_forms: Mapped[list[NameForm]] = relationship(back_populates="language")


class Source(Base):
    __tablename__ = "source"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_name: Mapped[str] = mapped_column(String(256), unique=True, nullable=False)
    license: Mapped[str] = mapped_column(String(128), nullable=False, default="unknown")
    license_url: Mapped[str] = mapped_column(Text, nullable=False, default="")
    citation_bibtex: Mapped[str] = mapped_column(Text, nullable=False, default="")
    retrieved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    lexemes: Mapped[list[Lexeme]] = relationship(back_populates="source")
    name_forms: Mapped[list[NameForm]] = relationship(back_populates="source")
    cognate_links: Mapped[list[CognateLink]] = relationship(back_populates="source")


class Lexeme(Base):
    __tablename__ = "lexeme"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    external_id: Mapped[str] = mapped_column(String(256), nullable=False, default="")
    language_id: Mapped[int] = mapped_column(ForeignKey("language.id"), nullable=False)
    source_id: Mapped[int] = mapped_column(ForeignKey("source.id"), nullable=False)
    concept_id: Mapped[str] = mapped_column(String(256), nullable=False, default="")
    lemma: Mapped[str] = mapped_column(String(512), nullable=False)
    orthography: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    phonetic_raw: Mapped[str] = mapped_column(Text, nullable=False, default="")
    phonetic_canonical: Mapped[str] = mapped_column(Text, nullable=False, default="")
    transcription_type: Mapped[str] = mapped_column(
        String(32), nullable=False, default="unknown"
    )
    sound_class: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    provenance = mapped_column(JSONB, nullable=True)

    language: Mapped[Language] = relationship(back_populates="lexemes")
    source: Mapped[Source] = relationship(back_populates="lexemes")

    __table_args__ = (
        Index("ix_lexeme_concept", "concept_id"),
        Index("ix_lexeme_language", "language_id"),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_lexeme_confidence"),
    )


class NameEntity(Base):
    __tablename__ = "name_entity"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False)
    external_ids = mapped_column(JSONB, nullable=True)
    location = mapped_column(Geometry("POINT", srid=4326), nullable=True)

    name_forms: Mapped[list[NameForm]] = relationship(back_populates="name_entity")

    __table_args__ = (
        CheckConstraint(
            "entity_type IN ('place', 'person', 'deity', 'ethnonym', 'other')",
            name="ck_entity_type",
        ),
    )


class NameForm(Base):
    __tablename__ = "name_form"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name_entity_id: Mapped[int] = mapped_column(ForeignKey("name_entity.id"), nullable=False)
    language_id: Mapped[int] = mapped_column(ForeignKey("language.id"), nullable=False)
    source_id: Mapped[int] = mapped_column(ForeignKey("source.id"), nullable=False)
    name_string: Mapped[str] = mapped_column(String(512), nullable=False)
    ipa_raw: Mapped[str] = mapped_column(Text, nullable=False, default="")
    ipa_canonical: Mapped[str] = mapped_column(Text, nullable=False, default="")
    provenance = mapped_column(JSONB, nullable=True)

    name_entity: Mapped[NameEntity] = relationship(back_populates="name_forms")
    language: Mapped[Language] = relationship(back_populates="name_forms")
    source: Mapped[Source] = relationship(back_populates="name_forms")


class CognateLink(Base):
    __tablename__ = "cognate_link"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lexeme_id_a: Mapped[int] = mapped_column(ForeignKey("lexeme.id"), nullable=False)
    lexeme_id_b: Mapped[int] = mapped_column(ForeignKey("lexeme.id"), nullable=False)
    source_id: Mapped[int] = mapped_column(ForeignKey("source.id"), nullable=False)
    concept_id: Mapped[str] = mapped_column(String(256), nullable=False, default="")
    relationship_type: Mapped[str] = mapped_column(String(64), nullable=False, default="cognate_candidate")
    score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    method: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    threshold_used: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    evidence = mapped_column(JSONB, nullable=True)

    source: Mapped[Source] = relationship(back_populates="cognate_links")

    __table_args__ = (
        CheckConstraint("lexeme_id_a < lexeme_id_b", name="ck_link_ordering"),
        CheckConstraint(
            "relationship_type IN ('cognate_inherited', 'similarity_only', 'cognate_candidate', 'borrowing')",
            name="ck_relationship_type",
        ),
        UniqueConstraint("lexeme_id_a", "lexeme_id_b", "method", name="uq_link_pair_method"),
        Index("ix_cognate_link_concept", "concept_id"),
    )


class CognateSetTable(Base):
    __tablename__ = "cognate_set"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    external_id: Mapped[str] = mapped_column(String(256), nullable=False, default="")
    concept_id: Mapped[str] = mapped_column(String(256), nullable=False, default="")
    method: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    quality = mapped_column(JSONB, nullable=True)

    members: Mapped[list[CognateSetMember]] = relationship(back_populates="cognate_set")

    __table_args__ = (
        Index("ix_cognate_set_concept", "concept_id"),
    )


class CognateSetMember(Base):
    __tablename__ = "cognate_set_member"

    cognate_set_id: Mapped[int] = mapped_column(
        ForeignKey("cognate_set.id"), primary_key=True
    )
    lexeme_id: Mapped[int] = mapped_column(
        ForeignKey("lexeme.id"), primary_key=True
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="member")

    cognate_set: Mapped[CognateSetTable] = relationship(back_populates="members")
