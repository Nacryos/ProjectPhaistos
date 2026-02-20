"""Initial schema with all 8 tables.

Revision ID: 001
Create Date: 2024-01-01 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Language table
    op.create_table(
        "language",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("glottocode", sa.String(8), nullable=False),
        sa.Column("name", sa.String(256), nullable=False, server_default=""),
        sa.Column("iso639_3", sa.String(3), nullable=True),
        sa.Column("family_glottocode", sa.String(8), nullable=True),
        sa.Column("classification_path", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("glottocode"),
    )
    op.execute("SELECT AddGeometryColumn('language', 'location', 4326, 'POINT', 2)")

    # Source table
    op.create_table(
        "source",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("source_name", sa.String(256), nullable=False),
        sa.Column("license", sa.String(128), nullable=False, server_default="unknown"),
        sa.Column("license_url", sa.Text(), nullable=False, server_default=""),
        sa.Column("citation_bibtex", sa.Text(), nullable=False, server_default=""),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source_name"),
    )

    # Lexeme table
    op.create_table(
        "lexeme",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("external_id", sa.String(256), nullable=False, server_default=""),
        sa.Column("language_id", sa.Integer(), nullable=False),
        sa.Column("source_id", sa.Integer(), nullable=False),
        sa.Column("concept_id", sa.String(256), nullable=False, server_default=""),
        sa.Column("lemma", sa.String(512), nullable=False),
        sa.Column("orthography", sa.String(512), nullable=False, server_default=""),
        sa.Column("ipa_raw", sa.Text(), nullable=False, server_default=""),
        sa.Column("ipa_canonical", sa.Text(), nullable=False, server_default=""),
        sa.Column("sound_class", sa.String(512), nullable=False, server_default=""),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("provenance", postgresql.JSONB(), nullable=True),
        sa.ForeignKeyConstraint(["language_id"], ["language.id"]),
        sa.ForeignKeyConstraint(["source_id"], ["source.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_lexeme_confidence"),
    )
    op.create_index("ix_lexeme_concept", "lexeme", ["concept_id"])
    op.create_index("ix_lexeme_language", "lexeme", ["language_id"])

    # Name entity table
    op.create_table(
        "name_entity",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("entity_type", sa.String(64), nullable=False),
        sa.Column("external_ids", postgresql.JSONB(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "entity_type IN ('place', 'person', 'deity', 'ethnonym', 'other')",
            name="ck_entity_type",
        ),
    )
    op.execute("SELECT AddGeometryColumn('name_entity', 'location', 4326, 'POINT', 2)")

    # Name form table
    op.create_table(
        "name_form",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name_entity_id", sa.Integer(), nullable=False),
        sa.Column("language_id", sa.Integer(), nullable=False),
        sa.Column("source_id", sa.Integer(), nullable=False),
        sa.Column("name_string", sa.String(512), nullable=False),
        sa.Column("ipa_raw", sa.Text(), nullable=False, server_default=""),
        sa.Column("ipa_canonical", sa.Text(), nullable=False, server_default=""),
        sa.Column("provenance", postgresql.JSONB(), nullable=True),
        sa.ForeignKeyConstraint(["name_entity_id"], ["name_entity.id"]),
        sa.ForeignKeyConstraint(["language_id"], ["language.id"]),
        sa.ForeignKeyConstraint(["source_id"], ["source.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    # Cognate link table
    op.create_table(
        "cognate_link",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("lexeme_id_a", sa.Integer(), nullable=False),
        sa.Column("lexeme_id_b", sa.Integer(), nullable=False),
        sa.Column("source_id", sa.Integer(), nullable=False),
        sa.Column("concept_id", sa.String(256), nullable=False, server_default=""),
        sa.Column("relationship_type", sa.String(64), nullable=False, server_default="cognate_candidate"),
        sa.Column("score", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("method", sa.String(64), nullable=False, server_default=""),
        sa.Column("threshold_used", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("evidence", postgresql.JSONB(), nullable=True),
        sa.ForeignKeyConstraint(["lexeme_id_a"], ["lexeme.id"]),
        sa.ForeignKeyConstraint(["lexeme_id_b"], ["lexeme.id"]),
        sa.ForeignKeyConstraint(["source_id"], ["source.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint("lexeme_id_a < lexeme_id_b", name="ck_link_ordering"),
        sa.CheckConstraint(
            "relationship_type IN ('cognate_inherited', 'similarity_only', 'cognate_candidate', 'borrowing')",
            name="ck_relationship_type",
        ),
        sa.UniqueConstraint("lexeme_id_a", "lexeme_id_b", "method", name="uq_link_pair_method"),
    )
    op.create_index("ix_cognate_link_concept", "cognate_link", ["concept_id"])

    # Cognate set table
    op.create_table(
        "cognate_set",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("external_id", sa.String(256), nullable=False, server_default=""),
        sa.Column("concept_id", sa.String(256), nullable=False, server_default=""),
        sa.Column("method", sa.String(64), nullable=False, server_default=""),
        sa.Column("quality", postgresql.JSONB(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_cognate_set_concept", "cognate_set", ["concept_id"])

    # Cognate set member table
    op.create_table(
        "cognate_set_member",
        sa.Column("cognate_set_id", sa.Integer(), nullable=False),
        sa.Column("lexeme_id", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(32), nullable=False, server_default="member"),
        sa.ForeignKeyConstraint(["cognate_set_id"], ["cognate_set.id"]),
        sa.ForeignKeyConstraint(["lexeme_id"], ["lexeme.id"]),
        sa.PrimaryKeyConstraint("cognate_set_id", "lexeme_id"),
    )


def downgrade() -> None:
    op.drop_table("cognate_set_member")
    op.drop_table("cognate_set")
    op.drop_table("cognate_link")
    op.drop_table("name_form")
    op.drop_table("name_entity")
    op.drop_table("lexeme")
    op.drop_table("source")
    op.drop_table("language")
