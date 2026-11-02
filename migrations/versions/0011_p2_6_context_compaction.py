"""Persist conversation context summaries for P2-6.

Revision ID: 0011_p2_6_context_compaction
Revises: 0010_p2_3_personal_knowledge
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0011_p2_6_context_compaction"
down_revision = "0010_p2_3_personal_knowledge"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "conversation_contexts",
        sa.Column("conversation_id", sa.String(length=128), nullable=False),
        sa.Column("tenant_id", sa.String(length=128), nullable=True),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column("summary_json", sa.JSON(), nullable=False),
        sa.Column("summary_text", sa.Text(), nullable=False),
        sa.Column("summary_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("covered_until_message_id", sa.Integer(), nullable=True),
        sa.Column("estimated_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("source_model", sa.String(length=128), nullable=False, server_default="deterministic"),
        sa.Column("content_hash", sa.String(length=128), nullable=False, server_default=""),
        sa.Column("fallback", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("conversation_id"),
    )
    op.create_index("ix_conversation_contexts_tenant_id", "conversation_contexts", ["tenant_id"], unique=False)
    op.create_index("ix_conversation_contexts_user_id", "conversation_contexts", ["user_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_conversation_contexts_user_id", table_name="conversation_contexts")
    op.drop_index("ix_conversation_contexts_tenant_id", table_name="conversation_contexts")
    op.drop_table("conversation_contexts")
