"""Add structured, user-confirmable long-term memory items."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0013_structured_memory"
down_revision = "0012_context_budget_metadata"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "memory_items",
        sa.Column("id", sa.String(length=128), nullable=False),
        sa.Column("tenant_id", sa.String(length=128), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column("memory_type", sa.String(length=32), nullable=False),
        sa.Column("memory_key", sa.String(length=128), nullable=False),
        sa.Column("memory_value", sa.JSON(), nullable=False),
        sa.Column("display_text", sa.String(length=512), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="proposed"),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0"),
        sa.Column("sensitivity", sa.String(length=32), nullable=False, server_default="low"),
        sa.Column("source_conversation_id", sa.String(length=128), nullable=True),
        sa.Column("source_message_id", sa.Integer(), nullable=True),
        sa.Column("source_excerpt", sa.Text(), nullable=False, server_default=""),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("superseded_by", sa.String(length=128), nullable=True),
        sa.Column("content_hash", sa.String(length=128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    for name, columns in {
        "ix_memory_items_tenant_id": ["tenant_id"],
        "ix_memory_items_user_id": ["user_id"],
        "ix_memory_items_memory_type": ["memory_type"],
        "ix_memory_items_memory_key": ["memory_key"],
        "ix_memory_items_status": ["status"],
        "ix_memory_items_source_conversation_id": ["source_conversation_id"],
        "ix_memory_items_content_hash": ["content_hash"],
        "ix_memory_items_expires_at": ["expires_at"],
    }.items():
        op.create_index(name, "memory_items", columns)


def downgrade() -> None:
    for name in (
        "ix_memory_items_expires_at",
        "ix_memory_items_content_hash",
        "ix_memory_items_source_conversation_id",
        "ix_memory_items_status",
        "ix_memory_items_memory_key",
        "ix_memory_items_memory_type",
        "ix_memory_items_user_id",
        "ix_memory_items_tenant_id",
    ):
        op.drop_index(name, table_name="memory_items")
    op.drop_table("memory_items")
