"""Add conservative context-budget metadata to persisted summaries."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0012_context_budget_metadata"
down_revision = "0011_p2_6_context_compaction"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "conversation_contexts",
        sa.Column("estimated_tokens_before", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "conversation_contexts",
        sa.Column("estimated_tokens_after", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "conversation_contexts",
        sa.Column("compression_ratio", sa.Float(), nullable=False, server_default="0"),
    )
    op.add_column(
        "conversation_contexts",
        sa.Column("compaction_reason", sa.String(length=64), nullable=False, server_default="none"),
    )
    op.add_column(
        "conversation_contexts",
        sa.Column("compaction_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "conversation_contexts",
        sa.Column("overflow_prevented", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "conversation_contexts",
        sa.Column("estimation_mode", sa.String(length=64), nullable=False, server_default="conservative_estimate"),
    )


def downgrade() -> None:
    op.drop_column("conversation_contexts", "estimation_mode")
    op.drop_column("conversation_contexts", "overflow_prevented")
    op.drop_column("conversation_contexts", "compaction_count")
    op.drop_column("conversation_contexts", "compaction_reason")
    op.drop_column("conversation_contexts", "compression_ratio")
    op.drop_column("conversation_contexts", "estimated_tokens_after")
    op.drop_column("conversation_contexts", "estimated_tokens_before")
