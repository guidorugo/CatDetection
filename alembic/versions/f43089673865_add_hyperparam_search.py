"""add hyperparam search

Revision ID: f43089673865
Revises: f98fa953c41b
Create Date: 2026-03-01 00:38:10.656963

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f43089673865'
down_revision: Union[str, None] = 'f98fa953c41b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('hyperparam_searches',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('status', sa.String(length=20), nullable=False),
    sa.Column('param_grid', sa.Text(), nullable=False),
    sa.Column('training_location', sa.String(length=20), nullable=False),
    sa.Column('base_config', sa.Text(), nullable=True),
    sa.Column('total_trials', sa.Integer(), nullable=False),
    sa.Column('completed_trials', sa.Integer(), nullable=False),
    sa.Column('failed_trials', sa.Integer(), nullable=False),
    sa.Column('best_trial_id', sa.Integer(), nullable=True),
    sa.Column('best_metric', sa.Float(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    # SQLite doesn't support ALTER ADD CONSTRAINT, so add columns without FK
    # The FK relationship is enforced at the ORM level
    op.add_column('training_jobs', sa.Column('search_id', sa.Integer(), nullable=True))
    op.add_column('training_jobs', sa.Column('trial_number', sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column('training_jobs', 'trial_number')
    op.drop_column('training_jobs', 'search_id')
    op.drop_table('hyperparam_searches')
