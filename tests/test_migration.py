"""Verifies the bare-bones auto-migration in src/database.py: init_db()
must add a column that exists on the model but not yet on an
already-created table, instead of erroring or silently doing nothing."""

import os

os.environ.setdefault("GEMINI_API_KEY", "test-dummy-key")
os.environ.setdefault("JWT_SECRET", "test-dummy-secret")
os.environ["GROQ_API_KEY"] = ""

from sqlalchemy import create_engine, inspect, text  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from src import database as db_module  # noqa: E402


def test_init_db_adds_missing_column_to_existing_table(tmp_path, monkeypatch):
    db_path = tmp_path / "old_schema.db"
    engine = create_engine(f"sqlite:///{db_path}")

    # Simulate a database created before the `provider` column existed:
    # create every table except with `decisions.provider` left out.
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT, "
            "password_hash TEXT, created_at TEXT)"
        ))
        conn.execute(text(
            "CREATE TABLE tickets (id INTEGER PRIMARY KEY, user_id INTEGER, "
            "message TEXT, order_value_inr REAL, days_since_delivery INTEGER, "
            "days_since_dispatch INTEGER, product_type TEXT, opened_status TEXT, "
            "order_status TEXT, created_at TEXT)"
        ))
        conn.execute(text(
            "CREATE TABLE decisions (id INTEGER PRIMARY KEY, ticket_id INTEGER, "
            "action TEXT, reason TEXT, confidence REAL, sources TEXT, created_at TEXT)"
        ))
    engine.dispose()

    monkeypatch.setattr(db_module, "engine", create_engine(f"sqlite:///{db_path}"))
    monkeypatch.setattr(
        db_module,
        "SessionLocal",
        sessionmaker(autocommit=False, autoflush=False, bind=db_module.engine),
    )

    inspector_before = inspect(db_module.engine)
    assert "provider" not in {c["name"] for c in inspector_before.get_columns("decisions")}

    db_module.init_db()

    inspector_after = inspect(db_module.engine)
    columns_after = {c["name"] for c in inspector_after.get_columns("decisions")}
    assert "provider" in columns_after

    # The table must still be usable after the migration - insert and read back.
    with db_module.engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO decisions (ticket_id, action, reason, confidence, sources, "
            "provider, created_at) VALUES (1, 'APPROVE_RETURN', 'x', 0.9, '[]', "
            "'groq', '2025-01-01')"
        ))
        row = conn.execute(text("SELECT provider FROM decisions WHERE ticket_id = 1")).fetchone()
    assert row[0] == "groq"
