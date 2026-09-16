import os

# Config validation requires these to be set and non-placeholder; tests never
# make real Gemini calls (make_decision is stubbed below), so dummy values
# are fine. Must be set before any `src.*` module is imported, since
# src.config validates them at import time.
os.environ.setdefault("GEMINI_API_KEY", "test-dummy-key")
os.environ.setdefault("JWT_SECRET", "test-dummy-secret")

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src import api as api_module
from src.database import Base, get_db
from src.rate_limit import reset_rate_limits
from src.schemas import LLMDecision


@pytest.fixture()
def client(monkeypatch):
    # User ids restart from 1 in each test's fresh in-memory database, but
    # the rate limiter's state is a module-level global keyed by user id, so
    # it must be reset between tests to avoid cross-test bleed.
    reset_rate_limits()

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    api_module.app.dependency_overrides[get_db] = override_get_db

    # The decision pipeline calls the real Gemini API; tests should not
    # depend on network access or a valid API key, so it is stubbed here.
    def fake_make_decision(ticket: dict):
        decision = LLMDecision(
            action="NEEDS_MORE_INFORMATION",
            confidence=0.75,
            reason="Stubbed decision for testing.",
            sources=["returns.md"],
        )
        return decision, []

    monkeypatch.setattr(api_module, "make_decision", fake_make_decision)

    with TestClient(api_module.app) as test_client:
        yield test_client

    api_module.app.dependency_overrides.clear()


def register_and_login(client: TestClient, email: str, password: str = "password123") -> str:
    client.post("/register", json={"email": email, "password": password})
    resp = client.post("/login", json={"email": email, "password": password})
    return resp.json()["access_token"]
