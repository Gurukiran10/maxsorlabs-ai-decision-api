"""Verifies the global exception handler turns an unexpected error into a
clean JSON response instead of leaking a raw traceback to the client."""

import os

os.environ.setdefault("GEMINI_API_KEY", "test-dummy-key")
os.environ.setdefault("JWT_SECRET", "test-dummy-secret")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src import api as api_module  # noqa: E402


def test_unhandled_exception_returns_clean_json(monkeypatch):
    @api_module.app.get("/__boom")
    def boom():
        raise RuntimeError("something genuinely unexpected")

    # raise_server_exceptions=False is required here: by default TestClient
    # re-raises exceptions that escape a route so you see them in your test
    # run, which would defeat the point of testing this handler.
    with TestClient(api_module.app, raise_server_exceptions=False) as client:
        resp = client.get("/__boom")

    assert resp.status_code == 500
    body = resp.json()
    assert body["detail"] == "Internal server error."
    assert "error_id" in body and len(body["error_id"]) == 12
    # The raw exception message must never reach the client.
    assert "something genuinely unexpected" not in resp.text
