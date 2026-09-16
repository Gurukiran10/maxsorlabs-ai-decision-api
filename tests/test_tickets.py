from src import api as api_module
from src.rate_limit import MAX_REQUESTS_PER_WINDOW
from tests.conftest import register_and_login


def create_ticket(client, token, message="My order arrived damaged."):
    return client.post(
        "/tickets",
        json={"message": message, "order_value_inr": 1500, "product_type": "non_food"},
        headers={"Authorization": f"Bearer {token}"},
    )


def test_create_ticket_persists_decision(client):
    token = register_and_login(client, "alice@example.com")
    resp = create_ticket(client, token)
    assert resp.status_code == 201
    body = resp.json()
    assert body["decision"] is not None
    assert body["decision"]["action"] == "NEEDS_MORE_INFORMATION"
    assert 0.0 <= body["decision"]["confidence"] <= 1.0
    assert isinstance(body["decision"]["sources"], list)


def test_list_tickets_returns_only_own_tickets(client):
    alice_token = register_and_login(client, "alice@example.com")
    bob_token = register_and_login(client, "bob@example.com")

    create_ticket(client, alice_token, "Alice's ticket")
    create_ticket(client, bob_token, "Bob's ticket")

    resp = client.get("/tickets", headers={"Authorization": f"Bearer {alice_token}"})
    assert resp.status_code == 200
    messages = [t["message"] for t in resp.json()]
    assert messages == ["Alice's ticket"]


def test_user_cannot_read_another_users_ticket(client):
    """Authorization check: Alice's token must not be able to fetch Bob's ticket."""
    alice_token = register_and_login(client, "alice@example.com")
    bob_token = register_and_login(client, "bob@example.com")

    bob_ticket = create_ticket(client, bob_token, "Bob's private ticket").json()
    bob_ticket_id = bob_ticket["id"]

    resp = client.get(
        f"/tickets/{bob_ticket_id}", headers={"Authorization": f"Bearer {alice_token}"}
    )

    assert resp.status_code == 404  # not 200, and not leaking existence via 403

    # Bob himself can still read it.
    own_resp = client.get(
        f"/tickets/{bob_ticket_id}", headers={"Authorization": f"Bearer {bob_token}"}
    )
    assert own_resp.status_code == 200
    assert own_resp.json()["message"] == "Bob's private ticket"


def test_get_ticket_requires_auth(client):
    token = register_and_login(client, "alice@example.com")
    ticket_id = create_ticket(client, token).json()["id"]
    resp = client.get(f"/tickets/{ticket_id}")
    assert resp.status_code in (401, 403)


def test_invalid_product_type_rejected(client):
    token = register_and_login(client, "alice@example.com")
    resp = client.post(
        "/tickets",
        json={"message": "hi", "product_type": "banana"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 422


def test_negative_order_value_rejected(client):
    token = register_and_login(client, "alice@example.com")
    resp = client.post(
        "/tickets",
        json={"message": "hi", "order_value_inr": -100},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 422


def test_product_type_is_case_insensitive(client):
    token = register_and_login(client, "alice@example.com")
    resp = client.post(
        "/tickets",
        json={"message": "hi", "product_type": "NON_FOOD"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 201
    assert resp.json()["product_type"] == "non_food"


def test_rate_limit_enforced_per_user(client):
    token = register_and_login(client, "alice@example.com")

    for _ in range(MAX_REQUESTS_PER_WINDOW):
        resp = create_ticket(client, token)
        assert resp.status_code == 201

    over_limit_resp = create_ticket(client, token)
    assert over_limit_resp.status_code == 429


def test_rate_limit_is_per_user_not_global(client):
    alice_token = register_and_login(client, "alice@example.com")
    bob_token = register_and_login(client, "bob@example.com")

    for _ in range(MAX_REQUESTS_PER_WINDOW):
        assert create_ticket(client, alice_token).status_code == 201

    assert create_ticket(client, alice_token).status_code == 429
    # Bob has his own quota and is unaffected by Alice's usage.
    assert create_ticket(client, bob_token).status_code == 201


def test_failed_decision_does_not_persist_orphan_ticket(client, monkeypatch):
    """If the decision pipeline fails, no ticket row should be left behind
    without a decision attached."""
    token = register_and_login(client, "alice@example.com")

    def failing_make_decision(ticket: dict):
        raise RuntimeError("simulated LLM/embedding outage")

    monkeypatch.setattr(api_module, "make_decision", failing_make_decision)

    resp = create_ticket(client, token)
    assert resp.status_code == 502

    list_resp = client.get("/tickets", headers={"Authorization": f"Bearer {token}"})
    assert list_resp.json() == []
