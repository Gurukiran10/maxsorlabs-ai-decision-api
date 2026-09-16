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
