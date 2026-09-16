from tests.conftest import register_and_login


def test_register_creates_user(client):
    resp = client.post("/register", json={"email": "alice@example.com", "password": "password123"})
    assert resp.status_code == 201
    assert resp.json()["email"] == "alice@example.com"
    assert "password" not in resp.json()
    assert "password_hash" not in resp.json()


def test_register_duplicate_email_rejected(client):
    client.post("/register", json={"email": "alice@example.com", "password": "password123"})
    resp = client.post("/register", json={"email": "alice@example.com", "password": "password123"})
    assert resp.status_code == 409


def test_login_wrong_password_rejected(client):
    client.post("/register", json={"email": "alice@example.com", "password": "password123"})
    resp = client.post("/login", json={"email": "alice@example.com", "password": "wrongpass"})
    assert resp.status_code == 401


def test_login_returns_jwt(client):
    client.post("/register", json={"email": "alice@example.com", "password": "password123"})
    resp = client.post("/login", json={"email": "alice@example.com", "password": "password123"})
    assert resp.status_code == 200
    assert resp.json()["token_type"] == "bearer"
    assert len(resp.json()["access_token"]) > 20


def test_me_requires_token(client):
    resp = client.get("/me")
    assert resp.status_code in (401, 403)


def test_me_returns_current_user(client):
    token = register_and_login(client, "alice@example.com")
    resp = client.get("/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["email"] == "alice@example.com"


def test_invalid_token_rejected(client):
    resp = client.get("/me", headers={"Authorization": "Bearer not-a-real-token"})
    assert resp.status_code == 401


def test_register_rejects_password_with_no_number(client):
    resp = client.post(
        "/register", json={"email": "alice@example.com", "password": "alletters"}
    )
    assert resp.status_code == 422


def test_register_rejects_password_with_no_letter(client):
    resp = client.post(
        "/register", json={"email": "alice@example.com", "password": "12345678"}
    )
    assert resp.status_code == 422


def test_register_rejects_too_short_password(client):
    resp = client.post("/register", json={"email": "alice@example.com", "password": "ab1"})
    assert resp.status_code == 422


def test_register_rejects_password_over_bcrypt_limit(client):
    resp = client.post(
        "/register", json={"email": "alice@example.com", "password": "a1" * 40}
    )
    assert resp.status_code == 422


def test_register_accepts_valid_password(client):
    resp = client.post(
        "/register", json={"email": "alice@example.com", "password": "Valid1Pass"}
    )
    assert resp.status_code == 201
