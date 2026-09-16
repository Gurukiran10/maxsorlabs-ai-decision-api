import os
import re

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def validate_password(password: str) -> str | None:
    """Mirrors src/schemas.py's UserRegister.password_complexity validator,
    so the same rule is enforced client-side (instant feedback) and
    server-side (the actual source of truth)."""
    if len(password) < 8:
        return "Password must be at least 8 characters."
    if len(password) > 72:
        return "Password must be 72 characters or fewer."
    if not re.search(r"[A-Za-z]", password):
        return "Password must contain at least one letter."
    if not re.search(r"\d", password):
        return "Password must contain at least one number."
    return None

st.set_page_config(page_title="AI Support Decision Assistant", page_icon="🎫", layout="wide")

# Streamlit shows a "Press Enter to submit form"/"Press Enter to apply" hint
# under text inputs inside a form. It's a built-in UI affordance, not
# something this app renders, and there's no public config flag to turn it
# off - hiding the element it renders into is the only way.
st.markdown(
    '<style>div[data-testid="InputInstructions"] { display: none; }</style>',
    unsafe_allow_html=True,
)

PROVIDER_LABELS = {
    "gemini": "🟢 Primary AI (Gemini)",
    "groq": "🟡 Backup AI (Groq)",
    "cache": "⚡ Instant — seen this exact ticket before",
    "retrieval_gate": "🔎 Answered automatically, no AI call needed",
    "fallback": "🔴 AI temporarily unavailable — generic response",
}

ACTION_LABELS = {
    "APPROVE_RETURN": "Return Approved",
    "REJECT_OUTSIDE_WINDOW": "Not Eligible — Outside the Return Window",
    "REJECT_OPENED_ITEM": "Not Eligible — Item Already Opened",
    "REJECT_FOOD_RETURN": "Not Eligible — Food Items Can't Be Returned",
    "APPROVE_REFUND_OR_REPLACEMENT": "Refund or Replacement Approved",
    "REQUEST_PHOTOS": "Photos Needed Before We Can Approve",
    "APPROVE_REPLACEMENT": "Replacement Approved",
    "REQUEST_DEFECT_EVIDENCE": "More Evidence Needed Before We Can Approve",
    "REPLACE_CORRECT_ITEM": "Correct Item Will Be Sent",
    "CANCEL_AND_REFUND": "Order Cancelled & Refunded",
    "CANNOT_CANCEL_AFTER_DISPATCH": "Can't Cancel — Already Shipped",
    "WAIT_AND_TRACK": "Please Wait — Still Within Normal Delivery Time",
    "OPEN_SHIPPING_INVESTIGATION": "Shipping Investigation Opened",
    "OFFER_REPLACEMENT_OR_REFUND": "Replacement or Refund Offered",
    "NEEDS_MORE_INFORMATION": "More Information Needed",
}

SOURCE_LABELS = {
    "cancellations.md": "Cancellation Policy",
    "damaged_goods.md": "Damaged Goods Policy",
    "defective_products.md": "Defective Product Policy",
    "returns.md": "Returns Policy",
    "shipping.md": "Shipping & Delivery Policy",
    "wrong_item.md": "Wrong Item Policy",
}


class BackendUnreachable(Exception):
    pass


def _request(method: str, path: str, **kwargs) -> requests.Response:
    try:
        return requests.request(method, f"{API_BASE_URL}{path}", timeout=30, **kwargs)
    except requests.exceptions.ConnectionError as exc:
        raise BackendUnreachable(
            f"Can't reach the backend at {API_BASE_URL}. Is `uvicorn src.api:app` running?"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise BackendUnreachable("The backend took too long to respond (timed out).") from exc


def api_post(path: str, json_body: dict, auth: bool = False) -> requests.Response:
    headers = {"Authorization": f"Bearer {st.session_state.token}"} if auth else {}
    return _request("POST", path, json=json_body, headers=headers)


def api_get(path: str) -> requests.Response:
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    return _request("GET", path, headers=headers)


def error_detail(resp: requests.Response, fallback: str) -> str:
    try:
        return resp.json().get("detail", fallback)
    except ValueError:
        return fallback


def backend_status() -> bool:
    try:
        resp = _request("GET", "/health")
        return resp.status_code == 200
    except BackendUnreachable:
        return False


if "token" not in st.session_state:
    st.session_state.token = None
if "email" not in st.session_state:
    st.session_state.email = None


def login_register_page():
    _, center, _ = st.columns([1, 1.4, 1])
    with center:
        card = st.container(border=True)
        card.markdown("## 🎫 Support Decision Assistant")
        card.caption("Policy-grounded AI recommendations for support tickets.")

        tab_login, tab_register = card.tabs(["Sign in", "Create account"])

        with tab_login:
            with st.form("login_form"):
                email = st.text_input("Email", key="login_email", placeholder="you@example.com")
                password = st.text_input("Password", type="password", key="login_password")
                submitted = st.form_submit_button("Sign in", use_container_width=True)
            if submitted:
                if not email or not password:
                    st.error("Please enter both email and password.")
                else:
                    try:
                        resp = api_post("/login", {"email": email, "password": password})
                    except BackendUnreachable as exc:
                        st.error(str(exc))
                    else:
                        if resp.status_code == 200:
                            st.session_state.token = resp.json()["access_token"]
                            st.session_state.email = email
                            st.rerun()
                        else:
                            st.error(error_detail(resp, "Login failed"))

        with tab_register:
            with st.form("register_form"):
                email = st.text_input(
                    "Email", key="register_email", placeholder="you@example.com"
                )
                password = st.text_input(
                    "Password (min 8 chars, at least one letter and one number)",
                    type="password",
                    key="register_password",
                )
                confirm_password = st.text_input(
                    "Confirm password", type="password", key="register_confirm_password"
                )
                submitted = st.form_submit_button("Create account", use_container_width=True)
            if submitted:
                password_error = validate_password(password)
                if not email:
                    st.error("Enter a valid email address.")
                elif password_error:
                    st.error(password_error)
                elif password != confirm_password:
                    st.error("Passwords don't match.")
                else:
                    try:
                        resp = api_post("/register", {"email": email, "password": password})
                    except BackendUnreachable as exc:
                        st.error(str(exc))
                    else:
                        if resp.status_code == 201:
                            st.success("Account created. Switch to 'Sign in' to log in.")
                        else:
                            st.error(error_detail(resp, "Registration failed"))


def render_decision(decision: dict | None):
    if decision is None:
        st.warning("No decision recorded for this ticket.")
        return

    action = decision["action"]
    action_label = ACTION_LABELS.get(action, action)
    if action == "NEEDS_MORE_INFORMATION":
        st.warning(f"**{action_label}**")
    elif action.startswith("REJECT") or action == "CANNOT_CANCEL_AFTER_DISPATCH":
        st.error(f"**{action_label}**")
    else:
        st.success(f"**{action_label}**")

    col1, col2 = st.columns([1, 2])
    with col1:
        confidence = decision["confidence"]
        st.metric("Confidence", f"{confidence:.0%}")
        st.progress(min(max(confidence, 0.0), 1.0))
    with col2:
        provider = decision.get("provider")
        if provider:
            st.caption("Answered by")
            st.write(PROVIDER_LABELS.get(provider, provider))

    st.write(decision["reason"])
    if decision["sources"]:
        friendly_sources = [SOURCE_LABELS.get(s, s) for s in decision["sources"]]
        st.caption("Based on: " + ", ".join(friendly_sources))


def new_decision_page():
    st.header("Submit a Support Ticket")
    st.caption(
        "Just describe the issue in your own words. Add order details below only if you "
        "have them — the policies key off order value, delivery timing, and product type, "
        "so filling these in helps the AI decide with fewer follow-up questions."
    )
    with st.form("ticket_form"):
        message = st.text_area(
            "Describe the issue *",
            height=100,
            placeholder="e.g. My order arrived damaged, the box was crushed",
        )
        st.caption("*Required. Everything below is optional.")

        with st.expander("📋 Order details (optional — improves accuracy)"):
            col1, col2 = st.columns(2)
            with col1:
                order_value = st.number_input(
                    "Order value (INR)", min_value=0.0, step=100.0, value=0.0
                )
                days_since_delivery = st.text_input(
                    "Days since delivery (leave blank if not delivered)"
                )
                product_type = st.selectbox(
                    "Product type", ["", "food", "non_food", "mixed", "unknown"]
                )
            with col2:
                days_since_dispatch = st.text_input(
                    "Days since dispatch (leave blank if not dispatched)"
                )
                opened_status = st.selectbox("Opened status", ["", "opened", "unopened", "unknown"])
                order_status = st.selectbox(
                    "Order status", ["", "processing", "dispatched", "delivered", "unknown"]
                )

        submitted = st.form_submit_button("Get AI Decision", use_container_width=True)

    if not submitted:
        return

    if not message.strip():
        st.error("Please describe the issue.")
        return

    for label, value in (
        ("Days since delivery", days_since_delivery),
        ("Days since dispatch", days_since_dispatch),
    ):
        if value.strip() and not value.strip().lstrip("-").isdigit():
            st.error(f"{label} must be a whole number, or left blank.")
            return

    payload = {
        "message": message,
        "order_value_inr": order_value or None,
        "days_since_delivery": int(days_since_delivery) if days_since_delivery.strip() else None,
        "days_since_dispatch": int(days_since_dispatch) if days_since_dispatch.strip() else None,
        "product_type": product_type or None,
        "opened_status": opened_status or None,
        "order_status": order_status or None,
    }
    try:
        with st.spinner("Retrieving policy context and consulting the AI..."):
            resp = api_post("/tickets", payload, auth=True)
    except BackendUnreachable as exc:
        st.error(str(exc))
        return

    if resp.status_code == 201:
        st.success("Decision generated.")
        render_decision(resp.json().get("decision"))
    elif resp.status_code == 429:
        st.warning(error_detail(resp, "Too many requests — please wait a moment and try again."))
    elif resp.status_code == 502:
        st.error(error_detail(resp, "The AI decision service failed. Please try again."))
    else:
        st.error(error_detail(resp, "Failed to create ticket"))


ACTION_FILTER_ALL = "All actions"


def history_page():
    st.header("Your Ticket History")
    try:
        resp = api_get("/tickets")
    except BackendUnreachable as exc:
        st.error(str(exc))
        return

    if resp.status_code != 200:
        st.error(error_detail(resp, "Failed to load history."))
        return

    tickets = resp.json()
    if not tickets:
        st.info("No tickets yet. Submit one under 'New Decision'.")
        return

    actions = sorted({t["action"] for t in tickets if t.get("action")})
    selected_action = st.selectbox("Filter by action", [ACTION_FILTER_ALL] + actions)
    if selected_action != ACTION_FILTER_ALL:
        tickets = [t for t in tickets if t.get("action") == selected_action]

    st.caption(f"{len(tickets)} ticket(s)")

    for t in tickets:
        label = f"#{t['id']} · {t['message'][:60]} · {t.get('action') or 'pending'}"
        with st.expander(label):
            try:
                detail_resp = api_get(f"/tickets/{t['id']}")
            except BackendUnreachable as exc:
                st.error(str(exc))
                continue

            if detail_resp.status_code != 200:
                st.error(error_detail(detail_resp, "Failed to load this ticket."))
                continue

            detail = detail_resp.json()
            st.write(f"**Message:** {detail['message']}")
            st.write(
                f"Order value: {detail['order_value_inr']} | "
                f"Days since delivery: {detail['days_since_delivery']} | "
                f"Days since dispatch: {detail['days_since_dispatch']}"
            )
            st.write(
                f"Product type: {detail['product_type']} | "
                f"Opened: {detail['opened_status']} | "
                f"Order status: {detail['order_status']}"
            )
            render_decision(detail.get("decision"))


def main():
    if not st.session_state.token:
        login_register_page()
        return

    with st.sidebar:
        st.write(f"Logged in as **{st.session_state.email}**")
        if st.button("Log out", use_container_width=True):
            st.session_state.token = None
            st.session_state.email = None
            st.rerun()

        st.divider()
        page = st.radio("Navigate", ["New Decision", "History"])

        st.divider()
        if backend_status():
            st.caption("🟢 Backend online")
        else:
            st.caption("🔴 Backend unreachable")

    if page == "New Decision":
        new_decision_page()
    else:
        history_page()


main()
