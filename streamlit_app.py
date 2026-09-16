import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="AI Support Decision Assistant", page_icon="🎫")


def api_post(path: str, json_body: dict, auth: bool = False) -> requests.Response:
    headers = {"Authorization": f"Bearer {st.session_state.token}"} if auth else {}
    return requests.post(f"{API_BASE_URL}{path}", json=json_body, headers=headers, timeout=30)


def api_get(path: str) -> requests.Response:
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    return requests.get(f"{API_BASE_URL}{path}", headers=headers, timeout=30)


if "token" not in st.session_state:
    st.session_state.token = None
if "email" not in st.session_state:
    st.session_state.email = None


def login_register_page():
    st.title("🎫 AI Support Decision Assistant")
    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Login")
        if submitted:
            resp = api_post("/login", {"email": email, "password": password})
            if resp.status_code == 200:
                st.session_state.token = resp.json()["access_token"]
                st.session_state.email = email
                st.rerun()
            else:
                st.error(resp.json().get("detail", "Login failed"))

    with tab_register:
        with st.form("register_form"):
            email = st.text_input("Email", key="register_email")
            password = st.text_input(
                "Password (min 8 characters)", type="password", key="register_password"
            )
            submitted = st.form_submit_button("Register")
        if submitted:
            resp = api_post("/register", {"email": email, "password": password})
            if resp.status_code == 201:
                st.success("Account created. Please log in.")
            else:
                st.error(resp.json().get("detail", "Registration failed"))


def render_decision(decision: dict | None):
    if decision is None:
        st.warning("No decision recorded for this ticket.")
        return
    action = decision["action"]
    if action == "NEEDS_MORE_INFORMATION":
        st.warning(f"**Action:** {action}")
    elif action.startswith("REJECT") or action == "CANNOT_CANCEL_AFTER_DISPATCH":
        st.error(f"**Action:** {action}")
    else:
        st.success(f"**Action:** {action}")
    st.metric("Confidence", f"{decision['confidence']:.0%}")
    st.write(f"**Reasoning:** {decision['reason']}")
    if decision["sources"]:
        st.caption("Sources: " + ", ".join(decision["sources"]))


def new_decision_page():
    st.header("Submit a Support Ticket")
    with st.form("ticket_form"):
        message = st.text_area("Describe the issue", height=100)
        col1, col2 = st.columns(2)
        with col1:
            order_value = st.number_input("Order value (INR)", min_value=0.0, step=100.0, value=0.0)
            days_since_delivery = st.text_input("Days since delivery (leave blank if not delivered)")
            product_type = st.selectbox("Product type", ["", "food", "non_food", "mixed", "unknown"])
        with col2:
            days_since_dispatch = st.text_input("Days since dispatch (leave blank if not dispatched)")
            opened_status = st.selectbox("Opened status", ["", "opened", "unopened", "unknown"])
            order_status = st.selectbox(
                "Order status", ["", "processing", "dispatched", "delivered", "unknown"]
            )
        submitted = st.form_submit_button("Get AI Decision")

    if submitted:
        if not message.strip():
            st.error("Please describe the issue.")
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
        with st.spinner("Retrieving policy context and consulting the AI..."):
            resp = api_post("/tickets", payload, auth=True)
        if resp.status_code == 201:
            st.success("Decision generated.")
            render_decision(resp.json().get("decision"))
        else:
            st.error(resp.json().get("detail", "Failed to create ticket"))


def history_page():
    st.header("Your Ticket History")
    resp = api_get("/tickets")
    if resp.status_code != 200:
        st.error("Failed to load history.")
        return
    tickets = resp.json()
    if not tickets:
        st.info("No tickets yet. Submit one under 'New Decision'.")
        return

    for t in tickets:
        label = f"#{t['id']} · {t['message'][:60]} · {t.get('action') or 'pending'}"
        with st.expander(label):
            detail_resp = api_get(f"/tickets/{t['id']}")
            if detail_resp.status_code == 200:
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

    st.sidebar.write(f"Logged in as **{st.session_state.email}**")
    if st.sidebar.button("Log out"):
        st.session_state.token = None
        st.session_state.email = None
        st.rerun()

    page = st.sidebar.radio("Navigate", ["New Decision", "History"])
    if page == "New Decision":
        new_decision_page()
    else:
        history_page()


main()
