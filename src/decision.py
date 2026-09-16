"""LLM-backed decision step: retrieve policy context, ask Gemini for a
structured decision, validate it, and fall back safely on failure."""

import json
import logging

from google import genai

from src.config import GEMINI_API_KEY
from src.retrieval import retrieve
from src.schemas import LLMDecision

logger = logging.getLogger(__name__)

GENERATION_MODEL = "gemini-2.0-flash"

SYSTEM_PROMPT = """You are a support-ticket decision assistant for an e-commerce company.

You will be given a customer support ticket (with any structured order details
available) and the most relevant excerpts from the company's policy documents.

Decide the single correct action using ONLY the provided policy context and
ticket facts. Do not use outside knowledge or assumptions about facts not given.

If the ticket does not contain enough information to apply the policy
confidently (e.g. missing delivery date, order value, product type, or
opened/unopened status when the relevant policy requires it), you MUST return
the action "NEEDS_MORE_INFORMATION" rather than guessing.

Respond with ONLY a single JSON object, no markdown fences, no commentary, in
exactly this shape:
{
  "action": "<one of the allowed action codes>",
  "confidence": <float between 0 and 1>,
  "reason": "<one or two sentences citing the specific policy rule applied>",
  "sources": ["<policy filename(s) used>"]
}

Allowed action codes: APPROVE_RETURN, REJECT_OUTSIDE_WINDOW, REJECT_OPENED_ITEM,
REJECT_FOOD_RETURN, APPROVE_REFUND_OR_REPLACEMENT, REQUEST_PHOTOS,
APPROVE_REPLACEMENT, REQUEST_DEFECT_EVIDENCE, REPLACE_CORRECT_ITEM,
CANCEL_AND_REFUND, CANNOT_CANCEL_AFTER_DISPATCH, WAIT_AND_TRACK,
OPEN_SHIPPING_INVESTIGATION, OFFER_REPLACEMENT_OR_REFUND, NEEDS_MORE_INFORMATION.
"""

FALLBACK_DECISION = LLMDecision(
    action="NEEDS_MORE_INFORMATION",
    confidence=0.5,
    reason="The system could not confidently determine an action from the available information.",
    sources=[],
)

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def _build_ticket_facts(ticket: dict) -> str:
    lines = [f"Message: {ticket['message']}"]
    for key in (
        "order_value_inr",
        "days_since_delivery",
        "days_since_dispatch",
        "product_type",
        "opened_status",
        "order_status",
    ):
        value = ticket.get(key)
        lines.append(f"{key}: {value if value is not None else 'unknown'}")
    return "\n".join(lines)


def _build_context(chunks: list[dict]) -> str:
    return "\n".join(f"- [{c['doc']}] {c['text']}" for c in chunks)


def _extract_json(raw_text: str) -> dict:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def _call_llm(prompt: str) -> str:
    client = _get_client()
    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config={"temperature": 0.1, "response_mime_type": "application/json"},
    )
    return response.text


def make_decision(ticket: dict) -> tuple[LLMDecision, list[dict]]:
    """Returns (validated decision, retrieved chunks used as context)."""
    retrieved = retrieve(ticket["message"], top_k=5)
    context = _build_context(retrieved)
    facts = _build_ticket_facts(ticket)

    prompt = f"{SYSTEM_PROMPT}\n\nPolicy context:\n{context}\n\nTicket:\n{facts}\n\nJSON decision:"

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            raw = _call_llm(prompt)
            parsed = _extract_json(raw)
            decision = LLMDecision.model_validate(parsed)
            return decision, retrieved
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning("Decision attempt %d failed: %s", attempt + 1, exc)

    logger.error("Falling back to NEEDS_MORE_INFORMATION after LLM failures: %s", last_error)
    return FALLBACK_DECISION, retrieved
