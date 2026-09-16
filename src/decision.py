"""LLM-backed decision step: retrieve policy context, ask Gemini for a
structured decision, validate it, and fall back safely on failure."""

import hashlib
import json
import logging

from google import genai

from src.config import GEMINI_API_KEY, RETRIEVAL_CACHE_DIR
from src.retrieval import retrieve
from src.schemas import LLMDecision

logger = logging.getLogger(__name__)

# Identical tickets (same message + same structured facts) always deserve the
# same answer, and the sample data set genuinely contains repeated tickets
# (e.g. many customers submitting the exact same change-of-mind return
# message). Caching by content hash makes those repeats free, instant, and
# deterministic, instead of paying for and re-rolling a fresh LLM call every
# time. This is a simple on-disk JSON cache with no TTL/invalidation, which
# is a reasonable scope for a small assignment; a real system would put this
# behind a proper cache (Redis, etc.) with expiry tied to policy-doc changes.
DECISION_CACHE_PATH = RETRIEVAL_CACHE_DIR / "decision_cache.json"


def _cache_key(ticket: dict) -> str:
    canonical = json.dumps(ticket, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_decision_cache() -> dict:
    if not DECISION_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(DECISION_CACHE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_decision_cache(cache: dict) -> None:
    DECISION_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    DECISION_CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")

GENERATION_MODEL = "gemini-3.6-flash"

# Minimum cosine similarity the best-matching policy chunk must clear before
# the LLM is even consulted. Empirically, genuinely relevant tickets score
# ~0.70-0.76 against this knowledge base, while off-topic messages top out
# around ~0.55-0.58 (see DEVELOPMENT.md). This is a cheap, deterministic
# guardrail against relying on the LLM alone to notice it has no grounding.
MIN_RETRIEVAL_SCORE = 0.62

UNGROUNDED_DECISION = LLMDecision(
    action="NEEDS_MORE_INFORMATION",
    confidence=0.9,
    reason="The ticket does not appear to relate to any known policy area (returns, "
    "damaged goods, shipping, cancellations, defects, or wrong item).",
    sources=[],
)

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
    key = _cache_key(ticket)
    cache = _load_decision_cache()
    cached = cache.get(key)
    if cached is not None:
        logger.info("Decision cache hit for ticket hash %s", key[:12])
        return LLMDecision.model_validate(cached["decision"]), cached["retrieved"]

    retrieved = retrieve(ticket["message"], top_k=5)

    if not retrieved or retrieved[0]["score"] < MIN_RETRIEVAL_SCORE:
        logger.info(
            "Top retrieval score %.3f below threshold %.2f; skipping LLM call.",
            retrieved[0]["score"] if retrieved else -1.0,
            MIN_RETRIEVAL_SCORE,
        )
        _cache_decision(cache, key, UNGROUNDED_DECISION, retrieved)
        return UNGROUNDED_DECISION, retrieved

    context = _build_context(retrieved)
    facts = _build_ticket_facts(ticket)

    prompt = f"{SYSTEM_PROMPT}\n\nPolicy context:\n{context}\n\nTicket:\n{facts}\n\nJSON decision:"

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            raw = _call_llm(prompt)
            parsed = _extract_json(raw)
            decision = LLMDecision.model_validate(parsed)
            _cache_decision(cache, key, decision, retrieved)
            return decision, retrieved
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning("Decision attempt %d failed: %s", attempt + 1, exc)

    # Deliberately not cached: a failure here is usually transient (rate
    # limit, network blip), and caching it would permanently freeze a bad
    # answer for a ticket that could be answered correctly once the
    # underlying issue clears.
    logger.error("Falling back to NEEDS_MORE_INFORMATION after LLM failures: %s", last_error)
    return FALLBACK_DECISION, retrieved


def _cache_decision(
    cache: dict, key: str, decision: LLMDecision, retrieved: list[dict]
) -> None:
    cache[key] = {"decision": decision.model_dump(mode="json"), "retrieved": retrieved}
    _save_decision_cache(cache)
