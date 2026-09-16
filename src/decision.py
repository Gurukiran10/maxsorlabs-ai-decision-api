"""LLM-backed decision step: retrieve policy context, ask Gemini for a
structured decision, validate it, and fall back safely on failure."""

import hashlib
import json
import logging
import time

from google import genai
from groq import Groq

from src.config import GEMINI_API_KEY, GROQ_API_KEY, RETRIEVAL_CACHE_DIR
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

GEMINI_MODEL = "gemini-3.6-flash"
# Fallback provider used when Gemini fails (typically free-tier quota
# exhaustion, which is a real failure mode this project hit during
# development, not a hypothetical one). Groq's free tier is generous and
# entirely separate infrastructure/quota from Google's, so it's a genuine
# redundancy path rather than just a second call to the same limit.
GROQ_MODEL = "openai/gpt-oss-120b"

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

_gemini_client: genai.Client | None = None
_groq_client: Groq | None = None


def _get_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client


def _get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


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


def _call_gemini(prompt: str) -> str:
    client = _get_client()
    start = time.monotonic()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={"temperature": 0.1, "response_mime_type": "application/json"},
    )
    latency_ms = (time.monotonic() - start) * 1000

    usage = response.usage_metadata
    logger.info(
        "llm_call provider=gemini model=%s latency_ms=%.0f prompt_tokens=%s "
        "output_tokens=%s total_tokens=%s",
        GEMINI_MODEL,
        latency_ms,
        getattr(usage, "prompt_token_count", None),
        getattr(usage, "candidates_token_count", None),
        getattr(usage, "total_token_count", None),
    )
    return response.text


def _call_groq(prompt: str) -> str:
    client = _get_groq_client()
    start = time.monotonic()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    latency_ms = (time.monotonic() - start) * 1000

    usage = response.usage
    logger.info(
        "llm_call provider=groq model=%s latency_ms=%.0f prompt_tokens=%s "
        "output_tokens=%s total_tokens=%s",
        GROQ_MODEL,
        latency_ms,
        getattr(usage, "prompt_tokens", None),
        getattr(usage, "completion_tokens", None),
        getattr(usage, "total_tokens", None),
    )
    return response.choices[0].message.content


def _providers() -> list[tuple[str, callable]]:
    """Ordered list of (provider_name, call_fn) to try. Gemini gets two
    attempts (covers a transient blip or a one-off malformed response); Groq
    is only included - also with two attempts - if a key is configured, so
    the app degrades gracefully to Gemini-only behavior without one."""
    attempts = [("gemini", _call_gemini), ("gemini", _call_gemini)]
    if GROQ_API_KEY:
        attempts += [("groq", _call_groq), ("groq", _call_groq)]
    return attempts


def make_decision(ticket: dict) -> tuple[LLMDecision, list[dict]]:
    """Returns (validated decision, retrieved chunks used as context)."""
    key = _cache_key(ticket)
    cache = _load_decision_cache()
    cached = cache.get(key)
    if cached is not None:
        logger.info("Decision cache hit for ticket hash %s", key[:12])
        decision = LLMDecision.model_validate(cached["decision"])
        decision.provider = "cache"
        return decision, cached["retrieved"]

    retrieved = retrieve(ticket["message"], top_k=5)

    if not retrieved or retrieved[0]["score"] < MIN_RETRIEVAL_SCORE:
        logger.info(
            "Top retrieval score %.3f below threshold %.2f; skipping LLM call.",
            retrieved[0]["score"] if retrieved else -1.0,
            MIN_RETRIEVAL_SCORE,
        )
        decision = UNGROUNDED_DECISION.model_copy(update={"provider": "retrieval_gate"})
        _cache_decision(cache, key, decision, retrieved)
        return decision, retrieved

    context = _build_context(retrieved)
    facts = _build_ticket_facts(ticket)

    prompt = f"{SYSTEM_PROMPT}\n\nPolicy context:\n{context}\n\nTicket:\n{facts}\n\nJSON decision:"

    last_error: Exception | None = None
    for attempt, (provider_name, call_fn) in enumerate(_providers(), start=1):
        try:
            raw = call_fn(prompt)
            parsed = _extract_json(raw)
            decision = LLMDecision.model_validate(parsed)
            decision.provider = provider_name
            _cache_decision(cache, key, decision, retrieved)
            return decision, retrieved
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning(
                "Decision attempt %d (provider=%s) failed: %s", attempt, provider_name, exc
            )

    # Deliberately not cached: a failure here is usually transient (rate
    # limit, network blip), and caching it would permanently freeze a bad
    # answer for a ticket that could be answered correctly once the
    # underlying issue clears.
    logger.error("Falling back to NEEDS_MORE_INFORMATION after LLM failures: %s", last_error)
    decision = FALLBACK_DECISION.model_copy(update={"provider": "fallback"})
    return decision, retrieved


def _cache_decision(
    cache: dict, key: str, decision: LLMDecision, retrieved: list[dict]
) -> None:
    cache[key] = {"decision": decision.model_dump(mode="json"), "retrieved": retrieved}
    _save_decision_cache(cache)
