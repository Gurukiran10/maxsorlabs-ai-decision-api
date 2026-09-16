"""Unit tests for the decision pipeline's validation, retry, and safety-net
logic, with both the Gemini client and the retriever mocked out."""

import os

os.environ.setdefault("GEMINI_API_KEY", "test-dummy-key")
os.environ.setdefault("JWT_SECRET", "test-dummy-secret")

from unittest.mock import MagicMock, patch  # noqa: E402

import pytest  # noqa: E402

from src import decision as decision_module  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_decision_cache(tmp_path, monkeypatch):
    """Point the on-disk decision cache at a throwaway path so tests don't
    read/write the real retrieval_cache/decision_cache.json, and don't leak
    cache hits between tests that reuse the same ticket fixtures."""
    monkeypatch.setattr(decision_module, "DECISION_CACHE_PATH", tmp_path / "decision_cache.json")


SAMPLE_TICKET = {
    "message": "My order arrived damaged.",
    "order_value_inr": 1500,
    "days_since_delivery": 2,
    "days_since_dispatch": None,
    "product_type": "non_food",
    "opened_status": "unopened",
    "order_status": "delivered",
}

RELEVANT_CHUNKS = [
    {"doc": "damaged_goods.md", "text": "Damage must be reported within 7 days.", "score": 0.8},
    {"doc": "damaged_goods.md", "text": "Orders under 2000 need no photos.", "score": 0.75},
]


def _fake_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.text = text
    return resp


def _fake_groq_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=text))]
    return resp


@patch.object(decision_module, "retrieve", return_value=RELEVANT_CHUNKS)
@patch.object(decision_module, "_get_client")
def test_valid_json_response_is_accepted(mock_get_client, mock_retrieve):
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _fake_response(
        '{"action": "APPROVE_REFUND_OR_REPLACEMENT", "confidence": 0.9, '
        '"reason": "Under the ?2,000 threshold, no photos required.", '
        '"sources": ["damaged_goods.md"]}'
    )
    mock_get_client.return_value = mock_client

    decision, retrieved = decision_module.make_decision(SAMPLE_TICKET)

    assert decision.action.value == "APPROVE_REFUND_OR_REPLACEMENT"
    assert decision.confidence == 0.9
    assert retrieved == RELEVANT_CHUNKS
    mock_client.models.generate_content.assert_called_once()


@patch.object(decision_module, "retrieve", return_value=RELEVANT_CHUNKS)
@patch.object(decision_module, "_get_client")
def test_malformed_json_falls_back_after_retries(mock_get_client, mock_retrieve):
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _fake_response("not valid json at all")
    mock_get_client.return_value = mock_client

    decision, _ = decision_module.make_decision(SAMPLE_TICKET)

    assert decision.action.value == "NEEDS_MORE_INFORMATION"
    assert mock_client.models.generate_content.call_count == 2  # one retry


@patch.object(decision_module, "retrieve", return_value=RELEVANT_CHUNKS)
@patch.object(decision_module, "_get_client")
def test_schema_violating_json_falls_back(mock_get_client, mock_retrieve):
    mock_client = MagicMock()
    # "confidence" out of the valid [0, 1] range should fail Pydantic validation.
    mock_client.models.generate_content.return_value = _fake_response(
        '{"action": "APPROVE_RETURN", "confidence": 5.0, "reason": "x", "sources": []}'
    )
    mock_get_client.return_value = mock_client

    decision, _ = decision_module.make_decision(SAMPLE_TICKET)

    assert decision.action.value == "NEEDS_MORE_INFORMATION"


@patch.object(decision_module, "retrieve")
@patch.object(decision_module, "_get_client")
def test_weak_retrieval_match_skips_llm_entirely(mock_get_client, mock_retrieve):
    """An off-topic ticket whose best policy match is below the similarity
    threshold should never reach the LLM at all."""
    mock_retrieve.return_value = [
        {"doc": "shipping.md", "text": "irrelevant", "score": 0.40},
    ]
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    decision, _ = decision_module.make_decision(
        {**SAMPLE_TICKET, "message": "What is the weather today?"}
    )

    assert decision.action.value == "NEEDS_MORE_INFORMATION"
    mock_client.models.generate_content.assert_not_called()


@patch.object(decision_module, "retrieve", return_value=[])
@patch.object(decision_module, "_get_client")
def test_empty_retrieval_result_skips_llm(mock_get_client, mock_retrieve):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    decision, _ = decision_module.make_decision(SAMPLE_TICKET)

    assert decision.action.value == "NEEDS_MORE_INFORMATION"
    mock_client.models.generate_content.assert_not_called()


@patch.object(decision_module, "retrieve", return_value=RELEVANT_CHUNKS)
@patch.object(decision_module, "_get_client")
def test_markdown_fenced_json_is_parsed(mock_get_client, mock_retrieve):
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _fake_response(
        '```json\n{"action": "REQUEST_PHOTOS", "confidence": 0.85, '
        '"reason": "Above threshold.", "sources": ["damaged_goods.md"]}\n```'
    )
    mock_get_client.return_value = mock_client

    decision, _ = decision_module.make_decision(SAMPLE_TICKET)

    assert decision.action.value == "REQUEST_PHOTOS"


@patch.object(decision_module, "retrieve", return_value=RELEVANT_CHUNKS)
@patch.object(decision_module, "_get_client")
def test_identical_ticket_is_served_from_cache(mock_get_client, mock_retrieve):
    """Submitting the exact same ticket twice should only call the LLM once;
    the second call should carry provider="cache" (everything else about
    the decision - action, confidence, reason, sources - stays identical)."""
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _fake_response(
        '{"action": "APPROVE_RETURN", "confidence": 0.88, '
        '"reason": "Unopened non-food item within the return window.", '
        '"sources": ["returns.md"]}'
    )
    mock_get_client.return_value = mock_client

    first_decision, _ = decision_module.make_decision(SAMPLE_TICKET)
    second_decision, _ = decision_module.make_decision(SAMPLE_TICKET)

    assert first_decision.provider == "gemini"
    assert second_decision.provider == "cache"
    assert first_decision.model_dump(exclude={"provider"}) == second_decision.model_dump(
        exclude={"provider"}
    )
    mock_client.models.generate_content.assert_called_once()


@patch.object(decision_module, "retrieve", return_value=RELEVANT_CHUNKS)
@patch.object(decision_module, "_get_client")
def test_different_ticket_is_not_served_from_cache(mock_get_client, mock_retrieve):
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _fake_response(
        '{"action": "APPROVE_RETURN", "confidence": 0.88, "reason": "x", "sources": []}'
    )
    mock_get_client.return_value = mock_client

    decision_module.make_decision(SAMPLE_TICKET)
    decision_module.make_decision({**SAMPLE_TICKET, "message": "A completely different issue."})

    assert mock_client.models.generate_content.call_count == 2


@patch.object(decision_module, "retrieve", return_value=RELEVANT_CHUNKS)
@patch.object(decision_module, "_get_client")
def test_llm_failure_is_not_cached(mock_get_client, mock_retrieve):
    """A transient failure (fallback) must not be cached, so the same ticket
    can succeed on a later attempt once the underlying issue clears."""
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _fake_response("not json")
    mock_get_client.return_value = mock_client

    decision_module.make_decision(SAMPLE_TICKET)  # falls back, 2 attempts

    mock_client.models.generate_content.return_value = _fake_response(
        '{"action": "APPROVE_RETURN", "confidence": 0.88, "reason": "x", "sources": []}'
    )
    second_decision, _ = decision_module.make_decision(SAMPLE_TICKET)

    assert second_decision.action.value == "APPROVE_RETURN"
    assert mock_client.models.generate_content.call_count == 3  # 2 failed + 1 succeeded


@patch.object(decision_module, "retrieve", return_value=RELEVANT_CHUNKS)
@patch.object(decision_module, "_get_groq_client")
@patch.object(decision_module, "_get_client")
def test_falls_back_to_groq_when_gemini_exhausted(
    mock_get_gemini_client, mock_get_groq_client, mock_retrieve, monkeypatch
):
    monkeypatch.setattr(decision_module, "GROQ_API_KEY", "test-dummy-groq-key")

    mock_gemini = MagicMock()
    mock_gemini.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED")
    mock_get_gemini_client.return_value = mock_gemini

    mock_groq = MagicMock()
    mock_groq.chat.completions.create.return_value = _fake_groq_response(
        '{"action": "APPROVE_RETURN", "confidence": 0.8, '
        '"reason": "Served by the fallback provider.", "sources": ["returns.md"]}'
    )
    mock_get_groq_client.return_value = mock_groq

    decision, _ = decision_module.make_decision(SAMPLE_TICKET)

    assert decision.action.value == "APPROVE_RETURN"
    assert mock_gemini.models.generate_content.call_count == 2  # both gemini attempts used
    mock_groq.chat.completions.create.assert_called_once()


@patch.object(decision_module, "retrieve", return_value=RELEVANT_CHUNKS)
@patch.object(decision_module, "_get_groq_client")
@patch.object(decision_module, "_get_client")
def test_groq_not_used_when_no_key_configured(
    mock_get_gemini_client, mock_get_groq_client, mock_retrieve, monkeypatch
):
    """Without a GROQ_API_KEY, the app must behave exactly as it did before
    the fallback existed - Gemini-only, no attempt to construct a Groq
    client at all."""
    monkeypatch.setattr(decision_module, "GROQ_API_KEY", "")

    mock_gemini = MagicMock()
    mock_gemini.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED")
    mock_get_gemini_client.return_value = mock_gemini

    decision, _ = decision_module.make_decision(SAMPLE_TICKET)

    assert decision.action.value == "NEEDS_MORE_INFORMATION"  # FALLBACK_DECISION
    assert mock_gemini.models.generate_content.call_count == 2
    mock_get_groq_client.assert_not_called()


@patch.object(decision_module, "retrieve", return_value=RELEVANT_CHUNKS)
@patch.object(decision_module, "_get_groq_client")
@patch.object(decision_module, "_get_client")
def test_final_fallback_when_both_providers_fail(
    mock_get_gemini_client, mock_get_groq_client, mock_retrieve, monkeypatch
):
    monkeypatch.setattr(decision_module, "GROQ_API_KEY", "test-dummy-groq-key")

    mock_gemini = MagicMock()
    mock_gemini.models.generate_content.side_effect = Exception("gemini down")
    mock_get_gemini_client.return_value = mock_gemini

    mock_groq = MagicMock()
    mock_groq.chat.completions.create.side_effect = Exception("groq also down")
    mock_get_groq_client.return_value = mock_groq

    decision, _ = decision_module.make_decision(SAMPLE_TICKET)

    assert decision.action.value == "NEEDS_MORE_INFORMATION"  # FALLBACK_DECISION
    assert decision.confidence == 0.5
    assert mock_gemini.models.generate_content.call_count == 2
    assert mock_groq.chat.completions.create.call_count == 2
