"""Unit tests for the decision pipeline's validation, retry, and safety-net
logic, with both the Gemini client and the retriever mocked out."""

import os

os.environ.setdefault("GEMINI_API_KEY", "test-dummy-key")
os.environ.setdefault("JWT_SECRET", "test-dummy-secret")

from unittest.mock import MagicMock, patch  # noqa: E402

from src import decision as decision_module  # noqa: E402

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
