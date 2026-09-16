# Development Notes: AI Coding Agent Usage

This project was built with Claude Code as an active pair-programming tool
during implementation. This document is an honest account of how it was
used, and what was reviewed/decided by me rather than accepted as-is.

## What the agent was used for

- Scaffolding the FastAPI/SQLAlchemy/Pydantic layer (models, schemas, routes)
  from the assignment spec.
- Writing the RAG pipeline (chunking, embedding calls, cosine similarity
  retrieval) and the Gemini decision-generation prompt/validation logic.
- Writing the Streamlit UI screens and the pytest suite, including the
  cross-user authorization test.
- Drafting this README/documentation.

## What I reviewed and decided myself

- **The ticket data model.** The spec's minimum schema only lists a
  `message` field, but after reading the supplied policy docs and
  `tickets.csv`, it was clear the decisions are actually driven by
  structured facts (order value, day counts, product type, opened status)
  that the historical data always carries alongside the free-text message.
  I made the call to add these as explicit ticket fields rather than have
  the LLM try to extract/guess them from prose, and documented that
  trade-off in the README.
- **Chunking strategy.** I chose rule-per-chunk splitting (one chunk per
  numbered policy line) over fixed-size windows, since the source documents
  are short numbered lists and each rule is already a complete, retrievable
  unit.
- **Fallback behavior.** I decided that any LLM failure — malformed JSON,
  validation failure, or the model's own uncertainty — should resolve to
  `NEEDS_MORE_INFORMATION` rather than a retry loop or a default "safe"
  action, since a false approval/rejection is worse than asking for more
  info.
- **Authorization approach.** I verified that ticket access is scoped by
  `user_id` pulled from the JWT (never from the URL/request body), and wrote
  the test that proves Alice's token 404s on Bob's ticket rather than 403ing
  (to avoid confirming the ticket's existence to an unauthorized caller).
- All generated code was read line by line, run locally, and adjusted where
  it didn't match the actual data shape in `candidate_pack.zip` (e.g. the
  sample test cases and `tickets.csv` include structured fields, not just a
  message string).

## What I would explain in the interview

- Why cosine similarity + NumPy is sufficient here instead of a vector DB.
- Why JWT is passed via the `Authorization` header rather than a cookie.
- How the retry/fallback logic in `src/decision.py` prevents an
  unparseable LLM response from ever reaching the database.
- The trade-off of extending the ticket schema beyond the spec's minimum,
  and what I'd do differently with more time (e.g. a small structured-field
  extraction step so the UI could accept pure free text).
