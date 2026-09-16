# AI Support Ticket Decision Assistant

A small end-to-end app: a user registers/logs in (JWT), submits a support
ticket, and receives a structured, policy-grounded AI decision (action,
confidence, reasoning, sources). Built with FastAPI, SQLite, a local RAG
pipeline over the supplied policy documents, and Gemini.

## Architecture

```
Streamlit (UI)  --HTTP-->  FastAPI (auth, tickets, decisions)  -->  SQLite
                                        |
                                        v
                          retrieval.py (chunk + embed + cosine search)
                                        |
                                        v
                          decision.py (Gemini -> validated JSON -> DB)
```

Streamlit never touches the database directly — every read/write goes
through the FastAPI HTTP API.

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # on Windows
# source venv/bin/activate   # on macOS/Linux

pip install -r requirements.txt

copy .env.example .env       # or `cp` on macOS/Linux
```

Edit `.env` and set `GEMINI_API_KEY` (get one at
https://aistudio.google.com/u/0/api-keys) and a random `JWT_SECRET`.

Build the local embedding index for the policy documents (one-time, or
whenever `knowledge_base/` changes):

```bash
python -m scripts.ingest
```

## Running

Terminal 1 — backend:

```bash
uvicorn src.api:app --reload
```

Terminal 2 — frontend:

```bash
streamlit run streamlit_app.py
```

Open the Streamlit URL it prints, register a user, log in, and submit a
ticket under "New Decision".

## Running tests

```bash
pytest
```

Tests use an in-memory SQLite database and stub out the Gemini call, so
they run without a network connection or a valid API key. They cover the
auth flow and, notably, that one user's JWT cannot be used to read another
user's ticket (`tests/test_tickets.py::test_user_cannot_read_another_users_ticket`).

## Running the evaluation

```bash
python -m scripts.ingest   # if not already built
python -m eval.run_eval
```

This runs the pipeline against `eval/sample_test_cases.json` (from the
supplied candidate pack) and prints per-case pass/fail plus overall accuracy.
This requires a valid `GEMINI_API_KEY` since it calls the real LLM end to end.

## API

| Method | Endpoint         | Purpose                                   |
|--------|------------------|--------------------------------------------|
| POST   | `/register`      | Create a user account                     |
| POST   | `/login`         | Verify credentials, return a JWT          |
| GET    | `/me`            | Return the authenticated user             |
| POST   | `/tickets`       | Submit a ticket, generate an AI decision  |
| GET    | `/tickets`       | List the authenticated user's tickets     |
| GET    | `/tickets/{id}`  | Get one ticket + its decision             |

All endpoints except `/register` and `/login` require
`Authorization: Bearer <JWT>`.

## Design decisions worth knowing about

- **Ticket schema is richer than the minimum spec.** The spec's minimum
  `tickets` table only lists a `message` text field, but the supplied
  policies key off concrete facts (order value, days since delivery/dispatch,
  product type, opened/unopened, order status) that can't be reliably parsed
  out of free text. The ticket form captures these as explicit structured
  fields alongside the free-text message, and the decision step is grounded
  on both. This keeps the LLM from having to guess numeric/categorical facts
  it should just be told.
- **RAG chunking is rule-per-chunk, not fixed-size windows.** The knowledge
  base docs are short numbered-list policies; splitting on each numbered rule
  keeps every chunk atomic and topically self-contained, which is far more
  precise for retrieval than sliding-window chunking would be on text this
  short.
- **No vector DB.** ~30 short chunks total — a NumPy cosine-similarity scan
  over cached embeddings is simpler, faster to set up, and just as correct
  at this scale. Embeddings are cached to `retrieval_cache/` (gitignored) so
  they aren't recomputed on every request.
- **Structured-output validation and fallback.** The LLM is prompted to
  return only JSON matching a fixed schema; the response is parsed and
  validated against a Pydantic model before it's ever persisted. One retry is
  allowed on a malformed response; if it still fails (or the model itself
  can't confidently decide), the system persists `NEEDS_MORE_INFORMATION`
  rather than guessing or crashing.
- **Same 404 for "not found" and "not yours."** `/tickets/{id}` returns 404
  in both cases so the API never confirms/denies another user's ticket IDs
  to someone who isn't authorized to see them.

## Project structure

```
src/
  api.py         FastAPI app and routes
  auth.py        password hashing + JWT issue/verify
  config.py      environment/config loading
  database.py    SQLAlchemy models (users, tickets, decisions)
  decision.py    prompt building, Gemini call, validation, fallback
  retrieval.py   chunking, embedding, local vector search
  schemas.py     Pydantic request/response/LLM-output models
streamlit_app.py Streamlit UI (login/register, new decision, history)
knowledge_base/  supplied policy documents
data/            supplied historical tickets (reference only, not looked up)
eval/            sample test cases + evaluation runner
scripts/ingest.py builds the local embedding cache
tests/           pytest suite
```
