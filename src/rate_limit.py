"""Simple per-user sliding-window rate limiter for the ticket-creation
endpoint, which is the only endpoint that spends a real LLM call.

This is deliberately an in-memory, single-process limiter - it resets on
restart and doesn't coordinate across multiple backend instances. That's a
reasonable trade-off at this project's scope (SQLite, single uvicorn
process); a real multi-instance deployment would back this with Redis
instead. The point of including it at all is that a support-ticket endpoint
backed by a paid, quota-limited LLM API is exactly the kind of endpoint that
needs *some* abuse/cost protection, and the assignment itself ran into the
free-tier quota during development.
"""

import threading
import time
from collections import defaultdict

from fastapi import HTTPException, status

MAX_REQUESTS_PER_WINDOW = 10
WINDOW_SECONDS = 60

_lock = threading.Lock()
_requests_by_user: dict[int, list[float]] = defaultdict(list)


def check_ticket_rate_limit(user_id: int) -> None:
    now = time.monotonic()
    cutoff = now - WINDOW_SECONDS

    with _lock:
        timestamps = _requests_by_user[user_id]
        # Drop entries outside the current window.
        timestamps[:] = [t for t in timestamps if t > cutoff]

        if len(timestamps) >= MAX_REQUESTS_PER_WINDOW:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Too many tickets submitted. Limit is {MAX_REQUESTS_PER_WINDOW} "
                    f"per {WINDOW_SECONDS} seconds; please wait and try again."
                ),
            )

        timestamps.append(now)


def reset_rate_limits() -> None:
    """Test-only helper to clear rate-limit state between test runs."""
    with _lock:
        _requests_by_user.clear()
