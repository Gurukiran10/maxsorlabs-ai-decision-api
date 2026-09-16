import json
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from src.auth import (
    InvalidTokenError,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from src.database import Decision, Ticket, User, get_db, init_db
from src.decision import make_decision
from src.rate_limit import check_ticket_rate_limit
from src.schemas import (
    TicketCreate,
    TicketListItem,
    TicketResponse,
    TokenResponse,
    UserLogin,
    UserRegister,
    UserResponse,
)

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="AI Decision API", version="1.0.0", lifespan=lifespan)
bearer_scheme = HTTPBearer()


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Last-resort handler for anything not already turned into an
    HTTPException. Without this, an unexpected error (e.g. a DB constraint
    violation, a bug in a route) would leak a raw Python traceback to the
    client instead of a clean, generic error response. The real traceback
    still goes to the server logs, tagged with an error id the client can
    quote back for support/debugging without exposing internals."""
    error_id = uuid.uuid4().hex[:12]
    logging.getLogger(__name__).exception("Unhandled error [%s] on %s", error_id, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error.", "error_id": error_id},
    )


@app.get("/health")
def health():
    return {"status": "ok"}


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    try:
        payload = decode_access_token(credentials.credentials)
    except InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token"
        ) from exc

    user = db.query(User).filter(User.id == int(payload["sub"])).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


@app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(payload: UserRegister, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    user = User(email=payload.email, password_hash=hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.post("/login", response_model=TokenResponse)
def login(payload: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password"
        )
    token = create_access_token(user.id, user.email)
    return TokenResponse(access_token=token)


@app.get("/me", response_model=UserResponse)
def me(current_user: User = Depends(get_current_user)):
    return current_user


@app.post("/tickets", response_model=TicketResponse, status_code=status.HTTP_201_CREATED)
def create_ticket(
    payload: TicketCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    check_ticket_rate_limit(current_user.id)

    # mode="json" is important: TicketCreate's product_type/opened_status/
    # order_status are Enum members, and Enum.__str__ (even for a str-mixed
    # Enum) renders as "ProductType.FOOD", not "food". mode="json" dumps the
    # underlying .value instead, which is what both the DB column and the
    # LLM prompt need.
    ticket_data = payload.model_dump(mode="json")

    # The ticket is flushed (assigned an id) but not committed yet, so if the
    # decision pipeline raises, the rollback below discards the ticket too -
    # a ticket with no decision should never be persisted. Both rows land in
    # the database atomically, or neither does.
    ticket = Ticket(user_id=current_user.id, **ticket_data)
    db.add(ticket)
    db.flush()

    try:
        decision_result, _ = make_decision(ticket_data)
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        logging.getLogger(__name__).exception("Decision pipeline failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI decision service failed. Please try again.",
        ) from exc

    decision = Decision(
        ticket_id=ticket.id,
        action=decision_result.action.value,
        reason=decision_result.reason,
        confidence=decision_result.confidence,
        sources=json.dumps(decision_result.sources),
    )
    db.add(decision)
    db.commit()
    db.refresh(ticket)
    return ticket


@app.get("/tickets", response_model=list[TicketListItem])
def list_tickets(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    tickets = (
        db.query(Ticket)
        .filter(Ticket.user_id == current_user.id)
        .order_by(Ticket.created_at.desc())
        .all()
    )
    items = []
    for t in tickets:
        items.append(
            TicketListItem(
                id=t.id,
                message=t.message,
                created_at=t.created_at,
                action=t.decision.action if t.decision else None,
                confidence=t.decision.confidence if t.decision else None,
            )
        )
    return items


@app.get("/tickets/{ticket_id}", response_model=TicketResponse)
def get_ticket(
    ticket_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    ticket = (
        db.query(Ticket)
        .filter(Ticket.id == ticket_id, Ticket.user_id == current_user.id)
        .first()
    )
    if ticket is None:
        # Same 404 whether the ticket doesn't exist or belongs to another
        # user, so this endpoint never confirms another user's ticket IDs.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ticket not found")
    return ticket
