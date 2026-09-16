import json
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, status
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
    ticket = Ticket(user_id=current_user.id, **payload.model_dump())
    db.add(ticket)
    db.commit()
    db.refresh(ticket)

    try:
        decision_result, _ = make_decision(payload.model_dump())
    except Exception as exc:  # noqa: BLE001
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
