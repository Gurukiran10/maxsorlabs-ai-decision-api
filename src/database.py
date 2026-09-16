from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

from src.config import DATABASE_URL


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=utcnow, nullable=False)

    tickets = relationship("Ticket", back_populates="user", cascade="all, delete-orphan")


class Ticket(Base):
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    message = Column(Text, nullable=False)

    # Structured context fields. The spec's minimum schema only lists `message`,
    # but the supplied dataset and policies (₹ thresholds, day windows, opened
    # status, dispatch vs delivery) can't be decided from free text alone, so
    # the ticket form captures them explicitly rather than asking the LLM to
    # infer facts that should come from order data.
    order_value_inr = Column(Float, nullable=True)
    days_since_delivery = Column(Integer, nullable=True)
    days_since_dispatch = Column(Integer, nullable=True)
    product_type = Column(String, nullable=True)  # food | non_food | mixed | unknown
    opened_status = Column(String, nullable=True)  # opened | unopened | unknown
    order_status = Column(String, nullable=True)  # processing | dispatched | delivered | unknown

    created_at = Column(DateTime, default=utcnow, nullable=False)

    user = relationship("User", back_populates="tickets")
    decision = relationship(
        "Decision", back_populates="ticket", uselist=False, cascade="all, delete-orphan"
    )


class Decision(Base):
    __tablename__ = "decisions"

    id = Column(Integer, primary_key=True, index=True)
    ticket_id = Column(Integer, ForeignKey("tickets.id"), nullable=False, unique=True, index=True)
    action = Column(String, nullable=False)
    reason = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False)
    sources = Column(Text, nullable=False)  # JSON-encoded list[str]
    created_at = Column(DateTime, default=utcnow, nullable=False)

    ticket = relationship("Ticket", back_populates="decision")


engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
