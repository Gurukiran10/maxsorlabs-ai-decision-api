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
    inspect,
    text,
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
    # Which path actually produced this decision: gemini | groq | cache |
    # retrieval_gate | fallback. Not in the assignment's minimum schema, but
    # cheap to store and useful for demonstrating/debugging the multi-
    # provider fallback chain (see src/decision.py).
    provider = Column(String, nullable=True)
    created_at = Column(DateTime, default=utcnow, nullable=False)

    ticket = relationship("Ticket", back_populates="decision")


engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _add_missing_columns() -> None:
    """Bare-bones auto-migration: create_all() only creates missing tables,
    it never alters existing ones. A real project at this SQLite/no-Alembic
    scope would still eventually need *some* way to evolve a schema that's
    already been created on disk, so this adds any column that's on the
    model but missing from the actual table - a good-enough substitute for
    a full migration tool at this project's size, not a replacement for one
    at a larger scale.
    """
    inspector = inspect(engine)
    if "decisions" not in inspector.get_table_names():
        return
    existing_columns = {col["name"] for col in inspector.get_columns("decisions")}
    with engine.begin() as conn:
        for column in Decision.__table__.columns:
            if column.name not in existing_columns:
                col_type = column.type.compile(dialect=engine.dialect)
                conn.execute(text(f"ALTER TABLE decisions ADD COLUMN {column.name} {col_type}"))


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    _add_missing_columns()


def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
