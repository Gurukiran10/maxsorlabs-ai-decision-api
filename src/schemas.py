import re
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


class ProductType(str, Enum):
    FOOD = "food"
    NON_FOOD = "non_food"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class OpenedStatus(str, Enum):
    OPENED = "opened"
    UNOPENED = "unopened"
    UNKNOWN = "unknown"


class OrderStatus(str, Enum):
    PROCESSING = "processing"
    DISPATCHED = "dispatched"
    DELIVERED = "delivered"
    UNKNOWN = "unknown"


class ActionEnum(str, Enum):
    APPROVE_RETURN = "APPROVE_RETURN"
    REJECT_OUTSIDE_WINDOW = "REJECT_OUTSIDE_WINDOW"
    REJECT_OPENED_ITEM = "REJECT_OPENED_ITEM"
    REJECT_FOOD_RETURN = "REJECT_FOOD_RETURN"
    APPROVE_REFUND_OR_REPLACEMENT = "APPROVE_REFUND_OR_REPLACEMENT"
    REQUEST_PHOTOS = "REQUEST_PHOTOS"
    APPROVE_REPLACEMENT = "APPROVE_REPLACEMENT"
    REQUEST_DEFECT_EVIDENCE = "REQUEST_DEFECT_EVIDENCE"
    REPLACE_CORRECT_ITEM = "REPLACE_CORRECT_ITEM"
    CANCEL_AND_REFUND = "CANCEL_AND_REFUND"
    CANNOT_CANCEL_AFTER_DISPATCH = "CANNOT_CANCEL_AFTER_DISPATCH"
    WAIT_AND_TRACK = "WAIT_AND_TRACK"
    OPEN_SHIPPING_INVESTIGATION = "OPEN_SHIPPING_INVESTIGATION"
    OFFER_REPLACEMENT_OR_REFUND = "OFFER_REPLACEMENT_OR_REFUND"
    NEEDS_MORE_INFORMATION = "NEEDS_MORE_INFORMATION"


# --- Auth ---


class UserRegister(BaseModel):
    email: EmailStr
    # max_length=72 isn't arbitrary: bcrypt (see src/auth.py) only uses the
    # first 72 bytes of a password and silently ignores the rest, so two
    # different long passwords sharing the same first 72 bytes would hash
    # identically. Capping input length here surfaces that as a validation
    # error instead of a silent footgun.
    password: str = Field(min_length=8, max_length=72)

    @field_validator("password")
    @classmethod
    def password_complexity(cls, v: str) -> str:
        if not re.search(r"[A-Za-z]", v):
            raise ValueError("Password must contain at least one letter.")
        if not re.search(r"\d", v):
            raise ValueError("Password must contain at least one number.")
        return v


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: int
    email: str
    created_at: datetime

    model_config = {"from_attributes": True}


# --- Tickets ---


class TicketCreate(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    order_value_inr: Optional[float] = Field(default=None, ge=0)
    days_since_delivery: Optional[int] = Field(default=None, ge=0)
    days_since_dispatch: Optional[int] = Field(default=None, ge=0)
    product_type: Optional[ProductType] = None
    opened_status: Optional[OpenedStatus] = None
    order_status: Optional[OrderStatus] = None

    @field_validator("product_type", "opened_status", "order_status", mode="before")
    @classmethod
    def normalize_lower(cls, v):
        return v.lower().strip() if isinstance(v, str) else v


class DecisionResponse(BaseModel):
    id: int
    action: ActionEnum
    reason: str
    confidence: float
    sources: list[str]
    provider: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}

    @field_validator("sources", mode="before")
    @classmethod
    def parse_sources(cls, v):
        if isinstance(v, str):
            import json

            return json.loads(v)
        return v


class TicketResponse(BaseModel):
    id: int
    message: str
    order_value_inr: Optional[float]
    days_since_delivery: Optional[int]
    days_since_dispatch: Optional[int]
    product_type: Optional[str]
    opened_status: Optional[str]
    order_status: Optional[str]
    created_at: datetime
    decision: Optional[DecisionResponse] = None

    model_config = {"from_attributes": True}


class TicketListItem(BaseModel):
    id: int
    message: str
    created_at: datetime
    action: Optional[ActionEnum] = None
    confidence: Optional[float] = None

    model_config = {"from_attributes": True}


# --- LLM structured output contract ---


class LLMDecision(BaseModel):
    """Schema the LLM's raw JSON output must satisfy before it is trusted.

    `provider` is not part of the LLM's own output - it's set by
    src/decision.py afterward, to record which path actually produced this
    decision (gemini/groq/cache/retrieval_gate/fallback). Surfaced to the
    frontend so a Groq fallback or cache hit is visible, not silent.
    """

    action: ActionEnum
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=1000)
    sources: list[str] = Field(default_factory=list)
    provider: Optional[str] = None
