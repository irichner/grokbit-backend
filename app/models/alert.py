# app/models/alert.py
from pydantic import BaseModel, field_validator
from typing import List, Optional

class RuleAlert(BaseModel):
    coin: str
    condition: str
    value: float
    triggered: bool = False

    @field_validator("coin")
    @classmethod
    def validate_coin(cls, v):
        from app.services.price import COIN_ID_MAP, KNOWN_IDS
        if v.upper() not in COIN_ID_MAP and v.upper() not in KNOWN_IDS:
            raise ValueError("Invalid coin symbol")
        return v.upper()

    @field_validator("condition")
    @classmethod
    def validate_condition(cls, v):
        if v not in [">", "<"]:
            raise ValueError("Condition must be '>' or '<'")
        return v

    @field_validator("value")
    @classmethod
    def validate_value(cls, v):
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

class AIAlert(BaseModel):
    prompt: str
    result: Optional[str] = None
    triggered: bool = False

class AlertsRequest(BaseModel):
    rule_alerts: List[RuleAlert]
    ai_alerts: List[AIAlert]

class AlertCheckRequest(BaseModel):
    prompt: str