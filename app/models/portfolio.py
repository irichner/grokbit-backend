# app/models/portfolio.py
from pydantic import BaseModel, field_validator
from typing import List

class PortfolioItem(BaseModel):
    coin: str
    quantity: float
    cost_basis: float

    @field_validator("coin")
    @classmethod
    def validate_coin(cls, v):
        from app.services.price import COIN_ID_MAP, KNOWN_IDS
        if v.upper() not in COIN_ID_MAP and v.upper() not in KNOWN_IDS:
            raise ValueError("Invalid coin symbol")
        return v.upper()

    @field_validator("quantity", "cost_basis")
    @classmethod
    def validate_non_negative(cls, v):
        if v < 0:
            raise ValueError("Quantity and cost basis must be non-negative")
        return v

class PortfolioRequest(BaseModel):
    portfolio: List[PortfolioItem]