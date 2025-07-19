# app/models/other.py
from pydantic import BaseModel
from typing import Optional

class InsightRequest(BaseModel):
    coin: str
    provider: Optional[str] = None