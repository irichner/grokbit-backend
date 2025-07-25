# app/models/other.py
from pydantic import BaseModel
from typing import Optional

class InsightRequest(BaseModel):
    coin: str
    provider: Optional[str] = None
    stream: Optional[bool] = False

# Added from OriginalMain.py
class ForgotPasswordRequest(BaseModel):
    email: str

class ResetPasswordRequest(BaseModel):
    token: str
    password: str