# app/models/user.py
from pydantic import BaseModel, field_validator
import re
from typing import Optional, Dict, List

class User(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    username: str
    hashed_password: Optional[str] = None
    email: Optional[str] = None
    preferences: Dict = {"prompt_default_provider": "Groq", "summary_default_provider": "Groq", "api_keys": {"Groq": "", "Gemini": "", "HuggingFace": "", "Grok": "", "CoinGecko": ""}, "prompts": [], "portfolio_prompts": [], "alert_prompts": [], "models": {}, "refresh_rate": 60000, "market_coins": []}
    oauth_providers: Optional[Dict] = {}
    oauth_only: Optional[bool] = False
    tier: str = 'free'
    profile_image: Optional[str] = ''

    @field_validator("username")
    @classmethod
    def validate_username(cls, v):
        if not re.match(r"^[a-zA-Z0-9_]{3,20}$", v):
            raise ValueError("Username must be 3-20 characters, alphanumeric or underscore")
        return v

    @field_validator("email")
    @classmethod
    def validate_email(cls, v):
        if v and not re.match(r"[^@]+@[^@]+\.[^@]+", v):
            raise ValueError("Invalid email format")
        return v

class Token(BaseModel):
    access_token: str
    token_type: str