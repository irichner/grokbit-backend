# app/dependencies.py
from fastapi import HTTPException, Cookie
from jose import JWTError, jwt
from bson import ObjectId
from typing import Optional
from datetime import datetime, timedelta
from app.config import SECRET_KEY, ALGORITHM, cipher, users_collection, DEFAULT_MODELS

async def get_current_user(grokbit_token: str = Cookie(None)):
    credentials_exception = HTTPException(status_code=401, detail="Could not validate credentials")
    if not grokbit_token:
        raise credentials_exception
    try:
        payload = jwt.decode(grokbit_token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = await users_collection.find_one({"username": username})
    if not user:
        raise credentials_exception
    # Decrypt API keys
    if "preferences" in user and "api_keys" in user["preferences"]:
        decrypted_keys = {}
        for provider, enc_key in user["preferences"]["api_keys"].items():
            if enc_key:
                try:
                    decrypted_keys[provider] = cipher.decrypt(enc_key.encode()).decode()
                except InvalidToken:
                    decrypted_keys[provider] = ""
            else:
                decrypted_keys[provider] = ""
        user["preferences"]["api_keys"] = decrypted_keys
    # Ensure lists and models
    if "preferences" in user:
        user["preferences"]["prompts"] = user["preferences"].get("prompts", [])
        user["preferences"]["portfolio_prompts"] = user["preferences"].get("portfolio_prompts", [])
        user["preferences"]["alert_prompts"] = user["preferences"].get("alert_prompts", [])
        user["preferences"]["models"] = {**DEFAULT_MODELS, **user["preferences"].get("models", {})}
        user["preferences"]["prompt_default_provider"] = user["preferences"].get("prompt_default_provider", "Groq")
        user["preferences"]["summary_default_provider"] = user["preferences"].get("summary_default_provider", "Groq")
        user["preferences"]["refresh_rate"] = user["preferences"].get("refresh_rate", 60000)
        user["preferences"]["market_coins"] = user["preferences"].get("market_coins", [])
    return user