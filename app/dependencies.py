# app/dependencies.py
from fastapi import HTTPException, Depends, Cookie
from jose import JWTError, jwt
from app.config import SECRET_KEY, ALGORITHM
from app.database import users_collection, ObjectId
from cryptography.fernet import Fernet, InvalidToken
from app.config import ENCRYPTION_KEY, DEFAULT_MODELS
from app.utils.security import cipher

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