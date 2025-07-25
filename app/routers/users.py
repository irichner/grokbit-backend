# app/routers/users.py
from fastapi import APIRouter, Depends, Body, Header
from typing import Dict
from app.dependencies import get_current_user
from app.database import users_collection, ObjectId
from app.utils.security import cipher
from app.config import DEFAULT_MODELS
from requests import get
from huggingface_hub import list_models
import google.generativeai as genai
from fastapi import HTTPException
import logging
from pydantic import BaseModel
from cryptography.fernet import InvalidToken

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["users"])

class MarketCoin(BaseModel):
    symbol: str

def get_coingecko_details(api_key: str) -> Dict:
    details = {}
    if not api_key:
        details = {"plan": "free", "rate_limit": 10, "monthly_credits": "Unlimited", "usage": 0}
        return details
    # Try pro
    pro_base = "https://pro-api.coingecko.com/api/v3"
    headers_pro = {"x-cg-pro-api-key": api_key}
    resp_pro_ping = get(f"{pro_base}/ping", headers=headers_pro)
    if resp_pro_ping.ok:
        resp_pro_key = get(f"{pro_base}/key", headers=headers_pro)
        if resp_pro_key.ok:
            data = resp_pro_key.json()
            plan = data.get("plan_name", "unknown")
            monthly = data.get("monthly_credit_limit", 0)
            usage = data.get("usage_this_month", 0)
            rate_map = {
                "analyst": 500,
                "lite": 500,
                "pro": 1000,
                "enterprise": 1250,
            }
            rate_limit = rate_map.get(plan, 500)
            details = {"plan": plan, "rate_limit": rate_limit, "monthly_credits": monthly, "usage": usage}
        else:
            details = {"plan": "pro_unknown", "rate_limit": 500, "monthly_credits": "Unknown", "usage": "Unknown"}
    else:
        # Try demo
        demo_base = "https://api.coingecko.com/api/v3"
        headers_demo = {"x-cg-demo-api-key": api_key}
        resp_demo_ping = get(f"{demo_base}/ping", headers=headers_demo)
        if resp_demo_ping.ok:
            details = {"plan": "demo", "rate_limit": 30, "monthly_credits": 10000, "usage": "N/A"}
        else:
            details = {"plan": "free", "rate_limit": 10, "monthly_credits": "Unlimited", "usage": 0}
    return details

@router.post("/preferences")
async def update_preferences(preferences: Dict = Body(...), current_user: dict = Depends(get_current_user)):
    encrypted_api_keys = {}
    market_details = preferences.get("market_details", {})
    for provider, key in preferences.get("api_keys", {}).items():
        if provider == "CoinGecko":
            details = get_coingecko_details(key)
            market_details[provider] = details
            if key and details["plan"] != "free":
                encrypted_api_keys[provider] = cipher.encrypt(key.encode()).decode()
            else:
                encrypted_api_keys[provider] = ""
        else:
            if key:
                try:
                    encrypted_api_keys[provider] = cipher.encrypt(key.encode()).decode()
                    logger.info(f"Encrypted key for {provider}")
                except Exception as e:
                    logger.error(f"Encryption failed for {provider}: {str(e)}")
                    encrypted_api_keys[provider] = key  # fallback to plain if encryption fails
            else:
                encrypted_api_keys[provider] = ""
    # Clean up market_details for removed providers
    for prov in list(market_details.keys()):
        if prov not in encrypted_api_keys:
            del market_details[prov]
    preferences["api_keys"] = encrypted_api_keys
    preferences["market_details"] = market_details
    if "prompts" not in preferences:
        preferences["prompts"] = []
    if "portfolio_prompts" not in preferences:
        preferences["portfolio_prompts"] = []
    if "alert_prompts" not in preferences:
        preferences["alert_prompts"] = []
    if "models" not in preferences:
        preferences["models"] = DEFAULT_MODELS
    if "prompt_default_provider" not in preferences:
        preferences["prompt_default_provider"] = "Groq"
    if "summary_default_provider" not in preferences:
        preferences["summary_default_provider"] = "Groq"
    if "refresh_rate" not in preferences:
        preferences["refresh_rate"] = 60000
    if "market_coins" not in preferences:
        preferences["market_coins"] = []
    if "market_details" not in preferences:
        preferences["market_details"] = {}
    await users_collection.update_one({"_id": ObjectId(current_user["_id"])}, {"$set": {"preferences": preferences}})
    return {"message": "Preferences updated"}

@router.get("/preferences")
async def get_preferences(current_user: dict = Depends(get_current_user)):
    prefs = current_user.get("preferences", {})
    prefs["models"] = {**DEFAULT_MODELS, **prefs.get("models", {})}
    prefs["market_details"] = prefs.get("market_details", {})
    return prefs

@router.get("/models")
async def get_models(provider: str, x_api_key: str = Header(None), current_user: dict = Depends(get_current_user)):
    api_key = None
    if x_api_key:
        api_key = x_api_key
        logger.info(f"Using provided X-API-Key for {provider}")
    else:
        api_key_enc = current_user["preferences"]["api_keys"].get(provider)
        if api_key_enc:
            logger.info(f"Attempting decryption for {provider}, enc: {api_key_enc}")
            if api_key_enc.startswith('gAAAAA'):
                try:
                    api_key = cipher.decrypt(api_key_enc.encode()).decode()
                    logger.info(f"Decryption successful for {provider}")
                except InvalidToken:
                    logger.error(f"Decryption failed with InvalidToken for {provider}")
            else:
                api_key = api_key_enc
                logger.warning(f"Using plain API key for {provider}")
    if not api_key and provider != "CoinGecko":
        raise HTTPException(status_code=400, detail=f"No valid API key for {provider}")
    models = []
    details = {}
    try:
        if provider == "Groq":
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            response = get("https://api.groq.com/openai/v1/models", headers=headers)
            response.raise_for_status()
            data = response.json()
            models = [m["id"] for m in data.get("data", []) if m.get("active", True)]
        elif provider == "Gemini":
            genai.configure(api_key=api_key)
            gen_models = genai.list_models()
            models = [m.name for m in gen_models]
        elif provider == "HuggingFace":
            hf_models = list_models(limit=50, sort="downloads", direction=-1)
            models = [m.modelId for m in hf_models]
        elif provider == "Grok":
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            response = get("https://api.x.ai/v1/models", headers=headers)
            response.raise_for_status()
            data = response.json()
            models = [m["id"] for m in data.get("data", [])]
        elif provider == "CoinGecko":
            models = []
            details = get_coingecko_details(api_key)
            if api_key and details["plan"] == "free":
                raise HTTPException(status_code=400, detail="Invalid CoinGecko API key")
        else:
            raise HTTPException(status_code=400, detail="Unsupported provider")
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Failed to fetch models for {provider}: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to fetch models for {provider}: {str(e)}")
    return {"models": models, "details": details}

@router.get("/profile")
async def get_user_profile(current_user: dict = Depends(get_current_user)):
    return {
        "firstName": current_user.get("first_name", ""),
        "lastName": current_user.get("last_name", ""),
        "tier": current_user.get("tier", "free"),
        "profileImage": current_user.get("profile_image", ""),
        "email": current_user.get("email", "")
    }

@router.post("/market_coin")
async def add_market_coin(coin: MarketCoin, current_user: dict = Depends(get_current_user)):
    user_id = ObjectId(current_user["_id"])
    prefs = current_user["preferences"]
    market_coins = prefs.get("market_coins", [])
    upper = coin.symbol.upper()
    if upper not in market_coins:
        market_coins.append(upper)
        await users_collection.update_one({"_id": user_id}, {"$set": {"preferences.market_coins": market_coins}})
    return {"message": "Coin added"}

@router.delete("/market_coin/{symbol}")
async def remove_market_coin(symbol: str, current_user: dict = Depends(get_current_user)):
    user_id = ObjectId(current_user["_id"])
    prefs = current_user["preferences"]
    market_coins = prefs.get("market_coins", [])
    upper = symbol.upper()
    if upper in market_coins:
        market_coins.remove(upper)
        await users_collection.update_one({"_id": user_id}, {"$set": {"preferences.market_coins": market_coins}})
    return {"message": "Coin removed"}