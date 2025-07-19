# app/routers/users.py
from fastapi import APIRouter, Depends, Body
from typing import Dict
from app.dependencies import get_current_user
from app.database import users_collection, ObjectId
from app.utils.security import cipher
from app.config import DEFAULT_MODELS
from app.services.ai import call_ai
from requests import get
from huggingface_hub import list_models
import google.generativeai as genai
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["users"])

@router.post("/preferences")
async def update_preferences(preferences: Dict = Body(...), current_user: dict = Depends(get_current_user)):
    encrypted_api_keys = {}
    for provider, key in preferences.get("api_keys", {}).items():
        if key:
            encrypted_api_keys[provider] = cipher.encrypt(key.encode()).decode()
        else:
            encrypted_api_keys[provider] = ""
    preferences["api_keys"] = encrypted_api_keys
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
    await users_collection.update_one({"_id": ObjectId(current_user["_id"])}, {"$set": {"preferences": preferences}})
    return {"message": "Preferences updated"}

@router.get("/preferences")
async def get_preferences(current_user: dict = Depends(get_current_user)):
    prefs = current_user.get("preferences", {})
    prefs["models"] = {**DEFAULT_MODELS, **prefs.get("models", {})}
    return prefs

@router.get("/models")
async def get_models(provider: str, current_user: dict = Depends(get_current_user)):
    api_key = current_user["preferences"]["api_keys"].get(provider)
    if not api_key and provider != "CoinGecko":
        raise HTTPException(status_code=400, detail=f"No API key for {provider}")
    models = []
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
        else:
            raise HTTPException(status_code=400, detail="Unsupported provider")
    except Exception as e:
        logger.error(f"Failed to fetch models for {provider}: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to fetch models for {provider}: {str(e)}")
    return {"models": models}

@router.get("/profile")
async def get_user_profile(current_user: dict = Depends(get_current_user)):
    return {
        "firstName": current_user.get("first_name", ""),
        "lastName": current_user.get("last_name", ""),
        "tier": current_user.get("tier", "free"),
        "profileImage": current_user.get("profile_image", ""),
        "email": current_user.get("email", "")
    }