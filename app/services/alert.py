# app/services/alert.py
from pywebpush import webpush, WebPushException
import json
from app.config import VAPID_PRIVATE_KEY, VAPID_PUBLIC_KEY
from app.database import users_collection
from app.services.ai import call_ai
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

async def send_push(user_id, message):
    user = await users_collection.find_one({"_id": user_id})
    sub = user.get("push_sub")
    if sub:
        try:
            webpush(
                subscription_info=sub,
                data=json.dumps({"title": "GrokBit Alert", "body": message}),
                vapid_private_key=VAPID_PRIVATE_KEY,
                vapid_public_key=VAPID_PUBLIC_KEY,
                vapid_claims={"sub": "mailto:israel.richner@gmail.com"}
            )
        except WebPushException as e:
            logger.error(f"Push failed: {e}")

async def check_ai_alert(prompt: str, current_user: dict):
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt required")
    user_prefs = current_user.get("preferences", {})
    api_key = user_prefs["api_keys"].get(user_prefs["prompt_default_provider"])
    model = user_prefs.get("models", {}).get(user_prefs["prompt_default_provider"])
    messages = [
        {"role": "system", "content": "You are a creative crypto alert generator. Analyze current trends, sentiment, or anomalies. If significant change detected, start response with 'ALERT:' followed by witty, useful message with action suggestion. Else, respond 'No alert'."},
        {"role": "user", "content": prompt}
    ]
    try:
        response = await call_ai(user_prefs["prompt_default_provider"], messages, api_key, model)
        if response.startswith("ALERT:"):
            await send_push(current_user["_id"], response)
            return {"alert": response}
        return {"alert": None}
    except Exception as e:
        logger.error(f"AI alert check failed: {e}")
        raise HTTPException(status_code=503, detail=f"API error: {str(e)}")