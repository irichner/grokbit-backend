# app/routers/admin.py
from fastapi import APIRouter, Depends, Body, HTTPException
from typing import Dict
from app.dependencies import get_current_user

router = APIRouter(prefix="/admin", tags=["admin"])

@router.get("/defaults")
async def get_defaults(current_user: dict = Depends(get_current_user)):
    if current_user.get("email") != "israel.richner@gmail.com":
        raise HTTPException(status_code=403, detail="Access denied")
    prefs = current_user.get("preferences", {})
    return {
        "prompts": prefs.get("prompts", []),
        "portfolio_prompts": prefs.get("portfolio_prompts", []),
        "alert_prompts": prefs.get("alert_prompts", []),
        "market_coins": prefs.get("market_coins", [])
    }

@router.post("/defaults")
async def update_defaults(defaults: Dict = Body(...), current_user: dict = Depends(get_current_user)):
    if current_user.get("email") != "israel.richner@gmail.com":
        raise HTTPException(status_code=403, detail="Access denied")
    prefs = current_user.get("preferences", {})
    prefs["prompts"] = defaults.get("prompts", [])
    prefs["portfolio_prompts"] = defaults.get("portfolio_prompts", [])
    prefs["alert_prompts"] = defaults.get("alert_prompts", [])
    prefs["market_coins"] = defaults.get("market_coins", [])
    await users_collection.update_one({"_id": current_user["_id"]}, {"$set": {"preferences": prefs}})
    return {"message": "Defaults updated"}