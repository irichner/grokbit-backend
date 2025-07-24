# app/routers/alerts.py
from fastapi import APIRouter, Depends, Body
from app.models.alert import AlertsRequest, AlertCheckRequest
from app.dependencies import get_current_user
from app.database import alerts_collection, ObjectId
from datetime import datetime
from app.services.alert import check_ai_alert, send_push
from typing import Dict

router = APIRouter(prefix="/alerts", tags=["alerts"])

@router.get("")
async def get_alerts(current_user: dict = Depends(get_current_user)):
    alerts = await alerts_collection.find_one({"user_id": str(current_user["_id"])})
    return {
        "rule_alerts": alerts.get("rule_alerts", []),
        "ai_alerts": alerts.get("ai_alerts", []),
        "triggered_alerts": alerts.get("triggered_alerts", [])
    }

@router.post("/save")
async def save_alerts(request: AlertsRequest, current_user: dict = Depends(get_current_user)):
    alerts_dict = {
        "user_id": str(current_user["_id"]),
        "rule_alerts": [alert.dict() for alert in request.rule_alerts],
        "ai_alerts": [alert.dict() for alert in request.ai_alerts],
        "updated_at": datetime.utcnow()
    }
    await alerts_collection.replace_one({"user_id": str(current_user["_id"])}, alerts_dict, upsert=True)
    return {"message": "Alerts saved"}

@router.post("/check")
async def check_ai_alert_route(request: AlertCheckRequest, current_user: dict = Depends(get_current_user)):
    return await check_ai_alert(request.prompt, current_user)

@router.options("/push/subscribe")
async def options_push_subscribe():
    return {"status": "ok"}

@router.post("/push/subscribe")
async def push_subscribe(sub: Dict = Body(...), current_user: dict = Depends(get_current_user)):
    await users_collection.update_one({"_id": current_user["_id"]}, {"$set": {"push_sub": sub}})
    return {"success": True}