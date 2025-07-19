# app/routers/portfolio.py
from fastapi import APIRouter, Depends, Body, HTTPException
from app.models.portfolio import PortfolioRequest
from app.dependencies import get_current_user
from app.database import portfolios_collection, ObjectId
from app.services.price import get_price
from datetime import datetime
from app.services.ai import call_ai
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/portfolio", tags=["portfolio"])

@router.get("")
async def get_portfolio(current_user: dict = Depends(get_current_user)):
    portfolio = await portfolios_collection.find_one({"user_id": str(current_user["_id"])})
    items = portfolio.get("items", []) if portfolio else []
    updated_items = []
    for item in items:
        updated_item = {
            "coin": item.get("coin", ""),
            "quantity": item.get("quantity", item.get("amount", 0)),
            "cost_basis": item.get("cost_basis", 0)
        }
        updated_items.append(updated_item)
    return {"portfolio": updated_items}

@router.get("/history")
async def get_portfolio_history(current_user: dict = Depends(get_current_user)):
    return []

@router.post("/save")
async def save_portfolio(request: PortfolioRequest, current_user: dict = Depends(get_current_user)):
    if not request.portfolio:
        return {"message": "Portfolio saved"}
    portfolio_dict = {
        "user_id": str(current_user["_id"]),
        "items": [item.dict() for item in request.portfolio],
        "updated_at": datetime.utcnow()
    }
    await portfolios_collection.replace_one({"user_id": str(current_user["_id"])}, portfolio_dict, upsert=True)
    return {"message": "Portfolio saved"}

@router.post("")
async def get_portfolio_insight(request: PortfolioRequest, current_user: dict = Depends(get_current_user)):
    if not request.portfolio:
        raise HTTPException(status_code=400, detail="Portfolio required")
    portfolio_dict = {
        "user_id": str(current_user["_id"]),
        "items": [item.dict() for item in request.portfolio],
        "updated_at": datetime.utcnow()
    }
    await portfolios_collection.replace_one({"user_id": str(current_user["_id"])}, portfolio_dict, upsert=True)
    
    total_value = 0.0
    portfolio_str = "Your portfolio: "
    for item in request.portfolio:
        price_data = await get_price(item.coin)
        price = price_data['price']
        value = price * item.quantity
        total_value += value
        portfolio_str += f"{item.quantity} {item.coin} (${value:.2f}, cost basis ${item.cost_basis:.2f}), "
    portfolio_str = portfolio_str.rstrip(", ") + "."
    user_prefs = current_user.get("preferences", {})
    api_key = user_prefs["api_keys"].get(user_prefs["prompt_default_provider"])
    model = user_prefs.get("models", {}).get(user_prefs["prompt_default_provider"])
    messages = [
        {"role": "system", "content": "You are a crypto portfolio advisor."},
        {"role": "user", "content": f"{portfolio_str} Total value: ${total_value:.2f}. Provide a concise suggestion under 1000 characters."}
    ]
    try:
        suggestion = await call_ai(user_prefs["prompt_default_provider"], messages, api_key, model)
        return {"total_value": total_value, "suggestion": suggestion}
    except Exception as e:
        logger.error(f"Portfolio insight call failed: {e}")
        raise HTTPException(status_code=503, detail=f"API error: {str(e)}")

@router.post("/optimizations")
async def get_optimizations(request: PortfolioRequest, current_user: dict = Depends(get_current_user)):
    messages = [{"role": "user", "content": f"Optimize portfolio {request.portfolio}"}]
    opt = await call_ai(current_user['preferences']['prompt_default_provider'], messages, current_user['preferences']['api_keys'][current_user['preferences']['prompt_default_provider']])
    return {"optimizations": [opt]}

@router.post("/tax")
async def get_tax(request: PortfolioRequest, current_user: dict = Depends(get_current_user)):
    messages = [{"role": "user", "content": f"Generate tax report for {request.portfolio}"}]
    rep = await call_ai(current_user['preferences']['prompt_default_provider'], messages, current_user['preferences']['api_keys'][current_user['preferences']['prompt_default_provider']])
    return {"reports": [rep]}