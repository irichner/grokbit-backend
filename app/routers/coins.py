# app/routers/coins.py
from fastapi import APIRouter, Depends, HTTPException
import requests
import logging
from app.dependencies import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/coins", tags=["coins"])

@router.get("/markets")
async def get_coins_markets(vs_currency: str = "usd", ids: str = "", current_user: dict = Depends(get_current_user)):
    coingecko_key = current_user["preferences"]["api_keys"].get("CoinGecko")
    try:
        url = f"https://api.coingecko.com/api/v3/coins/markets?vs_currency={vs_currency}"
        if ids:
            url += f"&ids={ids}"
        headers = {"x-cg-demo-api-key": coingecko_key} if coingecko_key else None
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to fetch coins markets: {e}")
        raise HTTPException(status_code=503, detail="Failed to fetch coins markets")