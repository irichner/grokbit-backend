# app/routers/news.py
from fastapi import APIRouter
import logging

from app.services.news import get_crypto_news

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/news", tags=["news"])

@router.get("/")
def fetch_news(coin: str = None, limit: int = 10):
    logger.info(f"News request received: coin={coin}, limit={limit}")
    news = get_crypto_news(coin, limit)
    return {"news": news, "count": len(news), "coin": coin}