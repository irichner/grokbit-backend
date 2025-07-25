# app/routers/insights.py
from fastapi import APIRouter, Depends, Body, HTTPException
from fastapi.responses import StreamingResponse
from app.models.other import InsightRequest
from app.dependencies import get_current_user
from app.services.ai import call_ai
from app.services.price import get_price
from cachetools import TTLCache
from typing import Dict
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/insights", tags=["insights"])

sentiment_cache = TTLCache(maxsize=100, ttl=300)

@router.post("")
async def get_insights(request: InsightRequest, current_user: dict = Depends(get_current_user)):
    if not request.coin:
        raise HTTPException(status_code=400, detail="Coin required")
    user_prefs = current_user.get("preferences", {})
    is_summary = request.coin.startswith("Summarize this user prompt in 3-4 words: ")
    selected_provider = user_prefs["summary_default_provider"] if is_summary else user_prefs["prompt_default_provider"]
    api_key = user_prefs["api_keys"].get(selected_provider)
    model = user_prefs.get("models", {}).get(selected_provider)
    messages = [
        {"role": "system", "content": "You are a crypto market assistant."},
        {"role": "user", "content": request.coin}
    ]
    try:
        if request.stream:
            async def stream_gen():
                gen = await call_ai(selected_provider, messages, api_key, model, stream=True)
                for chunk in gen:
                    yield chunk
            return StreamingResponse(stream_gen(), media_type="text/plain")
        else:
            insight = await call_ai(selected_provider, messages, api_key, model, stream=False)
            return {"insight": insight}
    except Exception as e:
        logger.error(f"AI call failed for {selected_provider}: {e}")
        raise HTTPException(status_code=503, detail=f"API error: {str(e)}")

@router.get("/sentiment")
async def get_sentiment(coin: str, current_user: dict = Depends(get_current_user)):
    cache_key = f"sentiment_{coin}"
    if cache_key in sentiment_cache:
        return sentiment_cache[cache_key]
    # Placeholder for X search
    tweets = []
    sentiment_prompt = f"Analyze sentiment for {coin} from these tweets: {tweets}"
    insight = await call_ai(current_user['preferences']['prompt_default_provider'], [{"role": "user", "content": sentiment_prompt}], current_user['preferences']['api_keys'][current_user['preferences']['prompt_default_provider']])
    score = 0.8
    result = {"score": score, "summary": insight}
    sentiment_cache[cache_key] = result
    return result

@router.get("/predictions")
async def get_predictions(coin: str, current_user: dict = Depends(get_current_user)):
    messages = [{"role": "user", "content": f"Predict market for {coin}"}]
    prediction = await call_ai(current_user['preferences']['prompt_default_provider'], messages, current_user['preferences']['api_keys'][current_user['preferences']['prompt_default_provider']])
    return {"predictions": [prediction]}

@router.post("/recommendations")
async def get_recommendations(request: Dict = Body(...), current_user: dict = Depends(get_current_user)):
    risk = request.get("risk")
    portfolio = request.get("portfolio")
    messages = [{"role": "user", "content": f"Recommend investments for {risk} risk with portfolio {portfolio}"}]
    rec = await call_ai(current_user['preferences']['prompt_default_provider'], messages, current_user['preferences']['api_keys'][current_user['preferences']['prompt_default_provider']])
    return {"recommendations": [rec]}

@router.get("/tutorials")
async def get_tutorials(topic: str, current_user: dict = Depends(get_current_user)):
    messages = [{"role": "user", "content": f"Generate tutorial for {topic}"}]
    tut = await call_ai(current_user['preferences']['prompt_default_provider'], messages, current_user['preferences']['api_keys'][current_user['preferences']['prompt_default_provider']])
    return {"tutorials": [tut]}

@router.get("/nfts")
async def get_nfts(coin: str, current_user: dict = Depends(get_current_user)):
    messages = [{"role": "user", "content": f"Value NFTs related to {coin}"}]
    val = await call_ai(current_user['preferences']['prompt_default_provider'], messages, current_user['preferences']['api_keys'][current_user['preferences']['prompt_default_provider']])
    return {"valuations": [val]}

@router.get("/defi")
async def get_defi(coin: str, current_user: dict = Depends(get_current_user)):
    messages = [{"role": "user", "content": f"DeFi strategies for {coin}"}]
    strat = await call_ai(current_user['preferences']['prompt_default_provider'], messages, current_user['preferences']['api_keys'][current_user['preferences']['prompt_default_provider']])
    return {"strategies": [strat]}

@router.post("/query")
async def get_query(request: Dict = Body(...), current_user: dict = Depends(get_current_user)):
    query = request.get("query")
    messages = [{"role": "user", "content": query}]
    res = await call_ai(current_user['preferences']['prompt_default_provider'], messages, current_user['preferences']['api_keys'][current_user['preferences']['prompt_default_provider']])
    return {"results": [res]}

@router.get("/risk")
async def get_risk(token: str, current_user: dict = Depends(get_current_user)):
    messages = [{"role": "user", "content": f"Assess risk for {token}"}]
    ass = await call_ai(current_user['preferences']['prompt_default_provider'], messages, current_user['preferences']['api_keys'][current_user['preferences']['prompt_default_provider']])
    return {"assessments": [ass]}

@router.get("/charts")
async def get_charts(coin: str, current_user: dict = Depends(get_current_user)):
    messages = [{"role": "user", "content": f"Analyze chart for {coin}"}]
    ana = await call_ai(current_user['preferences']['prompt_default_provider'], messages, current_user['preferences']['api_keys'][current_user['preferences']['prompt_default_provider']])
    return {"analyses": [ana]}

@router.get("/bots")
async def get_bots(current_user: dict = Depends(get_current_user)):
    return {"configs": []}

@router.get("/fraud")
async def get_fraud(current_user: dict = Depends(get_current_user)):
    return {"alerts": []}

@router.get("/forum")
async def get_forum(current_user: dict = Depends(get_current_user)):
    return {"posts": []}

@router.post("/forum")
async def post_forum(request: Dict = Body(...), current_user: dict = Depends(get_current_user)):
    return {"success": True}