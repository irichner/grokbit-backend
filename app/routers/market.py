# app/routers/market.py
from fastapi import APIRouter, Depends, Body, HTTPException
from app.dependencies import get_current_user
from app.services.price import COIN_ID_MAP, get_price
from typing import Dict, List
import requests
import json
import logging
from app.utils.security import cipher
import redis
import os

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/market", tags=["market"])

redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
redis_client = redis.Redis.from_url(redis_url)

KNOWN_IDS = {
    'BNB': 'bnb',
    'BTC': 'bitcoin',
    'ETH': 'ethereum',
    'USDT': 'tether',
    'SOL': 'solana',
    'DOGE': 'dogecoin'
}

def get_coingecko_config(user):
    provider = "CoinGecko"
    api_key_enc = user["preferences"]["api_keys"].get(provider, '')
    api_key = ''
    if api_key_enc:
        try:
            api_key = cipher.decrypt(api_key_enc.encode()).decode()
        except:
            api_key = api_key_enc
    plan = user["preferences"].get("market_details", {}).get(provider, {}).get("plan", "free")
    rate_limit = user["preferences"].get("market_details", {}).get(provider, {}).get("rate_limit", 10)
    if plan == "free":
        base = "https://api.coingecko.com/api/v3"
        headers = {}
    elif plan == "demo":
        base = "https://api.coingecko.com/api/v3"
        headers = {"x-cg-demo-api-key": api_key}
    else:
        base = "https://pro-api.coingecko.com/api/v3"
        headers = {"x-cg-pro-api-key": api_key}
    return base, headers, plan, rate_limit

@router.post("/prices")
async def get_prices(request: Dict = Body(...), current_user: dict = Depends(get_current_user)):
    coins = request.get("coins", [])
    result = {}
    if not coins:
        return result
    base, headers, plan, rate_limit = get_coingecko_config(current_user)
    try:
        cg_ids = [KNOWN_IDS.get(c.upper()) or COIN_ID_MAP.get(c.upper()) for c in coins if KNOWN_IDS.get(c.upper()) or COIN_ID_MAP.get(c.upper())]
        if cg_ids:
            await check_cg_rate_limit(current_user, base, plan, rate_limit)
            ids_str = ','.join(cg_ids)
            response = requests.get(f"{base}/simple/price?ids={ids_str}&vs_currencies=usd&include_24hr_change=true", headers=headers)
            response.raise_for_status()
            if response.status_code == 429:
                raise HTTPException(status_code=429, detail="Rate limit exceeded for CoinGecko API. Please try again later.")
            data = response.json()
            for original_coin, cg_id in zip([c for c in coins if KNOWN_IDS.get(c.upper()) or COIN_ID_MAP.get(c.upper())], cg_ids):
                price = data.get(cg_id, {}).get("usd", 0.0)
                change24h = data.get(cg_id, {}).get("usd_24h_change", 0.0)
                result[original_coin] = {'price': price, 'change24h': change24h}
    except Exception as e:
        logger.error(f"CoinGecko batch failed: {e}")
    missing_coins = [c for c in coins if c not in result or result[c]['price'] == 0.0]
    if missing_coins:
        try:
            symbols = [c.upper() + 'USDT' for c in missing_coins]
            symbols_str = json.dumps(symbols)
            response = requests.get(f"https://api.binance.com/api/v3/ticker/24hr?symbols={symbols_str}")
            response.raise_for_status()
            if response.status_code == 429:
                raise HTTPException(status_code=429, detail="Rate limit exceeded for Binance API. Please try again later.")
            data = response.json()
            for item in data:
                symbol = item.get('symbol', '')[:-4]
                price = float(item.get('lastPrice', 0.0))
                change24h = float(item.get("priceChangePercent", 0.0))
                if symbol in [m.upper() for m in missing_coins]:
                    result[symbol] = {'price': price, 'change24h': change24h}
        except Exception as e:
            logger.error(f"Binance batch failed: {e}")
    missing_coins = [c for c in coins if c not in result or result[c]['price'] == 0.0]
    if missing_coins:
        from app.config import CRYPTO_RANK_API_KEY
        if not CRYPTO_RANK_API_KEY:
            logger.warning("CRYPTO_RANK_API_KEY is missing, skipping CryptoRank API")
        else:
            try:
                symbols_str = ','.join([m.upper() for m in missing_coins])
                response = requests.get(f"https://api.cryptorank.io/v1/currencies?api_key={CRYPTO_RANK_API_KEY}&symbols={symbols_str}")
                response.raise_for_status()
                data = response.json().get("data", [])
                for item in data:
                    symbol = item.get('symbol', '').upper()
                    price = item.get("values", {}).get("USD", {}).get("price", 0.0)
                    change24h = item.get("values", {}).get("USD", {}).get("percentChange24h", 0.0)
                    if symbol in [m.upper() for m in missing_coins]:
                        result[symbol] = {'price': price, 'change24h': change24h}
            except Exception as e:
                logger.error(f"CryptoRank batch failed: {e}")
    for coin in coins:
        if coin not in result or result[coin]['price'] == 0.0:
            result[coin] = await get_price(coin, api_key)
    return result

@router.get("/coins")
async def get_coins(current_user: dict = Depends(get_current_user)):
    return list(COIN_ID_MAP.keys())

@router.get("/coins/list")
async def get_coins_list(current_user: dict = Depends(get_current_user)):
    base, headers, plan, rate_limit = get_coingecko_config(current_user)
    try:
        await check_cg_rate_limit(current_user, base, plan, rate_limit)
        response = requests.get(f"{base}/coins/list", headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to fetch coins list: {e}")
        raise HTTPException(status_code=503, detail="Failed to fetch coins list")

@router.post("/coins/markets")
async def get_coins_markets(data: Dict = Body(...), current_user: dict = Depends(get_current_user)):
    ids = data.get('ids', [])
    vs_currency = data.get('vs_currency', 'usd')
    base, headers, plan, rate_limit = get_coingecko_config(current_user)
    try:
        results = []
        missing_ids = []
        id_to_index = {id_: idx for idx, id_ in enumerate(ids)}
        for id_ in ids:
            key = f"market:{id_}:{vs_currency}"
            cached = None
            try:
                cached = redis_client.get(key)
            except Exception as redis_err:
                logger.error(f"Redis error: {redis_err}")
            if cached:
                results.append((id_, json.loads(cached)))
            else:
                missing_ids.append(id_)
        if missing_ids:
            for i in range(0, len(missing_ids), 250):
                batch = missing_ids[i:i+250]
                batch_str = ','.join(batch)
                url = f"{base}/coins/markets?vs_currency={vs_currency}&ids={batch_str}&price_change_percentage=1h%2C24h%2C7d"
                await check_cg_rate_limit(current_user, base, plan, rate_limit)
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                batch_data = response.json()
                for item in batch_data:
                    item_key = f"market:{item['id']}:{vs_currency}"
                    try:
                        redis_client.set(item_key, json.dumps(item), ex=30)
                    except Exception as redis_err:
                        logger.error(f"Redis error: {redis_err}")
                    results.append((item['id'], item))
        results.sort(key=lambda x: id_to_index[x[0]])
        final_results = [data for id_, data in results]
        return final_results
    except Exception as e:
        logger.error(f"Failed to fetch coins markets: {e}")
        raise HTTPException(status_code=503, detail="Failed to fetch coins markets")

@router.get("/global")
async def get_global(current_user: dict = Depends(get_current_user)):
    key = "market:global"
    cached = None
    try:
        cached = redis_client.get(key)
    except Exception as redis_err:
        logger.error(f"Redis error: {redis_err}")
    if cached:
        return json.loads(cached)
    base, headers, plan, rate_limit = get_coingecko_config(current_user)
    try:
        await check_cg_rate_limit(current_user, base, plan, rate_limit)
        url = f"{base}/global"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        try:
            redis_client.set(key, json.dumps(data), ex=30)
        except Exception as redis_err:
            logger.error(f"Redis error: {redis_err}")
        return data
    except Exception as e:
        logger.error(f"Failed to fetch global data: {e}")
        raise HTTPException(status_code=503, detail="Failed to fetch global data")
    
async def check_cg_rate_limit(user, base, plan, rate_limit):
    if 'coingecko.com' not in base or plan not in ['free', 'demo']:
        return
    effective_limit = rate_limit // 2
    user_id = str(user['_id']) if plan == 'demo' else 'free'
    key = f"cg_rate:{user_id}"
    pipe = redis_client.pipeline()
    pipe.incr(key)
    pipe.expire(key, 60)
    count = pipe.execute()[0]
    if count > effective_limit:
        raise HTTPException(429, "Internal rate limit exceeded; try again later.")