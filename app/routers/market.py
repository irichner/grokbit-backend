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

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/market", tags=["market"])

redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

KNOWN_IDS = {
    'BNB': 'bnb',
    'BTC': 'bitcoin',
    'ETH': 'ethereum',
    'USDT': 'tether',
    'SOL': 'solana'
}

@router.post("/prices")
async def get_prices(request: Dict = Body(...), current_user: dict = Depends(get_current_user)):
    coins = request.get("coins", [])
    result = {}
    if not coins:
        return result
    coingecko_key = current_user["preferences"]["api_keys"].get("CoinGecko", '')
    if coingecko_key:
        try:
            coingecko_key = cipher.decrypt(coingecko_key.encode()).decode()
        except:
            pass  # Use as is if decryption fails
    try:
        cg_ids = [KNOWN_IDS.get(c.upper()) or COIN_ID_MAP.get(c.upper()) for c in coins if KNOWN_IDS.get(c.upper()) or COIN_ID_MAP.get(c.upper())]
        if cg_ids:
            ids_str = ','.join(cg_ids)
            headers = {"x-cg-demo-api-key": coingecko_key} if coingecko_key else None
            response = requests.get(f"https://api.coingecko.com/api/v3/simple/price?ids={ids_str}&vs_currencies=usd&include_24hr_change=true", headers=headers)
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
            result[coin] = await get_price(coin, coingecko_key)
    return result

@router.get("/coins")
async def get_coins(current_user: dict = Depends(get_current_user)):
    return list(COIN_ID_MAP.keys())

@router.get("/coins/list")
async def get_coins_list(current_user: dict = Depends(get_current_user)):
    coingecko_key = current_user["preferences"]["api_keys"].get("CoinGecko", '')
    if coingecko_key:
        try:
            coingecko_key = cipher.decrypt(coingecko_key.encode()).decode()
        except:
            pass  # Use as is if decryption fails
    try:
        headers = {"x-cg-demo-api-key": coingecko_key} if coingecko_key else None
        response = requests.get("https://api.coingecko.com/api/v3/coins/list", headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to fetch coins list: {e}")
        raise HTTPException(status_code=503, detail="Failed to fetch coins list")

@router.post("/coins/markets")
async def get_coins_markets(data: Dict = Body(...), current_user: dict = Depends(get_current_user)):
    ids = data.get('ids', [])
    vs_currency = data.get('vs_currency', 'usd')
    coingecko_key = current_user["preferences"]["api_keys"].get("CoinGecko", '')
    if coingecko_key:
        try:
            coingecko_key = cipher.decrypt(coingecko_key.encode()).decode()
        except:
            pass  # Use as is if decryption fails
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
            headers = {"x-cg-demo-api-key": coingecko_key} if coingecko_key else None
            for i in range(0, len(missing_ids), 250):
                batch = missing_ids[i:i+250]
                batch_str = ','.join(batch)
                url = f"https://api.coingecko.com/api/v3/coins/markets?vs_currency={vs_currency}&ids={batch_str}&price_change_percentage=1h%2C24h%2C7d"
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
    coingecko_key = current_user["preferences"]["api_keys"].get("CoinGecko", '')
    if coingecko_key:
        try:
            coingecko_key = cipher.decrypt(coingecko_key.encode()).decode()
        except:
            pass  # Use as is if decryption fails
    try:
        url = "https://api.coingecko.com/api/v3/global"
        headers = {"x-cg-demo-api-key": coingecko_key} if coingecko_key else None
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