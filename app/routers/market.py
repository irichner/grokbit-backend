# app/routers/market.py
from fastapi import APIRouter, Depends, Body, HTTPException
from app.dependencies import get_current_user
from app.services.price import COIN_ID_MAP, get_price
from typing import Dict
import requests
import json
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/market", tags=["market"])

@router.post("/prices")
async def get_prices(request: Dict = Body(...), current_user: dict = Depends(get_current_user)):
    coins = request.get("coins", [])
    result = {}
    if not coins:
        return result
    coingecko_key = current_user["preferences"]["api_keys"].get("CoinGecko")
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
    coingecko_key = current_user["preferences"]["api_keys"].get("CoinGecko")
    try:
        headers = {"x-cg-demo-api-key": coingecko_key} if coingecko_key else None
        response = requests.get("https://api.coingecko.com/api/v3/coins/list", headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to fetch coins list: {e}")
        raise HTTPException(status_code=503, detail="Failed to fetch coins list")

@router.get("/coins/markets")
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