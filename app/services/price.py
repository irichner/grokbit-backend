# app/services/price.py
import requests
from fastapi import HTTPException
from cachetools import TTLCache
from app.config import CRYPTO_RANK_API_KEY
import logging

logger = logging.getLogger(__name__)

price_cache = TTLCache(maxsize=100, ttl=60)

COIN_ID_MAP = {}
KNOWN_IDS = {
    'BTC': 'bitcoin',
    'ETH': 'ethereum',
    'XRP': 'ripple',
    'USDT': 'tether',
    'BNB': 'bnb',
    'SOL': 'solana',
}

async def startup_load_coins():
    global COIN_ID_MAP
    try:
        response = requests.get("https://api.coingecko.com/api/v3/coins/list")
        response.raise_for_status()
        COIN_ID_MAP = {item["symbol"].upper(): item["id"] for item in response.json()}
        for symbol, id in KNOWN_IDS.items():
            COIN_ID_MAP[symbol] = id
        logger.info("Successfully loaded CoinGecko coin list")
    except Exception as e:
        logger.error(f"Failed to load CoinGecko coin list: {e}")

async def get_price(coin: str, coingecko_key: str = None) -> dict:
    coin_upper = coin.upper()
    cache_key = coin_upper
    if cache_key in price_cache:
        return price_cache[cache_key]
    cg_id = KNOWN_IDS.get(coin_upper) or COIN_ID_MAP.get(coin_upper)
    if not cg_id:
        logger.warning(f"No CoinGecko ID for coin: {coin_upper}")
        return {'price': 0.0, 'change24h': 0.0}
    try:
        headers = {"x-cg-demo-api-key": coingecko_key} if coingecko_key else None
        response = requests.get(f"https://api.coingecko.com/api/v3/simple/price?ids={cg_id}&vs_currencies=usd&include_24hr_change=true", headers=headers)
        response.raise_for_status()
        if response.status_code == 429:
            raise HTTPException(status_code=429, detail="Rate limit exceeded for CoinGecko API. Please try again later.")
        data = response.json()
        price = data.get(cg_id, {}).get("usd", 0.0)
        change24h = data.get(cg_id, {}).get("usd_24h_change", 0.0)
        price_cache[cache_key] = {'price': price, 'change24h': change24h}
        return price_cache[cache_key]
    except Exception as e:
        logger.error(f"CoinGecko price fetch failed for {coin_upper}: {e}")
        try:
            response = requests.get(f"https://api.binance.com/api/v3/ticker/24hr?symbol={coin_upper}USDT")
            response.raise_for_status()
            if response.status_code == 429:
                raise HTTPException(status_code=429, detail="Rate limit exceeded for Binance API. Please try again later.")
            data = response.json()
            price = float(data.get("lastPrice", 0.0))
            change24h = float(data.get("priceChangePercent", 0.0))
            price_cache[cache_key] = {'price': price, 'change24h': change24h}
            return price_cache[cache_key]
        except Exception as e:
            logger.error(f"Binance price fetch failed for {coin_upper}: {e}")
            if not CRYPTO_RANK_API_KEY:
                logger.warning("CRYPTO_RANK_API_KEY is missing, skipping CryptoRank API")
                return {'price': 0.0, 'change24h': 0.0}
            try:
                response = requests.get(f"https://api.cryptorank.io/v1/currencies?api_key={CRYPTO_RANK_API_KEY}&symbols={coin_upper}")
                response.raise_for_status()
                data = response.json().get("data", [])
                if data:
                    price = data[0].get("values", {}).get("USD", {}).get("price", 0.0)
                    change24h = data[0].get("values", {}).get("USD", {}).get("percentChange24h", 0.0)
                    price_cache[cache_key] = {'price': price, 'change24h': change24h}
                    return price_cache[cache_key]
                return {'price': 0.0, 'change24h': 0.0}
            except Exception as e:
                logger.error(f"CryptoRank price fetch failed for {coin_upper}: {e}")
                return {'price': 0.0, 'change24h': 0.0}