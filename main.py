# main.py
from fastapi import FastAPI, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import requests
import json
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from typing import List, Optional, Dict
from groq import Groq
import google.generativeai as genai
from huggingface_hub import InferenceClient
from cryptography.fernet import Fernet, InvalidToken

load_dotenv()

app = FastAPI()

# CORS setup (preserved from current)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Explicitly allow all for dev; tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB setup
MONGO_URI = os.getenv("MONGO_URI")  # Set in .env or Render
client = AsyncIOMotorClient(MONGO_URI)
db = client.grokbit
users_collection = db.users
portfolios_collection = db.portfolios
alerts_collection = db.alerts  # New collection for alerts

# No more server fallback keys
# Removed GROQ_API_KEY etc.

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")  # For JWT
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")  # For API key encryption; base64-encoded Fernet key
CRYPTO_RANK_API_KEY = os.getenv("CRYPTO_RANK_API_KEY")  # Optional for CryptoRank fallback
if not ENCRYPTION_KEY:
    raise ValueError("ENCRYPTION_KEY must be set in env")
cipher = Fernet(ENCRYPTION_KEY.encode())

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

COIN_ID_MAP = {}  # Global map for CoinGecko symbol to id

KNOWN_IDS = {
    'BTC': 'bitcoin',
    'ETH': 'ethereum',
    'XRP': 'ripple',
    # Add more common coins if needed, e.g., 'USDT': 'tether'
}

@app.on_event("startup")
async def startup_event():
    global COIN_ID_MAP
    try:
        response = requests.get("https://api.coingecko.com/api/v3/coins/list")
        response.raise_for_status()
        COIN_ID_MAP = {item["symbol"].upper(): item["id"] for item in response.json()}
    except Exception as e:
        print(f"Failed to load CoinGecko coin list: {e}")

async def get_price(coin: str) -> float:
    coin_upper = coin.upper()
    cg_id = KNOWN_IDS.get(coin_upper) or COIN_ID_MAP.get(coin_upper)
    if not cg_id:
        return 0.0
    try:
        response = requests.get(f"https://api.coingecko.com/api/v3/simple/price?ids={cg_id}&vs_currencies=usd")
        response.raise_for_status()
        if response.status_code == 429:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        data = response.json()
        return data.get(cg_id, {}).get("usd", 0.0)
    except Exception:
        # Fallback to Binance
        try:
            response = requests.get(f"https://api.binance.com/api/v3/ticker/price?symbol={coin_upper}USDT")
            response.raise_for_status()
            if response.status_code == 429:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            data = response.json()
            return float(data.get("price", 0.0))
        except Exception:
            # Fallback to CryptoRank (requires API key)
            try:
                if not CRYPTO_RANK_API_KEY:
                    return 0.0
                response = requests.get(f"https://api.cryptorank.io/v1/currencies?api_key={CRYPTO_RANK_API_KEY}&symbols={coin_upper}")
                response.raise_for_status()
                if response.status_code == 429:
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")
                data = response.json().get("data", [])
                if data:
                    return data[0].get("values", {}).get("USD", {}).get("price", 0.0)
                return 0.0
            except Exception:
                return 0.0

class User(BaseModel):
    first_name: str
    last_name: str
    username: str
    hashed_password: str
    preferences: Dict = {"default_provider": "Groq", "api_keys": {"Groq": "", "Gemini": "", "HuggingFace": "", "Grok": ""}, "prompts": [], "portfolio_prompts": [], "alert_prompts": []}

class Token(BaseModel):
    access_token: str
    token_type: str

class InsightRequest(BaseModel):
    coin: str

class PortfolioItem(BaseModel):
    coin: str
    quantity: float
    cost_basis: float

class PortfolioRequest(BaseModel):
    portfolio: List[PortfolioItem]

# New Models for Alerts
class RuleAlert(BaseModel):
    coin: str
    condition: str  # '>' or '<'
    value: float
    triggered: bool = False

class AIAlert(BaseModel):
    prompt: str
    result: Optional[str] = None
    triggered: bool = False

class AlertsRequest(BaseModel):
    rule_alerts: List[RuleAlert]
    ai_alerts: List[AIAlert]

class AlertCheckRequest(BaseModel):
    prompt: str

# Auth helpers
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(status_code=401, detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = await users_collection.find_one({"username": username})
    if not user:
        raise credentials_exception
    # Decrypt API keys on fetch
    if "preferences" in user and "api_keys" in user["preferences"]:
        decrypted_keys = {}
        for provider, enc_key in user["preferences"]["api_keys"].items():
            if enc_key:
                try:
                    decrypted_keys[provider] = cipher.decrypt(enc_key.encode()).decode()
                except InvalidToken:
                    decrypted_keys[provider] = ""  # Handle invalid/old keys gracefully
            else:
                decrypted_keys[provider] = ""
        user["preferences"]["api_keys"] = decrypted_keys
    # Ensure prompts is always a list
    if "preferences" in user and "prompts" not in user["preferences"]:
        user["preferences"]["prompts"] = []
    if "preferences" in user and "portfolio_prompts" not in user["preferences"]:
        user["preferences"]["portfolio_prompts"] = []
    if "preferences" in user and "alert_prompts" not in user["preferences"]:
        user["preferences"]["alert_prompts"] = []
    return user

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# AI call helper (uses user key if provided, else server fallback)
async def call_ai(provider: str, messages: list, user_api_key: Optional[str] = None):
    api_key = user_api_key
    if not api_key:
        raise ValueError(f"{provider} API key required")
    if provider == "Groq":
        groq_client = Groq(api_key=api_key)
        response = groq_client.chat.completions.create(messages=messages, model="llama-3.1-8b-instant", stream=False, temperature=0)
        return response.choices[0].message.content
    elif provider == "Gemini":
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel("gemini-2.5-flash")
        response = gemini_model.generate_content(messages[-1]["content"])
        return response.text
    elif provider == "HuggingFace":
        hf_client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.3", token=api_key)
        return hf_client.text_generation(messages[-1]["content"], max_new_tokens=500)
    elif provider == "Grok":
        url = "https://api.x.ai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"messages": messages, "model": "grok-4", "stream": False, "temperature": 0}
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "No response")
    raise ValueError("Invalid provider")

@app.post("/register")
async def register(user_data: Dict = Body(...)):
    first_name = user_data.get("first_name")
    last_name = user_data.get("last_name")
    username = user_data.get("username")
    password = user_data.get("password")
    if await users_collection.find_one({"username": username}):
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = pwd_context.hash(password)
    # Fetch irichner preferences
    irichner = await users_collection.find_one({"username": "irichner"})
    default_prefs = {
        "default_provider": "Groq",
        "api_keys": {"Groq": "", "Gemini": "", "HuggingFace": "", "Grok": ""},
        "prompts": [],
        "portfolio_prompts": [],
        "alert_prompts": []
    }
    if irichner and "preferences" in irichner:
        prefs = irichner["preferences"]
        default_prefs["prompts"] = prefs.get("prompts", [])
        default_prefs["portfolio_prompts"] = prefs.get("portfolio_prompts", [])
        default_prefs["alert_prompts"] = prefs.get("alert_prompts", [])
        # api_keys remain empty
    user_dict = {
        "first_name": first_name,
        "last_name": last_name,
        "username": username,
        "hashed_password": hashed_password,
        "preferences": default_prefs
    }
    await users_collection.insert_one(user_dict)
    return {"message": "User registered"}

@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await users_collection.find_one({"username": form_data.username})
    if not user or not pwd_context.verify(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    access_token = create_access_token({"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/preferences")
async def update_preferences(preferences: Dict = Body(...), current_user: dict = Depends(get_current_user)):
    # Encrypt API keys before saving
    encrypted_api_keys = {}
    for provider, key in preferences.get("api_keys", {}).items():
        if key:
            encrypted_api_keys[provider] = cipher.encrypt(key.encode()).decode()
        else:
            encrypted_api_keys[provider] = ""
    preferences["api_keys"] = encrypted_api_keys
    # Ensure prompts is a list
    if "prompts" not in preferences:
        preferences["prompts"] = []
    if "portfolio_prompts" not in preferences:
        preferences["portfolio_prompts"] = []
    if "alert_prompts" not in preferences:
        preferences["alert_prompts"] = []
    await users_collection.update_one({"_id": ObjectId(current_user["_id"])}, {"$set": {"preferences": preferences}})
    return {"message": "Preferences updated"}

@app.get("/preferences")
async def get_preferences(current_user: dict = Depends(get_current_user)):
    return current_user.get("preferences", {"default_provider": "Groq", "api_keys": {"Groq": "", "Gemini": "", "HuggingFace": "", "Grok": ""}, "prompts": [], "portfolio_prompts": [], "alert_prompts": []})

@app.get("/portfolio")
async def get_portfolio(current_user: dict = Depends(get_current_user)):
    portfolio = await portfolios_collection.find_one({"user_id": str(current_user["_id"])})
    items = portfolio.get("items", []) if portfolio else []
    # For backward compatibility, map old 'amount' to 'quantity' and set cost_basis to 0 if missing
    updated_items = []
    for item in items:
        updated_item = {
            "coin": item.get("coin", ""),
            "quantity": item.get("quantity", item.get("amount", 0)),
            "cost_basis": item.get("cost_basis", 0)
        }
        updated_items.append(updated_item)
    return {"portfolio": updated_items}

@app.post("/portfolio/save")
async def save_portfolio(request: PortfolioRequest, current_user: dict = Depends(get_current_user)):
    if not request.portfolio:
        return {"message": "Portfolio saved"}  # Empty is allowed
    portfolio_dict = {"user_id": str(current_user["_id"]), "items": [item.dict() for item in request.portfolio], "updated_at": datetime.utcnow()}
    await portfolios_collection.replace_one({"user_id": str(current_user["_id"])}, portfolio_dict, upsert=True)
    return {"message": "Portfolio saved"}

@app.post("/insights")
async def get_insights(request: InsightRequest, current_user: dict = Depends(get_current_user)):
    if not request.coin:
        raise HTTPException(status_code=400, detail="Coin required")
    user_prefs = current_user.get("preferences", {"default_provider": "Groq", "api_keys": {"Groq": "", "Gemini": "", "HuggingFace": "", "Grok": ""}})
    api_key = user_prefs["api_keys"].get(user_prefs["default_provider"])
    messages = [
        {"role": "system", "content": "You are a crypto market assistant."},
        {"role": "user", "content": request.coin}
    ]
    try:
        insight = await call_ai(user_prefs["default_provider"], messages, api_key)
        return {"insight": insight}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"API error: {str(e)}")

@app.post("/portfolio")
async def get_portfolio_insight(request: PortfolioRequest, current_user: dict = Depends(get_current_user)):
    if not request.portfolio:
        raise HTTPException(status_code=400, detail="Portfolio required")
    # Save to DB
    portfolio_dict = {"user_id": str(current_user["_id"]), "items": [item.dict() for item in request.portfolio], "updated_at": datetime.utcnow()}
    await portfolios_collection.replace_one({"user_id": str(current_user["_id"])}, portfolio_dict, upsert=True)
    
    # Calculate total_value
    total_value = 0.0
    portfolio_str = "Your portfolio: "
    for item in request.portfolio:
        price = await get_price(item.coin)
        value = price * item.quantity
        total_value += value
        portfolio_str += f"{item.quantity} {item.coin} (${value:.2f}, cost basis ${item.cost_basis:.2f}), "
    portfolio_str = portfolio_str.rstrip(", ") + "."

    user_prefs = current_user.get("preferences", {"default_provider": "Groq", "api_keys": {"Groq": "", "Gemini": "", "HuggingFace": "", "Grok": ""}})
    api_key = user_prefs["api_keys"].get(user_prefs["default_provider"])
    messages = [
        {"role": "system", "content": "You are a crypto portfolio advisor."},
        {"role": "user", "content": f"{portfolio_str} Total value: ${total_value:.2f}. Provide a concise suggestion under 1000 characters."}
    ]
    try:
        suggestion = await call_ai(user_prefs["default_provider"], messages, api_key)
        return {"total_value": total_value, "suggestion": suggestion}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"API error: {str(e)}")

@app.post("/prices")
async def get_prices(request: Dict = Body(...), current_user: dict = Depends(get_current_user)):
    coins = request.get("coins", [])
    result = {}
    if not coins:
        return result

    # Batch for CoinGecko
    try:
        cg_ids = [KNOWN_IDS.get(c.upper()) or COIN_ID_MAP.get(c.upper()) for c in coins if KNOWN_IDS.get(c.upper()) or COIN_ID_MAP.get(c.upper())]
        if cg_ids:
            ids_str = ','.join(cg_ids)
            response = requests.get(f"https://api.coingecko.com/api/v3/simple/price?ids={ids_str}&vs_currencies=usd")
            response.raise_for_status()
            data = response.json()
            for original_coin, cg_id in zip([c for c in coins if KNOWN_IDS.get(c.upper()) or COIN_ID_MAP.get(c.upper())], cg_ids):
                result[original_coin] = data.get(cg_id, {}).get("usd", 0.0)
    except Exception as e:
        print(f"CoinGecko batch failed: {e}")

    # For missing prices, fallback to Binance batch
    missing_coins = [c for c in coins if result.get(c, 0.0) == 0.0]
    if missing_coins:
        try:
            symbols = [c.upper() + 'USDT' for c in missing_coins]
            symbols_str = json.dumps(symbols)
            response = requests.get(f"https://api.binance.com/api/v3/ticker/price?symbols={symbols_str}")
            response.raise_for_status()
            data = response.json()
            for item in data:
                symbol = item.get('symbol', '')[:-4]  # Remove USDT
                price = float(item.get('price', 0.0))
                if symbol in [m.upper() for m in missing_coins]:
                    result[symbol] = price
        except Exception as e:
            print(f"Binance batch failed: {e}")

    # For still missing, fallback to CryptoRank batch
    missing_coins = [c for c in coins if result.get(c, 0.0) == 0.0]
    if missing_coins and CRYPTO_RANK_API_KEY:
        try:
            symbols_str = ','.join([m.upper() for m in missing_coins])
            response = requests.get(f"https://api.cryptorank.io/v1/currencies?api_key={CRYPTO_RANK_API_KEY}&symbols={symbols_str}")
            response.raise_for_status()
            data = response.json().get("data", [])
            for item in data:
                symbol = item.get('symbol', '').upper()
                price = item.get("values", {}).get("USD", {}).get("price", 0.0)
                if symbol in [m.upper() for m in missing_coins]:
                    result[symbol] = price
        except Exception as e:
            print(f"CryptoRank batch failed: {e}")

    # If any still 0, fallback to per-coin original get_price (though batches should cover)
    for coin in coins:
        if result.get(coin, 0.0) == 0.0:
            result[coin] = await get_price(coin)

    return result

@app.get("/coins")
async def get_coins(current_user: dict = Depends(get_current_user)):
    return list(COIN_ID_MAP.keys())

# New: Get Alerts
@app.get("/alerts")
async def get_alerts(current_user: dict = Depends(get_current_user)):
    alerts = await alerts_collection.find_one({"user_id": str(current_user["_id"])})
    return {
        "rule_alerts": alerts.get("rule_alerts", []),
        "ai_alerts": alerts.get("ai_alerts", []),
        "triggered_alerts": alerts.get("triggered_alerts", [])  # For history
    }

# New: Save Alerts
@app.post("/alerts/save")
async def save_alerts(request: Dict = Body(...), current_user: dict = Depends(get_current_user)):
    alerts_dict = {
        "user_id": str(current_user["_id"]),
        "rule_alerts": request.get("rule_alerts", []),
        "ai_alerts": request.get("ai_alerts", []),
        "updated_at": datetime.utcnow()
    }
    await alerts_collection.replace_one({"user_id": str(current_user["_id"])}, alerts_dict, upsert=True)
    return {"message": "Alerts saved"}

# New: Check AI Alert (Creative logic: Simulate trigger if AI detects 'alert' worthy content)
@app.post("/alerts/check")
async def check_ai_alert(request: AlertCheckRequest, current_user: dict = Depends(get_current_user)):
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt required")
    user_prefs = current_user.get("preferences", {"default_provider": "Groq", "api_keys": {"Groq": "", "Gemini": "", "HuggingFace": "", "Grok": ""}})
    api_key = user_prefs["api_keys"].get(user_prefs["default_provider"])
    messages = [
        {"role": "system", "content": "You are a creative crypto alert generator. Analyze current trends, sentiment, or anomalies. If significant change detected, start response with 'ALERT:' followed by witty, useful message with action suggestion. Else, respond 'No alert'."},
        {"role": "user", "content": request.prompt}
    ]
    try:
        response = await call_ai(user_prefs["default_provider"], messages, api_key)
        if response.startswith("ALERT:"):
            return {"alert": response}  # Triggered
        return {"alert": None}  # Not triggered
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"API error: {str(e)}")

# New: Delete User
@app.post("/delete_user")
async def delete_user(delete_data: Dict = Body(...), current_user: dict = Depends(get_current_user)):
    username = delete_data.get("username")
    password = delete_data.get("password")
    if username != current_user["username"] or not pwd_context.verify(password, current_user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    # Delete user
    await users_collection.delete_one({"_id": ObjectId(current_user["_id"])})
    # Delete associated data
    await portfolios_collection.delete_one({"user_id": str(current_user["_id"])})
    await alerts_collection.delete_one({"user_id": str(current_user["_id"])})
    return {"message": "Account deleted"}

# Preserve root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to GrokBit API"}