# main.py
from fastapi import FastAPI, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import requests
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

# Env vars for free APIs (set in Render) - server fallbacks
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
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
    cg_id = COIN_ID_MAP.get(coin_upper)
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
    preferences: Dict = {"default_provider": "Groq", "api_keys": {"Groq": "", "Gemini": "", "HuggingFace": "", "Grok": ""}, "prompts": [], "portfolio_prompts": []}

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
    return user

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# AI call helper (uses user key if provided, else server fallback)
async def call_ai(provider: str, messages: list, user_api_key: Optional[str] = None):
    api_key = user_api_key
    if provider == "Groq":
        api_key = api_key or GROQ_API_KEY
        if not api_key:
            raise ValueError("Groq API key required")
        groq_client = Groq(api_key=api_key)
        response = groq_client.chat.completions.create(messages=messages, model="llama-3.1-8b-instant", stream=False, temperature=0)
        return response.choices[0].message.content
    elif provider == "Gemini":
        api_key = api_key or GEMINI_API_KEY
        if not api_key:
            raise ValueError("Gemini API key required")
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel("gemini-2.5-flash")
        response = gemini_model.generate_content(messages[-1]["content"])
        return response.text
    elif provider == "HuggingFace":
        api_key = api_key or HF_TOKEN
        if not api_key:
            raise ValueError("HuggingFace token required")
        hf_client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.3", token=api_key)
        return hf_client.text_generation(messages[-1]["content"], max_new_tokens=500)
    elif provider == "Grok":
        api_key = api_key or os.getenv("XAI_API_KEY")
        if not api_key:
            raise ValueError("Grok API key required")
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
    user_dict = {
        "first_name": first_name,
        "last_name": last_name,
        "username": username,
        "hashed_password": hashed_password,
        "preferences": {"default_provider": "Groq", "api_keys": {"Groq": "", "Gemini": "", "HuggingFace": "", "Grok": ""}, "prompts": [], "portfolio_prompts": []}
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
    await users_collection.update_one({"_id": ObjectId(current_user["_id"])}, {"$set": {"preferences": preferences}})
    return {"message": "Preferences updated"}

@app.get("/preferences")
async def get_preferences(current_user: dict = Depends(get_current_user)):
    return current_user.get("preferences", {"default_provider": "Groq", "api_keys": {"Groq": "", "Gemini": "", "HuggingFace": "", "Grok": ""}, "prompts": [], "portfolio_prompts": []})

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
    for coin in coins:
        result[coin] = await get_price(coin)
    return result

@app.get("/coins")
async def get_coins(current_user: dict = Depends(get_current_user)):
    return list(COIN_ID_MAP.keys())

# Preserve root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to GrokBit API"}