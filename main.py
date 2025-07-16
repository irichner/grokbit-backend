import asyncio
import logging
import re
from fastapi import FastAPI, HTTPException, Depends, Body, Request, Response, Cookie
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, validator
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
import time
from huggingface_hub import list_models
from authlib.integrations.starlette_client import OAuth
from starlette.middleware.sessions import SessionMiddleware
import secrets
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from cachetools import TTLCache
import redis
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from pywebpush import webpush, WebPushException
import stripe

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL", "https://grokbit.ai")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session middleware for OAuth state and PKCE
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY"))

# OAuth setup
oauth = OAuth()
providers = ['google', 'github']

# Provider configurations
oauth.register(
    name='google',
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_id=os.getenv('CLIENT_ID_GOOGLE'),
    client_secret=os.getenv('CLIENT_SECRET_GOOGLE'),
    client_kwargs={'scope': 'openid email profile', 'code_challenge_method': 'S256'}
)

oauth.register(
    name='github',
    client_id=os.getenv('CLIENT_ID_GITHUB'),
    client_secret=os.getenv('CLIENT_SECRET_GITHUB'),
    authorize_url='https://github.com/login/oauth/authorize',
    access_token_url='https://github.com/login/oauth/access_token',
    client_kwargs={'scope': 'user:email', 'code_challenge_method': 'S256'},
    userinfo_endpoint='https://api.github.com/user'
)

# MongoDB setup
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MONGO_URI must be set in environment variables")
client = AsyncIOMotorClient(MONGO_URI)
db = client.grokbit
users_collection = db.users
portfolios_collection = db.portfolios
alerts_collection = db.alerts

# Ensure MongoDB indexes
async def create_indexes():
    await users_collection.create_index("username", unique=True)
    await users_collection.create_index("oauth_providers.google.sub")
    await users_collection.create_index("oauth_providers.github.sub")
    await portfolios_collection.create_index("user_id")
    await alerts_collection.create_index("user_id")

SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("SECRET_KEY must be set in environment variables")
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
if not ENCRYPTION_KEY:
    raise ValueError("ENCRYPTION_KEY must be set in environment variables")
cipher = Fernet(ENCRYPTION_KEY.encode())

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Price cache
price_cache = TTLCache(maxsize=100, ttl=60)

COIN_ID_MAP = {}
KNOWN_IDS = {
    'BTC': 'bitcoin',
    'ETH': 'ethereum',
    'XRP': 'ripple',
}

DEFAULT_MODELS = {
    "Groq": os.getenv("DEFAULT_MODEL_GROQ", "llama-3.1-8b-instant"),
    "Gemini": os.getenv("DEFAULT_MODEL_GEMINI", "gemini-2.5-flash"),
    "HuggingFace": os.getenv("DEFAULT_MODEL_HF", "mistralai/Mistral-7B-Instruct-v0.3"),
    "Grok": os.getenv("DEFAULT_MODEL_GROK", "grok-4-0709")
}

providers_order = ["Groq", "Gemini", "HuggingFace", "Grok"]

@app.on_event("startup")
async def startup_event():
    global COIN_ID_MAP
    try:
        response = requests.get("https://api.coingecko.com/api/v3/coins/list")
        response.raise_for_status()
        COIN_ID_MAP = {item["symbol"].upper(): item["id"] for item in response.json()}
        logger.info("Successfully loaded CoinGecko coin list")
    except Exception as e:
        logger.error(f"Failed to load CoinGecko coin list: {e}")
    await create_indexes()

async def get_price(coin: str) -> float:
    coin_upper = coin.upper()
    cache_key = coin_upper
    if cache_key in price_cache:
        return price_cache[cache_key]
    cg_id = KNOWN_IDS.get(coin_upper) or COIN_ID_MAP.get(coin_upper)
    if not cg_id:
        logger.warning(f"No CoinGecko ID for coin: {coin_upper}")
        return 0.0
    try:
        response = requests.get(f"https://api.coingecko.com/api/v3/simple/price?ids={cg_id}&vs_currencies=usd")
        response.raise_for_status()
        if response.status_code == 429:
            raise HTTPException(status_code=429, detail="Rate limit exceeded for CoinGecko API. Please try again later.")
        data = response.json()
        price = data.get(cg_id, {}).get("usd", 0.0)
        price_cache[cache_key] = price
        return price
    except Exception as e:
        logger.error(f"CoinGecko price fetch failed for {coin_upper}: {e}")
        try:
            response = requests.get(f"https://api.binance.com/api/v3/ticker/price?symbol={coin_upper}USDT")
            response.raise_for_status()
            if response.status_code == 429:
                raise HTTPException(status_code=429, detail="Rate limit exceeded for Binance API. Please try again later.")
            data = response.json()
            price = float(data.get("price", 0.0))
            price_cache[cache_key] = price
            return price
        except Exception as e:
            logger.error(f"Binance price fetch failed for {coin_upper}: {e}")
            CRYPTO_RANK_API_KEY = os.getenv("CRYPTO_RANK_API_KEY")
            if not CRYPTO_RANK_API_KEY:
                logger.warning("CRYPTO_RANK_API_KEY is missing, skipping CryptoRank API")
                return 0.0
            try:
                response = requests.get(f"https://api.cryptorank.io/v1/currencies?api_key={CRYPTO_RANK_API_KEY}&symbols={coin_upper}")
                response.raise_for_status()
                data = response.json().get("data", [])
                if data:
                    price = data[0].get("values", {}).get("USD", {}).get("price", 0.0)
                    price_cache[cache_key] = price
                    return price
                return 0.0
            except Exception as e:
                logger.error(f"CryptoRank price fetch failed for {coin_upper}: {e}")
                return 0.0

class User(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    username: str
    hashed_password: Optional[str] = None
    email: Optional[str] = None
    preferences: Dict = {"default_provider": "Groq", "api_keys": {"Groq": "", "Gemini": "", "HuggingFace": "", "Grok": ""}, "prompts": [], "portfolio_prompts": [], "alert_prompts": [], "models": DEFAULT_MODELS}
    oauth_providers: Optional[Dict] = {}
    oauth_only: Optional[bool] = False
    tier: str = 'free'

    @validator("username")
    def validate_username(cls, v):
        if not re.match(r"^[a-zA-Z0-9_]{3,20}$", v):
            raise ValueError("Username must be 3-20 characters, alphanumeric or underscore")
        return v

    @validator("email", check_fields=False)
    def validate_email(cls, v):
        if v and not re.match(r"[^@]+@[^@]+\.[^@]+", v):
            raise ValueError("Invalid email format")
        return v

class Token(BaseModel):
    access_token: str
    token_type: str

class InsightRequest(BaseModel):
    coin: str
    provider: Optional[str] = None

class PortfolioItem(BaseModel):
    coin: str
    quantity: float
    cost_basis: float

    @validator("coin")
    def validate_coin(cls, v):
        if v.upper() not in COIN_ID_MAP and v.upper() not in KNOWN_IDS:
            raise ValueError("Invalid coin symbol")
        return v.upper()

    @validator("quantity", "cost_basis")
    def validate_non_negative(cls, v):
        if v < 0:
            raise ValueError("Quantity and cost basis must be non-negative")
        return v

class PortfolioRequest(BaseModel):
    portfolio: List[PortfolioItem]

class RuleAlert(BaseModel):
    coin: str
    condition: str
    value: float
    triggered: bool = False

    @validator("coin")
    def validate_coin(cls, v):
        if v.upper() not in COIN_ID_MAP and v.upper() not in KNOWN_IDS:
            raise ValueError("Invalid coin symbol")
        return v.upper()

    @validator("condition")
    def validate_condition(cls, v):
        if v not in [">", "<"]:
            raise ValueError("Condition must be '>' or '<'")
        return v

    @validator("value")
    def validate_value(cls, v):
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

class AIAlert(BaseModel):
    prompt: str
    result: Optional[str] = None
    triggered: bool = False

class AlertsRequest(BaseModel):
    rule_alerts: List[RuleAlert]
    ai_alerts: List[AIAlert]

class AlertCheckRequest(BaseModel):
    prompt: str

async def get_current_user(grokbit_token: str = Cookie(None)):
    credentials_exception = HTTPException(status_code=401, detail="Could not validate credentials")
    if not grokbit_token:
        raise credentials_exception
    try:
        payload = jwt.decode(grokbit_token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = await users_collection.find_one({"username": username})
    if not user:
        raise credentials_exception
    # Decrypt API keys
    if "preferences" in user and "api_keys" in user["preferences"]:
        decrypted_keys = {}
        for provider, enc_key in user["preferences"]["api_keys"].items():
            if enc_key:
                try:
                    decrypted_keys[provider] = cipher.decrypt(enc_key.encode()).decode()
                except InvalidToken:
                    decrypted_keys[provider] = ""
            else:
                decrypted_keys[provider] = ""
        user["preferences"]["api_keys"] = decrypted_keys
    # Ensure lists and models
    if "preferences" in user:
        user["preferences"]["prompts"] = user["preferences"].get("prompts", [])
        user["preferences"]["portfolio_prompts"] = user["preferences"].get("portfolio_prompts", [])
        user["preferences"]["alert_prompts"] = user["preferences"].get("alert_prompts", [])
        user["preferences"]["models"] = {**DEFAULT_MODELS, **user["preferences"].get("models", {})}
    return user

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def call_ai(provider: str, messages: list, user_api_key: Optional[str] = None, model: str = None):
    api_key = user_api_key
    if not api_key:
        raise ValueError(f"{provider} API key required")
    if not model:
        model = DEFAULT_MODELS.get(provider)
    if provider == "Groq":
        groq_client = Groq(api_key=api_key)
        response = groq_client.chat.completions.create(messages=messages, model=model, stream=False, temperature=0)
        return response.choices[0].message.content
    elif provider == "Gemini":
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel(model)
        response = gemini_model.generate_content(messages[-1]["content"])
        return response.text
    elif provider == "HuggingFace":
        hf_client = InferenceClient(model=model, token=api_key)
        return hf_client.text_generation(messages[-1]["content"], max_new_tokens=500)
    elif provider == "Grok":
        url = "https://api.x.ai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"messages": messages, "model": model, "stream": False, "temperature": 0}
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=20)
                response.raise_for_status()
                data = response.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "No response")
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(3)
                    continue
                else:
                    raise ValueError(f"Grok API request failed after {max_retries} attempts: {str(e)}")
    raise ValueError("Invalid provider")

@app.post("/register")
@limiter.limit("5/minute")
async def register(request: Request, user_data: Dict = Body(...)):
    first_name = user_data.get("first_name")
    last_name = user_data.get("last_name")
    username = user_data.get("username")
    password = user_data.get("password")
    if await users_collection.find_one({"username": username}):
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = pwd_context.hash(password)
    irichner = await users_collection.find_one({"username": "irichner"})
    default_prefs = {
        "default_provider": "Groq",
        "api_keys": {"Groq": "", "Gemini": "", "HuggingFace": "", "Grok": ""},
        "prompts": [],
        "portfolio_prompts": [],
        "alert_prompts": [],
        "models": DEFAULT_MODELS
    }
    if irichner and "preferences" in irichner:
        prefs = irichner["preferences"]
        default_prefs["prompts"] = prefs.get("prompts", [])
        default_prefs["portfolio_prompts"] = prefs.get("portfolio_prompts", [])
        default_prefs["alert_prompts"] = prefs.get("alert_prompts", [])
    user_dict = {
        "first_name": first_name,
        "last_name": last_name,
        "username": username,
        "hashed_password": hashed_password,
        "preferences": default_prefs,
        "oauth_providers": {},
        "oauth_only": False,
        "tier": "free"
    }
    await users_collection.insert_one(user_dict)
    # Send verification email
    message = Mail(
        from_email='no-reply@grokbit.ai',
        to_emails=username,
        subject='Verify Your GrokBit Email',
        html_content='<strong>Click the link to verify: [verification link here]</strong>'  # Add real link logic if needed
    )
    try:
        sendgrid_client.send(message)
    except Exception as e:
        logger.error(f"Email send failed: {e}")
    logger.info(f"User registered: {username}")
    return {"message": "User registered"}

@app.post("/token")
@limiter.limit("5/minute")
async def login(response: Response, request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    user = await users_collection.find_one({"username": form_data.username})
    if not user or user.get("oauth_only") or not pwd_context.verify(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    access_token = create_access_token({"sub": user["username"]})
    secure = os.getenv("ENV") == "prod" or not os.getenv("BACKEND_URL", "").startswith("http://localhost")
    response.set_cookie(
    key="grokbit_token",
    value=access_token,
    httponly=True,
    secure=secure,
    samesite='lax',
    max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    domain=".grokbit.ai"
)
    logger.info(f"User logged in: {form_data.username}")
    return {"success": True}

@app.get("/oauth/{provider}/login")
@limiter.limit("5/minute")
async def oauth_login(request: Request, provider: str):
    if provider not in providers:
        raise HTTPException(status_code=404, detail="Provider not supported")
    oauth_provider = oauth.create_client(provider)
    redirect_uri = os.getenv("BACKEND_URL", "https://grokbit-backend.onrender.com") + f"/oauth/{provider}/callback"
    return await oauth_provider.authorize_redirect(request, redirect_uri)

@app.get("/oauth/{provider}/callback")
@limiter.limit("5/minute")
async def oauth_callback(request: Request, provider: str):
    if provider not in providers:
        raise HTTPException(status_code=404, detail="Provider not supported")
    oauth_provider = oauth.create_client(provider)
    try:
        token = await oauth_provider.authorize_access_token(request)
    except Exception as e:
        logger.error(f"OAuth token fetch failed for {provider}: {e}")
        return RedirectResponse(url=os.getenv('FRONTEND_URL', 'http://localhost:3000') + '/login?error=auth_failed')
    try:
        if provider == 'google':
            user_info = await oauth_provider.userinfo(token=token)
        else:
            userinfo_endpoint = oauth_provider.userinfo_endpoint
            user_resp = await oauth_provider.get(userinfo_endpoint, token=token)
            user_info = user_resp.json()
        sub = user_info.get('id') or user_info.get('sub')
        email = user_info.get('email')
        name = user_info.get('name') or user_info.get('username', '')
        if not sub:
            raise ValueError("No user ID from provider")
        if email and not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            raise ValueError("Invalid email format from provider")
        if not re.match(r"^[a-zA-Z0-9_ ]{0,50}$", name):
            name = ""
    except Exception as e:
        logger.error(f"OAuth userinfo fetch failed for {provider}: {e}")
        return RedirectResponse(url=os.getenv('FRONTEND_URL', 'http://localhost:3000') + '/login?error=userinfo_failed')
    query = {"oauth_providers." + provider + ".sub": sub}
    if email:
        query = {"$or": [query, {"email": email}]}
    user = await users_collection.find_one(query)
    refresh_enc = cipher.encrypt(token.get("refresh_token", "").encode()).decode() if token.get("refresh_token") else None
    if user:
        if provider not in user.get("oauth_providers", {}):
            await users_collection.update_one(
                {"_id": user["_id"]},
                {"$set": {f"oauth_providers.{provider}": {"sub": sub, "refresh_token": refresh_enc}}}
            )
    else:
        username = f"{provider}_{sub[:10]}"
        if await users_collection.find_one({"username": username}):
            username += secrets.token_hex(4)
        new_user = {
            "username": username,
            "hashed_password": None,
            "email": email,
            "first_name": name.split()[0] if name else "",
            "last_name": " ".join(name.split()[1:]) if name else "",
            "preferences": {
                "default_provider": "Groq",
                "api_keys": {"Groq": "", "Gemini": "", "HuggingFace": "", "Grok": ""},
                "prompts": [],
                "portfolio_prompts": [],
                "alert_prompts": [],
                "models": DEFAULT_MODELS
            },
            "oauth_providers": {provider: {"sub": sub, "refresh_token": refresh_enc}},
            "oauth_only": True,
            "tier": "free"
        }
        insert_result = await users_collection.insert_one(new_user)
        user = await users_collection.find_one({"_id": insert_result.inserted_id})
    access_token = create_access_token({"sub": user["username"]})
    secure = os.getenv("ENV") == "prod" or not os.getenv("BACKEND_URL", "").startswith("http://localhost")
    response = RedirectResponse(url=os.getenv('FRONTEND_URL', 'http://localhost:3000'))
    response.set_cookie(
        key="grokbit_token",
        value=access_token,
        httponly=True,
        secure=secure,
        samesite='lax',
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )
    logger.info(f"OAuth login successful for {provider}: {user['username']}")
    return response

@app.get("/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    response = RedirectResponse(url=os.getenv('FRONTEND_URL', 'http://localhost:3000') + '/login')
    for prov, data in current_user.get("oauth_providers", {}).items():
        refresh_enc = data.get("refresh_token")
        if refresh_enc:
            try:
                refresh = cipher.decrypt(refresh_enc.encode()).decode()
                if prov == 'google':
                    requests.post('https://oauth2.googleapis.com/revoke', params={'token': refresh})
                elif prov == 'github':
                    # GitHub no revoke for OAuth tokens
                    pass
            except Exception as e:
                logger.error(f"Failed to revoke {prov} token: {e}")
    response.delete_cookie("grokbit_token", domain=".grokbit.ai")
    logger.info(f"User logged out: {current_user['username']}")
    return response

@app.get("/check_auth")
async def check_auth(current_user: dict = Depends(get_current_user)):
    return {"authenticated": True}

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
    # Ensure models
    if "models" not in preferences:
        preferences["models"] = DEFAULT_MODELS
    await users_collection.update_one({"_id": ObjectId(current_user["_id"])}, {"$set": {"preferences": preferences}})
    logger.info(f"Preferences updated for user: {current_user['username']}")
    return {"message": "Preferences updated"}

@app.get("/preferences")
async def get_preferences(current_user: dict = Depends(get_current_user)):
    prefs = current_user.get("preferences", {
        "default_provider": "Groq",
        "api_keys": {"Groq": "", "Gemini": "", "HuggingFace": "", "Grok": ""},
        "prompts": [],
        "portfolio_prompts": [],
        "alert_prompts": [],
        "models": DEFAULT_MODELS
    })
    # Ensure models
    prefs["models"] = {**DEFAULT_MODELS, **prefs.get("models", {})}
    return prefs

@app.get("/models")
async def get_models(provider: str, current_user: dict = Depends(get_current_user)):
    api_key = current_user["preferences"]["api_keys"].get(provider)
    if not api_key:
        raise HTTPException(status_code=400, detail=f"No API key for {provider}")
    models = []
    try:
        if provider == "Groq":
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            response = requests.get("https://api.groq.com/openai/v1/models", headers=headers)
            response.raise_for_status()
            data = response.json()
            models = [m["id"] for m in data.get("data", []) if m.get("active", True)]
        elif provider == "Gemini":
            genai.configure(api_key=api_key)
            gen_models = genai.list_models()
            models = [m.name for m in gen_models]
        elif provider == "HuggingFace":
            hf_models = list_models(limit=50, sort="downloads", direction=-1)
            models = [m.modelId for m in hf_models]
        elif provider == "Grok":
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            response = requests.get("https://api.x.ai/v1/models", headers=headers)
            response.raise_for_status()
            data = response.json()
            models = [m["id"] for m in data.get("data", [])]
        else:
            raise HTTPException(status_code=400, detail="Unsupported provider")
    except Exception as e:
        logger.error(f"Failed to fetch models for {provider}: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to fetch models for {provider}: {str(e)}")
    return {"models": models}

@app.get("/portfolio")
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

@app.post("/portfolio/save")
async def save_portfolio(request: PortfolioRequest, current_user: dict = Depends(get_current_user)):
    if not request.portfolio:
        return {"message": "Portfolio saved"}  # Empty is allowed
    portfolio_dict = {
        "user_id": str(current_user["_id"]),
        "items": [item.dict() for item in request.portfolio],
        "updated_at": datetime.utcnow()
    }
    await portfolios_collection.replace_one({"user_id": str(current_user["_id"])}, portfolio_dict, upsert=True)
    logger.info(f"Portfolio saved for user: {current_user['username']}")
    return {"message": "Portfolio saved"}

@app.post("/insights")
async def get_insights(request: InsightRequest, current_user: dict = Depends(get_current_user)):
    if not request.coin:
        raise HTTPException(status_code=400, detail="Coin required")
    user_prefs = current_user.get("preferences", {
        "default_provider": "Groq",
        "api_keys": {"Groq": "", "Gemini": "", "HuggingFace": "", "Grok": ""}
    })
    default_provider = user_prefs["default_provider"]
    configured_providers = [p for p in providers_order if user_prefs["api_keys"].get(p)]
    is_summary = request.coin.startswith("Summarize this user prompt in 3-4 words: ")
    if len(configured_providers) > 1 and is_summary:
        default_index = configured_providers.index(default_provider)
        selected_provider = configured_providers[(default_index + 1) % len(configured_providers)]
    else:
        selected_provider = request.provider or default_provider
    api_key = user_prefs["api_keys"].get(selected_provider)
    model = user_prefs.get("models", {}).get(selected_provider)
    messages = [
        {"role": "system", "content": "You are a crypto market assistant."},
        {"role": "user", "content": request.coin}
    ]
    try:
        insight = await call_ai(selected_provider, messages, api_key, model)
        return {"insight": insight}
    except Exception as e:
        logger.error(f"AI call failed for {selected_provider}: {e}")
        raise HTTPException(status_code=503, detail=f"API error: {str(e)}")

@app.post("/portfolio")
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
        price = await get_price(item.coin)
        value = price * item.quantity
        total_value += value
        portfolio_str += f"{item.quantity} {item.coin} (${value:.2f}, cost basis ${item.cost_basis:.2f}), "
    portfolio_str = portfolio_str.rstrip(", ") + "."

    user_prefs = current_user.get("preferences", {
        "default_provider": "Groq",
        "api_keys": {"Groq": "", "Gemini": "", "HuggingFace": "", "Grok": ""}
    })
    api_key = user_prefs["api_keys"].get(user_prefs["default_provider"])
    model = user_prefs.get("models", {}).get(user_prefs["default_provider"])
    messages = [
        {"role": "system", "content": "You are a crypto portfolio advisor."},
        {"role": "user", "content": f"{portfolio_str} Total value: ${total_value:.2f}. Provide a concise suggestion under 1000 characters."}
    ]
    try:
        suggestion = await call_ai(user_prefs["default_provider"], messages, api_key, model)
        return {"total_value": total_value, "suggestion": suggestion}
    except Exception as e:
        logger.error(f"Portfolio insight call failed: {e}")
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
            if response.status_code == 429:
                raise HTTPException(status_code=429, detail="Rate limit exceeded for CoinGecko API. Please try again later.")
            data = response.json()
            for original_coin, cg_id in zip([c for c in coins if KNOWN_IDS.get(c.upper()) or COIN_ID_MAP.get(c.upper())], cg_ids):
                price = data.get(cg_id, {}).get("usd", 0.0)
                result[original_coin] = price
                price_cache[original_coin.upper()] = price
    except Exception as e:
        logger.error(f"CoinGecko batch failed: {e}")

    # For missing prices, fallback to Binance batch
    missing_coins = [c for c in coins if result.get(c, 0.0) == 0.0]
    if missing_coins:
        try:
            symbols = [c.upper() + 'USDT' for c in missing_coins]
            symbols_str = json.dumps(symbols)
            response = requests.get(f"https://api.binance.com/api/v3/ticker/price?symbols={symbols_str}")
            response.raise_for_status()
            if response.status_code == 429:
                raise HTTPException(status_code=429, detail="Rate limit exceeded for Binance API. Please try again later.")
            data = response.json()
            for item in data:
                symbol = item.get('symbol', '')[:-4]  # Remove USDT
                price = float(item.get('price', 0.0))
                if symbol in [m.upper() for m in missing_coins]:
                    result[symbol] = price
                    price_cache[symbol] = price
        except Exception as e:
            logger.error(f"Binance batch failed: {e}")

    # For still missing, fallback to CryptoRank batch
    missing_coins = [c for c in coins if result.get(c, 0.0) == 0.0]
    if missing_coins:
        CRYPTO_RANK_API_KEY = os.getenv("CRYPTO_RANK_API_KEY")
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
                    if symbol in [m.upper() for m in missing_coins]:
                        result[symbol] = price
                        price_cache[symbol] = price
            except Exception as e:
                logger.error(f"CryptoRank batch failed: {e}")

    # If any still 0, fallback to per-coin get_price
    for coin in coins:
        if result.get(coin, 0.0) == 0.0:
            result[coin] = await get_price(coin)

    return result

@app.get("/coins")
async def get_coins(current_user: dict = Depends(get_current_user)):
    return list(COIN_ID_MAP.keys())

@app.get("/alerts")
async def get_alerts(current_user: dict = Depends(get_current_user)):
    alerts = await alerts_collection.find_one({"user_id": str(current_user["_id"])})
    return {
        "rule_alerts": alerts.get("rule_alerts", []),
        "ai_alerts": alerts.get("ai_alerts", []),
        "triggered_alerts": alerts.get("triggered_alerts", [])
    }

@app.post("/alerts/save")
async def save_alerts(request: AlertsRequest, current_user: dict = Depends(get_current_user)):
    alerts_dict = {
        "user_id": str(current_user["_id"]),
        "rule_alerts": [alert.dict() for alert in request.rule_alerts],
        "ai_alerts": [alert.dict() for alert in request.ai_alerts],
        "updated_at": datetime.utcnow()
    }
    await alerts_collection.replace_one({"user_id": str(current_user["_id"])}, alerts_dict, upsert=True)
    logger.info(f"Alerts saved for user: {current_user['username']}")
    return {"message": "Alerts saved"}

@app.post("/alerts/check")
async def check_ai_alert(request: AlertCheckRequest, current_user: dict = Depends(get_current_user)):
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt required")
    user_prefs = current_user.get("preferences", {
        "default_provider": "Groq",
        "api_keys": {"Groq": "", "Gemini": "", "HuggingFace": "", "Grok": ""}
    })
    api_key = user_prefs["api_keys"].get(user_prefs["default_provider"])
    model = user_prefs.get("models", {}).get(user_prefs["default_provider"])
    messages = [
        {"role": "system", "content": "You are a creative crypto alert generator. Analyze current trends, sentiment, or anomalies. If significant change detected, start response with 'ALERT:' followed by witty, useful message with action suggestion. Else, respond 'No alert'."},
        {"role": "user", "content": request.prompt}
    ]
    try:
        response = await call_ai(user_prefs["default_provider"], messages, api_key, model)
        if response.startswith("ALERT:"):
            await send_push(current_user["_id"], response)
            return {"alert": response}  # Triggered
        return {"alert": None}  # Not triggered
    except Exception as e:
        logger.error(f"AI alert check failed: {e}")
        raise HTTPException(status_code=503, detail=f"API error: {str(e)}")

@app.post("/delete_user")
@limiter.limit("5/minute")
async def delete_user(request: Request, delete_data: Dict = Body(...), current_user: dict = Depends(get_current_user)):
    username = delete_data.get("username")
    password = delete_data.get("password")
    if username != current_user["username"] or not pwd_context.verify(password, current_user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    await users_collection.delete_one({"_id": ObjectId(current_user["_id"])})
    await portfolios_collection.delete_one({"user_id": str(current_user["_id"])})
    await alerts_collection.delete_one({"user_id": str(current_user["_id"])})
    logger.info(f"User deleted: {username}")
    return {"message": "Account deleted"}

@app.get("/sentiment")
async def get_sentiment(coin: str, current_user: dict = Depends(get_current_user)):
    cache_key = f"sentiment_{coin}"
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    # Placeholder for X search - integrate real X API if available
    tweets = []  # Replace with real fetch: e.g., tweets = x_keyword_search(query=f"{coin} sentiment", limit=20)
    sentiment_prompt = f"Analyze sentiment for {coin} from these tweets: {tweets}"
    insight = await call_ai(current_user['preferences']['default_provider'], [{"role": "user", "content": sentiment_prompt}], current_user['preferences']['api_keys'][current_user['preferences']['default_provider']])
    score = 0.8  # Parse from insight
    result = {"score": score, "summary": insight}
    redis_client.set(cache_key, json.dumps(result), ex=300)
    return result

@app.post("/push/subscribe")
async def push_subscribe(sub: Dict = Body(...), current_user: dict = Depends(get_current_user)):
    await users_collection.update_one({"_id": current_user["_id"]}, {"$set": {"push_sub": sub}})
    return {"success": True}

async def send_push(user_id, message):
    user = await users_collection.find_one({"_id": ObjectId(user_id)})
    sub = user.get("push_sub")
    if sub:
        try:
            webpush(
                subscription_info=sub,
                data=json.dumps({"title": "GrokBit Alert", "body": message}),
                vapid_private_key=os.getenv("VAPID_PRIVATE_KEY"),
                vapid_public_key=os.getenv("VAPID_PUBLIC_KEY"),
                vapid_claims={"sub": "mailto:israel.richner@gmail.com"}
            )
        except WebPushException as e:
            logger.error(f"Push failed: {e}")

@app.post("/create-checkout-session")
async def create_checkout_session(current_user: dict = Depends(get_current_user)):
    session = stripe.checkout.Session.create(
        payment_method_types=['card'],
        line_items=[{'price': 'prod_SgizAO3lVhxP2F', 'quantity': 1}],  # Your product ID
        mode='subscription',
        success_url='https://grokbit.ai/success',
        cancel_url='https://grokbit.ai/cancel',
        client_reference_id=str(current_user["_id"])
    )
    return {"id": session.id}

@app.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, os.getenv("STRIPE_WEBHOOK_SECRET"))
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError as e:
        raise HTTPException(status_code=400, detail="Invalid signature")
    if event['type'] == 'checkout.session.completed':
        user_id = event['data']['object']['client_reference_id']
        await users_collection.update_one({"_id": ObjectId(user_id)}, {"$set": {"tier": "premium"}})
    return {"success": True}

@app.get("/user/tier")
async def get_tier(current_user: dict = Depends(get_current_user)):
    return {"tier": current_user.get("tier", "free")}

@app.get("/")
def read_root():
    return {"message": "Welcome to GrokBit API"}