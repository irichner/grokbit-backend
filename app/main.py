# app/main.py
import asyncio
import logging
import re
from fastapi import FastAPI, HTTPException, Depends, Body, Request, Response, Cookie
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, field_validator
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
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from pywebpush import webpush, WebPushException
import stripe
import pwnedpasswords

# Routers import
from app.routers import auth, users, portfolio, alerts, market, payments, admin, insights, root

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

# Include routers
app.include_router(root.router)
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(portfolio.router)
app.include_router(alerts.router)
app.include_router(market.router)
app.include_router(payments.router)
app.include_router(admin.router)
app.include_router(insights.router)

# Rate limiting
def get_rate_limit_key(request: Request):
    username = request.form.get('username', '') if request.method == 'POST' else ''
    return f"{get_remote_address(request)}:{username}"

limiter = Limiter(key_func=get_rate_limit_key)
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
REFRESH_TOKEN_EXPIRE_MINUTES = 1440  # 1 day

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Price cache
price_cache = TTLCache(maxsize=100, ttl=60)

# Sentiment cache
sentiment_cache = TTLCache(maxsize=100, ttl=300)

COIN_ID_MAP = {}
KNOWN_IDS = {
    'BTC': 'bitcoin',
    'ETH': 'ethereum',
    'XRP': 'ripple',
    'USDT': 'tether',
    'BNB': 'bnb',
    'SOL': 'solana',
}

DEFAULT_MODELS = {
    "Groq": os.getenv("DEFAULT_MODEL_GROQ", "llama-3.1-8b-instant"),
    "Gemini": os.getenv("DEFAULT_MODEL_GEMINI", "gemini-2.5-flash"),
    "HuggingFace": os.getenv("DEFAULT_MODEL_HF", "mistralai/Mistral-7B-Instruct-v0.3"),
    "Grok": os.getenv("DEFAULT_MODEL_GROK", "grok-3"),
    "CoinGecko": "N/A"
}

providers_order = ["Groq", "Gemini", "HuggingFace", "Grok", "CoinGecko"]

@app.on_event("startup")
async def startup_event():
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
    await create_indexes()

async def get_price(coin: str, coingecko_key: str = None) -> float:
    coin_upper = coin.upper()
    cache_key = coin_upper
    if cache_key in price_cache:
        return price_cache[cache_key]
    cg_id = KNOWN_IDS.get(coin_upper) or COIN_ID_MAP.get(coin_upper)
    if not cg_id:
        logger.warning(f"No CoinGecko ID for coin: {coin_upper}")
        return 0.0
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
            CRYPTO_RANK_API_KEY = os.getenv("CRYPTO_RANK_API_KEY")
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

class User(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    username: str
    hashed_password: Optional[str] = None
    email: Optional[str] = None
    preferences: Dict = {"prompt_default_provider": "Groq", "summary_default_provider": "Groq", "api_keys": {"Groq": "", "Gemini": "", "HuggingFace": "", "Grok": "", "CoinGecko": ""}, "prompts": [], "portfolio_prompts": [], "alert_prompts": [], "models": DEFAULT_MODELS, "refresh_rate": 60000, "market_coins": []}
    oauth_providers: Optional[Dict] = {}
    oauth_only: Optional[bool] = False
    tier: str = 'free'
    profile_image: Optional[str] = ''
    verified: bool = False

    @field_validator("username")
    def validate_username(cls, v):
        if not re.match(r"^[a-zA-Z0-9_]{3,20}$", v):
            raise ValueError("Username must be 3-20 characters, alphanumeric or underscore")
        return v

    @field_validator("email", check_fields=False)
    def validate_email(cls, v):
        if v and not re.match(r"[^@]+@[^@]+\.[^@]+", v):
            raise ValueError("Invalid email format")
        return v

class Token(BaseModel):
    access_token: str
    token_type: str
    refresh_token: str

class InsightRequest(BaseModel):
    coin: str
    provider: Optional[str] = None

class PortfolioItem(BaseModel):
    coin: str
    quantity: float
    cost_basis: float

    @field_validator("coin")
    def validate_coin(cls, v):
        if v.upper() not in COIN_ID_MAP and v.upper() not in KNOWN_IDS:
            raise ValueError("Invalid coin symbol")
        return v.upper()

    @field_validator("quantity", "cost_basis")
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

    @field_validator("coin")
    def validate_coin(cls, v):
        if v.upper() not in COIN_ID_MAP and v.upper() not in KNOWN_IDS:
            raise ValueError("Invalid coin symbol")
        return v.upper()

    @field_validator("condition")
    def validate_condition(cls, v):
        if v not in [">", "<"]:
            raise ValueError("Condition must be '>' or '<'")
        return v

    @field_validator("value")
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

class ForgotPasswordRequest(BaseModel):
    email: str

class ResetPasswordRequest(BaseModel):
    token: str
    password: str

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
        user["preferences"]["prompt_default_provider"] = user["preferences"].get("prompt_default_provider", "Groq")
        user["preferences"]["summary_default_provider"] = user["preferences"].get("summary_default_provider", "Groq")
        user["preferences"]["refresh_rate"] = user["preferences"].get("refresh_rate", 60000)
        user["preferences"]["market_coins"] = user["preferences"].get("market_coins", [])
    return user

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def create_refresh_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=REFRESH_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def create_verification_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=1)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def create_reset_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=1)
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
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                data = response.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "No response")
            except requests.exceptions.RequestException as e:
                logger.error(f"Grok API request error: {e}")
                status_code = None
                if hasattr(e, 'response') and e.response is not None:
                    status_code = e.response.status_code
                if status_code == 503:
                    logger.error(f"Grok API 503 error: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(5)
                        continue
                    else:
                        raise HTTPException(status_code=503, detail="Grok API unavailable after retries")
                if attempt < max_retries - 1:
                    time.sleep(5)
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
    if len(password) < 12 or len(password) > 64:
        raise HTTPException(status_code=400, detail="Password must be 12-64 characters")
    pwned = pwnedpasswords.check(password)
    if pwned > 0:
        raise HTTPException(status_code=400, detail="Password has been breached; choose another")
    if await users_collection.find_one({"username": username}):
        raise HTTPException(status_code=400, detail="Username taken")
    hashed_password = pwd_context.hash(password)
    admin_user = await users_collection.find_one({"username": "irichner"})
    default_prefs = {"prompt_default_provider": "Groq", "summary_default_provider": "Groq", "api_keys": {"Groq": "", "Gemini": "", "HuggingFace": "", "Grok": "", "CoinGecko": ""}, "prompts": [], "portfolio_prompts": [], "alert_prompts": [], "models": DEFAULT_MODELS, "refresh_rate": 60000, "market_coins": []}
    if admin_user and "preferences" in admin_user:
        prefs = admin_user["preferences"]
        default_prefs["prompts"] = prefs.get("prompts", [])
        default_prefs["portfolio_prompts"] = prefs.get("portfolio_prompts", [])
        default_prefs["alert_prompts"] = prefs.get("alert_prompts", [])
        default_prefs["market_coins"] = prefs.get("market_coins", [])
    user_dict = {
        "first_name": first_name,
        "last_name": last_name,
        "username": username,
        "hashed_password": hashed_password,
        "preferences": default_prefs,
        "oauth_providers": {},
        "oauth_only": False,
        "tier": "free",
        "profile_image": "",
        "verified": False
    }
    await users_collection.insert_one(user_dict)
    token = create_verification_token({"sub": username})
    verification_url = f"{os.getenv('FRONTEND_URL', 'http://localhost:3000')}/verify?token={token}"
    message = Mail(
        from_email='no-reply@grokbit.ai',
        to_emails=username,
        subject='Verify Your GrokBit Email',
        html_content=f'<strong>Click the link to verify: <a href="{verification_url}">Verify</a></strong>'
    )
    try:
        sendgrid_client = SendGridAPIClient(os.getenv('SENDGRID_API_KEY'))
        sendgrid_client.send(message)
    except Exception as e:
        logger.error(f"Email send failed: {e}")
    logger.info(f"User registered: {username}")
    return {"message": "User registered"}

@app.get("/verify")
async def verify(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        user = await users_collection.find_one({"username": username})
        if user:
            await users_collection.update_one({"username": username}, {"$set": {"verified": True}})
            return {"message": "Email verified"}
        raise HTTPException(status_code=400, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=400, detail="Invalid token")

@app.post("/resend-verification")
async def resend_verification(current_user: dict = Depends(get_current_user)):
    if current_user["verified"]:
        raise HTTPException(status_code=400, detail="Already verified")
    token = create_verification_token({"sub": current_user["username"]})
    verification_url = f"{os.getenv('FRONTEND_URL', 'http://localhost:3000')}/verify?token={token}"
    message = Mail(
        from_email='no-reply@grokbit.ai',
        to_emails=current_user["username"],
        subject='Verify Your GrokBit Email',
        html_content=f'<strong>Click the link to verify: <a href="{verification_url}">Verify</a></strong>'
    )
    try:
        sendgrid_client = SendGridAPIClient(os.getenv('SENDGRID_API_KEY'))
        sendgrid_client.send(message)
    except Exception as e:
        logger.error(f"Resend failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to resend")
    return {"message": "Verification resent"}

@app.post("/token")
@limiter.limit("5/minute")
async def login(response: Response, request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    user = await users_collection.find_one({"username": form_data.username})
    if not user or user.get("oauth_only") or not pwd_context.verify(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not user["verified"]:
        raise HTTPException(status_code=403, detail="Verify email first")
    access_token = create_access_token({"sub": user["username"]})
    refresh_token = create_refresh_token({"sub": user["username"]})
    secure = os.getenv("ENV") == "prod" or not os.getenv("BACKEND_URL", "").startswith("http://localhost")
    response.set_cookie(
        key="grokbit_token",
        value=access_token,
        httponly=True,
        secure=secure,
        samesite='strict',
        path='/',
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        domain=".grokbit.ai"
    )
    response.set_cookie(
        key="grokbit_refresh",
        value=refresh_token,
        httponly=True,
        secure=secure,
        samesite='strict',
        path='/',
        max_age=REFRESH_TOKEN_EXPIRE_MINUTES * 60,
        domain=".grokbit.ai"
    )
    logger.info(f"User logged in: {form_data.username}")
    return {"success": True}

@app.post("/refresh")
async def refresh(response: Response, grokbit_refresh: str = Cookie(None)):
    credentials_exception = HTTPException(status_code=401, detail="Invalid refresh token")
    if not grokbit_refresh:
        raise credentials_exception
    try:
        payload = jwt.decode(grokbit_refresh, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = await users_collection.find_one({"username": username})
    if not user:
        raise credentials_exception
    access_token = create_access_token({"sub": username})
    secure = os.getenv("ENV") == "prod" or not os.getenv("BACKEND_URL", "").startswith("http://localhost")
    response.set_cookie(
        key="grokbit_token",
        value=access_token,
        httponly=True,
        secure=secure,
        samesite='strict',
        path='/',
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        domain=".grokbit.ai"
    )
    return {"success": True}

@app.post("/forgot-password")
@limiter.limit("5/minute")
async def forgot_password(request: ForgotPasswordRequest):
    user = await users_collection.find_one({"email": request.email})
    if not user:
        return {"message": "If email exists, reset link sent"}
    token = create_reset_token({"sub": user["username"]})
    reset_url = f"{os.getenv('FRONTEND_URL', 'http://localhost:3000')}/reset?token={token}"
    message = Mail(
        from_email='no-reply@grokbit.ai',
        to_emails=request.email,
        subject='Reset Your GrokBit Password',
        html_content=f'<strong>Click to reset: <a href="{reset_url}">Reset</a></strong>'
    )
    try:
        sendgrid_client = SendGridAPIClient(os.getenv('SENDGRID_API_KEY'))
        sendgrid_client.send(message)
    except Exception as e:
        logger.error(f"Reset email failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to send reset")
    return {"message": "If email exists, reset link sent"}

@app.post("/reset-password")
@limiter.limit("5/minute")
async def reset_password(request: ResetPasswordRequest):
    try:
        payload = jwt.decode(request.token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        user = await users_collection.find_one({"username": username})
        if not user:
            raise HTTPException(status_code=400, detail="Invalid token")
        if len(request.password) < 12 or len(request.password) > 64:
            raise HTTPException(status_code=400, detail="Password must be 12-64 characters")
        pwned = pwnedpasswords.check(request.password)
        if pwned > 0:
            raise HTTPException(status_code=400, detail="Password has been breached; choose another")
        hashed_password = pwd_context.hash(request.password)
        await users_collection.update_one({"username": username}, {"$set": {"hashed_password": hashed_password}})
        return {"message": "Password reset"}
    except JWTError:
        raise HTTPException(status_code=400, detail="Invalid token")

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
        profile_image = user_info.get('picture', '')
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
            "preferences": {"prompt_default_provider": "Groq", "summary_default_provider": "Groq", "api_keys": {"Groq": "", "Gemini": "", "HuggingFace": "", "Grok": "", "CoinGecko": ""}, "prompts": [], "portfolio_prompts": [], "alert_prompts": [], "models": DEFAULT_MODELS, "refresh_rate": 60000, "market_coins": []},
            "oauth_providers": {provider: {"sub": sub, "refresh_token": refresh_enc}},
            "oauth_only": True,
            "tier": "free",
            "profile_image": profile_image,
            "verified": True  # OAuth verified
        }
        await users_collection.insert_one(new_user)
        user = await users_collection.find_one({"username": username})
    access_token = create_access_token({"sub": user["username"]})
    secure = os.getenv("ENV") == "prod" or not os.getenv("BACKEND_URL", "").startswith("http://localhost")
    response = RedirectResponse(url=os.getenv('FRONTEND_URL', 'http://localhost:3000'))
    response.set_cookie(
        key="grokbit_token",
        value=access_token,
        httponly=True,
        secure=secure,
        samesite='strict',
        path='/',
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        domain=".grokbit.ai"
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
                    # GitHub no revoke
                    pass
            except Exception as e:
                logger.error(f"Failed to revoke {prov} token: {e}")
    response.delete_cookie("grokbit_token", domain=".grokbit.ai")
    response.delete_cookie("grokbit_refresh", domain=".grokbit.ai")
    logger.info(f"User logged out: {current_user['username']}")
    return response

@app.get("/check_auth")
async def check_auth(current_user: dict = Depends(get_current_user)):
    return {"authenticated": True}

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