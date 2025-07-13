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

load_dotenv()

app = FastAPI()

# CORS setup (preserved from current)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://grokbit-frontend-pua2ymczb-israel-richners-projects.vercel.app", "https://grokbit.ai", "*"],
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
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class User(BaseModel):
    username: str
    hashed_password: str
    preferences: Dict = {"default_provider": "Groq", "api_keys": {"Groq": "", "Gemini": "", "HuggingFace": "", "Grok": ""}}

class Token(BaseModel):
    access_token: str
    token_type: str

class InsightRequest(BaseModel):
    coin: str

class PortfolioItem(BaseModel):
    coin: str
    amount: float

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
async def register(form_data: OAuth2PasswordRequestForm = Depends()):
    if await users_collection.find_one({"username": form_data.username}):
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = pwd_context.hash(form_data.password)
    user_dict = {"username": form_data.username, "hashed_password": hashed_password, "preferences": {"default_provider": "Groq", "api_keys": {"Groq": "", "Gemini": "", "HuggingFace": "", "Grok": ""}}}
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
    await users_collection.update_one({"_id": ObjectId(current_user["_id"])}, {"$set": {"preferences": preferences}})
    return {"message": "Preferences updated"}

@app.get("/preferences")
async def get_preferences(current_user: dict = Depends(get_current_user)):
    return current_user.get("preferences", {"default_provider": "Groq", "api_keys": {"Groq": "", "Gemini": "", "HuggingFace": "", "Grok": ""}})

@app.get("/portfolio")
async def get_portfolio(current_user: dict = Depends(get_current_user)):
    portfolio = await portfolios_collection.find_one({"user_id": str(current_user["_id"])})
    return {"portfolio": portfolio.get("items", []) if portfolio else []}

@app.post("/insights")
async def get_insights(request: InsightRequest, current_user: dict = Depends(get_current_user)):
    if not request.coin:
        raise HTTPException(status_code=400, detail="Coin required")
    user_prefs = current_user.get("preferences", {"default_provider": "Groq", "api_keys": {"Groq": "", "Gemini": "", "HuggingFace": "", "Grok": ""}})
    api_key = user_prefs["api_keys"].get(user_prefs["default_provider"])
    messages = [
        {"role": "system", "content": "You are a crypto market assistant."},
        {"role": "user", "content": f"Provide a market insight for {request.coin} based on current trends in bullet points, keeping the response under 300 characters."}
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
    
    # Calculate total_value (preserved from current)
    coin_prices = {"BTC": 60000, "ETH": 3000}  # Example prices
    total_value = 0
    portfolio_str = "Your portfolio: "
    for item in request.portfolio:
        price = coin_prices.get(item.coin.upper(), 0)
        value = price * item.amount
        total_value += value
        portfolio_str += f"{item.amount} {item.coin} (${value:.2f}), "
    portfolio_str = portfolio_str.rstrip(", ") + "."

    user_prefs = current_user.get("preferences", {"default_provider": "Groq", "api_keys": {"Groq": "", "Gemini": "", "HuggingFace": "", "Grok": ""}})
    api_key = user_prefs["api_keys"].get(user_prefs["default_provider"])
    messages = [
        {"role": "system", "content": "You are a crypto portfolio advisor."},
        {"role": "user", "content": f"{portfolio_str} Total value: ${total_value:.2f}. Provide a concise suggestion under 100 characters."}
    ]
    try:
        suggestion = await call_ai(user_prefs["default_provider"], messages, api_key)
        return {"total_value": total_value, "suggestion": suggestion}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"API error: {str(e)}")

# Preserve root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to GrokBit API"}