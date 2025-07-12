from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import requests
from fastapi.security import OAuth2PasswordBearer

load_dotenv()

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://grokbit-frontend-pua2ymczb-israel-richners-projects.vercel.app", "https://grokbit.ai", "*"],  # Add your Vercel URL, domain, and wildcard for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OAuth2 scheme for token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Simple in-memory token storage (replace with database in production)
fake_users_db = {"xai_user": {"hashed_password": "fakehashedsecretkeyfornow", "token": "fake-token-123"}}

async def get_current_user(token: str = Depends(oauth2_scheme)):
    user = fake_users_db.get("xai_user")
    if not user or user["token"] != token:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    return user

class InsightRequest(BaseModel):
    coin: str

class PortfolioItem(BaseModel):
    coin: str
    amount: float

class PortfolioRequest(BaseModel):
    portfolio: list[PortfolioItem]

@app.get("/")
def read_root():
    return {"message": "Welcome to GrokBit API"}

@app.post("/insights")
async def get_insights(request: InsightRequest, current_user: dict = Depends(get_current_user)):
    if not request.coin:
        raise HTTPException(status_code=400, detail="Coin required")
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="API key not configured")
    url = "https://api.x.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "messages": [
            {"role": "system", "content": "You are a crypto market assistant."},
            {"role": "user", "content": f"Provide a market insight for {request.coin} based on current trends, keeping the response under 100 characters."}
        ],
        "model": "grok-3-latest",
        "stream": False,
        "temperature": 0
    }
    print(f"Request URL: {url}, Payload: {payload}")
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        print(f"Response Status: {response.status_code}, Text: {response.text}")
        response.raise_for_status()
        data = response.json()
        insight = data.get("choices", [{}])[0].get("message", {}).get("content", f"No insight available for {request.coin}")
        return {"coin": request.coin, "insight": insight}
    except requests.exceptions.RequestException as e:
        print(f"API Error: {str(e)}")
        raise HTTPException(status_code=503, detail=f"API error: {str(e)}")

@app.post("/portfolio")
async def get_portfolio_insight(request: PortfolioRequest, current_user: dict = Depends(get_current_user)):
    if not request.portfolio or len(request.portfolio) == 0:
        raise HTTPException(status_code=400, detail="Portfolio required")
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="API key not configured")
    url = "https://api.x.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    coin_prices = {"BTC": 60000, "ETH": 3000}  # Example prices in USD
    total_value = 0
    portfolio_str = "Your portfolio: "
    for item in request.portfolio:
        price = coin_prices.get(item.coin.upper(), 0)
        if price > 0:
            value = price * item.amount
            total_value += value
            portfolio_str += f"{item.amount} {item.coin} (${value:.2f}), "
    portfolio_str = portfolio_str.rstrip(", ") + "."

    payload = {
        "messages": [
            {"role": "system", "content": "You are a crypto portfolio advisor."},
            {"role": "user", "content": f"{portfolio_str} Total value: ${total_value:.2f}. Provide a concise suggestion under 100 characters."}
        ],
        "model": "grok-3-latest",
        "stream": False,
        "temperature": 0
    }
    print(f"Request URL: {url}, Payload: {payload}")
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        print(f"Response Status: {response.status_code}, Text: {response.text}")
        response.raise_for_status()
        data = response.json()
        suggestion = data.get("choices", [{}])[0].get("message", {}).get("content", "No suggestion available.")
        return {"total_value": total_value, "suggestion": suggestion}
    except requests.exceptions.RequestException as e:
        print(f"API Error: {str(e)}")
        raise HTTPException(status_code=503, detail=f"API error: {str(e)}")

@app.get("/token")
async def get_token():
    return {"access_token": fake_users_db["xai_user"]["token"], "token_type": "bearer"}