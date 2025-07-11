from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env

app = FastAPI()

# CORS setup for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InsightRequest(BaseModel):
    coin: str  # e.g., "BTC"

@app.get("/")
def read_root():
    return {"message": "Welcome to GrokBit API"}

@app.post("/insights")
def get_insights(request: InsightRequest):
    # Placeholder for xAI API integration
    # In real implementation, call xAI API here for market insights
    if not request.coin:
        raise HTTPException(status_code=400, detail="Coin required")
    return {"coin": request.coin, "insight": "Sample: Market is bullish!"}  # Replace with real AI call

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)