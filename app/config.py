# app/config.py
import os
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("SECRET_KEY must be set")

ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
if not ENCRYPTION_KEY:
    raise ValueError("ENCRYPTION_KEY must be set")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MONGO_URI must be set")

FRONTEND_URL = os.getenv("FRONTEND_URL", "https://grokbit.ai")
BACKEND_URL = os.getenv("BACKEND_URL", "https://grokbit-backend.onrender.com")
ENV = os.getenv("ENV", "dev")

CLIENT_ID_GOOGLE = os.getenv("CLIENT_ID_GOOGLE")
CLIENT_SECRET_GOOGLE = os.getenv("CLIENT_SECRET_GOOGLE")
CLIENT_ID_GITHUB = os.getenv("CLIENT_ID_GITHUB")
CLIENT_SECRET_GITHUB = os.getenv("CLIENT_SECRET_GITHUB")

CRYPTO_RANK_API_KEY = os.getenv("CRYPTO_RANK_API_KEY")

VAPID_PRIVATE_KEY = os.getenv("VAPID_PRIVATE_KEY")
VAPID_PUBLIC_KEY = os.getenv("VAPID_PUBLIC_KEY")

STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")

DEFAULT_MODELS = {
    "Groq": os.getenv("DEFAULT_MODEL_GROQ", "llama-3.1-8b-instant"),
    "Gemini": os.getenv("DEFAULT_MODEL_GEMINI", "gemini-2.5-flash"),
    "HuggingFace": os.getenv("DEFAULT_MODEL_HF", "mistralai/Mistral-7B-Instruct-v0.3"),
    "Grok": os.getenv("DEFAULT_MODEL_GROK", "grok-3"),
    "CoinGecko": "N/A"
}

providers_order = ["Groq", "Gemini", "HuggingFace", "Grok", "CoinGecko"]