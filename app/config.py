# app/config.py
import os
from dotenv import load_dotenv
from cryptography.fernet import Fernet

load_dotenv()

# Constants from OriginalMain.py
SECRET_KEY = os.getenv("SECRET_KEY")
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
cipher = Fernet(ENCRYPTION_KEY.encode()) if ENCRYPTION_KEY else None
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_MINUTES = 1440  # Added
CLIENT_ID_GOOGLE = os.getenv("CLIENT_ID_GOOGLE")
CLIENT_SECRET_GOOGLE = os.getenv("CLIENT_SECRET_GOOGLE")
CLIENT_ID_GITHUB = os.getenv("CLIENT_ID_GITHUB")
CLIENT_SECRET_GITHUB = os.getenv("CLIENT_SECRET_GITHUB")
BACKEND_URL = os.getenv("BACKEND_URL", "https://grokbit-backend.onrender.com")
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://grokbit.ai")
ENV = os.getenv("ENV", "dev")
MONGO_URI = os.getenv("MONGO_URI")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
VAPID_PRIVATE_KEY = os.getenv("VAPID_PRIVATE_KEY")
VAPID_PUBLIC_KEY = os.getenv("VAPID_PUBLIC_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
CRYPTO_RANK_API_KEY = os.getenv("CRYPTO_RANK_API_KEY")
DEFAULT_MODELS = {
    "Groq": os.getenv("DEFAULT_MODEL_GROQ", "llama-3.1-8b-instant"),
    "Gemini": os.getenv("DEFAULT_MODEL_GEMINI", "gemini-2.5-flash"),
    "HuggingFace": os.getenv("DEFAULT_MODEL_HF", "mistralai/Mistral-7B-Instruct-v0.3"),
    "Grok": os.getenv("DEFAULT_MODEL_GROK", "grok-3"),
    "CoinGecko": "N/A",
    "OpenAI": "gpt-4o",
    "Anthropic": "claude-3-5-sonnet-20240620",
    "Mistral": "mistral-large-latest",
    "Cohere": "command",
    "Perplexity": "llama-3-sonar-large-32k-online",
    "Replicate": "meta/llama-2-70b-chat",
    "Together": "togethercomputer/llama-2-70b-chat",
    "Fireworks": "accounts/fireworks/models/mixtral-8x7b-instruct",
    "Novita": "novita/llama-2-70b"
}
providers_order = ["Groq", "Gemini", "HuggingFace", "Grok", "CoinGecko", "OpenAI", "Anthropic", "Mistral", "Cohere", "Perplexity", "Replicate", "Together", "Fireworks", "Novita"]
# Add stripe.api_key if needed: stripe.api_key = os.getenv("STRIPE_SECRET_KEY")