# app/main.py
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from app.config import FRONTEND_URL, SECRET_KEY
from app.database import create_indexes
from app.services.price import startup_load_coins
from app.routers import auth, users, portfolio, alerts, insights, market, payments, admin, root
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup_load_coins()
    await create_indexes()
    yield

app = FastAPI(lifespan=lifespan)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

app.include_router(auth.router)
app.include_router(users.router)
app.include_router(portfolio.router)
app.include_router(alerts.router)
app.include_router(insights.router)
app.include_router(market.router)
app.include_router(payments.router)
app.include_router(admin.router)
app.include_router(root.router)