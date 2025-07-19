# app/database.py
from motor.motor_asyncio import AsyncIOMotorClient
from app.config import MONGO_URI
from bson import ObjectId
import asyncio

client = AsyncIOMotorClient(MONGO_URI)
db = client.grokbit
users_collection = db.users
portfolios_collection = db.portfolios
alerts_collection = db.alerts

async def create_indexes():
    await users_collection.create_index("username", unique=True)
    await users_collection.create_index("oauth_providers.google.sub")
    await users_collection.create_index("oauth_providers.github.sub")
    await portfolios_collection.create_index("user_id")
    await alerts_collection.create_index("user_id")