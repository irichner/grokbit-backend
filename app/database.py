# app/database.py
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from .config import MONGO_URI

client = AsyncIOMotorClient(MONGO_URI)
db = client.grokbit
users_collection = db.users
portfolios_collection = db.portfolios
alerts_collection = db.alerts