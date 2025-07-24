# app/utils/security.py
from cryptography.fernet import Fernet
from app.config import ENCRYPTION_KEY
import logging

logger = logging.getLogger(__name__)
logger.info(f"Loaded ENCRYPTION_KEY: {ENCRYPTION_KEY}")

cipher = Fernet(ENCRYPTION_KEY.encode())