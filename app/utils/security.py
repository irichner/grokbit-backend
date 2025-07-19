# app/utils/security.py
from cryptography.fernet import Fernet
from app.config import ENCRYPTION_KEY

cipher = Fernet(ENCRYPTION_KEY.encode())