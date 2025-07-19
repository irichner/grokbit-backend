# app/routers/root.py
from fastapi import APIRouter

router = APIRouter(tags=["root"])

@router.get("/")
def read_root():
    return {"message": "Welcome to GrokBit API"}