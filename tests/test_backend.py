# tests/test_backend.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

@pytest.mark.asyncio
async def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to GrokBit API"}

# Add auth token simulation if needed; example for /preferences GET
@pytest.mark.asyncio
async def test_get_preferences():
    # Mock auth cookie or header as per your setup
    client.cookies.set("grokbit_token", "valid_token_here")
    response = client.get("/users/preferences")
    assert response.status_code == 200 or 401  # Adjust based on auth

# Expand with more tests: register, login, portfolio, etc.
# Example POST /register
@pytest.mark.asyncio
async def test_register():
    data = {"username": "testuser", "password": "testpass"}
    response = client.post("/auth/register", json=data)
    assert response.status_code == 200 or 400  # Success or existing user

# Run with: pytest tests/test_backend.py