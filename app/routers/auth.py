# backend/app/routers/auth.py
from fastapi import APIRouter, HTTPException, Depends, Body, Request, Response, Cookie
from fastapi.security import OAuth2PasswordRequestForm
from authlib.integrations.starlette_client import OAuth
from starlette.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from app.config import CLIENT_ID_GOOGLE, CLIENT_SECRET_GOOGLE, CLIENT_ID_GITHUB, CLIENT_SECRET_GITHUB, BACKEND_URL, FRONTEND_URL, ENV, SECRET_KEY
from app.database import users_collection, ObjectId
from app.models.user import User
from app.services.auth import create_access_token, pwd_context, ACCESS_TOKEN_EXPIRE_MINUTES
from app.utils.security import cipher
from app.utils.email import send_verification_email
from app.dependencies import get_current_user
from slowapi import Limiter
from slowapi.util import get_remote_address
import secrets
import re
import requests
from typing import Dict
import logging
import traceback

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

oauth = OAuth()
providers = ['google', 'github']

oauth.register(
    name='google',
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_id=CLIENT_ID_GOOGLE,
    client_secret=CLIENT_SECRET_GOOGLE,
    client_kwargs={'scope': 'openid email profile', 'code_challenge_method': 'S256'}
)

oauth.register(
    name='github',
    client_id=CLIENT_ID_GITHUB,
    client_secret=CLIENT_SECRET_GITHUB,
    authorize_url='https://github.com/login/oauth/authorize',
    access_token_url='https://github.com/login/oauth/access_token',
    client_kwargs={'scope': 'user:email', 'code_challenge_method': 'S256'},
    userinfo_endpoint='https://api.github.com/user'
)

limiter = Limiter(key_func=get_remote_address)

@router.post("/register")
@limiter.limit("5/minute")
async def register(request: Request, user_data: Dict = Body(...)):
    first_name = user_data.get("first_name")
    last_name = user_data.get("last_name")
    username = user_data.get("username")
    password = user_data.get("password")
    if await users_collection.find_one({"username": username}):
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = pwd_context.hash(password)
    admin_user = await users_collection.find_one({"email": "israel.richner@gmail.com"})
    default_prefs = User().preferences  # Use default from model
    if admin_user and "preferences" in admin_user:
        prefs = admin_user["preferences"]
        default_prefs["prompts"] = prefs.get("prompts", [])
        default_prefs["portfolio_prompts"] = prefs.get("portfolio_prompts", [])
        default_prefs["alert_prompts"] = prefs.get("alert_prompts", [])
        default_prefs["market_coins"] = prefs.get("market_coins", [])
    user_dict = {
        "first_name": first_name,
        "last_name": last_name,
        "username": username,
        "hashed_password": hashed_password,
        "preferences": default_prefs,
        "oauth_providers": {},
        "oauth_only": False,
        "tier": "free",
        "profile_image": ""
    }
    await users_collection.insert_one(user_dict)
    await send_verification_email(username)
    return {"message": "User registered"}

@router.post("/token")
@limiter.limit("5/minute")
async def login(response: Response, request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    user = await users_collection.find_one({"username": form_data.username})
    if not user or user.get("oauth_only") or not pwd_context.verify(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    access_token = create_access_token({"sub": user["username"]})
    secure = ENV == "prod" or not BACKEND_URL.startswith("http://localhost")
    response.set_cookie(
        key="grokbit_token",
        value=access_token,
        httponly=True,
        secure=secure,
        samesite='lax',
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        domain=".grokbit.ai"
    )
    return {"success": True}

@router.get("/oauth/{provider}/login", name="oauth_login")
@limiter.limit("5/minute")
async def oauth_login(request: Request, provider: str):
    logger.info(f"Login request: provider={provider}, URL={request.url}")
    if provider not in providers:
        logger.error(f"Provider not supported: {provider}")
        raise HTTPException(status_code=404, detail="Provider not supported")
    oauth_provider = oauth.create_client(provider)
    redirect_uri = request.url_for('oauth_callback', provider=provider)
    try:
        return await oauth_provider.authorize_redirect(request, redirect_uri)
    except Exception as e:
        logger.error(f"OAuth login error for {provider}: {str(e)}\n{traceback.format_exc()}")
        return RedirectResponse(url=FRONTEND_URL + '/login?error=auth_failed')

@router.get("/oauth/{provider}/callback", name="oauth_callback")
@limiter.limit("5/minute")
async def oauth_callback(request: Request, provider: str):
    logger.info(f"Callback received: provider={provider}, URL={request.url}, Query params={request.query_params}")
    if provider not in providers:
        logger.error(f"Provider not supported: {provider}")
        raise HTTPException(status_code=404, detail="Provider not supported")
    oauth_provider = oauth.create_client(provider)
    try:
        token = await oauth_provider.authorize_access_token(request)
        logger.debug(f"Token received: {token}")
    except Exception as e:
        logger.error(f"OAuth token error for {provider}: {str(e)}\n{traceback.format_exc()}")
        return RedirectResponse(url=FRONTEND_URL + '/login?error=auth_failed')
    try:
        if provider == 'google':
            user_info = await oauth_provider.userinfo(token=token)
            logger.debug(f"User info: {user_info}")
        else:
            user_resp = await oauth_provider.get(oauth_provider.userinfo_endpoint, token=token)
            user_info = user_resp.json()
            logger.debug(f"User info: {user_info}")
        sub = user_info.get('id') or user_info.get('sub')
        email = user_info.get('email')
        name = user_info.get('name') or user_info.get('username', '')
        profile_image = user_info.get('picture', '')
        if not sub:
            raise ValueError("No user ID from provider")
        if email and not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            raise ValueError("Invalid email format from provider")
        if not re.match(r"^[a-zA-Z0-9_ ]{0,50}$", name):
            name = ""
    except Exception as e:
        logger.error(f"Userinfo error for {provider}: {str(e)}\n{traceback.format_exc()}")
        return RedirectResponse(url=FRONTEND_URL + '/login?error=userinfo_failed')
    query = {"oauth_providers." + provider + ".sub": sub}
    if email:
        query = {"$or": [query, {"email": email}]}
    user = await users_collection.find_one(query)
    refresh_enc = cipher.encrypt(token.get("refresh_token", "").encode()).decode() if token.get("refresh_token") else None
    if user:
        if provider not in user.get("oauth_providers", {}):
            await users_collection.update_one(
                {"_id": user["_id"]},
                {"$set": {f"oauth_providers.{provider}": {"sub": sub, "refresh_token": refresh_enc}}}
            )
    else:
        username = f"{provider}_{sub[:10]}"
        if await users_collection.find_one({"username": username}):
            username += secrets.token_hex(4)
        new_user = User(
            username=username,
            email=email,
            first_name=name.split()[0] if name else "",
            last_name=" ".join(name.split()[1:]) if name else "",
            hashed_password=None,
            oauth_providers={provider: {"sub": sub, "refresh_token": refresh_enc}},
            oauth_only=True,
            profile_image=profile_image
        ).dict(exclude_none=True)
        await users_collection.insert_one(new_user)
        user = await users_collection.find_one({"username": username})
    access_token = create_access_token({"sub": user["username"]})
    secure = ENV == "prod" or not BACKEND_URL.startswith("http://localhost")
    response = RedirectResponse(url=FRONTEND_URL)
    response.set_cookie(
        key="grokbit_token",
        value=access_token,
        httponly=True,
        secure=secure,
        samesite='lax',
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )
    return response

@router.get("/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    response = RedirectResponse(url=FRONTEND_URL + '/login')
    for prov, data in current_user.get("oauth_providers", {}).items():
        refresh_enc = data.get("refresh_token")
        if refresh_enc:
            try:
                refresh = cipher.decrypt(refresh_enc.encode()).decode()
                if prov == 'google':
                    requests.post('https://oauth2.googleapis.com/revoke', params={'token': refresh})
                elif prov == 'github':
                    pass
            except Exception as e:
                logger.error(f"Logout error for {prov}: {str(e)}\n{traceback.format_exc()}")
    response.delete_cookie("grokbit_token", domain=".grokbit.ai")
    return response

@router.get("/check_auth")
async def check_auth(current_user: dict = Depends(get_current_user)):
    return {"authenticated": True}

@router.post("/delete_user")
@limiter.limit("5/minute")
async def delete_user(request: Request, delete_data: Dict = Body(...), current_user: dict = Depends(get_current_user)):
    username = delete_data.get("username")
    password = delete_data.get("password")
    if username != current_user["username"] or not pwd_context.verify(password, current_user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    await users_collection.delete_one({"_id": ObjectId(current_user["_id"])})
    await portfolios_collection.delete_one({"user_id": str(current_user["_id"])})
    await alerts_collection.delete_one({"user_id": str(current_user["_id"])})
    return {"message": "Account deleted"}