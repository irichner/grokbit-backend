# app/routers/auth.py
from fastapi import APIRouter, HTTPException, Depends, Body, Request, Response, Cookie
from fastapi.security import OAuth2PasswordRequestForm
from authlib.integrations.starlette_client import OAuth
from starlette.responses import RedirectResponse
from app.config import CLIENT_ID_GOOGLE, CLIENT_SECRET_GOOGLE, CLIENT_ID_GITHUB, CLIENT_SECRET_GITHUB, BACKEND_URL, FRONTEND_URL, ENV, SECRET_KEY, ACCESS_TOKEN_EXPIRE_MINUTES, REFRESH_TOKEN_EXPIRE_MINUTES
from app.database import users_collection, portfolios_collection, alerts_collection, ObjectId
from app.models.user import User
from app.models.other import ForgotPasswordRequest, ResetPasswordRequest
from app.services.auth import create_access_token, create_refresh_token, create_verification_token, create_reset_token, pwd_context
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
import pwnedpasswords

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
    if len(password) < 12 or len(password) > 64:
        raise HTTPException(status_code=400, detail="Password must be 12-64 characters")
    pwned = pwnedpasswords.check(password)
    if pwned > 0:
        raise HTTPException(status_code=400, detail="Password has been breached; choose another")
    if await users_collection.find_one({"username": username}):
        raise HTTPException(status_code=400, detail="Username taken")
    hashed_password = pwd_context.hash(password)
    admin_user = await users_collection.find_one({"username": "irichner"})
    default_prefs = {"prompt_default_provider": "Groq", "summary_default_provider": "Groq", "api_keys": {"Groq": "", "Gemini": "", "HuggingFace": "", "Grok": "", "CoinGecko": ""}, "prompts": [], "portfolio_prompts": [], "alert_prompts": [], "models": DEFAULT_MODELS, "refresh_rate": 60000, "market_coins": []}
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
        "profile_image": "",
        "verified": False
    }
    await users_collection.insert_one(user_dict)
    token = create_verification_token({"sub": username})
    verification_url = f"{FRONTEND_URL}/verify?token={token}"
    await send_verification_email(username, verification_url)
    logger.info(f"User registered: {username}")
    return {"message": "User registered"}

@router.get("/verify")
async def verify(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        user = await users_collection.find_one({"username": username})
        if user:
            await users_collection.update_one({"username": username}, {"$set": {"verified": True}})
            return {"message": "Email verified"}
        raise HTTPException(status_code=400, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=400, detail="Invalid token")

@router.post("/resend-verification")
async def resend_verification(current_user: dict = Depends(get_current_user)):
    if current_user["verified"]:
        raise HTTPException(status_code=400, detail="Already verified")
    token = create_verification_token({"sub": current_user["username"]})
    verification_url = f"{FRONTEND_URL}/verify?token={token}"
    await send_verification_email(current_user["username"], verification_url)
    return {"message": "Verification resent"}

@router.post("/token")
@limiter.limit("5/minute")
async def login(response: Response, request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    user = await users_collection.find_one({"username": form_data.username})
    if not user or user.get("oauth_only") or not pwd_context.verify(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not user["verified"]:
        raise HTTPException(status_code=403, detail="Verify email first")
    access_token = create_access_token({"sub": user["username"]})
    refresh_token = create_refresh_token({"sub": user["username"]})
    secure = ENV == "prod" or not BACKEND_URL.startswith("http://localhost")
    cookie_domain = ".grokbit.ai" if ENV == "prod" else None
    response.set_cookie(
        key="grokbit_token",
        value=access_token,
        httponly=True,
        secure=secure,
        samesite='strict',
        path='/',
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        domain=cookie_domain
    )
    response.set_cookie(
        key="grokbit_refresh",
        value=refresh_token,
        httponly=True,
        secure=secure,
        samesite='strict',
        path='/',
        max_age=REFRESH_TOKEN_EXPIRE_MINUTES * 60,
        domain=cookie_domain
    )
    logger.info(f"User logged in: {form_data.username}")
    return {"success": True}

@router.post("/refresh")
@limiter.limit("5/minute")
async def refresh(response: Response, grokbit_refresh: str = Cookie(None), request: Request = None):
    credentials_exception = HTTPException(status_code=401, detail="Invalid refresh token")
    if not grokbit_refresh:
        raise credentials_exception
    try:
        payload = jwt.decode(grokbit_refresh, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = await users_collection.find_one({"username": username})
    if not user:
        raise credentials_exception
    access_token = create_access_token({"sub": username})
    secure = ENV == "prod" or not BACKEND_URL.startswith("http://localhost")
    cookie_domain = ".grokbit.ai" if ENV == "prod" else None
    response.set_cookie(
        key="grokbit_token",
        value=access_token,
        httponly=True,
        secure=secure,
        samesite='strict',
        path='/',
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        domain=cookie_domain
    )
    return {"success": True}

@router.post("/forgot-password")
@limiter.limit("5/minute")
async def forgot_password(body: ForgotPasswordRequest = Body(...), request: Request = None):
    user = await users_collection.find_one({"email": body.email})
    if not user:
        return {"message": "If email exists, reset link sent"}
    token = create_reset_token({"sub": user["username"]})
    reset_url = f"{FRONTEND_URL}/reset?token={token}"
    message = Mail(
        from_email='no-reply@grokbit.ai',
        to_emails=body.email,
        subject='Reset Your GrokBit Password',
        html_content=f'<strong>Click to reset: <a href="{reset_url}">Reset</a></strong>'
    )
    try:
        sendgrid_client = SendGridAPIClient(os.getenv('SENDGRID_API_KEY'))
        sendgrid_client.send(message)
    except Exception as e:
        logger.error(f"Reset email failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to send reset")
    return {"message": "If email exists, reset link sent"}

@router.post("/reset-password")
@limiter.limit("5/minute")
async def reset_password(body: ResetPasswordRequest = Body(...), request: Request = None):
    try:
        payload = jwt.decode(body.token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        user = await users_collection.find_one({"username": username})
        if not user:
            raise HTTPException(status_code=400, detail="Invalid token")
        if len(body.password) < 12 or len(body.password) > 64:
            raise HTTPException(status_code=400, detail="Password must be 12-64 characters")
        pwned = pwnedpasswords.check(body.password)
        if pwned > 0:
            raise HTTPException(status_code=400, detail="Password has been breached; choose another")
        hashed_password = pwd_context.hash(body.password)
        await users_collection.update_one({"username": username}, {"$set": {"hashed_password": hashed_password}})
        return {"message": "Password reset"}
    except JWTError:
        raise HTTPException(status_code=400, detail="Invalid token")

@router.get("/oauth/{provider}/login")
@limiter.limit("5/minute")
async def oauth_login(request: Request, provider: str):
    if provider not in providers:
        raise HTTPException(status_code=404, detail="Provider not supported")
    oauth_provider = oauth.create_client(provider)
    redirect_uri = BACKEND_URL + f"/auth/oauth/{provider}/callback"
    return await oauth_provider.authorize_redirect(request, redirect_uri)

@router.get("/oauth/{provider}/callback")
@limiter.limit("5/minute")
async def oauth_callback(request: Request, provider: str):
    if provider not in providers:
        raise HTTPException(status_code=404, detail="Provider not supported")
    oauth_provider = oauth.create_client(provider)
    try:
        token = await oauth_provider.authorize_access_token(request)
    except Exception as e:
        logger.error(f"OAuth token fetch failed for {provider}: {e}")
        return RedirectResponse(url=FRONTEND_URL + '/login?error=auth_failed')
    try:
        if provider == 'google':
            user_info = await oauth_provider.userinfo(token=token)
        else:
            userinfo_endpoint = oauth_provider.userinfo_endpoint
            user_resp = await oauth_provider.get(userinfo_endpoint, token=token)
            user_info = user_resp.json()
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
        logger.error(f"OAuth userinfo fetch failed for {provider}: {e}")
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
        new_user = {
            "username": username,
            "hashed_password": None,
            "email": email,
            "first_name": name.split()[0] if name else "",
            "last_name": " ".join(name.split()[1:]) if name else "",
            "preferences": {"prompt_default_provider": "Groq", "summary_default_provider": "Groq", "api_keys": {"Groq": "", "Gemini": "", "HuggingFace": "", "Grok": "", "CoinGecko": ""}, "prompts": [], "portfolio_prompts": [], "alert_prompts": [], "models": DEFAULT_MODELS, "refresh_rate": 60000, "market_coins": []},
            "oauth_providers": {provider: {"sub": sub, "refresh_token": refresh_enc}},
            "oauth_only": True,
            "tier": "free",
            "profile_image": profile_image,
            "verified": True  # OAuth verified
        }
        await users_collection.insert_one(new_user)
        user = await users_collection.find_one({"username": username})
    access_token = create_access_token({"sub": user["username"]})
    secure = ENV == "prod" or not BACKEND_URL.startswith("http://localhost")
    cookie_domain = ".grokbit.ai" if ENV == "prod" else None
    response = RedirectResponse(url=FRONTEND_URL)
    response.set_cookie(
        key="grokbit_token",
        value=access_token,
        httponly=True,
        secure=secure,
        samesite='strict',
        path='/',
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        domain=cookie_domain
    )
    logger.info(f"OAuth login successful for {provider}: {user['username']}")
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
                    # GitHub no revoke
                    pass
            except Exception as e:
                logger.error(f"Failed to revoke {prov} token: {e}")
    response.delete_cookie("grokbit_token", domain=cookie_domain)
    response.delete_cookie("grokbit_refresh", domain=cookie_domain)
    logger.info(f"User logged out: {current_user['username']}")
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
    logger.info(f"User deleted: {username}")
    return {"message": "Account deleted"}