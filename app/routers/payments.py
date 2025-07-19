# app/routers/payments.py
from fastapi import APIRouter, Depends, Request
from app.dependencies import get_current_user
from app.database import users_collection, ObjectId
from app.config import STRIPE_WEBHOOK_SECRET
import stripe
from typing import Dict

router = APIRouter(prefix="/payments", tags=["payments"])

@router.post("/create-checkout-session")
async def create_checkout_session(current_user: dict = Depends(get_current_user)):
    session = stripe.checkout.Session.create(
        payment_method_types=['card'],
        line_items=[{'price': 'prod_SgizAO3lVhxP2F', 'quantity': 1}],
        mode='subscription',
        success_url='https://grokbit.ai/success',
        cancel_url='https://grokbit.ai/cancel',
        client_reference_id=str(current_user["_id"])
    )
    return {"id": session.id}

@router.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    if event['type'] == 'checkout.session.completed':
        user_id = event['data']['object']['client_reference_id']
        await users_collection.update_one({"_id": ObjectId(user_id)}, {"$set": {"tier": "premium"}})
    return {"success": True}