# app/utils/email.py
import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import logging

logger = logging.getLogger(__name__)

sendgrid_client = SendGridAPIClient(os.getenv("SENDGRID_API_KEY"))  # Assume key in env

async def send_verification_email(email: str, verification_url: str):
    message = Mail(
        from_email='no-reply@grokbit.ai',
        to_emails=email,
        subject='Verify Your GrokBit Email',
        html_content=f'<strong>Click the link to verify: <a href="{verification_url}">Verify</a></strong>'
    )
    try:
        sendgrid_client.send(message)
    except Exception as e:
        logger.error(f"Email send failed: {e}")