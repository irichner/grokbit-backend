# app/services/ai.py
from typing import Optional
from groq import Groq
import google.generativeai as genai
from huggingface_hub import InferenceClient
import requests
import time
from app.config import DEFAULT_MODELS

async def call_ai(provider: str, messages: list, user_api_key: Optional[str] = None, model: str = None):
    api_key = user_api_key
    if not api_key:
        raise ValueError(f"{provider} API key required")
    if not model:
        model = DEFAULT_MODELS.get(provider)
    if provider == "Groq":
        groq_client = Groq(api_key=api_key)
        response = groq_client.chat.completions.create(messages=messages, model=model, stream=False, temperature=0)
        return response.choices[0].message.content
    elif provider == "Gemini":
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel(model)
        response = gemini_model.generate_content(messages[-1]["content"])
        return response.text
    elif provider == "HuggingFace":
        hf_client = InferenceClient(model=model, token=api_key)
        return hf_client.text_generation(messages[-1]["content"], max_new_tokens=500)
    elif provider == "Grok":
        url = "https://api.x.ai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"messages": messages, "model": model, "stream": False, "temperature": 0}
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                data = response.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "No response")
            except requests.exceptions.RequestException as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Grok API request error: {e}")
                status_code = e.response.status_code if hasattr(e, 'response') and e.response else None
                if status_code == 503:
                    if attempt < max_retries - 1:
                        time.sleep(5)
                        continue
                    else:
                        raise HTTPException(status_code=503, detail="Grok API unavailable after retries")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                else:
                    raise ValueError(f"Grok API request failed after {max_retries} attempts: {str(e)}")
    raise ValueError("Invalid provider")