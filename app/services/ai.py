# app/services/ai.py
from typing import Optional, Generator
from groq import Groq
import google.generativeai as genai
from huggingface_hub import InferenceClient
import requests
import time
from app.config import DEFAULT_MODELS
import openai
from anthropic import Anthropic
from mistralai.client import MistralClient
from cohere import Client as CohereClient
from replicate import Client as ReplicateClient
from together import Together
import json
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

async def call_ai(provider: str, messages: list, user_api_key: Optional[str] = None, model: str = None, stream: bool = False):
    api_key = user_api_key
    if not api_key:
        raise ValueError(f"{provider} API key required")
    if not model:
        model = DEFAULT_MODELS.get(provider)
    if not stream:
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
        elif provider == "OpenAI":
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(messages=messages, model=model, stream=False, temperature=0)
            return response.choices[0].message.content
        elif provider == "Anthropic":
            client = Anthropic(api_key=api_key)
            response = client.messages.create(messages=messages, model=model, stream=False, temperature=0, max_tokens=500)
            return response.content[0].text
        elif provider == "Mistral":
            client = MistralClient(api_key=api_key)
            response = client.chat(messages=messages, model=model, stream=False, temperature=0)
            return response.choices[0].message.content
        elif provider == "Cohere":
            client = CohereClient(api_key=api_key)
            response = client.chat(messages=messages, model=model, stream=False, temperature=0)
            return response.text
        elif provider == "Perplexity":
            client = openai.OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
            response = client.chat.completions.create(messages=messages, model=model, stream=False, temperature=0)
            return response.choices[0].message.content
        elif provider == "Replicate":
            client = ReplicateClient(api_key=api_key)
            response = client.run(model, input={"prompt": messages[-1]["content"]})
            return ''.join(response)
        elif provider == "Together":
            client = Together(api_key=api_key)
            response = client.chat.completions.create(messages=messages, model=model, stream=False, temperature=0)
            return response.choices[0].message.content
        elif provider == "Fireworks":
            client = openai.OpenAI(api_key=api_key, base_url="https://api.fireworks.ai/inference/v1")
            response = client.chat.completions.create(messages=messages, model=model, stream=False, temperature=0)
            return response.choices[0].message.content
        elif provider == "Novita":
            client = openai.OpenAI(api_key=api_key, base_url="https://api.novita.ai/v3")
            response = client.chat.completions.create(messages=messages, model=model, stream=False, temperature=0)
            return response.choices[0].message.content
        raise ValueError("Invalid provider")
    else:
        # Streaming mode
        if provider == "Groq":
            groq_client = Groq(api_key=api_key)
            response = groq_client.chat.completions.create(messages=messages, model=model, stream=True, temperature=0)
            def gen():
                for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
            return gen()
        elif provider == "Gemini":
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel(model)
            response = gemini_model.generate_content(messages[-1]["content"], stream=True)
            def gen():
                for chunk in response:
                    yield chunk.text
            return gen()
        elif provider == "HuggingFace":
            hf_client = InferenceClient(model=model, token=api_key)
            response = hf_client.text_generation(messages[-1]["content"], max_new_tokens=500, stream=True)
            def gen():
                for chunk in response:
                    yield chunk
            return gen()
        elif provider == "Grok":
            url = "https://api.x.ai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {"messages": messages, "model": model, "stream": True, "temperature": 0}
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    response = requests.post(url, headers=headers, json=payload, stream=True, timeout=120)
                    response.raise_for_status()
                    def gen():
                        for line in response.iter_lines(decode_unicode=True):
                            if line.startswith('data: '):
                                data_str = line[6:]
                                if data_str == '[DONE]':
                                    break
                                try:
                                    data = json.loads(data_str)
                                    content = data.get('choices', [{}])[0].get('delta', {}).get('content')
                                    if content:
                                        yield content
                                except json.JSONDecodeError as je:
                                    logger.error(f"JSON decode error in Grok streaming: {je} - line: {line}")
                                    continue
                        response.close()
                    return gen()
                except requests.exceptions.RequestException as e:
                    logger.error(f"Grok streaming API request error: {e}")
                    status_code = e.response.status_code if hasattr(e, 'response') and e.response else None
                    if status_code in (429, 500, 502, 503, 504):
                        if attempt < max_retries - 1:
                            time.sleep(5)
                            continue
                        else:
                            raise HTTPException(status_code=status_code, detail="Grok API unavailable after retries")
                    if attempt < max_retries - 1:
                        time.sleep(5)
                        continue
                    else:
                        raise ValueError(f"Grok streaming API request failed after {max_retries} attempts: {str(e)}")
        elif provider == "OpenAI":
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(messages=messages, model=model, stream=True, temperature=0)
            def gen():
                for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
            return gen()
        elif provider == "Anthropic":
            client = Anthropic(api_key=api_key)
            response = client.messages.create(messages=messages, model=model, stream=True, temperature=0, max_tokens=500)
            def gen():
                for event in response:
                    if event.type == 'content_block_delta':
                        yield event.delta.text
            return gen()
        elif provider == "Mistral":
            client = MistralClient(api_key=api_key)
            response = client.chat(messages=messages, model=model, stream=True, temperature=0)
            def gen():
                for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
            return gen()
        elif provider == "Cohere":
            client = CohereClient(api_key=api_key)
            response = client.chat(messages=messages, model=model, stream=True, temperature=0)
            def gen():
                for event in response:
                    if event.event_type == "text-generation":
                        yield event.text
            return gen()
        elif provider == "Perplexity":
            client = openai.OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
            response = client.chat.completions.create(messages=messages, model=model, stream=True, temperature=0)
            def gen():
                for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
            return gen()
        elif provider == "Replicate":
            # Replicate streaming might require different handling; fallback to non-stream for now
            response = ''.join(client.run(model, input={"prompt": messages[-1]["content"]}))
            def gen():
                yield response
            return gen()
        elif provider == "Together":
            client = Together(api_key=api_key)
            response = client.chat.completions.create(messages=messages, model=model, stream=True, temperature=0)
            def gen():
                for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
            return gen()
        elif provider == "Fireworks":
            client = openai.OpenAI(api_key=api_key, base_url="https://api.fireworks.ai/inference/v1")
            response = client.chat.completions.create(messages=messages, model=model, stream=True, temperature=0)
            def gen():
                for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
            return gen()
        elif provider == "Novita":
            client = openai.OpenAI(api_key=api_key, base_url="https://api.novita.ai/v3")
            response = client.chat.completions.create(messages=messages, model=model, stream=True, temperature=0)
            def gen():
                for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
            return gen()
        raise ValueError("Invalid provider")