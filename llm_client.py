import os
import json
import time
from typing import Dict, Any
from openai import OpenAI

from config import NVIDIA_API_KEY

class UnifiedLLMClient:
    def __init__(self):
        # NVIDIA NIM Client (OpenAI SDK)
        self.nvidia_client = OpenAI(
            api_key=NVIDIA_API_KEY,
            base_url="https://integrate.api.nvidia.com/v1"
        )

    def generate_json(self, 
                      system_prompt: str, 
                      user_prompt: str, 
                      provider: str = "nvidia", 
                      model: str = "meta/llama-3.3-70b-instruct", 
                      temperature: float = 0.0,
                      retries: int = 2) -> Dict[str, Any]:
        """
        Routes JSON generation. Always routes using NVIDIA NIM since Groq has been removed.
        """
        for attempt in range(retries):
            try:
                return self._run_nvidia_json(system_prompt, user_prompt, model, temperature)
            except Exception as e:
                print(f"      [LLM] NVIDIA NIM JSON generation attempt {attempt + 1} failed: {e}")
                time.sleep(1)
        return {}

    def generate_text(self, 
                      system_prompt: str, 
                      user_prompt: str, 
                      provider: str = "nvidia",
                      model: str = "meta/llama-3.3-70b-instruct",
                      temperature: float = 0.0,
                      retries: int = 1) -> str:
        """
        Routes standard text generation. Always routes using NVIDIA NIM since Groq has been removed.
        """
        for attempt in range(retries):
            try:
                return self._run_nvidia_text(system_prompt, user_prompt, model, temperature)
            except Exception as e:
                print(f"      [LLM] NVIDIA NIM Text generation attempt {attempt + 1} failed: {e}")
                time.sleep(1)
        
        return "System Error: NVIDIA NIM text generation failed."

    def generate_text_stream(self, 
                             system_prompt: str, 
                             user_prompt: str, 
                             provider: str = "nvidia",
                             model: str = "meta/llama-3.3-70b-instruct",
                             temperature: float = 0.0,
                             callback=None) -> str:
        """
        Streams standard text generation from NVIDIA NIM and executes a callback with each token.
        """
        try:
            kwargs = {
                "model": model,
                "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                "temperature": temperature,
                "max_tokens": 4096,
                "stream": True
            }
            if "120b" in model:
                kwargs["extra_body"] = {"thinking": "medium"}
                
            res = self.nvidia_client.chat.completions.create(**kwargs)
            
            full_response = []
            for chunk in res:
                if not chunk.choices:
                    continue
                token = chunk.choices[0].delta.content or ""
                if token:
                    full_response.append(token)
                    if callback:
                        callback(token)
            return "".join(full_response)
        except Exception as e:
            print(f"      [LLM] Streaming failed ({e}). Falling back to non-stream...")
            return self.generate_text(system_prompt, user_prompt, provider, model, temperature)

    # ==========================================
    # PRIVATE HELPER METHODS (NVIDIA NIM API calls)
    # ==========================================
    def _run_nvidia_json(self, sys_prompt: str, user_prompt: str, model: str, temp: float) -> Dict:
        modified_sys = sys_prompt + "\n\nYou must respond ONLY with a valid JSON object."
        content = None
        
        try:
            kwargs = {
                "model": model,
                "messages": [{"role": "system", "content": modified_sys}, {"role": "user", "content": user_prompt}],
                "temperature": temp,
                "max_tokens": 4096,
                "response_format": {"type": "json_object"}
            }
            if "120b" in model:
                kwargs["extra_body"] = {"thinking": "medium"}
                
            res = self.nvidia_client.chat.completions.create(**kwargs)
            content = res.choices[0].message.content
        except Exception as e:
            # Fallback: Retry without response_format if model or endpoint rejects it
            print(f"      [LLM] Warning: response_format failed ({e}). Retrying without it...")
            fallback_kwargs = {
                "model": model,
                "messages": [{"role": "system", "content": modified_sys}, {"role": "user", "content": user_prompt}],
                "temperature": temp,
                "max_tokens": 4096
            }
            if "120b" in model:
                fallback_kwargs["extra_body"] = {"thinking": "medium"}
                
            res = self.nvidia_client.chat.completions.create(**fallback_kwargs)
            content = res.choices[0].message.content

        if not content:
            print(f"      [LLM] ⚠️ EMPTY CONTENT RESPONSE: {res}")
            raise ValueError("NVIDIA NIM returned empty or None content.")
            
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            import re
            # Find first { and last }
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise ValueError(f"Failed to parse JSON from response content: {content}")

    def _run_nvidia_text(self, sys_prompt: str, user_prompt: str, model: str, temp: float) -> str:
        kwargs = {
            "model": model,
            "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
            "temperature": temp,
            "max_tokens": 4096
        }
        if "120b" in model:
            kwargs["extra_body"] = {"thinking": "medium"}
            
        res = self.nvidia_client.chat.completions.create(**kwargs)
        return res.choices[0].message.content

    # ==========================================
    # EMBEDDING & RERANKING APIS
    # ==========================================
    def generate_embeddings(self, text: str, model: str = "nvidia/nv-embed-v1") -> list:
        """Generates embeddings using NVIDIA NIM API."""
        response = self.nvidia_client.embeddings.create(
            input=[text],
            model=model,
            encoding_format="float",
            extra_body={"input_type": "query", "truncate": "NONE"}
        )
        return response.data[0].embedding

    def rerank_passages(self, query: str, passages: list, model: str = "nv-rerank-qa-mistral-4b:1") -> list:
        """
        Calls NVIDIA reranking API.
        """
        import requests
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Accept": "application/json",
        }
        payload = {
            "model": model,
            "query": {"text": query},
            "passages": [{"text": p} for p in passages]
        }
        url = "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        return response.json().get('rankings', [])

# Instantiate the singleton client
llm_client = UnifiedLLMClient()