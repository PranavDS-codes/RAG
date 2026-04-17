import json
import time
from typing import Dict, Any
from openai import OpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from config import GROQ_API_KEY, NVIDIA_API_KEY

class UnifiedLLMClient:
    def __init__(self):
        # 1. Groq Client (LangChain)
        self.groq_api_key = GROQ_API_KEY
        
        # 2. NVIDIA NIM Client (OpenAI SDK)
        self.nvidia_client = OpenAI(
            api_key=NVIDIA_API_KEY,
            base_url="https://integrate.api.nvidia.com/v1"
        )

    # ==========================================
    # CORE ROUTING LOGIC
    # ==========================================
    def generate_json(self, 
                      system_prompt: str, 
                      user_prompt: str, 
                      primary_provider: str = "groq", 
                      groq_model: str = "llama-3.3-70b-versatile", 
                      nvidia_model: str = "qwen/qwen3-coder-480b-a35b-instruct",
                      retries: int = 2) -> Dict[str, Any]:
        """
        Dynamically routes JSON generation. If the primary provider fails, 
        it automatically switches to the backup provider.
        """
        # Set up the Primary and Backup targets
        if primary_provider.lower() == "nvidia":
            primary_name, primary_func, p_model = "NVIDIA NIM", self._run_nvidia_json, nvidia_model
            backup_name, backup_func, b_model = "Groq", self._run_groq_json, groq_model
        else:
            primary_name, primary_func, p_model = "Groq", self._run_groq_json, groq_model
            backup_name, backup_func, b_model = "NVIDIA NIM", self._run_nvidia_json, nvidia_model

        # --- ATTEMPT 1: PRIMARY PROVIDER ---
        for attempt in range(retries):
            try:
                return primary_func(system_prompt, user_prompt, p_model)
            except Exception as e:
                print(f"      [LLM] {primary_name} attempt {attempt + 1} failed: {e}")
                time.sleep(1)

        # --- ATTEMPT 2: BACKUP PROVIDER ---
        print(f"      [LLM] 🔄 Switching to Backup Provider [{backup_name}] using model: {b_model}...")
        try:
            return backup_func(system_prompt, user_prompt, b_model)
        except Exception as e:
            print(f"      [LLM] ❌ Backup Provider [{backup_name}] also failed: {e}")
            return {}

    def generate_text(self, 
                      system_prompt: str, 
                      user_prompt: str, 
                      primary_provider: str = "groq",
                      groq_model: str = "llama-3.3-70b-versatile",
                      nvidia_model: str = "qwen/qwen3-coder-480b-a35b-instruct",
                      temperature: float = 0.0,
                      retries: int = 1) -> str:
        """
        Dynamically routes standard text generation (e.g., final answer synthesis).
        """
        if primary_provider.lower() == "nvidia":
            primary_name, primary_func, p_model = "NVIDIA NIM", self._run_nvidia_text, nvidia_model
            backup_name, backup_func, b_model = "Groq", self._run_groq_text, groq_model
        else:
            primary_name, primary_func, p_model = "Groq", self._run_groq_text, groq_model
            backup_name, backup_func, b_model = "NVIDIA NIM", self._run_nvidia_text, nvidia_model

        for attempt in range(retries):
            try:
                return primary_func(system_prompt, user_prompt, p_model, temperature)
            except Exception as e:
                print(f"      [LLM] {primary_name} attempt {attempt + 1} failed: {e}")
                time.sleep(1)

        print(f"      [LLM] 🔄 Switching to Backup Provider [{backup_name}] using model: {b_model}...")
        try:
            return backup_func(system_prompt, user_prompt, b_model, temperature)
        except Exception as e:
            print(f"      [LLM] ❌ Backup Provider [{backup_name}] also failed: {e}")
            return "System Error: All generation providers failed to process the request."

    # ==========================================
    # PRIVATE HELPER METHODS (The actual API calls)
    # ==========================================
    def _run_groq_json(self, sys_prompt: str, user_prompt: str, model: str) -> Dict:
        llm = ChatGroq(temperature=0, model_name=model, api_key=self.groq_api_key, model_kwargs={"response_format": {"type": "json_object"}})
        res = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
        return json.loads(res.content)

    def _run_nvidia_json(self, sys_prompt: str, user_prompt: str, model: str) -> Dict:
        res = self.nvidia_client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0, max_tokens=2048, response_format={"type": "json_object"}
        )
        return json.loads(res.choices[0].message.content)

    def _run_groq_text(self, sys_prompt: str, user_prompt: str, model: str, temp: float) -> str:
        llm = ChatGroq(temperature=temp, model_name=model, api_key=self.groq_api_key)
        res = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
        return res.content

    def _run_nvidia_text(self, sys_prompt: str, user_prompt: str, model: str, temp: float) -> str:
        res = self.nvidia_client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
            temperature=temp, max_tokens=2048
        )
        return res.choices[0].message.content

# Instantiate the singleton client
llm_client = UnifiedLLMClient()