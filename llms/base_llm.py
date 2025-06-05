"""
This module contains a wrapper for LLM models, providing a unified interface for different LLM implementations. The wrapper facilitates local and remote model usage, including OpenAI's GPT models and local models like Microsoft Phi 4 via Hugging Face. The wrapper supports both synchronous and asynchronous calls, allowing for efficient scaling.
"""
import os
import asyncio
import aiohttp
from dotenv import load_dotenv
load_dotenv()


class BaseLLM:
    def __init__(self, backend="openai", model_name="gpt-4", temperature=0.3, **kwargs):
        self.backend = backend
        self.model_name = model_name
        self.temperature = temperature
        self.config = kwargs

        if backend == "openai":
            from langchain.chat_models import ChatOpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY required in env file")
            self.model = ChatOpenAI(model_name=model_name, temperature=temperature, openai_api_key=api_key)

        elif backend == "huggingface":
            from langchain.llms import HuggingFacePipeline
            from transformers import pipeline
            pipe = pipeline("text-generation", model=model_name, max_new_tokens=512, do_sample=False)
            self.model = HuggingFacePipeline(pipeline=pipe)

        elif backend == "gemini":
            self.api_key = os.getenv("GEMINI_API_KEY") or kwargs.get("api_key")
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY required in env file")

        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def invoke(self, prompt: str):
        if self.backend == "gemini":
            raise RuntimeError("Use ainvoke() for Gemini.")
        return self.model.invoke(prompt)

    async def ainvoke(self, prompt: str):
        if self.backend == "openai":
            return await self.model.ainvoke(prompt)

        elif self.backend == "huggingface":
            return await asyncio.to_thread(self.model.invoke, prompt)

        elif self.backend == "gemini":
            return await self._call_gemini_api(prompt)

    async def _call_gemini_api(self, prompt: str):
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}
        data = {"contents": [{"parts": [{"text": prompt}]}]}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as resp:
                result = await resp.json()
                return result["candidates"][0]["content"]["parts"][0]["text"]

