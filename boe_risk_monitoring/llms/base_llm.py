"""
This module contains a wrapper for LLM models, providing a unified interface for different LLM implementations. The wrapper facilitates local and remote model usage, including OpenAI's GPT models and local models like Microsoft Phi 4 via Hugging Face. The wrapper supports both synchronous and asynchronous calls, allowing for efficient scaling.
"""
from google import genai
from openai import OpenAI, AsyncOpenAI
import os
import asyncio
import aiohttp
from dotenv import load_dotenv
load_dotenv()


SUPPORTED_LLMS_DICT = {
    "openai": {
        "gpt-4o": {
            "context_window": 128000,
            "max_output": 16384,
        },
        "gpt-4.1": {
            "context_window": 1047576,
            "max_output": 32768,
        }
    },
    "gemini": {
        "gemini-2.0-flash": {
            "context_window": 1048576,
            "max_output": 8192,
        },
        "gemini-2.5-flash-preview-05-20": {
            "context_window": 1048576,
            "max_output": 65536,
        },
        "gemini-2.5-pro-preview-06-05": {
            "context_window": 1048576,
            "max_output": 65536,
        },
    }
}


class BaseLLM:
    def __init__(self, backend="openai", model_name="gpt-4o", temperature=0.3):
        if backend not in SUPPORTED_LLMS_DICT:
            raise ValueError(f"Unsupported backend. Supported backends: {SUPPORTED_LLMS_DICT.keys()}")
        if model_name not in SUPPORTED_LLMS_DICT[backend]:
            raise ValueError(
                f"Unsupported model. Supported models for backend {backend}: {list(SUPPORTED_LLMS_DICT[backend].keys())}")
        self.backend = backend
        self.model_name = model_name
        self.temperature = temperature
        self.context_window = SUPPORTED_LLMS_DICT[backend][model_name]['context_window']
        self.max_output = SUPPORTED_LLMS_DICT[backend][model_name]['max_output']

        if backend == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY required in env file")
            self.api_key = api_key

        elif backend == "gemini":
            api_key = os.getenv("GEMINI_API_KEY") or kwargs.get("api_key")
            if not api_key:
                raise ValueError("GEMINI_API_KEY required in env file")
            self.api_key = api_key

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def invoke(self, prompt: str):
        if self.backend == "openai":
            client = OpenAI(api_key=self.api_key)
            resp = client.responses.create(
                model=self.model_name,
                input=prompt,
                temperature=self.temperature,
            )
            return resp.output_text

        if self.backend == "gemini":
            client = genai.Client(api_key=self.api_key)
            resp = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            return resp.text

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    async def ainvoke(self, prompt: str):
        if self.backend == "openai":
            client = AsyncOpenAI(api_key=self.api_key)
            resp = await client.responses.create(
                model=self.model_name,
                input=prompt,
                temperature=self.temperature,
            )
            return resp.output_text

        elif self.backend == "gemini":
            return await self._call_gemini_api(prompt)

    async def _call_gemini_api(self, prompt: str):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        # headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                result = await resp.json()
                # Extract text from the response
                try:
                    content = result['candidates'][0]['content']['parts'][0]['text']
                    return content
                except (KeyError, IndexError) as e:
                    # Log the error and response for debugging
                    print(f"Error parsing response: {e}")
                    print(f"Unexpected response format: {result}")
                    return "Error: Unexpected response format"


async def invoke_async_llm(llm, prompts):
    tasks = [asyncio.create_task(llm.ainvoke(prompt)) for prompt in prompts]
    resps = await asyncio.gather(*tasks)
    return resps


if __name__ == "__main__":
    # Instantiate base LLM class with Open AI backend
    base_llm_openai = BaseLLM(backend="openai", model_name="gpt-4o")

    # Test invocation (this will charge to your account)
    resp1 = base_llm_openai.invoke("Tell me something interesting.")

    # Print out response
    print("PRINTING OPENAI SYNC RESPONSE>>>")
    print(resp1 + "\n")

    # Test inconvation (this will charge to your account)
    prompts = ["Why is the sky blue?", "Tell me a Dad joke."]
    resps2 = asyncio.run(invoke_async_llm(base_llm_openai, prompts))

    # Print out response
    print("PRINTING OPENAI ASYNC RESPONSES>>>")
    for resp in resps2:
        print("====================================================")
        print(resp + "\n")

    # Instantiate base LLM class with Gemini backend
    base_llm_gemini = BaseLLM(backend="gemini", model_name="gemini-2.0-flash")

    # Test invocation (this will charge to your account if not using one of the free to use models on the free tier)
    resp3 = base_llm_gemini.invoke("Tell me something interesting.")

    # Print out response
    print("PRINTING GEMINI SYNC RESPONSE>>>")
    print(resp3 + "\n")

    # Test async invocation (this will charge to your account if not using one of the free to use models on the free tier)
    prompts = ["Why is the sky blue?", "Tell me a Dad joke."]
    resps4 = asyncio.run(invoke_async_llm(base_llm_gemini, prompts))

    # Print out response
    print("PRINTING GEMINI ASYNC RESPONSES>>>")
    for resp in resps4:
        print("====================================================")
        print(resp + "\n")
