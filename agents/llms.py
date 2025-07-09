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
            "pdf_input": True,
        },
        "gpt-4.1": {
            "context_window": 1047576,
            "max_output": 32768,
            "pdf_input": True,
        },
        "gpt-4.1-mini": {
            "context_window": 1047576,
            "max_output": 32768,
            "pdf_input": True,
        },
        "gpt-o4-mini": {
            "context_window": 200000,
            "max_output": 100000,
            "pdf_input": True,
        }

    },
    # "gemini": {
    #     "gemini-2.0-flash": {
    #         "context_window": 1048576,
    #         "max_output": 8192,
    #         "pdf_input": True,
    #     },
    #     "gemini-2.5-flash-preview-05-20": {
    #         "context_window": 1048576,
    #         "max_output": 65536,
    #         "pdf_input": True,
    #     },
    #     "gemini-2.5-pro-preview-06-05": {
    #         "context_window": 1048576,
    #         "max_output": 65536,
    #         "pdf_input": True,
    #     },
    # }
}


class BaseLLM:
    def __init__(self, prompt, backend="openai", model_name="gpt-4o", stream=False):
        if backend not in SUPPORTED_LLMS_DICT:
            raise ValueError(f"Unsupported backend. Supported backends: {SUPPORTED_LLMS_DICT.keys()}")
        if model_name not in SUPPORTED_LLMS_DICT[backend]:
            raise ValueError(
                f"Unsupported model. Supported models for backend {backend}: {list(SUPPORTED_LLMS_DICT[backend].keys())}")
        self.prompt = prompt
        self.backend = backend
        self.model_name = model_name
        self.context_window = SUPPORTED_LLMS_DICT[backend][model_name]['context_window']
        self.max_output = SUPPORTED_LLMS_DICT[backend][model_name]['max_output']
        self.stream = stream

        if backend == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY required in env file")
            self.api_key = api_key

        # elif backend == "gemini":
        #     api_key = os.getenv("GEMINI_API_KEY") or kwargs.get("api_key")
        #     if not api_key:
        #         raise ValueError("GEMINI_API_KEY required in env file")
        #     self.api_key = api_key

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def invoke(self, prompt=None):
        prompt = prompt or self.prompt
        if self.backend == "openai":
            client = OpenAI(api_key=self.api_key)
            resp = client.responses.create(
                model=self.model_name,
                input=prompt,
                stream=self.stream,
            )
            return resp

        # if self.backend == "gemini":
        #     client = genai.Client(api_key=self.api_key)
        #     resp = client.models.generate_content(
        #         model=self.model_name,
        #         contents=prompt,
        #     )
        #     return resp.text

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    async def ainvoke(self, prompt=None):
        prompt = prompt or self.prompt
        if self.backend == "openai":
            client = AsyncOpenAI(api_key=self.api_key)
            resp = await client.responses.create(
                model=self.model_name,
                input=prompt,
                stream=self.stream,
            )
            # async for event in stream:
            #     yield event
            return resp

        # elif self.backend == "gemini":
        #     return await self._call_gemini_api(prompt)

    # async def _call_gemini_api(self, prompt: str):
    #     url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
    #     # headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}
    #     headers = {
    #         "Content-Type": "application/json"
    #     }
    #     payload = {
    #         "contents": [
    #             {
    #                 "parts": [
    #                     {
    #                         "text": prompt
    #                     }
    #                 ]
    #             }
    #         ]
    #     }
    #     async with aiohttp.ClientSession() as session:
    #         async with session.post(url, headers=headers, json=payload) as resp:
    #             result = await resp.json()
    #             # Extract text from the response
    #             try:
    #                 content = result['candidates'][0]['content']['parts'][0]['text']
    #                 return content
    #             except (KeyError, IndexError) as e:
    #                 # Log the error and response for debugging
    #                 print(f"Error parsing response: {e}")
    #                 print(f"Unexpected response format: {result}")
    #                 return "Error: Unexpected response format"


class OrchestratorLLM(BaseLLM):
    def __init__(self, prompt, response_schema, backend="openai", model_name="gpt-4.1"):
        super().__init__(prompt=prompt, backend=backend, model_name=model_name)
        if not isinstance(prompt, str):
            raise TypeError("Chunking prompt must be a string.")
        self.prompt = prompt
        self.response_schema = response_schema

    def invoke(self, prompt=None):
        prompt = prompt or self.prompt
        if self.backend == "openai":
            client = OpenAI(api_key=self.api_key)
            resp = client.responses.parse(
                model=self.model_name,
                input=prompt,
                text_format=self.response_schema,
            )
            return resp.output_parsed

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    async def ainvoke(self, prompt=None):
        prompt = prompt or self.prompt
        if self.backend == "openai":
            client = AsyncOpenAI(api_key=self.api_key)
            resp = await client.responses.parse(
                model=self.model_name,
                input=prompt,
                text_format=self.response_schema,
            )
            return resp.output_parsed

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

# Copy the OrchestratorLLM class
class VectorDBSearchLLM(OrchestratorLLM):
    """Identical to OrchestratorLLM but just a different name."""
    def __init__(self, prompt, response_schema, backend="openai", model_name="gpt-4.1"):
        super().__init__(prompt=prompt, response_schema=response_schema, backend=backend, model_name=model_name)

async def invoke_async_llm(llm, prompts):
    tasks = [asyncio.create_task(llm.ainvoke(prompt)) for prompt in prompts]
    resps = await asyncio.gather(*tasks)
    return resps
