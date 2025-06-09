from typing import List
from collections import defaultdict
import math

from openai import OpenAI
from google import genai

from boe_risk_monitoring.llms.base_llm import BaseLLM, SUPPORTED_LLMS_DICT

# SUPPORTED_CHUNKING_LLMS_DICT = defaultdict(lambda: defaultdict(dict))
# for backend in SUPPORTED_LLMS_DICT:
#     for model in SUPPORTED_LLMS_DICT[backend]:
#         context_window = SUPPORTED_LLMS_DICT[backend][model]["context_window"]
#         max_output = SUPPORTED_LLMS_DICT[backend][model]["context_window"]
#         if context_window > 1000000:
#             SUPPORTED_CHUNKING_LLMS_DICT[backend][model]["context_window"] = context_window
#             SUPPORTED_CHUNKING_LLMS_DICT[backend][model]["max_output"] = max_output


class ChunkingLLM(BaseLLM):
    def __init__(self, response_schema, backend="openai", model_name="gpt-4.1", temperature=0.3):
        # if backend not in SUPPORTED_CHUNKING_LLMS_DICT:
        #     raise ValueError(
        #         f"Unsupported backend for chunking. Only models with context window > 1000000 are supported: {SUPPORTED_CHUNKING_LLMS_DICT.keys()}")
        # if model_name not in SUPPORTED_MODELS_DICT[backend]:
        #     raise ValueError(
        #         f"Unsupported model for chunking. Only models with context window > 1000000 are supported: {list(SUPPORTED_CHUNKING_LLMS_DICT[backend].keys())}")
        super().__init__(backend=backend, model_name=model_name, temperature=temperature)
        self.response_schema = response_schema

    def invoke(self, prompt: str):
        if self.backend == "openai":
            client = OpenAI(api_key=self.api_key)
            resp = client.responses.parse(
                model=self.model_name,
                input=prompt,
                temperature=self.temperature,
                text_format=self.response_schema,
            )
            return resp.output_parsed

        if self.backend == "gemini":
            client = genai.Client(api_key=self.api_key)
            resp = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": self.response_schema,
                }
            )
            return resp.parsed

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _make_chunk_prompt(self, text):
        return (
            "You are an expert in document structuring and analysis. "
            "Split the following financial earnings transcript into **small, logically coherent chunks**. "
            "Each chunk should be **no more than one paragraph**, and ideally a single idea or statement. "
            "Break chunks at changes in **speaker**, **topic**, or shifts in financial focus. "
            "If a paragraph contains multiple ideas, split it into multiple smaller chunks. "
            "Preserve the original order of text.\n\n"
            # "Ensure clarity and minimalism in each chunk. Avoid combining multiple statements into one block.\n\n"
            f"Transcript:\n{text}"
        )

    def estimate_required_pre_chunks(self, text):
        # Approximate number of tokens of prompt + approximate response tokens
        prompt = self._make_chunk_prompt(text)
        n_words_prompt = len(prompt.split())
        # 100 tokens to 75 words rule of thumb, applying a 1.2 safety factor too
        n_tokens_prompt = int(n_words_prompt*4/3*1.2)
        # We expect the output chunks to have a similar order for number of tokens as the input prompt but we add 10% for structured response characters (e.g. curly brackets), applying a 1.2 safety factor too
        n_tokens_response = int(n_tokens_prompt*1.1*1.2)
        # Total input and output tokens
        input_output_tokens = (n_tokens_prompt + n_tokens_response)
        # Check against context_window
        context_window = self.context_window
        pre_chunks_context_window = 0
        pre_chunks_max_output = 0
        if input_output_tokens > context_window:
            print(f"Estimated prompt + response tokens = {input_output_tokens}. This exceeds the context window of {self.model_name} = {context_window}.")
            pre_chunks_context_window = math.ceil(input_output_tokens/context_window)
        max_output = self.max_output
        if n_tokens_response > max_output:
            print(f"Estimated response tokens = {n_tokens_response}. This exceeds the max output tokens of {self.model_name} = {max_output}.")
            pre_chunks_max_output = math.ceil(n_tokens_response/max_output)
        return pre_chunks_context_window, pre_chunks_max_output

    def chunk_transcript(self, text):
        prompt = self._make_chunk_prompt(text)
        return self.invoke(prompt)

    # async def chunk_transcript_async(self, text):
    #     prompt = self._make_chunk_prompt(text)
    #     return await self.ainvoke(prompt)
