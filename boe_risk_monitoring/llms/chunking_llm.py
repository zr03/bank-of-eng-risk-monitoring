from typing import List
from collections import defaultdict
import math

from openai import OpenAI
from google import genai

from boe_risk_monitoring.llms.base_llm import BaseLLM, SUPPORTED_LLMS_DICT

class ChunkingLLM(BaseLLM):
	def __init__(self, chunking_prompt, response_schema, backend="openai", model_name="gpt-4.1", temperature=0.3):
		super().__init__(backend=backend, model_name=model_name, temperature=temperature)
		if not isinstance(chunking_prompt, str):
			raise TypeError("Chunking prompt must be a string.")
		self.chunking_prompt = chunking_prompt
		self.response_schema = response_schema

	def invoke(self):
		if self.backend == "openai":
			client = OpenAI(api_key=self.api_key)
			resp = client.responses.parse(
				model=self.model_name,
				input=self.chunking_prompt,
				temperature=self.temperature,
				text_format=self.response_schema,
			)
			return resp.output_parsed

		if self.backend == "gemini":
			client = genai.Client(api_key=self.api_key)
			resp = client.models.generate_content(
				model=self.model_name,
				contents=self.chunking_prompt,
				config={
					"response_mime_type": "application/json",
					"response_schema": self.response_schema,
				}
			)
			return resp.parsed

		else:
			raise ValueError(f"Unsupported backend: {self.backend}")




	# def fetch_metadata(self, text, metadata_schema=None):
	# 	if not metadata_schema:
	# 		metadata_schema = self.metadata_schema
	# 	prompt = self._make_doc_metadata_prompt(text)
	# 	return self.invoke(prompt, metadata_schema)

	# def chunk_transcript(self, text, chunks_schema=None):
	# 	if not chunks_schema:
	# 		chunks_schema = self.chunks_schema
	# 	prompt = self._make_chunk_prompt(text)
	# 	return self.invoke(prompt, chunks_schema)
