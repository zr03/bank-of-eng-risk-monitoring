from typing import List
from collections import defaultdict
import math

from openai import OpenAI
from google import genai

from boe_risk_monitoring.llms.base_llm import BaseLLM

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

class SentimentAnalysisLLM(ChunkingLLM):
	"""Identical to ChunkingLLM but uses `sentiment_analysis_prompt` for clarity."""
	def __init__(self, sentiment_analysis_prompt, response_schema, backend="openai", model_name="gpt-4.1", temperature=0.3):
		super().__init__(chunking_prompt=sentiment_analysis_prompt, response_schema=response_schema, backend=backend, model_name=model_name, temperature=temperature)
class TopicLabellingLLM(ChunkingLLM):
    """Identical to ChunkingLLM but uses `topic_labelling_prompt` for clarity."""
    def __init__(self, topic_labelling_prompt, response_schema, backend="openai", model_name="gpt-4.1", temperature=0.3):
        super().__init__(chunking_prompt=topic_labelling_prompt, response_schema=response_schema, backend=backend, model_name=model_name, temperature=temperature)

class QuestionAnswerTaggingLLM(ChunkingLLM):
	"""Identical to ChunkingLLM but uses `q_and_a_tagging_prompt` for clarity."""
	def __init__(self, q_and_a_tagging_prompt, response_schema, backend="openai", model_name="gpt-4.1", temperature=0.3):
		super().__init__(chunking_prompt=q_and_a_tagging_prompt, response_schema=response_schema, backend=backend, model_name=model_name, temperature=temperature)

class EvasivenessTaggingLLM(ChunkingLLM):
	"""Identical to ChunkingLLM but uses `evasiveness_tagging_prompt` for clarity."""
	def __init__(self, evasiveness_tagging_prompt, response_schema, backend="openai", model_name="gpt-4.1", temperature=0.3):
		super().__init__(chunking_prompt=evasiveness_tagging_prompt, response_schema=response_schema, backend=backend, model_name=model_name, temperature=temperature)
