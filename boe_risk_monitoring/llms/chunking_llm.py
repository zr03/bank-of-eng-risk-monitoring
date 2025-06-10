from typing import List
from collections import defaultdict
import math

from openai import OpenAI
from google import genai

from boe_risk_monitoring.llms.base_llm import BaseLLM, SUPPORTED_LLMS_DICT

class ChunkingLLM(BaseLLM):
	def __init__(self, chunks_schema, metadata_schema, backend="openai", model_name="gpt-4.1", temperature=0.3):
		super().__init__(backend=backend, model_name=model_name, temperature=temperature)
		self.chunks_schema = chunks_schema
		self.metadata_schema = metadata_schema

	def invoke(self, prompt, response_schema):
		if self.backend == "openai":
			client = OpenAI(api_key=self.api_key)
			resp = client.responses.parse(
				model=self.model_name,
				input=prompt,
				temperature=self.temperature,
				text_format=response_schema,
			)
			return resp.output_parsed

		if self.backend == "gemini":
			client = genai.Client(api_key=self.api_key)
			resp = client.models.generate_content(
				model=self.model_name,
				contents=prompt,
				config={
					"response_mime_type": "application/json",
					"response_schema": response_schema,
				}
			)
			return resp.parsed

		else:
			raise ValueError(f"Unsupported backend: {self.backend}")

	def _make_doc_metadata_prompt(self, text):
		return (
			"You are an assistant that extracts structured metadata from financial earnings call transcripts."
			"From the text below, extract the following metadata:"
			"reporting year: the year of the reporting period."
			"reporting quarter: the quarter of the reporting period."
			"date: the date the earnings call occurred."
			f"Text:\n{text}"
		)

	def _make_chunk_prompt(self, text):
		return (
			"You are an expert in document structuring and analysis. "
			"Split the following financial earnings transcript into **small, logically coherent chunks**. "
			"Each chunk should be **no more than one paragraph**, and ideally a single idea or statement. "
			"Break chunks at changes in **speaker**, **topic**, or shifts in financial focus. "
			"If a paragraph contains multiple ideas, split it into multiple smaller chunks. "
			"Preserve the original order of text.\n"
			# "Ensure clarity and minimalism in each chunk. Avoid combining multiple statements into one block.\n\n"
			# "Additionally, for each chunk, provide the following:\n:"
			# "1. flag if the chunk contains a forward-looking statement (e.g., projections, outlook, or future plans)\n"
			# "Example:\n"
			# "Chunk: We expect revenues to grow 10% next quarter."
			# "forward_looking: true\n"
			# "Chunk: Revenue this quarter was $2.3 billion, a 5% increase year-over-year."
			# "forward_looking: false\n"
			# "Only flag if there is a clear statement about the future\n."
			# "2. List out any mentions of external or macroeconomic factors. Omit if not present.\n"
			# "Examples of macro factors include: interest rates, inflation, monetary policy, regulatory changes, geopolitical instability, FX rates, tariffs, etc.\n"
			# "If the factor doesn't fit any known category, describe it clearly (e.g. tariffs)\n"
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

	def fetch_metadata(self, text, metadata_schema=None):
		if not metadata_schema:
			metadata_schema = self.metadata_schema
		prompt = self._make_doc_metadata_prompt(text)
		return self.invoke(prompt, metadata_schema)

	def chunk_transcript(self, text, chunks_schema=None):
		if not chunks_schema:
			chunks_schema = self.chunks_schema
		prompt = self._make_chunk_prompt(text)
		return self.invoke(prompt, chunks_schema)

	# async def chunk_transcript_async(self, text):
	#     prompt = self._make_chunk_prompt(text)
	#     return await self.ainvoke(prompt)
