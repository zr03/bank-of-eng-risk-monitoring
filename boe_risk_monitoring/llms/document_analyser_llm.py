from typing import List
from collections import defaultdict
import math
import pathlib

from openai import OpenAI
from google import genai
from google.genai import types

from boe_risk_monitoring.llms.base_llm import BaseLLM, SUPPORTED_LLMS_DICT


class DocumentAnalyserLLM(BaseLLM):
	def __init__(self, input_pdf_path, results_schema, backend="openai", model_name="gpt-4.1", temperature=0.3):
		super().__init__(backend=backend, model_name=model_name, temperature=temperature)
		if not SUPPORTED_LLMS_DICT[backend][model_name]['pdf_input']:
			raise RuntimeError(f"The chosen mode {model_name} cannot take pdfs as input. Please choose a different model.")
		if not isinstance(input_pdf_path, pathlib.Path):
			raise ValueError("Input path must be of type pathlib.Path.")
		if not input_pdf_path.suffix == ".pdf":
			raise ValueError("Input path must be a PDF file.")
		self.input_pdf_path = input_pdf_path
		self.results_schema = results_schema

	def invoke(self, prompt, response_schema):
		# if self.backend == "openai":
		#     client = OpenAI(api_key=self.api_key)
		#     resp = client.responses.parse(
		#         model=self.model_name,
		#         input=prompt,
		#         temperature=self.temperature,
		#         text_format=response_schema,
		#     )
		#     return resp.output_parsed

		if self.backend == "gemini":
			client = genai.Client(api_key=self.api_key)
			resp = client.models.generate_content(
				model=self.model_name,
				contents=[
					types.Part.from_bytes(
						data=self.input_pdf_path.read_bytes(),
						mime_type='application/pdf',
					),
					prompt
					],
				config={
					"response_mime_type": "application/json",
					"response_schema": response_schema,
				}
			)
			return resp.parsed

		else:
			raise ValueError(f"Unsupported backend: {self.backend}")

	def _make_analysis_prompt(self):
		return (
			"You are an expert assistant analyzing corporate financial presentation slides."
			"Your task is to extract structured data from each slide based on text, graph and tabular content.\n\n"
			"Analyze each slide individually and extract the following:\n"
			"1. key points: Break the slide text into small, coherent chunks, each representing a single idea.\n"
			"2. graphs: For each graph on the slide (e.g., line or bar charts), return:\n"
			"- caption: a short descriptive title.\n"
			"- trend summary: a brief explanation of key trends or insights for the plotted variables.\n"
			"3. tables: For each table on the slide, return:\n"
			"- column_headings: list of column headers.\n"
			"- row_summaries: a list of structured objects, one per row, containing:\n"
			"- row_heading: the row heading e.g. Revenue.\n"
			"- row_number: the index position of the row in the table e.g. first row is 1.\n"
			"- preceding_quarter_trend_summary: text summary comparing reported quarter to previous quarter (e.g. 1Q25 vs 4Q24). Omit if not available.\n"
			"- year_on_year_quarter_trend_summary: text summary comparing reported quarter to same quarter last year (e.g. 1Q25 vs 1Q24). Omit if not available.\n"
			"- current_quarter_value: the value for the reported quarter. Omit if not available.\n"
			"- preceding_quarter_value: the value for the quarter preceding the reported quarter (e.g. 4Q24 if reported quarter is 1Q25). Omit if not available.\n"
			"- last_year_quarter_value: the value from the previous year for the same quarter as the reported quarter (e.g. 1Q24 if reported quarter is 1Q25). Omit if not available.\n"
			"- unit: the unit of the values in the current row (e.g. millions USD, %)"
			# "- preceding_quarter_pct_change: estimated percentage change between reported quarter and previous quarter (e.g. 1Q25 vs 4Q24). Omit if not available.\n"
			# "- year_on_year_quarter_pct_change: estimated percentage change between reported quarter and same quarter last year (e.g. 1Q25 vs 1Q24). Omit if not available.\n"
			# "- preceding_quarter_abs_change: estimated absolute change between reported quarter and previous quarter (e.g. 1Q25 vs 4Q24). Omit if not available.\n"
			# "- year_on_year_quarter_abs_change: estimated absolute change between reported quarter and same quarter last year (e.g. 1Q25 vs 1Q24). Omit if not available.\n\n"
		)

	def analyse_presentation(self, results_schema=None):
		if not results_schema:
			results_schema = self.results_schema
		prompt = self._make_analysis_prompt()
		return self.invoke(prompt, results_schema)
