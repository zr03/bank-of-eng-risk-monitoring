import os
from typing import List
from collections import defaultdict
import math
import pathlib
import base64

from openai import OpenAI
from google import genai
from google.genai import types

from boe_risk_monitoring.llms.base_llm import BaseLLM, SUPPORTED_LLMS_DICT


class DocumentAnalyserLLM(BaseLLM):
	def __init__(self, input_pdf_path, analysis_prompt, response_schema, backend="openai", model_name="gpt-4.1", temperature=0.3):
		super().__init__(backend=backend, model_name=model_name, temperature=temperature)
		if not SUPPORTED_LLMS_DICT[backend][model_name]['pdf_input']:
			raise RuntimeError(f"The chosen mode {model_name} cannot take pdfs as input. Please choose a different model.")
		if not isinstance(input_pdf_path, pathlib.Path):
			raise TypeError("Input path must be of type pathlib.Path.")
		if not input_pdf_path.suffix == ".pdf":
			raise ValueError("Input path must be a PDF file.")
		if not isinstance(analysis_prompt, str):
			raise TypeError("Analysis prompt must be a string.")

		self.input_pdf_path = input_pdf_path
		self.analysis_prompt = analysis_prompt
		self.response_schema = response_schema

	def invoke(self):
		if self.backend == "openai":
			with open(self.input_pdf_path, "rb") as f:
				data = f.read()
			base64_string = base64.b64encode(data).decode("utf-8")
			client = OpenAI(api_key=self.api_key)
			resp = client.responses.parse(
				model=self.model_name,
				input=[
						{
							"role": "user",
							"content": [
								{
									"type": "input_file",
									"filename": os.path.basename(self.input_pdf_path),
									"file_data": f"data:application/pdf;base64,{base64_string}",
								},
								{
									"type": "input_text",
									"text": self.analysis_prompt,
								},
							],
						},
					],
				temperature=self.temperature,
				text_format=self.response_schema,
			)
			return resp.output_parsed

		if self.backend == "gemini":
			client = genai.Client(api_key=self.api_key)
			resp = client.models.generate_content(
				model=self.model_name,
				contents=[
					types.Part.from_bytes(
						data=self.input_pdf_path.read_bytes(),
						mime_type='application/pdf',
					),
					self.analysis_prompt
					],
				config={
					"response_mime_type": "application/json",
					"response_schema": self.response_schema,
				}
			)
			return resp.parsed

		else:
			raise ValueError(f"Unsupported backend: {self.backend}")
