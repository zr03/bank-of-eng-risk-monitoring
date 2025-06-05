"""
This module contains different ETL classes for extracting data from financial documents published by banks.
"""

from abc import ABC, abstractmethod
import json
from pathlib import Path

class BaseETL(ABC):
	"""
	Abstract base class for ETL processes.
	Defines the interface for extracting, transforming, and loading data.
	"""

	@abstractmethod
	def extract(self):
		"""
		Extract data from the source.
		"""
		pass

	@abstractmethod
	def transform(self):
		"""
		Transform the extracted data.
		"""
		pass

	@abstractmethod
	def load(self):
		"""
		Load the transformed data into the target system.
		"""
		pass



class TranscriptsETL(BaseETL):
	def __init__(self, input_pdf_path, output_json_path, llm_model):
		# Check if input paths are valid
		if not isinstance(input_pdf_path, str) or not isinstance(output_json_path, str):
			raise ValueError("Input and output paths must be strings.")
		if not input_pdf_path.endswith('.pdf'):
			raise ValueError("Input path must be a PDF file.")
		if not output_json_path.endswith('.json'):
			raise ValueError("Output path must be a JSON file.")

		input_pdf_path = Path(input_pdf_path)
		output_json_path = Path(output_json_path)
		output_json_path_dir = output_json_path.parents[0]

		if not input_pdf_path.exists():
			raise FileNotFoundError(f"Input PDF file does not exist: {input_pdf_path}")

		if not output_json_path_dir.exists():
			raise FileNotFoundError(f"Output directory does not exist: {output_json_path_dir}")

		self.input_pdf_path = Path(input_pdf_path)
		self.output_json_path = Path(output_json_path)
		self.llm_model = llm_model

	def extract(self):
		# Example: extract text from PDF
		from PyPDF2 import PdfReader
		reader = PdfReader(str(self.input_pdf_path))
		text = "\n".join([page.extract_text() or "" for page in reader.pages])
		return text

	def transform(self, raw_text):
		# Use an LLM to perform contextual chunking
		# This could include Q&A chunking, topic segmentation, etc.
		chunks = self.llm_model.chunk_transcript(raw_text)
		return chunks

	def load(self, transformed_data):
		# Save the structured data to JSON
		with open(self.output_json_path, "w", encoding="utf-8") as f:
			json.dump(transformed_data, f, indent=2)
