"""
This module contains different ETL classes for extracting data from financial documents published by banks.
"""

from abc import ABC, abstractmethod
import os
import ast
from pathlib import Path
from typing import List, Literal, Optional
import datetime
from collections import defaultdict

from pydantic import BaseModel, Field
import pymupdf
import pandas as pd
# from langchain_core.messages.utils import count_tokens_approximately
# import tiktoken

from boe_risk_monitoring.llms.chunking_llm import ChunkingLLM
from boe_risk_monitoring.llms.document_analyser_llm import DocumentAnalyserLLM


class TranscriptChunk(BaseModel):
	text: str
	speaker: str
	role: Literal['CEO', 'CFO', 'Host', 'Analyst', 'Other']
	page: int
	section: Literal['Introduction', 'Disclaimer', 'Prepared remarks', 'Q and A', 'Conclusion']
	# forward_looking_statement: bool
	# macroeconomic_references: List[str]

class ChunkedTranscript(BaseModel):
	doc: List[TranscriptChunk]

class TranscriptMetadata(BaseModel):
	reporting_year: int = Field(...,ge=2021)
	reporting_quarter: int = Field(...,ge=1,le=4)
	date_of_call: str = Field(..., description="in format YYYY-MM-DD e.g. 2024-10-31")

class SlideTextChunk(BaseModel):
	text: str

class SlideGraphSummary(BaseModel):
	caption: str
	trend_summary: str

class SlideTableRowSummary(BaseModel):
	row_heading: str
	row_number: int
	preceding_quarter_trend_summary: Optional[str] = None
	year_on_year_quarter_trend_summary: Optional[str] = None
	current_quarter_value: Optional[float] = None
	preceding_quarter_value: Optional[float] = None
	last_year_quarter_value: Optional[float] = None
	unit: Optional[str] = None
	# preceding_quarter_pct_change: Optional[float] = None
	# year_on_year_pct_change: Optional[float] = None
	# preceding_quarter_abs_change: Optional[float] = None
	# year_on_year_pct_change: Optional[float] = None

class SlideTableSummary(BaseModel):
	caption: str
	column_headings: List[str]
	row_summaries: List[SlideTableRowSummary]

class Slide(BaseModel):
	title: str
	key_points: List[SlideTextChunk]
	graphs: List[SlideGraphSummary]
	tables: List[SlideTableSummary]
	slide_number: int
	section: Literal['Title', 'Vision', 'Financial Results', 'Outlook', 'Glossary', 'Footnotes']

class Presentation(BaseModel):
	doc: List[Slide]
	reporting_year: int = Field(...,ge=2021)
	reporting_quarter: int = Field(...,ge=1,le=4)
	date_of_presentation: str = Field(..., description="in format YYYY-MM-DD e.g. 2024-10-31")


class BaseETL(ABC):
	"""
	Abstract base class for ETL processes.
	Defines the interface for extracting, transforming, and loading data.
	"""

	@abstractmethod
	def extract(self, *args, **kwargs):
		"""
		Extract data from the source.
		"""

	@abstractmethod
	def transform(self, *args, **kwargs):
		"""
		Transform the extracted data.
		"""

	@abstractmethod
	def load(self, *args, **kwargs):
		"""
		Load the transformed data into the target system.
		"""


class TranscriptETL(BaseETL):
	def __init__(self, input_pdf_path):
		# Check if input paths are valid
		if not isinstance(input_pdf_path, str):
			raise ValueError("Input path must be a string.")
		if not input_pdf_path.endswith('.pdf'):
			raise ValueError("Input path must be a PDF file.")

		input_pdf_path = Path(input_pdf_path)
		if not input_pdf_path.exists():
			raise FileNotFoundError(f"Input PDF file does not exist: {input_pdf_path}")
		self.input_pdf_path = Path(input_pdf_path)

		self.llm_model = None
		self.output_dir_path = None
		self.output_file_path_csv = None
		self.output_file_path_parquet = None

	def extract(self):
		doc = pymupdf.open(self.input_pdf_path)
		full_text = ""
		for i, page in enumerate(doc):
			text = page.get_text(option="text")
			# Insert page metadata
			text = f"=== START OF PAGE {i+1} ===\n" + text + "\n"
			full_text += text
		return full_text

	def transform(self, raw_data, chunks_schema, metadata_schema, llm_backend="openai", llm_model_name="gpt-4o", temperature=0.3):
		# Instantiate the LLM
		self.llm_model = ChunkingLLM(
			chunks_schema=chunks_schema,
			metadata_schema=metadata_schema,
			backend=llm_backend,
			model_name=llm_model_name,
			temperature=temperature,
		)

		# Fetch the metadata
		transcript_metadata = self.llm_model.fetch_metadata(raw_data[:2000])
		reporting_period = "Q" + str(transcript_metadata.reporting_quarter) + "_" + str(transcript_metadata.reporting_year)
		date_of_call_dt = datetime.datetime.strptime(transcript_metadata.date_of_call, "%Y-%m-%d") # Use this to ensure date is in the right format, will error out otherwise

		# Check if pre-chunking is required given the length of the doc and model token limits
		pre_chunks_req_context_window, pre_chunks_req_max_output = self.llm_model.estimate_required_pre_chunks(raw_data)

		# Use an LLM to perform contextual chunking
		if max(pre_chunks_req_context_window, pre_chunks_req_max_output) == 0:
			chunked_transcript = self.llm_model.chunk_transcript(raw_data)
			n_chunks = len(chunked_transcript.doc)
			chunk_dicts = [dict(chunked_transcript.doc[i]) for i in range(n_chunks)]
			chunks_df = pd.DataFrame(chunk_dicts)
		else:
			print(f"Pre-chunking needed: estimated chunks based on context window = {pre_chunks_req_context_window}, estimated chunks based on max output tokens = {pre_chunks_req_max_output}")
			raise NotImplementedError("The selected model has limited context window or max output tokens. Pre chunking has not yet been implemented. Please choose a different model for now.")

		chunks_df['reporting_period'] = reporting_period
		chunks_df['date_of_call'] = date_of_call_dt
		return chunks_df

	def load(self, transformed_data, output_dir_path, file_name=None):
		if not isinstance(transformed_data, pd.DataFrame):
			raise TypeError("Data must be pandas dataframe")

		if not isinstance(output_dir_path, str):
			raise TypeError("Output directory path must be a string.")

		output_dir_path = Path(output_dir_path)
		# Make the directory if it does not exist already
		output_dir_path.mkdir(parents=True, exist_ok=True)
		self.output_dir_path = output_dir_path

		if file_name:
			if not isinstance(file_name, str):
				raise TypeError("File name must be a string if provided.")
			else:
				file_name_csv = Path(file_name + ".csv")
				file_name_parquet = Path(file_name + ".parquet")
		else:
			file_name = transformed_data['reporting_period'].iloc[0]
			file_name_csv = Path(file_name + ".csv")
			file_name_parquet = Path(file_name + ".parquet")

		# Construct full path
		full_path_csv = output_dir_path / file_name_csv
		full_path_parquet = output_dir_path / file_name_parquet
		self.output_file_path_csv = full_path_csv
		self.output_file_path_parquet = full_path_parquet

		# Save the data
		transformed_data.to_csv(full_path_csv, index=False)
		transformed_data.to_parquet(full_path_parquet)

class PresentationETL(BaseETL):
	def __init__(self, input_pdf_path):
		# Check if input paths are valid
		if not isinstance(input_pdf_path, str):
			raise ValueError("Input path must be a string.")
		if not input_pdf_path.endswith('.pdf'):
			raise ValueError("Input path must be a PDF file.")

		input_pdf_path = Path(input_pdf_path)
		if not input_pdf_path.exists():
			raise FileNotFoundError(f"Input PDF file does not exist: {input_pdf_path}")
		self.input_pdf_path = Path(input_pdf_path)

		self.llm_model = None
		self.output_dir_path = None
		self.output_file_paths_csv = []
		self.output_file_paths_parquet = []

	def extract(self):
		print("Skipping extraction step — using LLM-native document ingestion.")

	def transform(self, results_schema, llm_backend="openai", llm_model_name="gpt-4o", temperature=0.3):

		# Instantiate the LLM
		self.llm_model = DocumentAnalyserLLM(
			input_pdf_path=self.input_pdf_path,
			results_schema=results_schema,
			backend=llm_backend,
			model_name=llm_model_name,
			temperature=temperature,
		)

		# Run the analysis
		analysis_results = self.llm_model.analyse_presentation()
		text_df, graphs_df, tables_df = unpack_analysis_results(analysis_results)

		# Combine into single dictionary with keys as file name suffixes
		results_dict = {
			"text": text_df,
			"graphs": graphs_df,
			"tables": tables_df
		}
		return results_dict

	def load(self, transformed_data, output_dir_path):
		if not isinstance(transformed_data, dict):
			raise TypeError("Data must be a dictionary with key-value pairs each repesenting a file name (str) and associated data (pd.DataFrame)")

		file_names = list(transformed_data.keys())
		file_data = list(transformed_data.values())

		if not all([isinstance(x, str) for x in file_names]):
			raise TypeError("Keys in dictionary must all be strings representing file names")

		if not all([isinstance(x, pd.DataFrame) for x in file_data]):
			raise TypeError("Values in dictionary must all be dataframes representing data to be saved")

		if not isinstance(output_dir_path, str):
			raise TypeError("Output directory path must be a string.")

		output_dir_path = Path(output_dir_path)
		# Make the directory if it does not exist already
		output_dir_path.mkdir(parents=True, exist_ok=True)
		self.output_dir_path = output_dir_path

		# Construct full paths and save
		output_file_paths_csv = []
		output_file_paths_parquet = []
		for file_name in file_names:
			# save csv
			file_name_obj_csv = Path(file_name + ".csv")
			full_path_csv = output_dir_path / file_name_obj_csv
			output_file_paths_csv.append(full_path_csv)
			transformed_data[file_name].to_csv(full_path_csv, index=False)
			# save parquet
			file_name_obj_parquet = Path(file_name + ".parquet")
			full_path_parquet = output_dir_path / file_name_obj_parquet
			output_file_paths_parquet.append(full_path_parquet)
			transformed_data[file_name].to_parquet(full_path_parquet)

		self.output_file_paths_csv = output_file_paths_csv
		self.output_file_paths_parquet = output_file_paths_parquet

def unpack_analysis_results(analysis_results):
	"""
	This function takes the pydantic schema returned from document analysis LLM call and converts it
	to three dataframes capturing textual, graphical and tabular summaries.
	"""
	reporting_year = analysis_results.reporting_year
	reporting_quarter = analysis_results.reporting_quarter
	reporting_period = "Q" + str(reporting_quarter) + "_" + str(reporting_year)
	date_of_presentation = analysis_results.date_of_presentation
	main_results = analysis_results.doc
	text_data = defaultdict(list)
	graph_data = defaultdict(list)
	table_data = defaultdict(list)
	# Iterate through the schema
	for slide in main_results:
		slide_num = slide.slide_number
		section = slide.section
		for key_point in slide.key_points:
			text_data['text'].append(key_point.text)
			text_data['page'].append(slide_num)
			text_data['section'].append(section)
		for graph in slide.graphs:
			graph_data['caption'].append(graph.caption)
			graph_data['trend_summary'].append(graph.trend_summary)
			graph_data['page'].append(slide_num)
			graph_data['section'].append(section)
		for table in slide.tables:
			table_caption = table.caption
			table_col_headings = list(filter(bool, table.column_headings)) # Removes any empty strings/null values
			for row in table.row_summaries:
				table_data['caption'].append(table_caption)
				table_data['table_col_headings'].append(table_col_headings)
				table_data['row_heading'].append(row.row_heading)
				table_data['row_number'].append(row.row_number)
				table_data['preceding_quarter_trend_summary'].append(row.preceding_quarter_trend_summary)
				table_data['year_on_year_quarter_trend_summary'].append(row.year_on_year_quarter_trend_summary)
				table_data['current_quarter_value'].append(row.current_quarter_value)
				table_data['preceding_quarter_value'].append(row.preceding_quarter_value)
				table_data['unit'].append(row.unit)
				table_data['page'].append(slide_num)
				table_data['section'].append(section)
	# Construct dataframes
	text_df = pd.DataFrame(text_data)
	graphs_df = pd.DataFrame(graph_data)
	tables_df = pd.DataFrame(table_data)

	# Add global values
	text_df['reporting_period'] = reporting_period
	text_df['date_of_presentation'] = date_of_presentation
	graphs_df['reporting_period'] = reporting_period
	graphs_df['date_of_presentation'] = date_of_presentation
	tables_df['reporting_period'] = reporting_period
	tables_df['date_of_presentation'] = date_of_presentation

	return text_df, graphs_df, tables_df



if __name__ == "__main__":
	# # Instantiate the TranscriptETL class
	# input_pdf_path = os.path.join("data", "citigroup", "raw_docs", "transcripts", "Q1_2025.pdf")

	# transcript_etl = TranscriptETL(
	#     input_pdf_path=input_pdf_path,
	# )

	# # Run the extract method
	# extracted_text = transcript_etl.extract()

	# # Run the transform method
	# chunks_df = transcript_etl.transform(
	#     raw_data=extracted_text,
	#     chunks_schema=ChunkedTranscript,
	#     metadata_schema=TranscriptMetadata,
	#     # llm_backend="openai",
	#     # llm_model_name="gpt-4.1",
	#     llm_backend="gemini",
	#     # llm_model_name="gemini-2.5-pro-preview-06-05",
	#     llm_model_name="gemini-2.5-flash-preview-05-20",
	#     )

	# # Run the load method
	# output_dir_path = os.path.join("data", "citigroup", "processed", "transcripts")
	# transcript_etl.load(
	#     transformed_data=chunks_df,
	#     output_dir_path=output_dir_path,
	#     )

	# Instantiate the PresentationETL class
	input_pdf_path = os.path.join("data", "citigroup", "raw_docs", "presentations", "Q1_2025_presentation.pdf")
	presentation_etl = PresentationETL(
		input_pdf_path=input_pdf_path,
	)

	# Run the extract method
	extracted_text = presentation_etl.extract()

	# Run the transform method
	analysis_results_dict = presentation_etl.transform(
		results_schema=Presentation,
		llm_backend="gemini",
		llm_model_name="gemini-2.5-pro-preview-06-05",
	)

	# Run the load method
	output_dir_path = os.path.join("data", "citigroup", "processed", "presentations")
	presentation_etl.load(
		transformed_data=analysis_results_dict,
		output_dir_path=output_dir_path,
	)






