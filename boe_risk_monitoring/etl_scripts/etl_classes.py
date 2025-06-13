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
import boe_risk_monitoring.config as config

DATA_FOLDER = config.DATA_FOLDER

class TranscriptChunk(BaseModel):
	text: str
	speaker: str
	role: Literal['CEO', 'CFO', 'Host', 'Analyst', 'Other']
	page: int
	section: Literal['Introduction', 'Disclaimer', 'Prepared remarks', 'Q and A', 'Conclusion']

class TranscriptChunkQ4(BaseModel):
	text: str
	fiscal_period_ref: Literal['year', 'quarter', 'both']
	speaker: str
	role: Literal['CEO', 'CFO', 'Host', 'Analyst', 'Other']
	page: int
	section: Literal['Introduction', 'Disclaimer', 'Prepared remarks', 'Q and A', 'Conclusion']

class ChunkedTranscript(BaseModel):
	doc: List[TranscriptChunk]
	reporting_year: int = Field(...,ge=2021)
	reporting_quarter: int = Field(...,ge=1,le=4)
	date_of_call: str = Field(..., description="in format YYYY-MM-DD e.g. 2024-10-31")

class ChunkedTranscriptQ4(BaseModel):
	doc: List[TranscriptChunkQ4]
	reporting_year: int = Field(...,ge=2021)
	reporting_quarter: int = Field(...,ge=1,le=4)
	date_of_call: str = Field(..., description="in format YYYY-MM-DD e.g. 2024-10-31")

class SlideTextChunk(BaseModel):
	text: str

class SlideTextChunkQ4(BaseModel):
	text: str
	# fiscal_period_ref: Optional[Literal['year', 'quarter', 'both']] = None
	fiscal_period_ref: Literal['year', 'quarter', 'both']

class SlideGraphSummary(BaseModel):
	caption: str
	trend_summary: str

class SlideGraphSummaryQ4(BaseModel):
	caption: str
	trend_summary: str
	# comparison_scope: Optional[Literal['year', 'quarter']] = None
	comparison_scope: Literal['year', 'quarter']

class SlideTableRowSummary(BaseModel):
	row_heading: str
	row_number: int
	preceding_quarter_trend_summary: Optional[str] = None
	year_on_year_quarter_trend_summary: Optional[str] = None
	current_quarter_value: Optional[float] = None
	preceding_quarter_value: Optional[float] = None
	prev_year_same_quarter_value: Optional[float] = None
	unit: Optional[str] = None

class SlideTableRowSummaryQ4(BaseModel):
	row_heading: str
	row_number: int
	preceding_quarter_trend_summary: Optional[str] = None
	year_on_year_quarter_trend_summary: Optional[str] = None
	current_quarter_value: Optional[float] = None
	preceding_quarter_value: Optional[float] = None
	prev_year_same_quarter_value: Optional[float] = None
	preceding_year_trend_summary: Optional[str] = None
	current_year_value: Optional[float] = None
	preceding_year_value: Optional[float] = None
	unit: Optional[str] = None

class SlideTableSummary(BaseModel):
	# caption: str
	# column_headings: List[str]
	row_summaries: List[SlideTableRowSummary]

class SlideTableSummaryQ4(BaseModel):
	# caption: str
	# column_headings: List[str]
	row_summaries: List[SlideTableRowSummaryQ4]

class Slide(BaseModel):
	title: str
	key_points: List[SlideTextChunk]
	graphs: List[SlideGraphSummary]
	tables: List[SlideTableSummary]
	slide_number: int
	section: Literal['Title', 'Vision', 'Financial Results', 'Outlook', 'Glossary', 'Footnotes']

class SlideQ4(BaseModel):
	title: str
	key_points: List[SlideTextChunkQ4]
	graphs: List[SlideGraphSummaryQ4]
	tables: List[SlideTableSummaryQ4]
	slide_number: int
	section: Literal['Title', 'Vision', 'Financial Results', 'Outlook', 'Glossary', 'Footnotes']

class Presentation(BaseModel):
	doc: List[Slide]
	reporting_year: int = Field(...,ge=2021)
	reporting_quarter: int = Field(...,ge=1,le=4)
	date_of_presentation: str = Field(..., description="in format YYYY-MM-DD e.g. 2024-10-31")

class PresentationQ4(BaseModel):
	doc: List[SlideQ4]
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
	def __init__(self, input_pdf_path, is_q4_transcript=False):
		# Check if input paths are valid
		if not isinstance(input_pdf_path, str):
			raise TypeError("Input path must be a string.")
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
		self.is_q4_transcript = is_q4_transcript

	def extract(self):
		doc = pymupdf.open(self.input_pdf_path)
		full_text = ""
		for i, page in enumerate(doc):
			text = page.get_text(option="text")
			# Insert page metadata
			text = f"=== START OF PAGE {i+1} ===\n" + text + "\n"
			full_text += text
		return full_text

	def transform(self, raw_data, llm_backend="openai", llm_model_name="gpt-4o", temperature=0.3):

		if self.is_q4_transcript:
			# Use the Q4-specific schema and prompt
			chunking_prompt = self._make_chunk_prompt_q4(raw_data)
			response_schema = ChunkedTranscriptQ4
		else:
			# Use the standard schema and prompt
			chunking_prompt = self._make_chunk_prompt(raw_data)
			response_schema = ChunkedTranscript

		# Instantiate the LLM
		self.llm_model = ChunkingLLM(
			chunking_prompt=chunking_prompt,
			response_schema=response_schema,
			backend=llm_backend,
			model_name=llm_model_name,
			temperature=temperature,
		)

		# # Fetch the metadata
		# transcript_metadata = self.llm_model.fetch_metadata(raw_data[:2000])
		# reporting_period = "Q" + str(transcript_metadata.reporting_quarter) + "_" + str(transcript_metadata.reporting_year)
		# date_of_call_dt = datetime.datetime.strptime(transcript_metadata.date_of_call, "%Y-%m-%d") # Use this to ensure date is in the right format, will error out otherwise

		# Check if pre-chunking is required given the length of the doc and model token limits
		pre_chunks_req_context_window, pre_chunks_req_max_output = self.estimate_required_pre_chunks(raw_data)

		# Use an LLM to perform contextual chunking
		if max(pre_chunks_req_context_window, pre_chunks_req_max_output) == 0:
			chunked_transcript = self.llm_model.invoke()
			n_chunks = len(chunked_transcript.doc)
			chunk_dicts = [dict(chunked_transcript.doc[i]) for i in range(n_chunks)]
			chunks_df = pd.DataFrame(chunk_dicts)
			reporting_period = "Q" + str(chunked_transcript.reporting_quarter) + "_" + str(chunked_transcript.reporting_year)
			date_of_call_dt = datetime.datetime.strptime(chunked_transcript.date_of_call, "%Y-%m-%d") # Use this to ensure date is in the right format, will error out otherwise
			chunks_df['reporting_period'] = reporting_period
			chunks_df['date_of_call'] = date_of_call_dt
			if not self.is_q4_transcript:
				chunks_df.insert(1, 'fiscal_period_ref', 'quarter')  # Default to quarter for non-Q4 transcripts
		else:
			print(f"Pre-chunking needed: estimated chunks based on context window = {pre_chunks_req_context_window}, estimated chunks based on max output tokens = {pre_chunks_req_max_output}")
			raise NotImplementedError("The selected model has limited context window or max output tokens. Pre chunking has not yet been implemented. Please choose a different model for now.")

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
		context_window = self.llm_model.context_window
		pre_chunks_context_window = 0
		pre_chunks_max_output = 0
		if input_output_tokens > context_window:
			print(f"Estimated prompt + response tokens = {input_output_tokens}. This exceeds the context window of {self.model_name} = {context_window}.")
			pre_chunks_context_window = math.ceil(input_output_tokens/context_window)
		max_output = self.llm_model.max_output
		if n_tokens_response > max_output:
			print(f"Estimated response tokens = {n_tokens_response}. This exceeds the max output tokens of {self.model_name} = {max_output}.")
			pre_chunks_max_output = math.ceil(n_tokens_response/max_output)
		return pre_chunks_context_window, pre_chunks_max_output

	@staticmethod
	def _make_chunk_prompt(text):
		return (
			"You are an expert in document structuring and analysis. "
			"Split the following financial earnings transcript into **small, logically coherent chunks**. "
			"Each chunk should be **no more than one paragraph**, and ideally a single idea or statement. "
			"Break chunks at changes in **speaker**, **topic**, or shifts in financial focus. "
			"If a paragraph contains multiple ideas, split it into multiple smaller chunks. "
			"Preserve the original order of text.\n"
			f"Transcript:\n{text}"
		)

	@staticmethod
	def _make_chunk_prompt_q4(text):
		return (
			"You are an expert in document structuring and analysis. "
			"Split the following financial earnings transcript into **small, logically coherent chunks**. "
			"Each chunk should be **no more than one paragraph**, and ideally a single idea or statement. "
			"Break chunks at changes in **speaker**, **topic**, or shifts in financial focus. "
			"If a paragraph contains multiple ideas, split it into multiple smaller chunks. "
			"Preserve the original order of text."
			"For each chunk, indicate the fiscal period it refers to e.g. the quarter or the full fiscal year or both.\n"
			f"Transcript:\n{text}"
		)

class PresentationETL(BaseETL):
	def __init__(self, input_pdf_path, is_q4_presentation=False):
		# Check if input paths are valid
		if not isinstance(input_pdf_path, str):
			raise TypeError("Input path must be a string.")
		if not input_pdf_path.endswith('.pdf'):
			raise ValueError("Input path must be a PDF file.")
		if not isinstance(is_q4_presentation, bool):
			raise TypeError("is_q4_presentation must be a boolean value.")

		input_pdf_path = Path(input_pdf_path)
		if not input_pdf_path.exists():
			raise FileNotFoundError(f"Input PDF file does not exist: {input_pdf_path}")
		self.input_pdf_path = Path(input_pdf_path)

		self.llm_model = None
		self.output_dir_path = None
		self.output_file_paths_csv = []
		self.output_file_paths_parquet = []
		self.is_q4_presentation = is_q4_presentation


	def extract(self):
		print("Skipping extraction step — using LLM-native document ingestion.")
		return None

	def transform(self, llm_backend="openai", llm_model_name="gpt-4o", temperature=0.3):

		if self.is_q4_presentation:
			# Use the Q4-specific schema and prompt
			response_schema = PresentationQ4
			analysis_prompt = self._make_analysis_prompt_q4()
		else:
			# Use the standard schema and prompt
			response_schema = Presentation
			analysis_prompt = self._make_analysis_prompt()

		# Instantiate the LLM
		self.llm_model = DocumentAnalyserLLM(
			input_pdf_path=self.input_pdf_path,
			analysis_prompt=analysis_prompt,
			response_schema=response_schema,
			backend=llm_backend,
			model_name=llm_model_name,
			temperature=temperature,
		)

		# Run the analysis
		analysis_results = self.llm_model.invoke()
		text_df, graphs_df, tables_df = self.unpack_analysis_results(analysis_results)

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

		# Fetch the reporting period to prefix the file names
		reporting_period = transformed_data[file_names[0]]['reporting_period'].iloc[0]

		# Construct full paths and save
		output_file_paths_csv = []
		output_file_paths_parquet = []
		for file_name in file_names:
			# save csv
			file_name_obj_csv = Path(reporting_period + "_" + file_name + ".csv")
			full_path_csv = output_dir_path / file_name_obj_csv
			output_file_paths_csv.append(full_path_csv)
			transformed_data[file_name].to_csv(full_path_csv, index=False)
			# save parquet
			file_name_obj_parquet = Path(reporting_period + "_" + file_name + ".parquet")
			full_path_parquet = output_dir_path / file_name_obj_parquet
			output_file_paths_parquet.append(full_path_parquet)
			transformed_data[file_name].to_parquet(full_path_parquet)

		self.output_file_paths_csv = output_file_paths_csv
		self.output_file_paths_parquet = output_file_paths_parquet

	@staticmethod
	def _make_analysis_prompt():
		return (
			"You are an expert assistant analyzing corporate financial presentation slides."
			"Your task is to extract structured data from each slide based on text, graph and tabular content.\n\n"
			"Analyze each slide individually and extract the following:\n"
			"1. key points: Break the slide text into small, coherent chunks, each representing a single idea.\n"
			"2. graphs: For each graph on the slide (e.g., line or bar charts), return:\n"
			"- caption: a short descriptive title.\n"
			"- trend summary: a brief explanation of key trends or insights for the plotted variables.\n"
			"3. tables: For each table on the slide, return:\n"
			# "- column_headings: list of column headers.\n"
			"- row_summaries: a list of structured objects, one per row, containing:\n"
			"- row_heading: the row heading e.g. Revenue.\n"
			"- row_number: the index position of the row in the table e.g. first row is 1.\n"
			"- preceding_quarter_trend_summary: text summary of the change in performance between the reported quarter and the previous quarter (e.g. 1Q25 vs 4Q24). Omit if not available.\n"
			"- year_on_year_quarter_trend_summary: text summary of the change in performance between the reported quarter and the same quarter last year (e.g. 1Q25 vs 1Q24). Omit if not available.\n"
			"- current_quarter_value: the value for the reported quarter. Omit if not available.\n"
			"- preceding_quarter_value: the value for the **quarter immediately before** the reported one (e.g. 4Q24 value if reported quarter is 1Q25). Omit if not available.\n"
			"- prev_year_same_quarter_value: the value for the **same quarter in the prior year** (e.g. 1Q24 value if reported quarter is 1Q25). Omit if not available.\n"
			"- unit: the unit of the values in the current row (e.g. millions USD, %)"
			# "- preceding_quarter_pct_change: estimated percentage change between reported quarter and previous quarter (e.g. 1Q25 vs 4Q24). Omit if not available.\n"
			# "- year_on_year_quarter_pct_change: estimated percentage change between reported quarter and same quarter last year (e.g. 1Q25 vs 1Q24). Omit if not available.\n"
			# "- preceding_quarter_abs_change: estimated absolute change between reported quarter and previous quarter (e.g. 1Q25 vs 4Q24). Omit if not available.\n"
			# "- year_on_year_quarter_abs_change: estimated absolute change between reported quarter and same quarter last year (e.g. 1Q25 vs 1Q24). Omit if not available.\n\n"
		)

	@staticmethod
	def _make_analysis_prompt_q4():
		return (
			"You are an expert assistant analyzing corporate financial presentation slides."
			"Your task is to extract structured data from each slide based on text, graph and tabular content.\n\n"
			"Analyze each slide individually and extract the following:\n"
			"1. key points: Break the slide text into small, coherent chunks, each representing a single idea.\n"
			"- fiscal_period_ref: for each key point, indicate whether it refers to the quarter or the full fiscal year or both\n"
			"2. graphs: For each graph on the slide (e.g., line or bar charts), return:\n"
			"- caption: a short descriptive title.\n"
			"- trend_summary: a brief explanation of key trends or insights for the plotted variables.\n"
			"- comparison_scope: specify 'year' if the graph compares full fiscal years (e.g., FY24 vs FY23), or 'quarter' if it compares specific quarters (e.g., Q4 2024 vs Q4 2023 or Q4 vs Q3).\n"
			"3. tables: For each table on the slide, return:\n"
			# "- column_headings: list of column headers.\n"
			"- row_summaries: a list of structured objects, one per row, containing:\n"
			"- row_heading: the row heading e.g. Revenue.\n"
			"- row_number: the index position of the row in the table e.g. first row is 1.\n"
			"- preceding_quarter_trend_summary: text summary of the change in performance between the reported quarter and the previous quarter (e.g. 1Q25 vs 4Q24). Omit if not available.\n"
			"- year_on_year_quarter_trend_summary: text summary of the change in performance between the reported quarter and the same quarter last year (e.g. 1Q25 vs 1Q24). Omit if not available.\n"
			"- current_quarter_value: the value for the reported quarter. Omit if not available.\n"
			"- preceding_quarter_value: the value for the **quarter immediately before** the reported one (e.g. 4Q24 value if reported quarter is 1Q25). Omit if not available.\n"
			"- prev_year_same_quarter_value: the value for the **same quarter in the prior year** (e.g. 1Q24 value if reported quarter is 1Q25). Omit if not available.\n"
			"- preceding_year_trend_summary: text summary of the change in performance over the full fiscal year, comparing the reported year (e.g., FY25) to the previous fiscal year (e.g., FY24). Omit if not available.\n"
			"- current_year_value: the value for the reported year. Omit if not available.\n"
			"- preceding_year_value: the value for the **year immediately before** the reported one (e.g. FY23 value if reported year is FY24). Omit if not available.\n"
			"- unit: the unit of the values in the current row (e.g. millions USD, %)"
			# "- preceding_quarter_pct_change: estimated percentage change between reported quarter and previous quarter (e.g. 1Q25 vs 4Q24). Omit if not available.\n"
			# "- year_on_year_quarter_pct_change: estimated percentage change between reported quarter and same quarter last year (e.g. 1Q25 vs 1Q24). Omit if not available.\n"
			# "- preceding_quarter_abs_change: estimated absolute change between reported quarter and previous quarter (e.g. 1Q25 vs 4Q24). Omit if not available.\n"
			# "- year_on_year_quarter_abs_change: estimated absolute change between reported quarter and same quarter last year (e.g. 1Q25 vs 1Q24). Omit if not available.\n\n"
		)

	def unpack_analysis_results(self, analysis_results):
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
				if self.is_q4_presentation:
					text_data['fiscal_period_ref'].append(key_point.fiscal_period_ref)
				else:
					text_data['fiscal_period_ref'].append("quarter")
				text_data['page'].append(slide_num)
				text_data['section'].append(section)
			for graph in slide.graphs:
				graph_data['caption'].append(graph.caption)
				graph_data['trend_summary'].append(graph.trend_summary)
				if self.is_q4_presentation:
					graph_data['comparison_scope'].append(graph.comparison_scope)
				else:
					graph_data['comparison_scope'].append("quarter")
				graph_data['page'].append(slide_num)
				graph_data['section'].append(section)
			for table in slide.tables:
				# table_caption = table.caption
				# table_col_headings = list(filter(bool, table.column_headings)) # Removes any empty strings/null values
				for row in table.row_summaries:
					# table_data['caption'].append(table_caption)
					# table_data['table_col_headings'].append(table_col_headings)
					table_data['row_heading'].append(row.row_heading)
					table_data['row_number'].append(row.row_number)
					table_data['preceding_quarter_trend_summary'].append(row.preceding_quarter_trend_summary)
					table_data['year_on_year_quarter_trend_summary'].append(row.year_on_year_quarter_trend_summary)
					table_data['current_quarter_value'].append(row.current_quarter_value)
					table_data['preceding_quarter_value'].append(row.preceding_quarter_value)
					table_data['prev_year_same_quarter_value'].append(row.prev_year_same_quarter_value)
					if self.is_q4_presentation:
						table_data['preceding_year_trend_summary'].append(row.preceding_year_trend_summary)
						table_data['current_year_value'].append(row.current_year_value)
						table_data['preceding_year_value'].append(row.preceding_year_value)
					else:
						table_data['preceding_year_trend_summary'].append(None)
						table_data['current_year_value'].append(None)
						table_data['preceding_year_value'].append(None)
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

	class DataAggregationETL(BaseETL):
		"""This class provides methods to aggregate data from multiple ETL processes into single files ready for downstream analysis.
		"""
		def __init__(self, data_dir_path):
			# Check if input paths are valid
			if not isinstance(data_dir_path, str):
				raise TypeError("Data directory path must be a string.")
			data_dir_path = Path(data_dir_path)
			if not data_dir_path.exists():
				raise FileNotFoundError(f"Data directory does not exist: {data_dir_path}")
			self.data_dir_path = data_dir_path

		def extract(self):
			"""
			Extracts transcripts and presentations data from the data directory.
			Returns a nested dictionary with banks and document type as keys and associated lists of dataframes as their values.
			"""
			csv_files = list(self.data_dir_path.glob("**/*.csv"))
			parquet_files = list(self.data_dir_path.glob("**/*.parquet"))
			all_files = {file.stem: file for file in csv_files + parquet_files}
			return all_files

		pass

	class SupplementaryDataETL(BaseETL):
		pass

	class SharePriceDataETL(BaseETL):
		"""This class provides methods to extract share price data from Yahoo Finance API for a given company
		"""
		pass

	class VectorDBETL(BaseETL):
		"""This class provides methods to extract data from NLP outputs and store embeddings in a vector DB along with associated metadata.
		"""
		pass





if __name__ == "__main__":
	# Instantiate the TranscriptETL class
	input_pdf_path = os.path.join("data", "jpmorgan", "raw_docs", "transcripts", "Q3_2024.pdf")

	transcript_etl = TranscriptETL(
		input_pdf_path=input_pdf_path,
		is_q4_transcript=False,
	)

	# Run the extract method
	extracted_text = transcript_etl.extract()

	# Run the transform method
	chunks_df = transcript_etl.transform(
	    raw_data=extracted_text,
	    llm_backend="gemini",
	    llm_model_name="gemini-2.5-pro-preview-06-05",
	    # llm_model_name="gemini-2.5-flash-preview-05-20",
	    )

	# Run the load method
	output_dir_path = os.path.join("data", "jpmorgan", "processed", "transcripts")
	transcript_etl.load(
	    transformed_data=chunks_df,
	    output_dir_path=output_dir_path,
	    )

	# # Instantiate the PresentationETL class
	# input_pdf_path = os.path.join("data", "citigroup", "raw_docs", "presentations", "Q4_2024_presentation.pdf")
	# presentation_etl = PresentationETL(
	# 	input_pdf_path=input_pdf_path,
	# 	is_q4_presentation=True,
	# )

	# # Run the transform method (no need to run extract method for presentations)
	# analysis_results_dict = presentation_etl.transform(
	# 	llm_backend="gemini",
	# 	llm_model_name="gemini-2.5-pro-preview-06-05",
	# )

	# # Run the load method
	# output_dir_path = os.path.join("data", "citigroup", "processed", "presentations")
	# presentation_etl.load(
	# 	transformed_data=analysis_results_dict,
	# 	output_dir_path=output_dir_path,
	# )
