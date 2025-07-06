"""
This module contains different ETL classes for extracting data from financial documents published by banks.
"""

from abc import ABC, abstractmethod
import os
import ast
from pathlib import Path
from typing import List, Literal, Optional, Set
import datetime
from collections import defaultdict
from tqdm import tqdm
import time
import sqlite3

from pydantic import BaseModel, Field
import pymupdf
import numpy as np
import pandas as pd
import yfinance as yf
from pinecone import Pinecone, ServerlessSpec
# import torch
# from sentence_transformers import SentenceTransformer


from boe_risk_monitoring.llms.processing_llms import ChunkingLLM
from boe_risk_monitoring.llms.document_analyser_llm import DocumentAnalyserLLM
import boe_risk_monitoring.config as config
from boe_risk_monitoring.utils.utils import reformat_reporting_period

DATA_FOLDER_PATH = config.DATA_FOLDER_PATH
AGGREGATED_DATA_FOLDER_PATH = config.AGGREGATED_DATA_FOLDER_PATH
NEWS_DATA_FOLDER_PATH = config.NEWS_DATA_FOLDER_PATH
APP_DATA_FOLDER_PATH = config.APP_DATA_FOLDER_PATH

SHARE_PRICE_HISTORY_START_DATE = config.SHARE_PRICE_HISTORY_START_DATE
PERMISSIBLE_BANK_NAMES = config.PERMISSIBLE_BANK_NAMES
BANK_NAME_MAPPING = config.BANK_NAME_MAPPING
TICKER_MAPPING = config.TICKER_MAPPING
PERMISSIBLE_VECTOR_DB_PROVIDERS = config.PERMISSIBLE_VECTOR_DB_PROVIDERS
PERMISSIBLE_DOC_TYPES = ['transcripts', 'presentations_text', 'presentations_graphs', 'presentations_tables', 'news_text', 'news_graphs']

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

class ArticleTextChunk(BaseModel):
    text: str
    # is_peer_comparison: bool
    banks_referenced: Optional[List[Literal['citigroup', 'jpmorgan', 'bankofamerica', 'other']]] = None

class ArticleGraphSummary(BaseModel):
    caption: str
    trend_summary: str
    # is_peer_comparison: bool
    banks_referenced: Optional[List[Literal['citigroup', 'jpmorgan', 'bankofamerica', 'other']]] = None

class Article(BaseModel):
    author: str
    article_title: str
    publication_date: str = Field(..., description="in format YYYY-MM-DD e.g. 2024-10-31")
    text_chunks: List[ArticleTextChunk]
    graphs: Optional[List[ArticleGraphSummary]] = None

class ArticlesDoc(BaseModel):
    doc: List[Article]

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
        self.input_pdf_path = input_pdf_path

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
            print("Invoking LLM for chunking...")
            chunked_transcript = self.llm_model.invoke()
            print("LLM chunking completed.")
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
        self.input_pdf_path = input_pdf_path

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
        print("Invoking LLM for document analysis...")
        analysis_results = self.llm_model.invoke()
        print("LLM analysis completed.")
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

class NLPInputDataAggregationETL(BaseETL):
    """This class provides methods to aggregate data from multiple ETL processes into single files ready for NLP analysis.
    """
    def __init__(self, data_dir_path, news_dir_path):
        # Check if input paths are valid
        if not isinstance(data_dir_path, str) or not isinstance(news_dir_path, str):
            raise TypeError("Data directory and news directory paths must be strings.")
        data_dir_path = Path(data_dir_path)
        if not data_dir_path.exists():
            raise FileNotFoundError(f"Data directory does not exist: {data_dir_path}")
        self.data_dir_path = data_dir_path
        news_dir_path = Path(news_dir_path)
        if not news_dir_path.exists():
            raise FileNotFoundError(f"News directory does not exist: {news_dir_path}")
        self.news_dir_path = news_dir_path

    def extract(self):
        """
        Extracts transcripts, presentations and news data from the data directory.
        Returns a nested dictionary with banks and document type as keys and associated lists of dataframes as their values.
        """

        # First let's focus on transcripts and presentations
        bank_dirs = [bank_dir for bank_dir in self.data_dir_path.iterdir() if bank_dir.is_dir() and bank_dir.name in PERMISSIBLE_BANK_NAMES]
        all_files = defaultdict(lambda: defaultdict(list))  # Nested dictionary to hold dataframes
        for bank_dir in bank_dirs:
            transcripts_dir = bank_dir / "processed" / "transcripts"
            presentations_dir = bank_dir / "processed" / "presentations"
            # get all transcript parquet files
            transcript_files = list(transcripts_dir.glob("[Q1-4]*.parquet"))
            presentation_text_files = list(presentations_dir.glob("[Q1-4]*_text.parquet"))
            presentation_graph_files = list(presentations_dir.glob("[Q1-4]*_graphs.parquet"))
            presentation_table_files = list(presentations_dir.glob("[Q1-4]*_tables.parquet"))
            # Store the dataframes in the nested dictionary
            bank_name = bank_dir.name
            bank_clean_name = BANK_NAME_MAPPING[bank_name]
            for file in transcript_files:
                df = pd.read_parquet(file)
                df['bank'] = bank_clean_name  # Add bank name column
                df['document_type'] = 'transcript'
                all_files[bank_clean_name]['transcripts'].append(df)
            for file in presentation_text_files:
                df = pd.read_parquet(file)
                df['bank'] = bank_clean_name
                df['document_type'] = 'presentation'
                all_files[bank_clean_name]['presentations_text'].append(df)
            for file in presentation_graph_files:
                df = pd.read_parquet(file)
                df['bank'] = bank_clean_name
                df['document_type'] = 'presentation'
                all_files[bank_clean_name]['presentations_graphs'].append(df)
            for file in presentation_table_files:
                df = pd.read_parquet(file)
                df['bank'] = bank_clean_name
                df['document_type'] = 'presentation'
                all_files[bank_clean_name]['presentations_tables'].append(df)

        # Now let's focus on news articles
        news_processed_dir = self.news_dir_path / "processed"
        if not news_processed_dir.exists():
            raise FileNotFoundError(f"Processed news directory does not exist: {news_processed_dir}")
        news_text_files = list(news_processed_dir.glob("*_text.parquet"))
        news_graph_files = list(news_processed_dir.glob("*_graphs.parquet"))
        for file in news_text_files:
            df = pd.read_parquet(file)
            df['banks_referenced'] = df['banks_referenced'].replace(BANK_NAME_MAPPING)
            df['document_type'] = 'news'
            all_files['news']['news_text'].append(df)
        for file in news_graph_files:
            df = pd.read_parquet(file)
            df['banks_referenced'] = df['banks_referenced'].replace(BANK_NAME_MAPPING)
            df['document_type'] = 'news'
            all_files['news']['news_graphs'].append(df)

        return all_files

    def transform(self, raw_data):
        """
        Transform the extracted data into a single dataframe for each document type.
        Returns a dictionary with document types as keys and their associated dataframes as values.
        """
        if not isinstance(raw_data, dict):
            raise TypeError("Raw data must be a nested dictionary with banks and document types")

        transcripts_df = pd.DataFrame()
        presentation_text_df = pd.DataFrame()
        presentation_graphs_df = pd.DataFrame()
        presentation_tables_df = pd.DataFrame()
        news_text_df = pd.DataFrame()
        news_graphs_df = pd.DataFrame()

        clean_bank_names = [BANK_NAME_MAPPING[bank] for bank in PERMISSIBLE_BANK_NAMES]
        # Check if all banks in raw_data are permissible
        for bank in clean_bank_names:
            for doc_type in raw_data[bank].keys():
                if doc_type not in PERMISSIBLE_DOC_TYPES:
                    raise ValueError(f"Unsupported document type: {doc_type}. Supported types are {PERMISSIBLE_DOC_TYPES}.")
                if not isinstance(raw_data[bank][doc_type], list):
                    raise TypeError(f"Data for {bank} - {doc_type} must be a list of dataframes.")
                # Concatenate all dataframes for the document type
                df = pd.concat(raw_data[bank][doc_type], ignore_index=True)
                if doc_type == 'transcripts':
                    transcripts_df = pd.concat([transcripts_df, df], ignore_index=True)
                    transcripts_df['reference'] = transcripts_df['speaker'] + " (" + transcripts_df['role'] + ")\n" + transcripts_df['bank'] + ", " + transcripts_df['reporting_period'].str.replace("_", ", ") + " Earnings Call Transcript, Page " + transcripts_df['page'].astype(str)
                elif doc_type == 'presentations_text':
                    presentation_text_df = pd.concat([presentation_text_df, df], ignore_index=True)
                    presentation_text_df['reference'] = "Slide Text\n" + presentation_text_df['bank'] + ", "+ presentation_text_df['reporting_period'].str.replace("_", ", ") + ", Earnings Call Presentation, Slide " +  transcripts_df['page'].astype(str)
                elif doc_type == 'presentations_graphs':
                    presentation_graphs_df = pd.concat([presentation_graphs_df, df], ignore_index=True)
                    presentation_graphs_df['reference'] = "Graph (" + presentation_graphs_df['caption'] + ")\n" + presentation_graphs_df['bank'] + ", " + presentation_graphs_df['reporting_period'].str.replace("_", ", ") + ", Earnings Call Presentation, Slide " + presentation_graphs_df['page'].astype(str)
                elif doc_type == 'presentations_tables':
                    presentation_tables_df = pd.concat([presentation_tables_df, df], ignore_index=True)
                    presentation_tables_df['reference'] = "Table Row (" + presentation_tables_df['row_heading'] + ")\n" + presentation_tables_df['bank'] + ", " + presentation_tables_df['reporting_period'].str.replace("_", ", ") + ", Earnings Call Presentation, Slide " + presentation_tables_df['page'].astype(str)

        for doc_type in raw_data['news'].keys():
            if doc_type not in ['news_text', 'news_graphs']:
                raise ValueError(f"Unsupported document type: {doc_type}. Supported types for news are ['news_text', 'news_graphs'].")
            if not isinstance(raw_data['news'][doc_type], list):
                raise TypeError(f"Data for news - {doc_type} must be a list of dataframes.")
            # Concatenate all dataframes for the document type
            df = pd.concat(raw_data['news'][doc_type], ignore_index=True)
            if doc_type == 'news_text':
                news_text_df = pd.concat([news_text_df, df], ignore_index=True)
                news_text_df['reference'] = "News article by " + news_text_df['author'] + "\n" + "Title: " + news_text_df['article_title'] + ", published " + news_text_df['publication_date']
            elif doc_type == 'news_graphs':
                news_graphs_df = pd.concat([news_graphs_df, df], ignore_index=True)
                news_graphs_df['reference'] = "News article by " + news_graphs_df['author'] + "\n" + "Title: " + news_graphs_df['article_title'] + ", published " + news_graphs_df['publication_date']


        # Add some extra columns
        transcripts_df['source_type'] = 'internal'
        presentation_text_df['source_type'] = 'internal'
        presentation_graphs_df['source_type'] = 'internal'
        presentation_tables_df['source_type'] = 'internal'
        news_text_df['source_type'] = 'external'
        news_graphs_df['source_type'] = 'external'
        transcripts_df['is_comparative'] = False
        presentation_text_df['is_comparative'] = False
        presentation_graphs_df['is_comparative'] = False
        presentation_tables_df['is_comparative'] = False

        # Convert the date columns to string format
        transcripts_df['date_of_call'] = pd.to_datetime(transcripts_df['date_of_call']).dt.strftime('%Y-%m-%d')
        presentation_text_df['date_of_presentation'] = pd.to_datetime(presentation_text_df['date_of_presentation']).dt.strftime('%Y-%m-%d')
        presentation_graphs_df['date_of_presentation'] = pd.to_datetime(presentation_graphs_df['date_of_presentation']).dt.strftime('%Y-%m-%d')
        presentation_tables_df['date_of_presentation'] = pd.to_datetime(presentation_tables_df['date_of_presentation']).dt.strftime('%Y-%m-%d')

        # Rename the 'speaker'/'author' columns to 'source'
        transcripts_df.rename(columns={'speaker': 'source'}, inplace=True)
        news_text_df.rename(columns={'author': 'source'}, inplace=True)
        news_graphs_df.rename(columns={'author': 'source'}, inplace=True)

        # Rename the banks_referenced column to 'bank'
        news_text_df.rename(columns={'banks_referenced': 'bank'}, inplace=True)
        news_graphs_df.rename(columns={'banks_referenced': 'bank'}, inplace=True)

        # Now we'll create a specific dataframe which summarises all text components of the various documents
        # Transcripts
        transcripts_df2 = transcripts_df.copy()
        transcripts_df2 = transcripts_df2.rename(columns={'date_of_call': 'publication_date'})

        # Presentation text
        presentation_text_df2 = presentation_text_df.copy()
        presentation_text_df2 = presentation_text_df2.rename(columns={'date_of_presentation': 'publication_date'})

        # Presentation graphs
        presentation_graphs_df2 = presentation_graphs_df.copy()
        presentation_graphs_df2['trend_summary'] = presentation_graphs_df2['caption'] + ": " + presentation_graphs_df2['trend_summary']
        presentation_graphs_df2.drop(columns=['caption'], inplace=True)
        presentation_graphs_df2 = presentation_graphs_df2.rename(columns={
            'date_of_presentation': 'publication_date',
            "trend_summary": "text",
            "comparison_scope": "fiscal_period_ref",
            }
        )

        # Presentation tables
        presentation_tables_df2 = presentation_tables_df.copy()
        # Drop columns we don't need
        presentation_tables_df2.drop(columns=[
            'row_number',
            'current_quarter_value',
            'preceding_quarter_value',
            'prev_year_same_quarter_value',
            'current_year_value',
            'preceding_year_value',
            'unit'
            ], inplace=True)

        value_vars = ['preceding_quarter_trend_summary', 'year_on_year_quarter_trend_summary', 'preceding_year_trend_summary']
        id_vars = [col for col in presentation_tables_df2.columns if col not in value_vars]

        presentation_tables_df2 = presentation_tables_df2.melt(id_vars=id_vars, value_vars=value_vars, var_name="text_scope", value_name="text")
        presentation_tables_df2.dropna(subset=['text'], inplace=True)  # Remove rows with empty text
        presentation_tables_df2["text_scope"] = presentation_tables_df2['text_scope'].replace(
            {
                'preceding_quarter_trend_summary': 'preceding quarter trend',
                'year_on_year_quarter_trend_summary': 'year-on-year quarter trend',
                'preceding_year_trend_summary': 'preceding year trend',
            }
        )


        # Combine the row heading and text scope into the text column
        presentation_tables_df2['text'] = presentation_tables_df2['row_heading'] + ", " + presentation_tables_df2['text_scope'] + ": " + presentation_tables_df2['text']

        # Create the fiscal period reference for consistency
        presentation_tables_df2['fiscal_period_ref'] = presentation_tables_df2['text_scope'].apply(
            lambda x: 'quarter' if 'quarter' in x else 'year'
        )

        # Drop the now redundant columns
        presentation_tables_df2.drop(columns=['row_heading', 'text_scope'], inplace=True)

        # Reorder the columns
        cols_in_order = [col for col in presentation_tables_df2.columns if col != 'text' and col != 'fiscal_period_ref']
        presentation_tables_df2 = presentation_tables_df2[['text', 'fiscal_period_ref'] + cols_in_order]
        presentation_tables_df2 = presentation_tables_df2.rename(columns={'date_of_presentation': 'publication_date'})

        # News text
        news_text_df2 = news_text_df.copy()
        news_text_df2.drop(columns="article_title", inplace=True)
        # We'll map the news data to particular fiscal periods e.g. Q1_2025 based on the earnings call dates
        # We'll use the period between earnings calls with a 1 week shift to determine the fiscal period to map the news to
        news_publication_dates_df = news_text_df2[['bank','publication_date']].copy()
        news_publication_dates_df = news_publication_dates_df.drop_duplicates().reset_index(drop=True)
        # Convert publication_date to datetime
        news_publication_dates_df['publication_date_dt'] = pd.to_datetime(news_publication_dates_df['publication_date'])
        earnings_call_dates_df = transcripts_df[['bank', 'reporting_period', 'date_of_call']].copy()
        earnings_call_dates_df = earnings_call_dates_df.drop_duplicates()
        # Convert date_of_call to datetime
        earnings_call_dates_df['date_of_call_dt'] = pd.to_datetime(earnings_call_dates_df['date_of_call'])
        # Add a 1 week buffer to the earnings call dates (to account for the fact that news articles are often published after the earnings call)
        earnings_call_dates_df['date_of_call_dt'] += pd.Timedelta(weeks=1)

        # Map the publication dates to reporting periods
        news_publication_dates_df = self.map_news_to_reporting_periods(news_publication_dates_df, earnings_call_dates_df)

        # Merge the mapped reporting periods back into the news text dataframe
        news_text_df2 = news_text_df2.merge(news_publication_dates_df[['bank', 'publication_date', 'reporting_period']], on=['bank', 'publication_date'], how='left')

        # News graphs
        news_graphs_df2 = news_graphs_df.copy()
        news_graphs_df2['trend_summary'] = news_graphs_df2['caption'] + ": " + news_graphs_df2['trend_summary']
        news_graphs_df2.drop(columns=['caption', 'article_title'], inplace=True)
        news_graphs_df2 = news_graphs_df2.rename(columns={"trend_summary": "text"})
        # TODO: Refactor below
        # We'll map the news data to particular fiscal periods e.g. Q1_2025 based on the earnings call dates
        # We'll use the period between earnings calls with a 1 week shift to determine the fiscal period to map the news to
        news_publication_dates_df = news_graphs_df2[['bank','publication_date']].copy()
        news_publication_dates_df = news_publication_dates_df.drop_duplicates().reset_index(drop=True)
        # Convert publication_date to datetime
        news_publication_dates_df['publication_date_dt'] = pd.to_datetime(news_publication_dates_df['publication_date'])

        # Map the publication dates to reporting periods
        news_publication_dates_df = self.map_news_to_reporting_periods(news_publication_dates_df, earnings_call_dates_df)

        # Merge the mapped reporting periods back into the news graphs dataframe
        news_graphs_df2 = news_graphs_df2.merge(news_publication_dates_df[['bank', 'publication_date', 'reporting_period']], on=['bank', 'publication_date'], how='left')

        # Collect dfs to concatenate
        dfs_to_concat = [transcripts_df2, presentation_text_df2, presentation_graphs_df2, presentation_tables_df2,
                         news_text_df2, news_graphs_df2]
        all_text_df = pd.concat(dfs_to_concat, ignore_index=True)

        all_text_df['fiscal_period_ref'] = all_text_df['fiscal_period_ref'].astype('string')
        # Replace null values in source and role columns with empty string (for vector database compatibility)
        fillna_cols = ['fiscal_period_ref', 'role', 'page', 'section']
        # Fill NAs and cast the to string type
        all_text_df[fillna_cols] = all_text_df[fillna_cols].fillna('').astype(str)
        # Ensure publication_date is in string format
        all_text_df['publication_date'] = pd.to_datetime(all_text_df['publication_date']).dt.strftime('%Y-%m-%d')

        results_dict = {
            'transcripts': transcripts_df,
            'presentation_text': presentation_text_df,
            'presentation_graphs': presentation_graphs_df,
            'presentation_tables': presentation_tables_df,
            'news_text': news_text_df,
            'news_graphs': news_graphs_df,
            'all_text': all_text_df,
        }

        return results_dict


    def load(self, transformed_data, output_dir_path):
        if not isinstance(transformed_data, dict):
            raise TypeError("Transformed data must be a dictionary with document types as keys and their associated dataframes as values.")

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

        # Save each dataframe to parquet files
        for doc_type, df in transformed_data.items():
            file_name_csv = Path(doc_type + ".csv")
            file_name_parquet = Path(doc_type + ".parquet")
            full_path_csv = output_dir_path / file_name_csv
            full_path_parquet = output_dir_path / file_name_parquet
            df.to_csv(full_path_csv, index=False)
            df.to_parquet(full_path_parquet)

    @staticmethod
    def map_news_to_reporting_periods(news_publication_dates_df, earnings_call_dates_df):
            """
            Maps each news publication date to the appropriate reporting period based on earnings call dates.
            Returns a DataFrame with an added 'reporting_period' column.
            """
            # Make a copy of the earnings call dates DataFrame to avoid modifying the original
            earnings_call_dates_df2 = earnings_call_dates_df.copy()
            # Sort the earnings call dates by bank and date
            earnings_call_dates_df2 = earnings_call_dates_df2.sort_values(by=['bank', 'date_of_call_dt']).reset_index(drop=True)
            mapped_reporting_periods = []
            for _, row in news_publication_dates_df.iterrows():
                bank = row['bank']
                pub_date = row['publication_date_dt']
                earnings_call_dates_bank_df = earnings_call_dates_df2[earnings_call_dates_df2['bank'] == bank].copy().reset_index(drop=True)
                gt_pub_date_bool_srs = (earnings_call_dates_bank_df['date_of_call_dt'] > pub_date)
                if not gt_pub_date_bool_srs.any():
                    mapped_reporting_period = earnings_call_dates_bank_df['reporting_period'].iloc[-1]
                    mapped_reporting_periods.append(mapped_reporting_period)
                    continue
                boundary_idx = (earnings_call_dates_bank_df['date_of_call_dt'] > pub_date).idxmax()
                mapped_reporting_period = earnings_call_dates_bank_df.loc[boundary_idx, 'reporting_period']
                mapped_reporting_periods.append(mapped_reporting_period)
            news_publication_dates_df = news_publication_dates_df.copy()
            news_publication_dates_df['reporting_period'] = mapped_reporting_periods
            return news_publication_dates_df

class SupplementaryDataETL(BaseETL):
    pass

class SharePriceDataETL:
    """
    A class for extracting, transforming, visualizing, and exporting share price data
    for G-SIBs (Globally Systemically Important Banks) using data from Yahoo Finance.

    Features:
    - Downloads share price data for a specified bank and time range using the `yfinance` library.
    - Automatically detects the bank's full name and trading currency (with fallbacks).
    - Merges the KBW Nasdaq Global Bank Index (^GBKX) for benchmark comparison.
    - Allows for data export to CSV and plotting of both bank and index time series.

    Note:
    -----
    This class includes the KBW Nasdaq Global Bank Index (^GBKX) as a benchmark.
    This index tracks the performance of major global banks (G-SIBs) and is equally weighted.
    Read more: https://indexes.nasdaqomx.com/Index/Overview/GBKX

    Parameters:
    ----------
    ticker : str
        Yahoo Finance ticker symbol for the bank (e.g., "JPM" for JPMorgan Chase; see relevant bank names/tickers in the fallbacks below).
    start_date : str
        Start date for the time series (format: "YYYY-MM-DD").
    end_date : str
        End date for the time series (format: "YYYY-MM-DD").
    """
    def __init__(self, bank_name, ticker, start_date, end_date):
        self.ticker = ticker
        self.bank_name = bank_name
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.currency = None

    def extract(self):
        print(f"Extracting data for {self.ticker} from {self.start_date} to {self.end_date}...")
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        data = data.droplevel(1, axis="columns")
        self.data = data

        if self.currency is None:
            try:
                ticker_info = yf.Ticker(self.ticker).info
                self.currency = ticker_info.get("currency", None)
            except Exception as e:
                print(f"Could not retrieve metadata for {self.ticker}: {e}")  # Yahoo blocks Google. Retrieval doesn't work on Colab, but should work locally and on GitHub.
                self.currency = None

        # Fallbacks if info lookup failed
        if self.currency is None:
            fallback_currencies = {
                "JPM": "USD",
                "BAC": "USD",
                "HSBC": "GBP",
                "C": "USD",
                "BARC.L": "GBP",
                "GS": "USD",
                "UBS": "CHF",
                "SAN": "EUR",
                "MS": "USD",
            }
            self.currency = fallback_currencies.get(self.ticker, "Unknown")

    def transform(self):
        if self.data is not None and not self.data.empty:
            print("Transforming data...")
            self.data.reset_index(inplace=True)
            self.data = self.data[["Date", "Close"]]
            self.data["Bank"] = BANK_NAME_MAPPING[self.bank_name]
            self.data["Currency"] = self.currency

            # Download and merge Global Bank Index
            print("Downloading benchmark data (^GBKX)...")
            benchmark_data = yf.download("^GBKX", start=self.start_date, end=self.end_date)
            benchmark_data.reset_index(inplace=True)
            benchmark_data = benchmark_data.droplevel(1, axis="columns")  # Remove multi-index columns
            benchmark_data = benchmark_data[["Date", "Close"]].rename(columns={"Close": "GlobalBankIndex"})

            # Merge benchmark by Date
            self.data = pd.merge(self.data, benchmark_data, on="Date", how="left") # The left-merge keeps bank data intact.

            # Reorder columns
            self.data = self.data[["Bank", "Date", "Close", "Currency", "GlobalBankIndex"]]
        else:
            print("No data to transform. Run extract() first or check if data is empty.")

    def load(self, fpath):
        if self.data is not None and not self.data.empty:
            print(f"Saving data to {fpath}...")
            self.data.to_csv(fpath, index=False)
        else:
            print("No data to save. Run extract() and transform() first.")

    def export_to_csv(self, fpath):
        """Convenience method: Extracts, transforms, and saves to CSV."""
        self.extract()
        self.transform()
        self.load(fpath)


    def plot_time_series(self):
        self.extract()
        self.transform()

        if self.data is not None and not self.data.empty:
            print(f"Plotting time series for {self.bank_name} and Global Bank Index...")

            fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

            # Top chart: bank share price
            axes[0].plot(self.data["Date"], self.data["Close"], label=self.bank_name)
            axes[0].set_title(f"{self.bank_name} - Share Price Over Time")
            axes[0].set_ylabel(f"Close Price ({self.currency} per share)")
            axes[0].grid(True)
            axes[0].legend()

            # Bottom chart: index
            axes[1].plot(self.data["Date"], self.data["GlobalBankIndex"], label="KBW Nasdaq Global Bank Index", color="orange")
            axes[1].set_title("KBW Nasdaq Global Bank Index Over Time")
            axes[1].set_ylabel("Index Value (Equally Weighted)")
            axes[1].set_xlabel("Date")
            axes[1].grid(True)
            axes[1].legend()

            plt.tight_layout()
            plt.show()

        else:
            print("No data available to plot.")

class VectorDBETL(BaseETL):
    """
    This class provides methods to extract data from NLP outputs and store embeddings in a vector DB along with associated metadata.
    """
    def __init__(self, input_parquet_path, vector_db_provider, index_name):
        # Check if input paths are valid
        if not isinstance(input_parquet_path, str):
            raise TypeError("Input path must be a string.")
        if not input_parquet_path.endswith('.parquet'):
            raise ValueError("Input path must be a Parquet file.")
        input_parquet_path = Path(input_parquet_path)
        if not input_parquet_path.exists():
            raise FileNotFoundError(f"Input Parquet file does not exist: {input_parquet_path}")
        self.input_parquet_path = input_parquet_path

        if vector_db_provider not in PERMISSIBLE_VECTOR_DB_PROVIDERS:
            raise NotImplementedError(f"Unsupported vector DB provider: {vector_db_provider}. Supported providers are {PERMISSIBLE_VECTOR_DB_PROVIDERS}.")
        self.vector_db_provider = vector_db_provider

        if not isinstance(index_name, str):
            raise TypeError("Index name must be a string.")
        self.index_name = index_name
        # Placeholder for the embedding dimension
        self.embedding_dim = None

    def extract(self):
        """
        Extracts data from the input Parquet file.
        Returns a DataFrame containing the data.
        """
        print(f"Extracting data from {self.input_parquet_path}...")
        df = pd.read_parquet(self.input_parquet_path)
        return df

    def transform(self, raw_data, col_to_embed_name, embedding_model):
        """
        Transforms the extracted data by generating embeddings for the specified column. Any other columns in the raw data are retained as metadata.
        Returns a DataFrame with embeddings and associated metadata.
        """
        if not isinstance(raw_data, pd.DataFrame):
            raise TypeError("Raw data must be a Pandas DataFrame.")
        if col_to_embed_name not in raw_data.columns:
            raise ValueError(f"Column to embed '{col_to_embed_name}' not found in the raw data.")
        if not isinstance(embedding_model, SentenceTransformer):
            raise TypeError("Embedding model must be an instance of HuggingFace's SentenceTransformer.")

        # Store the embedding model instance
        self.embedding_dim = embedding_model.get_sentence_embedding_dimension()

        if not torch.cuda.is_available():
            print("CUDA is not available. Please ensure you have a compatible GPU and PyTorch installed with CUDA support. Running on CPU - this may be slow.")

        # Generate embeddings for the specified column, extract metadata and build vectors object for upsert
        print(f"Generating embeddings from column {col_to_embed_name}")
        text_embeddings_arr = embedding_model.encode(raw_data[col_to_embed_name].tolist())
        text_embeddings_list = text_embeddings_arr.tolist()  # Convert numpy array to list
        embeddings_idx_list = raw_data.index.astype(str).tolist()
        metadata_df = raw_data.drop(columns=[col_to_embed_name])
        if metadata_df.empty:
            metadata_dicts_list = []
        else:
            metadata_dicts_list = metadata_df.to_dict(orient='records')
        # Build vectors object for upsert
        vectors = self.build_vectors_object(embeddings_idx_list, text_embeddings_list, metadata_dicts_list)

        return vectors

    def load(self, transformed_data):
        if not isinstance(transformed_data, list) or len(transformed_data) == 0:
            raise TypeError("Transformed data must be a list of dictionaries each with 'id', 'values' and optionally 'metadata' keys to upsert into the vector DB.")
        for item in transformed_data:
            if not isinstance(item, dict):
                raise TypeError("Each item in transformed data must be a dictionary.")
            keys_list_set = set(item.keys())
            mandatory_keys_set = set(['id', 'values'])
            allowed_keys_set = mandatory_keys_set.union({'metadata'})
            # Check if 'id' and 'values' keys are present
            if not mandatory_keys_set.issubset(keys_list_set):
                raise ValueError("Each item in transformed data must contain 'id' and 'values' keys.")
            # Check if any other keys are present
            if not keys_list_set.issubset(allowed_keys_set):
                raise ValueError(f"Each item in transformed data can only contain keys {allowed_keys_set}. Found keys: {keys_list_set}.")

        if not self.embedding_dim:
            # Infer embedding model from the transformed data
            # Get first item to determine the embedding dimension
            self.embedding_dim = len(transformed_data[0]['values'])


        print("Initializing vector DB index")
        if self.vector_db_provider == "pinecone":
            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                raise ValueError("PINECONE_API_KEY required in env file")
            vector_db = Pinecone(api_key=api_key)
            # Check existing indices
            existing_indices = [idx.name for idx in vector_db.list_indexes()]
            if self.index_name in existing_indices:
                print(f"{self.index_name} already exists. Deleting and recreating...")
                vector_db.delete_index(self.index_name)
            vector_db.create_index(
                name=self.index_name,
                dimension=self.embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            # Connect to the index
            db_idx = vector_db.Index(self.index_name)

            # Upsert the data in batches
            print("Upserting data into Pinecone index...")
            self.batch_upsert(db_idx, transformed_data, batch_size=100)

        else:
            raise NotImplementedError(f"Vector DB provider {self.vector_db_provider} is not implemented yet.")


    @staticmethod
    def build_vectors_object(embeddings_idx_list, text_embeddings_list, metadata_dicts_list=[]):
        """
        Builds a list of vectors to be upserted into the vector DB.
        Each vector is a dictionary with 'id', 'values', and 'metadata'.
        """
        if not isinstance(embeddings_idx_list, list) or not isinstance(text_embeddings_list, list) or not isinstance(metadata_dicts_list, list):
            raise TypeError("All inputs must be lists.")

        if len(embeddings_idx_list) != len(text_embeddings_list):
            raise ValueError("Embeddings list must have the same length as index list.")

        if metadata_dicts_list:
            if len(metadata_dicts_list) != len(embeddings_idx_list):
                raise ValueError("Metadata list must have the same length as index and embeddings lists.")

        vectors = []
        if metadata_dicts_list:
            for idx, embedding, metadata in zip(embeddings_idx_list, text_embeddings_list, metadata_dicts_list):
                if not isinstance(metadata, dict):
                    raise TypeError("Metadata must be a dictionary.")
                if not isinstance(embedding, list):
                    raise TypeError("Embedding must be a list.")
                vector = {
                    "id": idx,
                    "values": embedding,
                    "metadata": metadata
                }
                vectors.append(vector)
        else:
            for idx, embedding in zip(embeddings_idx_list, text_embeddings_list):
                if not isinstance(embedding, list):
                    raise TypeError("Embedding must be a list.")
                vector = {
                    "id": idx,
                    "values": embedding,
                }
                vectors.append(vector)
        return vectors

    @staticmethod
    def batch_upsert(index, vectors, batch_size=100):
        for i in tqdm(range(0, len(vectors), batch_size), desc="Upserting to Pinecone"):
            batch = vectors[i:i + batch_size]
            response = index.upsert(batch)
            upserted_count = response.get("upserted_count", 0)
            if upserted_count != len(batch):
                print(f"Warning: Only {upserted_count}/{len(batch)} vectors upserted.(Batch start_idx: {i})")

class FinancialNewsETL(BaseETL):
    """
    This class provides methods to extract, transform, and load financial news articles stored in pdf format.

    The pdf files are fed directly to the LLM for analysis, so no explicity text extraction is performed in this class.
    """
    def __init__(self, input_pdf_path):
        # Check if input paths are valid
        if not isinstance(input_pdf_path, str):
            raise TypeError("Input path must be a string.")
        if not input_pdf_path.endswith('.pdf'):
            raise ValueError("Input path must be a PDF file.")

        input_pdf_path = Path(input_pdf_path)
        if not input_pdf_path.exists():
            raise FileNotFoundError(f"Input PDF file does not exist: {input_pdf_path}")
        self.input_pdf_path = input_pdf_path

        # Get the file name without the extension
        self.file_name = input_pdf_path.stem

        self.llm_model = None
        self.output_dir_path = None
        self.output_file_paths_csv = []
        self.output_file_paths_parquet = []

    def extract(self):
        print("Skipping extraction step — using LLM-native document ingestion.")
        return None

    def transform(self, llm_backend="openai", llm_model_name="gpt-4o", temperature=0.3):

        response_schema = ArticlesDoc
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
        print("Invoking LLM for document analysis...")
        analysis_results = self.llm_model.invoke()
        print("LLM analysis completed.")
        text_df, graphs_df = self.unpack_analysis_results(analysis_results)

        print("Reformatting output dataframes...")
        text_df = self.clean_output_df(text_df)
        graphs_df = self.clean_output_df(graphs_df)

        # Combine into single dictionary with keys as file name suffixes
        results_dict = {
            "text": text_df,
            "graphs": graphs_df,
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

        # Drop any file names where the associated dataframe is empty
        file_names = [file_name for file_name in file_names if not transformed_data[file_name].empty]

        if not file_names:
            print(f"{self.file_name} has no bank-specific data to save. Skipping load step.")

        output_dir_path = Path(output_dir_path)
        # Make the directory if it does not exist already
        output_dir_path.mkdir(parents=True, exist_ok=True)
        self.output_dir_path = output_dir_path

        # Get the original file name without the extension for prefixing
        prefix = self.file_name

        # Construct full paths and save
        output_file_paths_csv = []
        output_file_paths_parquet = []
        for file_name in file_names:
            # save csv
            file_name_obj_csv = Path(prefix + "_" + file_name + ".csv")
            full_path_csv = output_dir_path / file_name_obj_csv
            output_file_paths_csv.append(full_path_csv)
            transformed_data[file_name].to_csv(full_path_csv, index=False)
            # save parquet
            file_name_obj_parquet = Path(prefix + "_" + file_name + ".parquet")
            full_path_parquet = output_dir_path / file_name_obj_parquet
            output_file_paths_parquet.append(full_path_parquet)
            transformed_data[file_name].to_parquet(full_path_parquet)

        self.output_file_paths_csv = output_file_paths_csv
        self.output_file_paths_parquet = output_file_paths_parquet

    @staticmethod
    def _make_analysis_prompt():
        return (
            "You are a banking and finance expert  analysing financial news articles about bank firms.\n"
            "Your task is to extract structured data from each article based on text and graph content. There may be multiple articles in a given file.\n"
            "Analyze each article individually and extract the following:\n"
            "1. text: Break the article text into small, coherent chunks, each representing a single idea.\n"
            "For each chunk, return:\n"
            "- text: the chunk of text.\n"
            # "- is_peer_comparison: a boolean indicating if the text compares the performance of bank peers.\n"
            "- banks_referenced: a list of banks mentioned in the chunk (return None if no banks are mentioned, or a single-item list if only one bank is mentioned). Only include banks that are explicitly compared or whose performance is discussed in the chunk. List each bank mentioned **once only**. Map aliases (e.g., 'Citi' --> 'citigroup') as needed.\n"
            "2. graphs: For each graph on the slide (e.g., line or bar charts), return:\n"
            "- caption: a short descriptive title.\n"
            f"- trend summary: a brief explanation of key trends or insights for the plotted variables. If the graph compares peers, make sure to summarise the comparison. When relevant, **focus on comparing these banks**: {', '.join(BANK_NAME_MAPPING.values())}\n"
            # "- is_peer_comparison: a boolean indicating if the graph compares the performance of bank peers.\n"
            "- banks_referenced: a list of banks referenced in the graph (return None if no banks are referenced, or a single-item list if only one bank is referenced). Only include banks that are explicitly compared or whose performance is discussed in the chunk. List each bank mentioned **once only**. Map aliases (e.g., 'Citi' --> 'citigroup') as needed."
        )

    @staticmethod
    def unpack_analysis_results(analysis_results):
        """
        This function takes the pydantic schema returned from document analysis LLM call and converts it
        to three dataframes capturing textual and graphical summaries.
        """
        main_results = analysis_results.doc
        text_data = defaultdict(list)
        graph_data = defaultdict(list)
        # Iterate through the schema
        for article in main_results:
            author = article.author
            article_title = article.article_title
            publication_date = article.publication_date
            for text_chunk in article.text_chunks:
                text_data['text'].append(text_chunk.text)
                # text_data['is_peer_comparison'].append(text_chunk.is_peer_comparison)
                banks_referenced = text_chunk.banks_referenced
                if banks_referenced:
                    text_data['banks_referenced'].append(banks_referenced)
                else:
                    text_data['banks_referenced'].append(pd.NA)
                text_data['author'].append(author)
                text_data['article_title'].append(article_title)
                text_data['publication_date'].append(publication_date)

            if article.graphs:
                for graph in article.graphs:
                    graph_data['caption'].append(graph.caption)
                    graph_data['trend_summary'].append(graph.trend_summary)
                    # graph_data['is_peer_comparison'].append(graph.is_peer_comparison)
                    banks_referenced = graph.banks_referenced
                    if banks_referenced:
                        graph_data['banks_referenced'].append(banks_referenced)
                    else:
                        graph_data['banks_referenced'].append(pd.NA)
                    graph_data['author'].append(author)
                    graph_data['article_title'].append(article_title)
                    graph_data['publication_date'].append(publication_date)

        # Construct dataframes
        text_df = pd.DataFrame(text_data)
        graphs_df = pd.DataFrame(graph_data)

        # # Sort by publication date
        # text_df['publication_date'] = pd.to_datetime(text_df['publication_date'])
        # graphs_df['publication_date'] = pd.to_datetime(graphs_df['publication_date'])
        # text_df.sort_values(by='publication_date', inplace=True)
        # graphs_df.sort_values(by='publication_date', inplace=True)

        return text_df, graphs_df

    @staticmethod
    def clean_output_df(df):
        """
        Cleans the output dataframe to ensure it has the correct columns and data types.
        """
        # Check if the dataframe is empty
        if df.empty:
            print("Warning: The dataframe is empty. Returning an empty dataframe.")
            return df
        # Drop generic statements which don't mention specific banks
        df = df.dropna(subset=['banks_referenced'])
        # Check that all remaining entries in banks_referenced are lists
        if not df['banks_referenced'].apply(lambda x: isinstance(x, list)).all():
            raise ValueError("All non-NaN entries in 'banks_referenced' must be lists.")

        # Remove any duplicate items in lists
        df['banks_referenced'] = df['banks_referenced'].apply(lambda x: list(set(x)))

        # Drop text chunks that only reference "other" banks and none of our banks of interest
        excl_bool = df['banks_referenced'].apply(lambda x: len(x) == 1 and x[0] == "other")
        df = df[~excl_bool]

        # Early return if no valid data left
        if df.empty:
            return df

        # Add is_comparative columns if multiple banks from those we are interested in are mentioned
        df['is_comparative'] = df['banks_referenced'].apply(lambda x: sum([bank in x for bank in PERMISSIBLE_BANK_NAMES]) > 1)

        # Now we'll create duplicate rows for each bank mentioned in the text chunk
        df = df.explode('banks_referenced')

        # Drop rows which reference "other"
        df = df[df['banks_referenced'] != "other"]

        # Reset the index
        df.reset_index(drop=True, inplace=True)

        return df

class FinancialMetricsAggregationETL(BaseETL):
    """
    This class provides methods to aggregate financial metrics data from multiple sources into a single dataframe.
    It is designed to be used after NLPInputDataAggregationETL has already been run (as it requires knowledge of the dates of earnings calls from transcripts).
    """
    def __init__(self, data_dir_path, transcripts_parquet_path):
        # Check if input path is valid
        if not isinstance(data_dir_path, str):
            raise TypeError("Data directory path must be a string.")
        data_dir_path = Path(data_dir_path)
        if not data_dir_path.exists():
            raise FileNotFoundError(f"Data directory does not exist: {data_dir_path}")
        self.data_dir_path = data_dir_path
        # Check if transcripts parquet file is valid
        if not isinstance(transcripts_parquet_path, str) or not transcripts_parquet_path.endswith('.parquet'):
            raise TypeError("Transcripts path must be a string. It must point to a Parquet file.")
        transcripts_parquet_path = Path(transcripts_parquet_path)
        if not transcripts_parquet_path.exists():
            raise FileNotFoundError(f"Transcripts parquet file does not exist: {transcripts_parquet_path}")
        self.transcripts_parquet_path = transcripts_parquet_path

        # We'll check that the files we expect exist.
        bank_dirs = [bank_dir for bank_dir in self.data_dir_path.iterdir() if bank_dir.is_dir() and bank_dir.name in PERMISSIBLE_BANK_NAMES]
        all_files = defaultdict(lambda: defaultdict(dict))
        # Iterate through each bank directory
        for bank_dir in bank_dirs:
            # Get the name of the bank
            bank_name = bank_dir.name
            metrics_fpath = bank_dir / "processed" / "supplementary_data" / f"{bank_name}_cleaned_summary.xlsx"
            share_price_history_fpath = bank_dir / "processed" / "share_price_history" / "share_price_history.csv"
            if not metrics_fpath.exists():
                raise FileNotFoundError(f"Metrics file does not exist for {bank_name}: {metrics_fpath}")
            all_files[bank_name]["metrics"]["fpath"] = metrics_fpath
            if not share_price_history_fpath.exists():
                raise FileNotFoundError(f"Share price history file does not exist for {bank_name}: {share_price_history_fpath}")
            all_files[bank_name]["share_prices"]["fpath"] = share_price_history_fpath

        self.all_files = all_files




    def extract(self):
        """
        Extracts financial metrics data from the data directory.
        Returns a dictionary with keys as file names and values as DataFrames.
        """
        # First let's get the financial metrics and share price history files from the data directory.
        all_files = self.all_files
        for bank in all_files.keys():
            metrics_fpath = all_files[bank]["metrics"]["fpath"]
            share_price_history_fpath = all_files[bank]["share_prices"]["fpath"]
            df_metrics = pd.read_excel(metrics_fpath)
            df_metrics.columns = ["reporting_period"] + list(df_metrics.columns[1:])
            # Ensure reporting_period is in the right format
            split_srs = df_metrics['reporting_period'].str.split("Q")
            df_metrics['reporting_period'] = (split_srs.str.get(1).astype(int)+2000).astype(str) + "Q" + split_srs.str.get(0)
            df_share_price_history = pd.read_csv(share_price_history_fpath)
            df_share_price_history['Date'] = pd.to_datetime(df_share_price_history['Date']).dt.date
            all_files[bank]["metrics"]["data"] = df_metrics
            all_files[bank]["share_prices"]["data"] = df_share_price_history

        # Now let's get the earnings calls dates from the transcripts parquet file - this is an aggregated file so we will disaggregate to individul banks before storing in the dictionary.
        transcripts_df = pd.read_parquet(self.transcripts_parquet_path)
        # Ensure the transcripts DataFrame has the necessary columns
        expected_cols = ["reporting_period", "date_of_call", "bank"]
        if not all(col in transcripts_df.columns for col in expected_cols):
            raise ValueError(f"Transcripts parquet file must contain all of the following columns: {expected_cols}.")
        transcripts_df = transcripts_df[expected_cols]
        # Deduplicate
        transcripts_df = transcripts_df.drop_duplicates().reset_index(drop=True)
        # Convert date_of_call to datetime
        transcripts_df['date_of_call'] = pd.to_datetime(transcripts_df['date_of_call']).dt.date
        # Reformat reporting_period
        transcripts_df['reporting_period'] = reformat_reporting_period(transcripts_df['reporting_period'])
        # reporting_period_split_srs = transcripts_df['reporting_period'].str.split("_")
        # transcripts_df['reporting_period'] = reporting_period_split_srs.str.get(1).astype(str) + reporting_period_split_srs.str.get(0)
        for bank in all_files.keys():
            bank_name_clean = BANK_NAME_MAPPING[bank]
            # Filter the transcripts DataFrame for the current bank
            bank_transcripts_df = transcripts_df[transcripts_df['bank'] == bank_name_clean].copy()
            bank_transcripts_df.reset_index(drop=True, inplace=True)
            bank_transcripts_df = bank_transcripts_df.drop(columns=['bank'])
            if bank_transcripts_df.empty:
                raise ValueError(f"No transcripts data found for {bank}.")
            # Store the earnings call dates in the all_files dictionary
            all_files[bank]["earnings_calls_dates"]["data"] = bank_transcripts_df

        return all_files

    def transform(self, raw_data):
        """
        Transforms the extracted data by aggregating financial metrics.
        Returns a DataFrame with aggregated financial metrics.
        """
        # Check if raw_data is a dictionary with the expected structure
        if not isinstance(raw_data, dict):
            raise TypeError("Raw data must be a dictionary")

        # Check we have all the keys we need
        if not all(isinstance(bank_data, dict) and 'metrics' in bank_data and 'share_prices' in bank_data and 'earnings_calls_dates' in bank_data for bank_data in raw_data.values()):
            raise ValueError("Each bank's data must contain 'metrics', 'share_prices' and 'earnings_calls_dates' keys.")

        # Check for data keys and that the data is a DataFrame
        for bank in raw_data.keys():
            for data_type in ['metrics', 'share_prices', 'earnings_calls_dates']:
                if 'data' not in raw_data[bank][data_type]:
                    raise ValueError(f"Data for {bank} does not contain '{data_type}' data.")
                if not isinstance(raw_data[bank][data_type]['data'], pd.DataFrame):
                    raise TypeError(f"Data for {bank} under '{data_type}' must be a Pandas DataFrame.")


        metrics_dfs = []
        market_reaction_dfs = []
        for bank in raw_data:
            bank_clean_name = BANK_NAME_MAPPING[bank]
            # For the metrics data we simply concatenate
            metrics_df = raw_data[bank]["metrics"]["data"].copy()
            bank_clean_name = BANK_NAME_MAPPING[bank]
            metrics_df.insert(0, 'bank', bank_clean_name)  # Add bank name as first column
            metrics_dfs.append(metrics_df)

            # For share price reaction, we need to combine the share price history with the earnings call dates.
            share_price_df = raw_data[bank]["share_prices"]["data"].copy()
            earnings_calls_df = raw_data[bank]["earnings_calls_dates"]["data"].copy()
            share_prices_before = []
            share_prices_after = []
            for call_date in earnings_calls_df['date_of_call']:
                # Get the index of this day in the share price DataFrame
                earnings_call_idx = share_price_df[share_price_df['Date'] == call_date].index[0]
                # Get the share price 2 days before and after the earnings call
                share_prices_before.append(share_price_df.loc[earnings_call_idx-2, ['Close', 'GlobalBankIndex']].tolist())
                share_prices_after.append(share_price_df.loc[earnings_call_idx+2, ['Close', 'GlobalBankIndex']].tolist())

            earnings_calls_df[['share_price_before', 'index_before']] = share_prices_before
            earnings_calls_df[['share_price_after', 'index_after']] = share_prices_after
            earnings_calls_df['share_price_pct_change'] = (earnings_calls_df['share_price_after'] / earnings_calls_df['share_price_before'] - 1)*100
            earnings_calls_df['index_pct_change'] = (earnings_calls_df['index_after'] / earnings_calls_df['index_before'] - 1)*100
            earnings_calls_df['share_price_pct_change_relative_to_index'] = earnings_calls_df['share_price_pct_change'] - earnings_calls_df['index_pct_change']
            earnings_calls_df.insert(0, 'bank', bank_clean_name)
            keep_cols = ["bank", "reporting_period", "share_price_pct_change", "share_price_pct_change_relative_to_index"]
            earnings_calls_df = earnings_calls_df[keep_cols]
            market_reaction_dfs.append(earnings_calls_df)



        metrics_df = pd.concat(metrics_dfs, ignore_index=True)

        # We'll add LDR ratio to the metrics DataFrame as it is a better measure of risk than total loans by itself
        metrics_df['LDR ratio (%)'] = metrics_df['Total loans (billions of dollars)'] / metrics_df['Total deposits (billions of dollars)'] * 100
        # We'll drop the Total loans column as somewhat redundant now, we'll keep total deposits as it is a useful measure of liquidity risk by itself
        metrics_df.drop(columns=['Total loans (billions of dollars)'], inplace=True, errors='ignore')

        market_reaction_df = pd.concat(market_reaction_dfs, ignore_index=True)

        # Merge the two DataFrames on 'bank' and 'reporting_period'
        all_metrics_df = pd.merge(
            market_reaction_df, metrics_df,
            on=['bank', 'reporting_period'],
            how='left'
        )
        all_metrics_df.sort_values(by=['bank', 'reporting_period'], inplace=True)

        return all_metrics_df

    def load(self, transformed_data, output_dir_path):
        """Saves the transformed data to csv and parquet files in the app data folder."""
        if not isinstance(transformed_data, pd.DataFrame):
            raise TypeError("Transformed data must be a Pandas DataFrame.")
        if transformed_data.empty:
            raise ValueError("Transformed data is empty. Nothing to save.")
        if not isinstance(output_dir_path, str):
            raise TypeError("Output directory path must be a string.")
        output_dir_path = Path(output_dir_path)
        # Make the directory if it does not exist already
        output_dir_path.mkdir(parents=True, exist_ok=True)

        # Save the DataFrame to cav and parquet
        output_file_path_csv = output_dir_path / "financial_metrics.csv"
        output_file_path_parquet = output_dir_path / "financial_metrics.parquet"
        transformed_data.to_csv(output_file_path_csv, index=False)
        transformed_data.to_parquet(output_file_path_parquet, index=False)

class SQLDatabaseETL(BaseETL):
    """
    This class provides methods to extract financial metrics and sentiment data and store it in a SQL database.
    """
    raw_metrics_schema_dict = {
        "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "bank": "TEXT NOT NULL",
        "quarter": "TEXT NOT NULL",
        "category_type": "TEXT NOT NULL", # Risk or topic
        "category": "TEXT NOT NULL",  # Risk or topic name
        "metric_name": "TEXT NOT NULL", # E.g. sentiment
        "metric_value": "REAL NOT NULL",
        "rank": "INTEGER",

    }

    metric_trends_schema_dict = {
        "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "bank": "TEXT NOT NULL",
        "category_type": "TEXT NOT NULL",  # Risk or topic
        "category": "TEXT NOT NULL",  # Risk or topic name
        "metric_name": "TEXT NOT NULL",
        "trend_horizon": "TEXT NOT NULL",
        "slope": "REAL NOT NULL",
        "status": "TEXT NOT NULL",  # E.g. "improving", "deteriorating", "stable"
        "volatility": "REAL NOT NULL",
        "slope_rank": "INTEGER",
        "volatility_rank": "INTEGER",
    }

    # Mapping of metric names to their display names, risk categories and rank direction (False means higher has a lower rank i.e. is better)
    metric_name_mapping = {
        "share_price_pct_change": ["Earnings Call Share Price Change (%)", "General", False],
        "share_price_pct_change_relative_to_index": ["Earnings Call Share Price Change Relative to Global Index (%)", "General", False],
        "Noninterest Expense (millions of dollars)": ["Noninterest Expense (millions of dollars)", "Operational Risk", True],
        "Provision for Credit Losses (millions of dollars)": ["Provision for Credit Losses (millions of dollars)", "Asset Quality and Credit Risk", True],
        "Net income (millions of dollars)": ["Net Income (millions of dollars)", "Profitability", False],
        "Common Equity Tier 1 (CET1) Capital Ratio (%)": ["CET1 (%)", "Capital Adequacy", False],
        "Tier 1 Capital ratio ratio (%)": ["Tier 1 Ratio (%)", "Capital Adequacy", False],
        "Total Capital ratio (%)": ["TCR (%)", "Capital Adequacy", False],
        "Supplementary Leverage ratio (SLR) (%)": ["SLR (%)", "Capital Adequacy", False],
        "Return on average assets, ROA (%)": ["ROA (%)", "Profitability", False],
        "Return on average common equity (%)": ["ROCE (%)", "Profitability", False],
        "Return on average tangible common equity (RoTCE) (%)": ["RoTCE (%)", "Profitability", False],
        "Efficiency ratio(%)": ["Efficiency Ratio (%)", "Operational Risk", True],
        # "Total loans (billions of dollars)": ["Total Loans (billions of dollars)", "Asset Quality and Credit Risk", True],
        "LDR ratio (%)": ["LDR Ratio (%)", "Liquidity Risk", True],
        "Total deposits (billions of dollars)": ["Total Deposits (billions of dollars)", "Liquidity Risk", False],
        "Book value per share (dollars)": ["Book Value per Share ($)", "Strategic and Business Model Risk", False],
        "Tangible book value per share (dollars)": ["Tangible Book Value per Share ($)", "Strategic and Business Model Risk", False],
    }

    trend_periods_mapping = {
        'Short-term': 2,  # 2 quarters for short term
        'Medium-term': 4,  # 4 quarters for medium term
        'Long-term': 8  # 8 quarters for long term
    }

    def __init__(self, sql_db_fpath):
        # Check if db_url is a valid string
        if not isinstance(sql_db_fpath, str):
            raise TypeError("Database path must be a string.")
        if not sql_db_fpath.endswith('.db'):
            raise ValueError("Database path must point to a .db file.")
        sql_db_fpath = Path(sql_db_fpath)
        self.sql_db_fpath = sql_db_fpath
        # Create the directory if it does not exist
        sql_db_fpath.parent.mkdir(parents=True, exist_ok=True)
        if not sql_db_fpath.exists():
            print(f"Database file does not exist: {sql_db_fpath}. Creating a new database.")
        self.create_table("raw_metrics", SQLDatabaseETL.raw_metrics_schema_dict, unique_columns=['bank', 'quarter', 'category', 'metric_name'])
        self.create_table("metric_trends", SQLDatabaseETL.metric_trends_schema_dict, unique_columns=['bank', 'category', 'metric_name', 'trend_horizon'])

    def extract(self, sentiment_data_fpath, topic_relevance_data_fpath, metrics_data_fpath):
        """
        Extracts data from the specified files and returns a dictionary with the data.
        """
        if not isinstance(sentiment_data_fpath, str) or not sentiment_data_fpath.endswith('.parquet'):
            raise TypeError("Sentiment data file path must be a string pointing to a .parquet file.")
        if not isinstance(topic_relevance_data_fpath, str) or not topic_relevance_data_fpath.endswith('.parquet'):
            raise TypeError("Topic relevance data file path must be a string pointing to a .parquet file.")
        if not isinstance(metrics_data_fpath, str) or not metrics_data_fpath.endswith('.parquet'):
            raise TypeError("Metrics data file path must be a string pointing to a .parquet file.")

        sentiment_data_fpath = Path(sentiment_data_fpath)
        topic_relevance_data_fpath = Path(topic_relevance_data_fpath)
        metrics_data_fpath = Path(metrics_data_fpath)
        if not sentiment_data_fpath.exists():
            raise FileNotFoundError(f"Sentiment data file does not exist: {sentiment_data_fpath}. Make sure you have run the relevane ETL steps before this one.")
        if not topic_relevance_data_fpath.exists():
            raise FileNotFoundError(f"Topic relevance data file does not exist: {topic_relevance_data_fpath}. Make sure you have run the relevane ETL steps before this one.")
        if not metrics_data_fpath.exists():
            raise FileNotFoundError(f"Metrics data file does not exist: {metrics_data_fpath}. Make sure you have run the relevane ETL steps before this one.")

        sentiment_df = pd.read_parquet(sentiment_data_fpath)
        topic_relevance_df = pd.read_parquet(topic_relevance_data_fpath)
        metrics_df = pd.read_parquet(metrics_data_fpath)

        return {
            "sentiment": sentiment_df,
            "topic_relevance": topic_relevance_df,
            "metrics": metrics_df
        }



    def transform(self, raw_data):
        """Reshape and aggregate the sentiment, topic relevance and metrics data, including calculating trends."""

        # Let's start with the topic relevance data
        df_topic_relevance = raw_data['topic_relevance'].copy()
        df_topic_relevance = df_topic_relevance[df_topic_relevance['bank'] != "All"]
        df_topic_relevance = df_topic_relevance[df_topic_relevance['source_type'] != "all"]
        first_topic_col_idx = df_topic_relevance.columns.get_loc("risk_proportion") + 1
        topic_cols = df_topic_relevance.columns[first_topic_col_idx:-1].tolist()
        top_topics = self.get_top_k_topics(
            df_topic_relevance,
            topic_cols=topic_cols,
            top_k=20,
            last_n_quarters=2  # Consider the last 2 quarters
        )
        df_topic_relevance.rename(columns={'reporting_period': 'quarter', 'risk_category': 'category', 'risk_proportion': 'metric_value'}, inplace=True)
        keep_cols = ['bank', 'quarter', 'category', 'metric_value', 'source_type'] + top_topics
        df_topic_relevance_risk = df_topic_relevance[keep_cols]
        df_topic_relevance_topics = df_topic_relevance[[col for col in keep_cols if col not in ['category', 'metric_value']]].copy()
        df_topic_relevance_risk.drop(columns=top_topics, inplace=True)
        df_topic_relevance_risk['category_type'] = "Risk"
        df_topic_relevance_topics['category_type'] = "Topic"
        # Reshape the topics data
        df_topic_relevance_topics = df_topic_relevance_topics.melt(
            id_vars=['bank', 'quarter', 'source_type', 'category_type'],
            var_name='category',
            value_name='metric_value'
        )
        # Combine the risk and topics data
        df_topic_relevance_combined = pd.concat([df_topic_relevance_risk, df_topic_relevance_topics], ignore_index=True)
        df_topic_relevance_combined['metric_name'] = "Topic Prevalence - " + df_topic_relevance_combined['source_type'].str.capitalize()
        # Reorder the columns
        df_topic_relevance_combined = df_topic_relevance_combined[['bank', 'quarter', 'category_type', 'category', 'metric_name', 'metric_value', 'source_type']]
        df_topic_relevance_combined.drop(columns=['source_type'], inplace=True)
        # Add rank column - for topic prevalence we want higher values to have lower ranks
        #  TODO: NaNs should be handled upstream
        df_topic_relevance_combined['metric_value'] = df_topic_relevance_combined['metric_value'].fillna(0)  # Fill NaN values with 0 for ranking
        df_topic_relevance_combined['rank'] = df_topic_relevance_combined.groupby(['quarter', 'category', 'metric_name'])['metric_value'].rank(method='min', ascending=False).astype(int)


        # Now let's look at the sentiment data
        df_sentiment = raw_data['sentiment'].copy()
        df_sentiment = df_sentiment[df_sentiment['bank'] != "All"]
        df_sentiment = df_sentiment[df_sentiment['source_type'] != "all"]
        df_sentiment.rename(columns={'reporting_period': 'quarter', 'risk_category': 'category', 'sentiment_score': 'metric_value'}, inplace=True)
        keep_cols = ['bank', 'quarter', 'category', 'metric_value', 'source_type'] + top_topics
        df_sentiment_risk = df_sentiment[keep_cols]
        df_sentiment_topics = df_sentiment[[col for col in keep_cols if col not in ['category', 'metric_value']]].copy()
        df_sentiment_risk.drop(columns=top_topics, inplace=True)
        df_sentiment_risk['category_type'] = "Risk"
        df_sentiment_topics['category_type'] = "Topic"
        # Reshape the topics data
        df_sentiment_topics = df_sentiment_topics.melt(
            id_vars=['bank', 'quarter', 'source_type', 'category_type'],
            var_name='category',
            value_name='metric_value'
        )
        # Combine the risk and topics data
        df_sentiment_combined = pd.concat([df_sentiment_risk, df_sentiment_topics], ignore_index=True)
        df_sentiment_combined['metric_name'] = "Sentiment - " + df_sentiment_combined['source_type'].str.capitalize()
        # Reorder the columns
        df_sentiment_combined = df_sentiment_combined[['bank', 'quarter', 'category_type', 'category', 'metric_name', 'metric_value', 'source_type']]
        df_sentiment_combined.drop(columns=['source_type'], inplace=True)
        # Add rank column - for sentiment we want higher values to have lower ranks
        # TODO: check zero sentiment values and potential for NaNs upstream
        df_sentiment_combined['rank'] = df_sentiment_combined.groupby(['quarter', 'category', 'metric_name'])['metric_value'].rank(method='min', ascending=False).astype(int)

        # Now let's look at the financial metrics data
        df_metrics = raw_data['metrics'].copy()
        df_metrics['category_type'] = "Risk"
        df_metrics.rename(columns={'reporting_period': 'quarter'}, inplace=True)
        # Get the clean names and categories for the metrics
        clean_name_mapping = {k: v[0] for k, v in SQLDatabaseETL.metric_name_mapping.items()}
        risk_category_mapping = {k: v[1] for k, v in SQLDatabaseETL.metric_name_mapping.items()}
        rank_direction_mapping = {k: v[2] for k, v in SQLDatabaseETL.metric_name_mapping.items()}
        # Reshape the metrics data
        df_metrics = df_metrics.melt(
            id_vars=['bank', 'quarter', 'category_type'],
            var_name='metric_name',
            value_name='metric_value'
        )
        # Use the mapping to get the risk categories
        df_metrics['category'] = df_metrics['metric_name'].map(risk_category_mapping)
        # Reorder the columns
        df_metrics = df_metrics[['bank', 'quarter', 'category_type', 'category', 'metric_name', 'metric_value']]
        # Add rank column - for financial metrics we want higher values to have lower ranks if rank_direction is True, otherwise higher values have higher ranks
        # First split the dataframe into two based on the rank direction
        metrics_higher_better = [k for k in rank_direction_mapping.keys() if rank_direction_mapping[k] is False]
        metrics_lower_better = [k for k in rank_direction_mapping.keys() if k not in metrics_higher_better]
        df_metrics_higher_better = df_metrics[df_metrics['metric_name'].isin(metrics_higher_better)].copy()
        df_metrics_lower_better = df_metrics[df_metrics['metric_name'].isin(metrics_lower_better)].copy()
        df_metrics_higher_better['rank'] = df_metrics_higher_better.groupby(['quarter', 'category', 'metric_name'])['metric_value'].rank(method='min', ascending=False).astype(int)
        df_metrics_lower_better['rank'] = df_metrics_lower_better.groupby(['quarter', 'category', 'metric_name'])['metric_value'].rank(method='min', ascending=True).astype(int)
        # Concatenate the two dataframes back together
        df_metrics = pd.concat([df_metrics_higher_better, df_metrics_lower_better], ignore_index=True)
        # Sort the dataframe by bank, quarter and metric_name
        df_metrics.sort_values(by=['bank', 'quarter', 'metric_name'], inplace=True)

        # Concatenate the topic relevance, sentiment and metrics dataframes
        df_raw_metrics = pd.concat([df_topic_relevance_combined, df_sentiment_combined, df_metrics], ignore_index=True)

        # Now let's calculate the trends
        # Add Sentiment - Internal, Sentiment - External, Topic Prevalence - Internal and Topic Prevalence - External to rank_direction_mapping and metrics_higher_better lists
        rank_direction_mapping.update({
            "Sentiment - Internal": False,
            "Sentiment - External": False,
            "Topic Prevalence - Internal": False,
            "Topic Prevalence - External": False
        })
        metrics_higher_better.extend([
            "Sentiment - Internal",
            "Sentiment - External",
            "Topic Prevalence - Internal",
            "Topic Prevalence - External"
        ])
        df_metric_trends = df_raw_metrics.groupby(['bank', 'category', 'metric_name']).apply(self.calculate_trends, rank_direction_mapping, SQLDatabaseETL.trend_periods_mapping).reset_index(drop=True)

        # Add ranks for the trends
        # First split the dataframe into two based on the rank direction
        df_increasing_trend_better = df_metric_trends[df_metric_trends['metric_name'].isin(metrics_higher_better)].copy()
        df_decreasing_trend_better = df_metric_trends[df_metric_trends['metric_name'].isin(metrics_lower_better)].copy()
        df_increasing_trend_better['slope_rank'] = df_increasing_trend_better.groupby(['category', 'metric_name', 'trend_horizon'])['slope'].rank(method='min', ascending=False).astype(int)
        df_decreasing_trend_better['slope_rank'] = df_decreasing_trend_better.groupby(['category', 'metric_name', 'trend_horizon'])['slope'].rank(method='min', ascending=True).astype(int)
        # Concatenate the two dataframes back together
        df_metric_trends = pd.concat([df_increasing_trend_better, df_decreasing_trend_better], ignore_index=True)
        # Add volatility rank - for volatility we want lower values to have lower ranks
        df_metric_trends['volatility_rank'] = df_metric_trends.groupby(['category', 'metric_name', 'trend_horizon'])['volatility'].rank(method='min', ascending=True).astype(int)
        # Sort the trends dataframe by bank, metric_name and trend_horizon
        df_metric_trends.sort_values(by=['bank', 'category', 'metric_name', 'trend_horizon'], inplace=True)

        # Finally map the metric names to their clean names
        df_raw_metrics['metric_name'] = df_raw_metrics['metric_name'].replace(clean_name_mapping)
        df_metric_trends['metric_name'] = df_metric_trends['metric_name'].replace(clean_name_mapping)

        return {
            "raw_metrics": df_raw_metrics,
            "metric_trends": df_metric_trends,
        }

    def load(self, transformed_data):
        """
        Loads the transformed data into the SQL database.
        """

        if not isinstance(transformed_data, dict):
            raise TypeError("Transformed data must be a dictionary with keys 'raw_metrics' and 'metric_trends'.")
        if 'raw_metrics' not in transformed_data or 'metric_trends' not in transformed_data:
            raise ValueError("Transformed data must contain 'raw_metrics' and 'metric_trends' keys.")

        raw_metrics_df = transformed_data['raw_metrics']
        metric_trends_df = transformed_data['metric_trends']

        if not isinstance(raw_metrics_df, pd.DataFrame) or not isinstance(metric_trends_df, pd.DataFrame):
            raise TypeError("Transformed data must contain Pandas DataFrames for 'raw_metrics' and 'metric_trends'.")

        if raw_metrics_df.empty or metric_trends_df.empty:
            raise ValueError("Transformed data cannot be empty.")

        # Insert the raw metrics data into the raw_metrics table
        self.upsert_data(raw_metrics_df, "raw_metrics", match_columns=['bank', 'quarter', 'category', 'metric_name'])
        # Insert the metric trends data into the metric_trends table
        self.upsert_data(metric_trends_df, "metric_trends", match_columns=['bank', 'category', 'metric_name', 'trend_horizon'])



    def create_table(self, table_name, columns, unique_columns=None):
        """
        Creates a table in the SQL database with the specified name and columns.
        """
        if not isinstance(table_name, str):
            raise TypeError("Table name must be a string.")
        if not isinstance(columns, dict):
            raise TypeError("Columns must be a dictionary with column names as keys and data types as values.")
        if not columns:
            raise ValueError("Columns dictionary cannot be empty.")
        unique_clause = ''
        if unique_columns:
            unique_cols_str = ', '.join(unique_columns)
            unique_clause = f", UNIQUE({unique_cols_str})"

        with sqlite3.connect(self.sql_db_fpath) as conn:
            cursor = conn.cursor()
            columns_str = ', '.join([f"{col} {dtype}" for col, dtype in columns.items()])
            cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_str}{unique_clause})")
            conn.commit()

    def upsert_data(self, df, table_name, match_columns):
        """
        Upserts data into the specified table in the SQL database.
        The kwargs should match the columns of the table.
        """
        if not isinstance(table_name, str):
            raise TypeError("Table name must be a string.")
        if df.empty:
            raise ValueError("No data provided to upsert.")

        match_columns = list(match_columns)
        insert_columns = df.columns.tolist()
        if not set(match_columns).issubset(set(insert_columns)):
            raise ValueError(f"Match columns {match_columns} must be a subset of the table columns {insert_columns}.")

        if not len(insert_columns) > len(match_columns):
            raise ValueError("At least one column must be specified for updating, in addition to the match columns.")

        match_columns_str = ', '.join(match_columns)
        insert_columns_str = ', '.join(insert_columns)
        placeholders = ', '.join(['?'] * len(insert_columns))
        update_columns = [col for col in insert_columns if col not in match_columns]
        data_values = df[insert_columns].values.tolist()

        query_template = f"""
            INSERT INTO {table_name} ({insert_columns_str})
            VALUES ({placeholders})
            ON CONFLICT({match_columns_str}) DO UPDATE SET
                {', '.join([f"{col} = excluded.{col}" for col in update_columns])}
        """
        try:
            with sqlite3.connect(self.sql_db_fpath) as conn:
                cursor = conn.cursor()
                # executemany executes the same query template for each tuple in data_tuples
                cursor.executemany(query_template, data_values)
                conn.commit()
            print(f"Data upserted successfully into {table_name}.")
        except sqlite3.Error as e:
            raise RuntimeError(f"An error occurred while upserting data into {table_name}: {e}")



    @staticmethod
    def get_top_k_topics(df, topic_cols, top_k, last_n_quarters=4):
        """
        Returns the top k topics considering all banks in the last n quarters and the risk category they predominately belong to over the same period.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Data must be a Pandas DataFrame.")
        if not isinstance(topic_cols, list):
            raise TypeError("Topic columns must be a list of column names.")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("Top k must be a positive integer.")
        if not isinstance(last_n_quarters, int) or last_n_quarters <= 0:
            raise ValueError("Last n quarters must be a positive integer.")

        # Get the last n quarters from the reporting_period column
        quarters = df['reporting_period'].drop_duplicates().sort_values(ascending=False).head(last_n_quarters).tolist()
        dff = df[df['reporting_period'].isin(quarters)].copy()

        # We'll get the top topics based on the mean relevance score across all banks
        top_topics = dff[topic_cols].mean().sort_values(ascending=False).head(top_k). index.tolist()

        return top_topics

    @staticmethod
    def calculate_trends(df, rank_direction_mapping, trend_periods_dict):
        # Ensure sorted by quarter with most recent quarter last
        df = df.sort_values(by='quarter')
        # Initialize a dictionary to store results
        results = defaultdict(list)
        for trend_horizon, n_quarters in trend_periods_dict.items():
            results["bank"].append(df['bank'].iloc[0])
            results["category_type"].append(df['category_type'].iloc[0])
            results["category"].append(df['category'].iloc[0])
            metric_name = df['metric_name'].iloc[0]
            results["metric_name"].append(metric_name)
            results["trend_horizon"].append(trend_horizon)
            data = df['metric_value'].tail(n_quarters)
            x = np.arange(len(data))
            # Fit a linear regression model to the last n_quarters of data
            if len(data) < 2:
                slope = 0
            else:
                slope = np.polyfit(x, data, 1)[0]
            results["slope"].append(slope)
            # Get the status based on the slope, the starting and the rank direction
            starting_value = data.iloc[0]
            rank_direction = rank_direction_mapping[metric_name]
            if abs(slope/starting_value) < 0.01:
                status = "stable"
            # If rank direction is True, decreasing trend is better, similarly if rank direction is False, increasing trend is better
            elif (slope > 0 and rank_direction is False) or (slope < 0 and rank_direction is True):
                status = "improving"
            else:
                status = "deteriorating"
            results["status"].append(status)
            if len(data) < 2:
                volatility = 0
            else:
                volatility = data.std()
            results["volatility"].append(volatility)

        df_trends = pd.DataFrame(results)

        return df_trends


if __name__ == "__main__":
    # # Instantiate the TranscriptETL class
    # input_pdf_path = os.path.join("data", "bankofamerica", "raw_docs", "transcripts", "Q2_2023.pdf")

    # transcript_etl = TranscriptETL(
    # 	input_pdf_path=input_pdf_path,
    # 	is_q4_transcript=False,
    # )

    # # Run the extract method
    # extracted_text = transcript_etl.extract()

    # # Run the transform method
    # chunks_df = transcript_etl.transform(
    #     raw_data=extracted_text,
    #     llm_backend="gemini",
    #     llm_model_name="gemini-2.5-pro-preview-06-05",
    #     # llm_model_name="gemini-2.5-flash-preview-05-20",
    #     )

    # # Run the load method
    # output_dir_path = os.path.join("data", "bankofamerica", "processed", "transcripts")
    # transcript_etl.load(
    #     transformed_data=chunks_df,
    #     output_dir_path=output_dir_path,
    #     )

    # # Instantiate the PresentationETL class
    # input_pdf_path = os.path.join("data", "bankofamerica", "raw_docs", "presentations", "Q1_2023_presentation.pdf")
    # presentation_etl = PresentationETL(
    # 	input_pdf_path=input_pdf_path,
    # 	is_q4_presentation=False,
    # )

    # # Run the transform method (no need to run extract method for presentations)
    # analysis_results_dict = presentation_etl.transform(
    # 	llm_backend="gemini",
    # 	llm_model_name="gemini-2.5-pro-preview-06-05",
    # )

    # # Run the load method
    # output_dir_path = os.path.join("data", "bankofamerica", "processed", "presentations")
    # presentation_etl.load(
    # 	transformed_data=analysis_results_dict,
    # 	output_dir_path=output_dir_path,
    # )

    # # Instantiate the NLPInputDataAggregationETL class
    # data_aggregation_etl = NLPInputDataAggregationETL(
    # 	data_dir_path=DATA_FOLDER_PATH,
    #     news_dir_path=NEWS_DATA_FOLDER_PATH,
    # )

    # # Extract data
    # all_files = data_aggregation_etl.extract()

    # # Transform data
    # aggregated_data_dict = data_aggregation_etl.transform(raw_data=all_files)

    # # Load data
    # output_dir_path = AGGREGATED_DATA_FOLDER_PATH
    # data_aggregation_etl.load(transformed_data=aggregated_data_dict, output_dir_path=output_dir_path)

    # # Instantiate the SharePriceDataETL class
    # bank_name = "bankofamerica"
    # ticker = TICKER_MAPPING[bank_name]
    # today = datetime.datetime.now().date().strftime("%Y-%m-%d")
    # share_price_etl = SharePriceDataETL(
    #     ticker=ticker,
    #     bank_name=bank_name,
    #     start_date=SHARE_PRICE_HISTORY_START_DATE,
    #     end_date=today,
    # )

    # # Run the convenience method to extract, transform, and load data
    # output_fpath = os.path.join(DATA_FOLDER_PATH, bank_name, "processed", "share_price_history", "share_price_history.csv")
    # share_price_etl.export_to_csv(fpath=output_fpath)

    # # Instantiate the VectorDBETL class
    # input_parquet_path = os.path.join(AGGREGATED_DATA_FOLDER_PATH, "all_text.parquet")
    # vector_db_etl = VectorDBETL(
    #     input_parquet_path=input_parquet_path,
    #     vector_db_provider="pinecone",
    #     index_name="boe-text-embeddings"
    # )
    # # Extract data
    # raw_data = vector_db_etl.extract()

    # # Transform data
    # raw_data['date_of_earnings_call'] = pd.to_datetime(raw_data['date_of_earnings_call']).dt.strftime('%Y-%m-%d')
    # raw_data[['speaker','role']] = raw_data[['speaker','role']].fillna('')  # Fill NaN values in speaker and role columns with empty strings
    # col_to_embed_name = "text"
    # embedding_model_name = "mukaj/fin-mpnet-base"
    # embedding_model = SentenceTransformer(embedding_model_name)
    # vectors = vector_db_etl.transform(
    #     raw_data=raw_data,
    #     col_to_embed_name=col_to_embed_name,
    #     embedding_model=embedding_model
    # )
    # # Load data
    # vector_db_etl.load(vectors)

    # start = time.time()
    # # Instantiate the FinancialNewsETL class
    # # fname = "2025_01_15_blog_US bank earnings as it happened_ Shares jump as investors cheer bumper results.pdf"
    # fname="2025_05_20_JPMorgan London trader unfairly dismissed despite spoofing.pdf"
    # input_pdf_path = os.path.join(NEWS_DATA_FOLDER_PATH, "raw_docs", "jpmorgan_news", fname)

    # financial_news_etl = FinancialNewsETL(
    #     input_pdf_path=input_pdf_path,
    # )
    # # Run the transform method
    # analysis_results_dict = financial_news_etl.transform(
    #     llm_backend="gemini",
    #     llm_model_name="gemini-2.5-pro-preview-06-05",
    # )

    # # Run the load method
    # output_dir_path = os.path.join(NEWS_DATA_FOLDER_PATH, "processed")
    # financial_news_etl.load(
    #     transformed_data=analysis_results_dict,
    #     output_dir_path=output_dir_path,
    # )
    # elapsed_time = time.time() - start
    # print(f"Elapsed time for FinancialNewsETL: {elapsed_time:.2f} seconds")

    # # Instantiate the FinancialMetricsAggregationETL class
    # data_dir_path = DATA_FOLDER_PATH
    # transcripts_parquet_path = os.path.join(AGGREGATED_DATA_FOLDER_PATH, "transcripts.parquet")
    # financial_metrics_etl = FinancialMetricsAggregationETL(
    #     data_dir_path=data_dir_path,
    #     transcripts_parquet_path=transcripts_parquet_path,
    # )
    # # Extract data
    # raw_data = financial_metrics_etl.extract()

    # # Transform data
    # transformed_data = financial_metrics_etl.transform(raw_data=raw_data)

    # # Load data
    # output_dir_path = APP_DATA_FOLDER_PATH
    # financial_metrics_etl.load(transformed_data=transformed_data, output_dir_path=output_dir_path)

    # Instantiate the SQLDatabaseETL class
    sql_db_fpath = os.path.join(APP_DATA_FOLDER_PATH, "metrics.db")
    sql_db_etl = SQLDatabaseETL(sql_db_fpath)

    # Extract data
    sentiment_data_fpath = os.path.join(APP_DATA_FOLDER_PATH, "multi_topic_modelling_with_relevance_sentiment_quarter_agg_norm.parquet")
    topic_relevance_data_fpath = os.path.join(APP_DATA_FOLDER_PATH, "multi_topic_modelling_with_relevance_quarter_agg_norm.parquet")
    metrics_data_fpath = os.path.join(APP_DATA_FOLDER_PATH, "financial_metrics.parquet")
    raw_data = sql_db_etl.extract(
        sentiment_data_fpath=sentiment_data_fpath,
        topic_relevance_data_fpath=topic_relevance_data_fpath,
        metrics_data_fpath=metrics_data_fpath
    )

    # Transform data
    transformed_data = sql_db_etl.transform(raw_data=raw_data)

    # Load data
    sql_db_etl.load(transformed_data=transformed_data)




