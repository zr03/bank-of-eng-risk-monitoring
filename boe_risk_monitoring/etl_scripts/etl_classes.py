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
from tqdm import tqdm

from pydantic import BaseModel, Field
import pymupdf
import pandas as pd
import yfinance as yf
from pinecone import Pinecone, ServerlessSpec
# import torch
from sentence_transformers import SentenceTransformer


from boe_risk_monitoring.llms.processing_llms import ChunkingLLM
from boe_risk_monitoring.llms.document_analyser_llm import DocumentAnalyserLLM
import boe_risk_monitoring.config as config

DATA_FOLDER = config.DATA_FOLDER
AGGREGATED_DATA_FOLDER_NAME = config.AGGREGATED_DATA_FOLDER_NAME
SHARE_PRICE_HISTORY_START_DATE = config.SHARE_PRICE_HISTORY_START_DATE
PERMISSIBLE_BANK_NAMES = config.PERMISSIBLE_BANK_NAMES
BANK_NAME_MAPPING = config.BANK_NAME_MAPPING
TICKER_MAPPING = config.TICKER_MAPPING
PERMISSIBLE_VECTOR_DB_PROVIDERS = config.PERMISSIBLE_VECTOR_DB_PROVIDERS
PERMISSIBLE_DOC_TYPES = ['transcripts', 'presentations_text', 'presentations_graphs', 'presentations_tables']

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
        bank_dirs = [bank_dir for bank_dir in self.data_dir_path.iterdir() if bank_dir.is_dir() and bank_dir.name != AGGREGATED_DATA_FOLDER_NAME]
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

        for bank in raw_data.keys():
            for doc_type in raw_data[bank].keys():
                if doc_type not in PERMISSIBLE_DOC_TYPES:
                    raise ValueError(f"Unsupported document type: {doc_type}. Supported types are {PERMISSIBLE_DOC_TYPES}.")
                if not isinstance(raw_data[bank][doc_type], list):
                    raise TypeError(f"Data for {bank} - {doc_type} must be a list of dataframes.")
                # Concatenate all dataframes for the document type
                df = pd.concat(raw_data[bank][doc_type], ignore_index=True)
                if doc_type == 'transcripts':
                    transcripts_df = pd.concat([transcripts_df, df], ignore_index=True)
                    transcripts_df['source'] = transcripts_df['speaker'] + " (" + transcripts_df['role'] + ")\n" + transcripts_df['bank'] + ", " + transcripts_df['reporting_period'].str.replace("_", ", ") + " Earnings Call Transcript, Page " + transcripts_df['page'].astype(str)
                elif doc_type == 'presentations_text':
                    presentation_text_df = pd.concat([presentation_text_df, df], ignore_index=True)
                    presentation_text_df['source'] = "Slide Text\n" + presentation_text_df['bank'] + ", "+ presentation_text_df['reporting_period'].str.replace("_", ", ") + ", Earnings Call Presentation, Slide " +  transcripts_df['page'].astype(str)
                elif doc_type == 'presentations_graphs':
                    presentation_graphs_df = pd.concat([presentation_graphs_df, df], ignore_index=True)
                    presentation_graphs_df['source'] = "Graph (" + presentation_graphs_df['caption'] + ")\n" + presentation_graphs_df['bank'] + ", " + presentation_graphs_df['reporting_period'].str.replace("_", ", ") + ", Earnings Call Presentation, Slide " + presentation_graphs_df['page'].astype(str)
                elif doc_type == 'presentations_tables':
                    presentation_tables_df = pd.concat([presentation_tables_df, df], ignore_index=True)
                    presentation_tables_df['source'] = "Table Row (" + presentation_tables_df['row_heading'] + ")\n" + presentation_tables_df['bank'] + ", " + presentation_tables_df['reporting_period'].str.replace("_", ", ") + ", Earnings Call Presentation, Slide " + presentation_tables_df['page'].astype(str)

        # Convert the date columns to string format
        transcripts_df['date_of_call'] = pd.to_datetime(transcripts_df['date_of_call']).dt.strftime('%Y-%m-%d')
        presentation_text_df['date_of_presentation'] = pd.to_datetime(presentation_text_df['date_of_presentation']).dt.strftime('%Y-%m-%d')
        presentation_graphs_df['date_of_presentation'] = pd.to_datetime(presentation_graphs_df['date_of_presentation']).dt.strftime('%Y-%m-%d')
        presentation_tables_df['date_of_presentation'] = pd.to_datetime(presentation_tables_df['date_of_presentation']).dt.strftime('%Y-%m-%d')

        # Replace null values in speaker and role columns with empty string
        fillna_cols = ['speaker', 'role']
        presentation_text_df[fillna_cols] = presentation_text_df[fillna_cols].fillna('')
        presentation_graphs_df[fillna_cols] = presentation_graphs_df[fillna_cols].fillna('')
        presentation_tables_df[fillna_cols] = presentation_tables_df[fillna_cols].fillna('')

        # Now we'll create a specific dataframe which summarise all text components of the various documents

        # Transcripts
        transcripts_df2 = transcripts_df.copy()
        transcripts_df2 = transcripts_df2.rename(columns={'date_of_call': 'date_of_earnings_call'})

        # Presentation text
        presentation_text_df2 = presentation_text_df.copy()
        presentation_text_df2 = presentation_text_df2.rename(columns={'date_of_presentation': 'date_of_earnings_call'})

        # Presentation graphs
        presentation_graphs_df2 = presentation_graphs_df.copy()
        presentation_graphs_df2['trend_summary'] = presentation_graphs_df2['caption'] + ": " + presentation_graphs_df2['trend_summary']
        presentation_graphs_df2.drop(columns=['caption'], inplace=True)
        presentation_graphs_df2 = presentation_graphs_df2.rename(columns={
            'date_of_presentation': 'date_of_earnings_call',
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

        # ['row_headings', 'page', 'section', 'reporting_period', 'date_of_presentation', 'bank', 'document_type', 'source']
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
        presentation_tables_df2 = presentation_tables_df2.rename(columns={'date_of_presentation': 'date_of_earnings_call'})


        # Collect dfs to concatenate
        dfs_to_concat = [transcripts_df2, presentation_text_df2, presentation_graphs_df2, presentation_tables_df2]
        all_text_df = pd.concat(dfs_to_concat, ignore_index=True)

        all_text_df['fiscal_period_ref'] = all_text_df['fiscal_period_ref'].astype('string')
        all_text_df['date_of_earnings_call'] = pd.to_datetime(all_text_df['date_of_earnings_call']).dt.date

        results_dict = {
            'transcripts': transcripts_df,
            'presentation_text': presentation_text_df,
            'presentation_graphs': presentation_graphs_df,
            'presentation_tables': presentation_tables_df,
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


class SupplementaryDataETL(BaseETL):
    def __init__(self, *args, **kwargs):

        filenames_dict = {'jpmorgan': ["1Q21_Earnings_Supplement.xlsx","2Q21_Earnings_Supplement.xlsx",
                                       "3Q21_Earnings_Supplement.xlsx","4Q21_Earnings_Supplement.xlsx",
                                       "1q22-earnings-supplement.xlsx","2Q22-Earnings-Supplement.xlsx",
                                       "3Q22_Earnings_Supplement.xlsx","4q22-earnings-supplement-xls.xlsx",
                                       "1q23-earnings-supplement.xlsx","2q23-earnings-supplement.xlsx",
                                       "3q23-earnings-supplement.xlsx","4q23-earnings-supplement.xlsx",
                                       "1Q24_Earnings_Supplement.xlsx","2q24-earnings-supplement.xlsx",
                                       "3q24-earnings-supplement.xlsx","4q24-earnings-supplement.xlsx",
                                       "1q25-earnings-supplement.xlsx" ],
                          'bankofamerica': ["1Q21_Financial_Report.xlsx","2Q21_Financial_Report.xlsx",
                                            "3Q21_Financial_Report.xlsx","4Q21_Financial_Report.xlsx",
                                            "1Q22_Financial_Report.xlsx","2Q22_Financial_Report.xlsx",
                                            "3Q22_Financial_Report.xlsx","4Q22_Financial_Report.xlsx",
                                            "1Q23_Financial_Report.xlsx","2Q23_Financial_Report.xlsx",
                                            "3Q23_Financial_Report.xlsx","4Q23_Financial_Report.xlsx",
                                            "1Q24_Financial_Report.xlsx","2Q24_Financial_Report.xlsx",
                                            "3Q24_Financial_Report.xlsx","4Q24_Financial_Report.xlsx",
                                            "1Q25_Financial_Report.xlsx"],
                          'citigroup': ["2025fqtr1hstr.xlsx"]}

        self.bankofamerica_supp = 'bankofamerica_more_metrics.xlsx'
        
        fpaths_dict = defaultdict(list)
        for bank, fname_list in filenames_dict.items():
            for fname in fname_list:
                fpath = os.path.join(DATA_FOLDER, bank, "raw_docs", "supplementary_data", fname)
                if os.path.exists(fpath):
                    fpaths_dict[bank].append(fpath)
                else:
                    raise FileNotFoundError(f"File not found: {fpath}")

        self.fpaths_dict = fpaths_dict

        self.sheet_name = { 'jpmorgan': ['Page 2','Page 5'],
                            'bankofamerica': ['Consolidated Statement of Incom',
                                             'Consolidated Balance Sheet'],
                            'citigroup':'Summary'}
        self.bank = kwargs.pop("bank", None)
        self.extract_output = {'jpmorgan': None,
                               'bankofamerica': None,
                               'citigroup': None}
        self.transform_output = {'jpmorgan': None,
                               'bankofamerica': None,
                               'citigroup': None}

        new_column_names = [0]*15
        new_column_names[0] = 'Noninterest Expense (millions of dollars)'
        new_column_names[1] = 'Provision for Credit Losses (millions of dollars)'
        new_column_names[2] = 'Net income (millions of dollars)'
        new_column_names[3] = 'Common Equity Tier 1 (CET1) Capital ratio (%)'
        new_column_names[4] = 'Tier 1 Capital ratio ratio (%)'
        new_column_names[5] = 'Total Capital ratio (%)'
        new_column_names[6] = 'Supplementary Leverage ratio (SLR) (%)'
        new_column_names[7] = 'Return on average assets, ROA (%)'
        new_column_names[8] = 'Return on average common equity (%)'

        new_column_names[9]='Return on average tangible common equity (RoTCE) (%)'
        new_column_names[10]='Efficiency ratio(%)'
        new_column_names[11]='Total loans (billions of dollars)'
        new_column_names[12]='Total deposits (billions of dollars)'
        new_column_names[13]='Book value per share (dollars)'
        new_column_names[14]='Tangible book value per share (dollars)' 

        self.new_column_names = new_column_names                     
        super().__init__(*args, **kwargs)

    def extract(self, *args, **kwargs):
      """
      Extract data from the source.
      """
      if self.bank is None:
        # run all extracts
        self._extract_jpmorgan(*args, **kwargs)
        self._extract_bankofamerica(*args, **kwargs)
        self._extract_citigroup(*args, **kwargs)
      elif self.bank == "jpmorgan":
        self._extract_jpmorgan(*args, **kwargs)
      elif self.bank == "bankofamerica":
        self._extract_bankofamerica(*args, **kwargs)
      elif self.bank == "citigroup":
        self._extract_citigroup(*args, **kwargs)

    def _extract_jpmorgan_single(self, file_list, sheet_name):

      """
      Extract data from the source.
      """

      df_list = []
      quarters=[]
      for item in self.fpaths_dict['jpmorgan']:
        df_page2 = pd.read_excel(item, sheet_name=self.sheet_name['jpmorgan'], header=None, usecols="B:F", skiprows=6)
        header_rows = df_page2.iloc[1:4, 1:5]
        col_header = header_rows.fillna(method='ffill', axis=1).apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
        quarter_label = df_page2.iloc[0, 4]
        final_columns = ['Quarters'] + [f"{quarter_label} - {h}" for h in col_header]
        line_items = df_page2.iloc[2:39, 0:3].apply(lambda row: ' '.join(row.dropna().astype(str)).strip(), axis=1)
        data_values = df_page2.iloc[2:39, 4]
        data_values_cleaned = data_values.replace(r'\((.*?)\)', r'-\1', regex=True).astype(str).str.replace(',', '').str.replace('$', '', regex=False).str.replace('%', '', regex=False).str.strip()
        data_values_numeric = pd.to_numeric(data_values_cleaned, errors='coerce')
        df_list.append(data_values_numeric)
        quarters.append(final_columns[1][:4])
      lista = pd.DataFrame(df_list)
      lista.columns = line_items
      lista.index = quarters
      df_results = lista.dropna(axis=1)
      return df_results

    def _extract_jpmorgan(self, *args, **kwargs):
      """
      Extract data from the source.
      """
      df_results = {}
      for sheet in self.sheet_name['jpmorgan']:
        df_results[sheet] =\
            self._extract_jpmorgan_single(self.fpaths_dict['jpmorgan'], sheet)
        
      self.extract_output['jpmorgan'] = df_results
      

    def _extract_bankofamerica_single(self, file_list, sheet_name):
      df_list = []
      quarters=[]
      for item in file_list:
        df_page2 = pd.read_excel(item, sheet_name=sheet_name, header=None, usecols="A:B", skiprows=2)


        quarter_label = item[0:4]

        line_items = df_page2.iloc[1:, 0]
        data_values = df_page2.iloc[1:, 1]
        data_values_cleaned = data_values.replace(r'\((.*?)\)', r'-\1', regex=True).astype(str).str.replace(',', '').str.replace('$', '', regex=False).str.replace('%', '', regex=False).str.strip()
        data_values_numeric = pd.to_numeric(data_values_cleaned, errors='coerce')
        df_list.append(data_values_numeric)
        quarters.append(quarter_label)

      lista=pd.DataFrame(df_list)
      lista.columns=line_items

      #lista.rename(columns=mapper)
      lista.index=quarters
      df_results=lista.dropna(axis=1)
      df_results.head()
      return df_results
    def _extract_bankofamerica(self, *args, **kwargs):
      """
      Extract data from the source.
      """
      df_results = {}
      for sheet in self.sheet_name['bankofamerica']:
        df_results[sheet] =\
            self._extract_bankofamerica_single(self.fpaths_dict['bankofamerica'], sheet)
        
      self.extract_output['bankofamerica'] = df_results
    def _extract_citigroup(self, *args, **kwargs):
      """
      Extract data from the source.
      """
      file_path = self.fpaths_dict['citigroup'][0]
      df_raw = pd.read_excel(file_path, sheet_name=self.sheet_name['citigroup'], header=None)

      # Combine columns A–D (indices 0–3) for full row labels
      label_col = (
          df_raw[0]
          .combine_first(df_raw[1])
          .combine_first(df_raw[2])
          .combine_first(df_raw[3])
          .iloc[9:66]
          .astype(str)
          .str.strip()
      )

      # Extract time labels from rows 7 and 8 (index 6 and 7)
      data_cols = list(range(3, df_raw.shape[1], 2))
      quarters = df_raw.iloc[6, data_cols].astype(str).str.strip()
      years = df_raw.iloc[7, data_cols].astype(str).str.strip()
      time_labels = [f"{q} {y}" for q, y in zip(quarters, years)]
      time_labels2=[]
      for i in range(len(time_labels)):
          time_labels2.append(time_labels[i][0:2]+time_labels[i][-2:])

      # Extract data block and label it
      data_block = df_raw.iloc[9:66, data_cols]
      data_block.index = label_col
      data_block.columns = time_labels
      df = data_block.T

      # Clean number formatting
      def clean_number(x):
          if isinstance(x, str):
              x = x.replace(',', '').strip()
              if x.startswith('(') and x.endswith(')'):
                  x = '-' + x.strip('()')
          return pd.to_numeric(x, errors='coerce')

      df = df.applymap(clean_number)

      # Drop fully empty rows/columns
      df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
      df=df.drop(index=['Ye21','Ye22','Ye23','Ye24'])

      self.extract_output['citigroup'] = df



    def transform(self, *args, **kwargs):
      """
      transform data from the extracted data.
      """
      if self.bank is None:
        # transform all
        self._transform_jpmorgan(*args, **kwargs)
        self._transform_bankofamerica(*args, **kwargs)
        self._transform_citigroup(*args, **kwargs)
      elif self.bank == "jpmorgan":
        return self._transform_jpmorgan(*args, **kwargs)
      elif self.bank == "bankofamerica":
        return self._transform_bankofamerica(*args, **kwargs)
      elif self.bank == "citigroup":
        return self._transform_citigroup(*args, **kwargs)

    def _transform_jpmorgan(self, *args, **kwargs):
      """
      Transform the extracted data.
      """
      df_results = self.extract_output['jpmorgan']['Page 2']
      df_results5 = self.extract_output['jpmorgan']['Page 5']
      efficiency_ratio = df_results.iloc[:,1]/df_results.iloc[:,0]
      df2 = pd.DataFrame()
      df2 = df_results.iloc[:,[1,3,4,22,23,24,26,21,19,20]]
      df2['Efficiency ratio'] = efficiency_ratio
      df2['Loan'] = df_results5.iloc[:, 9].values.flatten() / 1000
      df2['deposit'] = df_results5.iloc[:, 17].values.flatten() / 1000
      df2['Book value per share'] = df_results.iloc[:, 16].values.flatten()
      df2['Tangible book value per share'] = df_results.iloc[:, 17].values.flatten()
      df2.columns = self.new_column_names
      #df_results.head()
      self.transform_output['jpmorgan']=df2

    def _transform_bankofamerica(self, *args, **kwargs):
      """
      Transform the extracted data.
      """
      df_more_metrics = pd.read_excel(self.bankofamerica_supp)
      df_more_metrics.set_index('Quarters',inplace=True)
      df_results = self.extract_output['bankofamerica']['Consolidated Statement of Incom']
                                             
      df_results_balance = self.extract_output['bankofamerica']['Consolidated Balance Sheet']
      df2 = pd.DataFrame()
      df2 = df_results.iloc[:,[16,8,19]].join(df_more_metrics.iloc[:,[0,1,2,3,4,5,6,7]]).join(df_results_balance.iloc[:,[9,22]]/1000).join(df_more_metrics.iloc[:,[8,9]])
      
      df2.columns = self.new_column_names
      self.transform_output['bankofamerica'] = df2
              

    def _transform_citigroup(self, *args, **kwargs):
      """
      Transform the extracted data.
      """
      self.transform_output['citigroup'] = self.extract_output['citigroup'].iloc[:,[1,6,13,24,25,26,27,28,29,31,33,36,37,39, 40 ]]
      df2 = self.transform_output['citigroup']
      df2.columns = self.new_column_names

    def load(self, *args, **kwargs):
      """
      Save transformed data.
      """
      if self.bank is None:
        # save all
        self._load('jpmorgan')
        self._load('bankofamerica')
        self._load('citigroup')
      elif self.bank == "jpmorgan":
        self._load('jpmorgan')
      elif self.bank == "bankofamerica":
        self._load('bankofamerica')
      elif self.bank == "citigroup":
        self._load('citigroup')

    def _load(self, bank):
      df_results = self.transform_output[bank]

      output_dir_path = os.path.join(DATA_FOLDER, bank, "processed", "supplementary_data")
      os.makedirs(output_dir_path, exist_ok=True)

      df_results.to_csv(os.path.join(output_dir_path, f"{bank}_metrics.csv"), index=True)
    #   df_results.to_parquet(os.path.join(output_dir_path, f"{bank}_metrics.parquet"))



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
    """This class provides methods to extract data from NLP outputs and store embeddings in a vector DB along with associated metadata.
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










if __name__ == "__main__":
    # # Instantiate the TranscriptETL class
    # input_pdf_path = os.path.join("data", "jpmorgan", "raw_docs", "transcripts", "Q3_2024.pdf")

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
    # output_dir_path = os.path.join("data", "jpmorgan", "processed", "transcripts")
    # transcript_etl.load(
    #     transformed_data=chunks_df,
    #     output_dir_path=output_dir_path,
    #     )

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

    # # Instantiate the DataAggregationETL class
    # data_aggregation_etl = DataAggregationETL(
    # 	data_dir_path=DATA_FOLDER,
    # )

    # # Extract data
    # all_files = data_aggregation_etl.extract()

    # # Transform data
    # aggregated_data_dict = data_aggregation_etl.transform(raw_data=all_files)

    # # Load data
    # output_dir_path = os.path.join(DATA_FOLDER, "aggregated")
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
    # output_fpath = os.path.join(DATA_FOLDER, bank_name, "processed", "share_price_history", "share_price_history.csv")
    # share_price_etl.export_to_csv(fpath=output_fpath)

    # # Instantiate the VectorDBETL class
    # input_parquet_path = os.path.join(DATA_FOLDER, "aggregated", "all_text.parquet")
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

    # Instantiate the SupplementaryDataETL class
    supplementary_data_etl = SupplementaryDataETL()
    # Extract data for all banks
    supplementary_data_etl.extract()
    # Transform data for all banks
    supplementary_data_etl.transform()
    # Load data for all banks
    supplementary_data_etl.load()

