"""
This module contains different ETL classes for extracting data from financial documents published by banks.
"""

from abc import ABC, abstractmethod
import os
import ast
from pathlib import Path
from typing import List, Literal
from pydantic import BaseModel
import pymupdf

import pandas as pd
# from langchain_core.messages.utils import count_tokens_approximately
# import tiktoken

from boe_risk_monitoring.llms.chunking_llm import ChunkingLLM


class TranscriptChunk(BaseModel):
    text: str
    speaker: str
    role: Literal['CEO', 'CFO', 'Host', 'analyst', 'other']
    page: int
    section: Literal['Introduction', 'Disclaimer', 'Prepared remarks', 'Q and A', 'Conclusion']
    # quarter: str
    # document: str


class ChunkedTranscript(BaseModel):
    doc: List[TranscriptChunk]

# class PreChunkMetadata(BaseModel):
#     section_summary: str,
#     speaker:

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


class TranscriptsETL(BaseETL):
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
        self.output_csv_path = None



    def extract(self):
        doc = pymupdf.open(self.input_pdf_path)
        full_text = ""
        for i, page in enumerate(doc):
            text = page.get_text(option="text")
            # Insert page metadata
            text = f"=== START OF PAGE {i+1} ===\n" + text + "\n"
            full_text += text
        return full_text

    def transform(self, raw_data, response_schema, llm_backend="openai", llm_model_name="gpt-4o", temperature=0.3):
        # Instantiate the LLM
        self.llm_model = ChunkingLLM(
            response_schema=response_schema,
            backend=llm_backend,
            model_name=llm_model_name,
            temperature=temperature,
        )

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

        return chunks_df

    def load(self, transformed_data, output_csv_path):
        if not isinstance(transformed_data, pd.DataFrame):
            raise TypeError("Data must be pandas dataframe")

        if not isinstance(output_csv_path, str):
            raise ValueError("Output path must be a string.")

        if not output_csv_path.endswith('.csv'):
            raise ValueError("Output path must be a CSV file.")

        output_csv_path = Path(output_csv_path)
        output_csv_path_dir = output_csv_path.parents[0]

        if not output_csv_path_dir.exists():
            raise FileNotFoundError(f"Output directory does not exist: {output_csv_path_dir}")

        self.output_csv_path = Path(output_csv_path)

        # Save the data to csv
        transformed_data.to_csv(self.output_csv_path, index=False)


if __name__ == "__main__":
    # Instantiate the TranscriptsETL class
    input_pdf_path = os.path.join("data", "citigroup", "raw_docs", "transcripts", "25Q1.pdf")

    transcript_etl = TranscriptsETL(
        input_pdf_path=input_pdf_path,
    )

    # Run the extract method
    extracted_text = transcript_etl.extract()

    # Run the transform method
    chunks_df = transcript_etl.transform(
        raw_data=extracted_text,
        response_schema=ChunkedTranscript,
        # llm_backend="openai",
        # llm_model_name="gpt-4.1",
        llm_backend="gemini",
        llm_model_name="gemini-2.5-pro-preview-06-05",
        # llm_model_name="gemini-2.5-flash-preview-05-20",
        )

    # Run the load method
    output_csv_path = os.path.join("data", "citigroup", "processed", "transcripts", "25Q1.csv")
    transcript_etl.load(
        transformed_data=chunks_df,
        output_csv_path=output_csv_path
        )


class PresentationsETL(BaseETL):
