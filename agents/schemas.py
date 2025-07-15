from typing import Literal, List, Optional
from typing_extensions import TypedDict
import sqlite3

import pandas as pd
from pydantic import BaseModel, Field, create_model

import app_config as config

DB_PATH = config.DB_PATH

class OrchestrationPlan(BaseModel):
    vector_db_retrieval_needed: bool
    metrics_db_retrieval_needed: bool

class Doc(TypedDict):
    text: str
    reference: str

class GraphState(TypedDict):
    user_query: str
    agents_to_run: List[Literal['vector_db_search', 'metrics_db_search']]
    retrieved_docs: Optional[List[Doc]]
    final_answer: Optional[str]

class VectorSearchInput(BaseModel):
    query: str = Field(..., description="User search query")
    role: Optional[Literal['CEO', 'CFO', 'Analyst']] = Field(None, description="Filter by source, e.g. 'CEO'")
    reporting_period: Optional[Literal["2023Q1", "2023Q2", "2023Q3", "2023Q4", "2024Q1", "2024Q2", "2024Q3", "2024Q4", "2025Q1"]] = Field(None, description="Filter by reporting period")
    bank: Optional[Literal["Citigroup", "Bank of America", "JPMorgan"]] = Field(None, description="Filter by bank name")
    # document_type: Optional[Literal["transcript", "presentation", "news"]] = Field(None, description="Filter by document type")
    source_type: Optional[Literal["internal", "external"]] = Field(None, description="'internal' to filter for sources originating from the bank itself e.g. transcripts, 'external' for news sources")
    is_comparative: bool = Field(False, description="whether the text compares banks")
    sentiment_score: Optional[Literal[-1, 0, 1]] = Field(None, description="Filter by sentiment of the text, -1 for negative, 0 for neutral, 1 for positive")


def construct_dynamic_schema(df: pd.DataFrame, model_name: str, required_fields: list = [], exclude_fields: list = [], categorical_fields: list =[], field_descriptions: dict = {}) -> BaseModel:
	"""
	Dynamically constructs a Pydantic model based on the DataFrame columns.
	This is useful for creating schemas that can adapt to updating data.
	"""
	fields = {}
	for col, dtype in df.dtypes.items():
		if col in exclude_fields:
			continue
		if col in categorical_fields:
			# For categorical fields, we use Literal with unique values
			unique_values = df[col].dropna().unique()
			typ = Literal[tuple(unique_values)]
		elif pd.api.types.is_integer_dtype(dtype):
			typ = Optional[int]
		elif pd.api.types.is_float_dtype(dtype):
			typ = Optional[float]
		elif pd.api.types.is_string_dtype(dtype): # Datetimes are often stored as strings in SQLite
			typ = Optional[str]
		else:
			raise ValueError(f"Unsupported data type '{dtype}' for column '{col}'. Supported types are: int, float, string.")
		if col in required_fields:
			default_value = ...  # Required fields should not have a default value
		else:
			default_value = None
		fields[col] = (typ, Field(default=default_value, description=field_descriptions.get(col, "")))
	return create_model(
		model_name,
		**fields,
	)

def generate_raw_metrics_search_schema() -> BaseModel:
	"""
	Generates a Pydantic model for searching raw metrics.
	This is a static schema based on the expected structure of the raw metrics table.
	"""
	with sqlite3.connect(DB_PATH) as conn:
		df_raw_metrics = pd.read_sql('SELECT * FROM raw_metrics', conn)

	return construct_dynamic_schema(
		df_raw_metrics,
		model_name="RawMetricsSearchSchema",
		categorical_fields=["category_type", "category", "metric_name"],
		field_descriptions={
			"bank": "Name of the bank",
			"quarter": "Reporting quarter e.g. '2023Q1', '2023Q2', etc.",
			"metric_name": "Name of the financial metric",
			"value": "Value of the financial metric"
		}
	)

if __name__ == "__main__":
	# Construct dynamic schemas
	RawMetricsSearchSchema = generate_raw_metrics_search_schema()
	x = RawMetricsSearchSchema(
		bank="Bank of America",
		quarter="2023Q1",
		category_type="Risk",
		category="Profitability",
		metric_name="ROCE (%)",
		rank=1,
	)



