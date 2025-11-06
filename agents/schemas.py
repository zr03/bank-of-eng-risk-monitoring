from typing import Literal, List, Optional, Union
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


class NumericFilter(BaseModel):
    field: str
    gt: Optional[float] = None
    gte: Optional[float] = None
    lt: Optional[float] = None
    lte: Optional[float] = None
    eq: Optional[float] = None

# class IntegerFilter(BaseModel):
# 	field: str
# 	gt: Optional[int] = None
# 	gte: Optional[int] = None
# 	lt: Optional[int] = None
# 	lte: Optional[int] = None
# 	eq: Optional[int] = None

class CategoricalFilter(BaseModel):
    field: str
    eq: Optional[str] = None
    in_: Optional[List[str]] = None

# class QueryFilter(BaseModel):
#     numeric_filters: Optional[List[NumericFilter]] = None
#     categorical_filters: Optional[List[CategoricalFilter]] = None

class LogicalGroup(BaseModel):
    op: Literal["AND", "OR"]
    filters: List[Union[NumericFilter, CategoricalFilter]]

class AdvancedQuery(BaseModel):
    conditions: List[LogicalGroup]

def build_condition(filter):
    if isinstance(filter, NumericFilter):
        parts = []
        if filter.eq is not None: parts.append(f"{filter.field} = {filter.eq}")
        if filter.gt is not None: parts.append(f"{filter.field} > {filter.gt}")
        if filter.gte is not None: parts.append(f"{filter.field} >= {filter.gte}")
        if filter.lt is not None: parts.append(f"{filter.field} < {filter.lt}")
        if filter.lte is not None: parts.append(f"{filter.field} <= {filter.lte}")
        return " AND ".join(parts)
    elif isinstance(filter, CategoricalFilter):
        if filter.eq is not None:
            return f"{filter.field} = '{filter.eq}'"
        elif filter.in_ is not None:
            vals = ", ".join(f"'{v}'" for v in filter.in_)
            return f"{filter.field} IN ({vals})"
    return ""

def build_sql_query(query: AdvancedQuery) -> str:
    groups = []
    for group in query.conditions:
        subconditions = [build_condition(f) for f in group.filters]
        group_clause = f" {group.op} ".join(subconditions)
        groups.append(f"({group_clause})")
    return " AND ".join(groups)

def construct_dynamic_schema(df: pd.DataFrame, model_name: str, required_fields: list = [], exclude_fields: list = [], categorical_fields: list =[], field_descriptions: dict = {}) -> BaseModel:
	"""
	Dynamically constructs a Pydantic model based on the DataFrame columns.
	This is useful for creating schemas that can adapt to updating data.
	"""
	filter_models = []
	for col, dtype in df.dtypes.items():
		if col in exclude_fields:
			continue
		field_desc = field_descriptions.get(col)
		if col in required_fields:
			default_value = ...  # Required fields should not have a default value
		else:
			default_value = None
		if col in categorical_fields:
			# For categorical fields, we use Literal with unique values
			valid_values_list = df[col].dropna().unique().tolist()
			filter_model_dict = {
				# "field": (str, Field(default_value, description=field_desc)),
				"eq": (Optional[Literal[*valid_values_list]], Field(None, description=f"Exact match for {col}")),
				"in_": (Optional[List[Literal[*valid_values_list]]], Field(None, description=f"List of values to match for {col}")),

			}
		# elif pd.api.types.is_integer_dtype(dtype):
		# 	filter_model_dict = {
		# 		"field": (str, Field(default_value, description=f"Filter for {col}")),
		# 		"gt": (Optional[int], Field(None, description=f"Greater than for {col}")),
		# 		"gte": (Optional[int], Field(None, description=f"Greater than or equal to for {col}")),
		# 		"lt": (Optional[int], Field(None, description=f"Less than for {col}")),
		# 		"lte": (Optional[int], Field(None, description=f"Less than or equal to for {col}")),
		# 		"eq": (Optional[int], Field(None, description=f"Exact match for {col}")),
		# 	}
		elif pd.api.types.is_float_dtype(dtype) or pd.api.types.is_integer_dtype(dtype):
			filter_model_dict = {
				# "field": (str, Field(..., description=field_desc)),
				"gt": (Optional[float], Field(None, description=f"Greater than for {col}")),
				"gte": (Optional[float], Field(None, description=f"Greater than or equal to for {col}")),
				"lt": (Optional[float], Field(None, description=f"Less than for {col}")),
				"lte": (Optional[float], Field(None, description=f"Less than or equal to for {col}")),
				"eq": (Optional[float], Field(None, description=f"Exact match for {col}")),
			}
		# elif pd.api.types.is_string_dtype(dtype): # Datetimes are often stored as strings in SQLite
		# 	typ = Optional[str]
		else:
			print(f"Unsupported data type '{dtype}' for column '{col}'. Supported types are: categorical, int and float. Skipping this column.")
			continue

		filter_model_name = f"{''.join([x.capitalize() for x in col.replace('_',' ').split()])}Filter"
		filter_model = create_model(
			filter_model_name,
			**	filter_model_dict,
			__doc__= f"Filter model for {col}: {field_desc}" if field_desc else f"Filter model for {col}"
		)
		filter_models.append(filter_model)

	# Construct the logical group model with all filter models
	logical_group_model = create_model(
		"LogicalGroup",
		op=(Literal["AND", "OR"], Field("AND", description="Logical operator for the group")),
		filters=(List[Union[*filter_models]], Field(..., description="List of filters in this logical group"))
	)

	# Construct the main query model
	advanced_query_model = create_model(
		model_name,
		conditions=(List[logical_group_model], Field(..., description="List of logical groups for the query"))
	)

	return advanced_query_model, {m.__name__: m for m in filter_models}

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
		categorical_fields=["bank", "category_type", "category", "metric_name", "quarter"],
		field_descriptions={
			"bank": "Name of the bank",
			"quarter": "Reporting quarter e.g. '2023Q1', '2023Q2', etc.",
			"metric_name": "Name of the financial metric",
			"value": "Value of the financial metric"
		},
		exclude_fields=["id"],
	)

if __name__ == "__main__":
	# Construct dynamic schemas
	RawMetricsSearchSchema, filter_models_dict = generate_raw_metrics_search_schema()
	RawMetricsSearchSchema(
		conditions=[
			{
				"op": "AND",
				"filters": [
					{"eq": "Citigroup"}, # BankFilter
					{"in_": ["2023Q1", "2023Q2"]}, # QuarterFilter
					# CategoryTypeFilter
					# CategoryFilter
					# MetricNameFilter
					# MetricValueFilter
					# RankFilter
				]
			}
		]
	)



