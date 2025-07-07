import os
import asyncio
from typing import Annotated, Literal, List, Optional
from typing_extensions import TypedDict

from pydantic import BaseModel, Field
from pinecone import Pinecone
from langgraph.graph import StateGraph
# from langgraph.graph.message import add_message
from langchain_core.tools import tool

from sentence_transformers import SentenceTransformer

vector_db_api_key = os.getenv("PINECONE_API_KEY")
pinecone_idx_name = os.getenv("PINECONE_INDEX_NAME")
embedding_model_name = "mukaj/fin-mpnet-base"
embedding_model = SentenceTransformer(embedding_model_name)

pc = Pinecone(api_key=vector_db_api_key)
index = pc.Index(pinecone_idx_name)

class VectorSearchInput(BaseModel):
    query: str = Field(..., description="User search query")
    role: Optional[Literal['CEO', 'CFO', 'Analyst']] = Field(None, description="Filter by source, e.g. 'CEO'")
    reporting_period: Optional[Literal["2023Q1", "2023Q2", "2023Q3", "2023Q4", "2024Q1", "2024Q2", "2024Q3", "2024Q4", "2025Q1"]] = Field(None, description="Filter by reporting period")
    bank: Optional[Literal["Citigroup", "Bank of America", "JPMorgan"]] = Field(None, description="Filter by bank name")
    source_type: Optional[Literal["internal", "external"]] = Field(None, description="'internal' for internal documents, 'external' for news sources")
    is_comparative: bool = Field(False, description="whether the text compares banks")
    sentiment_score: Optional[Literal[-1, 0, 1]] = Field(None, description="Filter by sentiment of the text, -1 for negative, 0 for neutral, 1 for positive")

def construct_metadata_filter_expr(search_schema: VectorSearchInput) -> dict:
    """Construct a filter expression for the vector database query."""
    filters = {}

    if search_schema.role is not None:
        filters["role"] = {"$eq": search_schema.role}
    if search_schema.reporting_period is not None:
        filters["reporting_period"] = {"$eq": search_schema.reporting_period}
    if search_schema.bank is not None:
        filters["bank"] = {"$eq": search_schema.bank}
    if search_schema.source_type is not None:
        filters["source_type"] = {"$eq": search_schema.source_type}
    if search_schema.sentiment_score is not None:
        filters["sentiment_score"] = {"$eq": search_schema.sentiment_score}
    if search_schema.is_comparative:
        filters["is_comparative"] = {"$eq": True}

    return filters

@tool
def vector_db_search(input: VectorSearchInput, top_k:int=5):
    """Call to search a vector database for relevant information."""
    query = input.query
    query_vector = embedding_model.encode(query).tolist()

    filters = construct_metadata_filter_expr(input)
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        filter=filters,
        include_metadata=True,
        include_values=False,

    )
    # return "\n".join([f"{match['score']:.2f}: {match['metadata']}" for match in results["matches"]])
    return results

# class State(TypedDict):
#     messages: Annotated[list, add_messages]

if __name__ == "__main__":
    res = vector_db_search.invoke(
        {
            "input":VectorSearchInput(query="What has JPMorgan said about it's liquidity?", bank="JPMorgan", source_type="internal", reporting_period="2025Q1"), "top_k":5
        }
    )
    res
