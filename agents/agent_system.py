import os
import asyncio
from typing import Annotated, Literal, List, Optional
from typing_extensions import TypedDict

from pydantic import BaseModel, Field
from pinecone import Pinecone
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer
# from langgraph.graph.message import add_message
# from langchain_core.tools import tool
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent

from sentence_transformers import SentenceTransformer

from agents.llms import BaseLLM, OrchestratorLLM, VectorDBSearchLLM

vector_db_api_key = os.getenv("PINECONE_API_KEY")
pinecone_idx_name = os.getenv("PINECONE_INDEX_NAME")
embedding_model_name = "mukaj/fin-mpnet-base"
embedding_model = SentenceTransformer(embedding_model_name)

pc = Pinecone(api_key=vector_db_api_key)
index = pc.Index(pinecone_idx_name)


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

    docs = [
        Doc(
            text=match["metadata"]["orig_text"],
            reference=match["metadata"]["reference"]
        )
        for match in results["matches"]
    ]
    return docs




def _make_orchestrator_prompt(user_query: str) -> str:
    """Construct the prompt for the orchestrator."""
    return f"""
    You are an expert in financial data retrieval and analysis, and your task is to determine which agents are needed to answer the user's query.
    User query: {user_query}
    You have access to the following agents:
    1. vector_db_search: Searches a vector database of statements from banks and news articles across different banks.
    2. metrics_db_search: Searches a metrics SQL database for financial metrics including trends and comparisons between banks.
    """

def _make_vector_db_search_prompt(user_query: str) -> str:
    """Construct the prompt for the vector database search."""
    return f"""
    You are an expert in financial data retrieval and your task is to provide the metadata filtering parameters for a downstream vector database search based on the user's query.
    User query: {user_query}
    The vector database contains statements from banks and news articles across different banks.
    Provide a short, focused phrase capturing the essence of the user's question for semantic search
    You can filter for the following fields:
    1. role
    2. reporting_period
    3. bank
    4. source_type
    5. sentiment_score
    6. is_comparative
    """

def _make_summarisation_prompt(retrieved_docs: str) -> str:
    """Construct the prompt for summarising the retrieved documents."""
    if not retrieved_docs:
        return "No documents to summarise."
    return (
        "You are an expert in summarising financial documents. Your task is to provide a concise summary of the retrieved documents based on the user's query and retrieved documents (which will include sources).\n"
        "Please summarise the following documents and extract any useful insights in relation to risk factors that may be at play using Prudential Risk Authority criteria:\n\n"
        f"{retrieved_docs}\n\n"
        "Remember to clearly cite sources at the end of the summary."
        )


async def orchestrator(state: GraphState):
    orchestrator_llm = OrchestratorLLM(
        prompt=_make_orchestrator_prompt(state["user_query"]),
        response_schema=OrchestrationPlan,
        backend="openai",
        model_name="gpt-4.1-mini"
    )
    response = await orchestrator_llm.ainvoke()
    agents_to_run = []
    if response.vector_db_retrieval_needed:
        agents_to_run.append("vector_db_search")
    if response.metrics_db_retrieval_needed:
        agents_to_run.append("metrics_db_search")
    return {"agents_to_run": agents_to_run}

async def retrieve_statements(state: GraphState):
    vector_db_search_llm = VectorDBSearchLLM(
        prompt=_make_vector_db_search_prompt(state["user_query"]),
        response_schema=VectorSearchInput,
        backend="openai",
        model_name="gpt-4.1-mini"
    )
    response = await vector_db_search_llm.ainvoke(state["user_query"])
    docs = vector_db_search(response, top_k=5)
    return {"retrieved_docs": docs}

async def summariser(state: GraphState):
    """Summarise the retrieved documents."""

    if not state.get("retrieved_docs"):
        summariser_llm = BaseLLM(
            prompt=f"You are an expert financial analyst who analyses risk factors of bank firms. Provide a generic response based on the user's query. {state['user_query']}, explain your role and guide them to ask a relevant question. Do not mention anything outside the scope of your role as a financial analyst.",
            backend="openai",
            model_name="gpt-4.1-mini",
            stream=True
        )
    else:
        documents_text = "\n".join(doc["text"] + "\n" + "Source: " + doc["reference"] for doc in state["retrieved_docs"])
        summariser_llm = BaseLLM(
            prompt=_make_summarisation_prompt(documents_text),
            backend="openai",
            model_name="gpt-4.1-mini",
            stream=True
        )

    stream_response = await summariser_llm.ainvoke()

    # Initialize stream writer and full response
    stream_writer = get_stream_writer()
    full_response = ""
    # Iterate over stream_response
    async for event in stream_response:
        if isinstance(event, ResponseTextDeltaEvent):
            token = event.delta
            stream_writer(token)  # Write token to stream
            full_response += token
    return {"final_answer": "".join(full_response)}


def decide_next_node(state: GraphState):
        if "vector_db_search" in state["agents_to_run"]:
            return "vector_db_search"
        return "summariser"

def build_graph():
    # Initialize the graph
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("orchestrator", orchestrator)
    graph.add_node("vector_db_search", retrieve_statements)
    # graph.add_node("metrics_db_search", metrics_db_search)  # Uncomment when metrics
    graph.add_node("summariser", summariser)

    # Add edges
    graph.add_conditional_edges("orchestrator", decide_next_node)
    graph.add_edge("vector_db_search", "summariser")

    # Transitions
    graph.set_entry_point("orchestrator")
    graph.set_finish_point("summariser")
    compiled_graph = graph.compile()
    return compiled_graph


async def stream_graph(graph, user_query):
    initial_state = {
        "user_query": user_query,
        "agents_to_run": [],
        "retrieved_docs": None,
        "final_answer": None,
    }
    async for output in graph.astream(
        initial_state,
        stream_mode="custom"
    ):
        print(output, end="", flush=True)

def run_graph(user_query):
    # Create the graph
    graph = build_graph()

    # Run the graph with a sample input
    asyncio.run(stream_graph(graph, user_query))

if __name__ == "__main__":
    # res = vector_db_search.invoke(
    #     {
    #         "input":VectorSearchInput(query="What has JPMorgan said about it's liquidity?", bank="JPMorgan", source_type="internal", reporting_period="2025Q1"), "top_k":5
    #     }
    # )
    user_query = "What has JPMorgan been saying about its liquidity in its earnings documents?"
    run_graph(user_query)

