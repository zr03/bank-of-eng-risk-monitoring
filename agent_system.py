from typing_extensions import TypedDict
from typing import Annotated

from langgraph.graph import StateGraph
from langgraph.graph.message import add_message
from langchain_core.tools import tool

from transformers import SentenceTransformer

vector_db_api_key = os.getenv("PINECONE_API_KEY")
pinecone_idx = 
embedding_model_name = "mukaj/fin-mpnet-base"
embedding_model = SentenceTransformer(embedding_model_name)

@tool
def vector_db_search(query: str):
    """Call to search a vector database for relevant information."""


    return

class State(TypedDict):
    messages: Annotated[list, add_messages]

if __name__ == "__main__":
	pass
