import logging
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import RETRIEVAL_TYPE, MMR_K, MMR_FETCH_K, MMR_LAMBDA

from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever

logging.basicConfig(level=logging.INFO)


def create_retriever(vector_db, llm, retrieval_type=None):
    """
    Create a retriever based on the specified type.
    
    Args:
        vector_db: ChromaDB vector database instance
        llm: Language model for multi-query generation
        retrieval_type: Type of retrieval ("mmr" or "multi_query"). Uses config default if None.
    
    Returns:
        Configured retriever instance
    """
    retrieval_type = retrieval_type or RETRIEVAL_TYPE
    
    if retrieval_type == "mmr":
        # MMR (Maximal Marginal Relevance) - balances relevance and diversity
        retriever = vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": MMR_K,
                "fetch_k": MMR_FETCH_K,
                "lambda_mult": MMR_LAMBDA
            }
        )
        logging.info(f"MMR Retriever created (k={MMR_K}, fetch_k={MMR_FETCH_K}, lambda={MMR_LAMBDA})")
    
    elif retrieval_type == "multi_query":
        # Multi-Query - generates multiple query variations
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI assistant. Generate five
            different versions of the given user question to retrieve relevant documents
            from a vector database. Provide these alternative questions separated by newlines.
            Original question: {question}""",
        )
        
        retriever = MultiQueryRetriever.from_llm(
            vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
        )
        logging.info("Multi-Query Retriever created")
    
    else:
        raise ValueError(f"Unknown retrieval type: {retrieval_type}. Use 'mmr' or 'multi_query'")
    
    return retriever
