import logging
import sys
from pathlib import Path
from typing import List
from json import dumps, loads

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import RETRIEVAL_TYPE, MMR_K, MMR_FETCH_K, MMR_LAMBDA, RRF_K, FUSION_QUERIES

from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

logging.basicConfig(level=logging.INFO)


def reciprocal_rank_fusion(results: List[List[Document]], k: int = 60) -> List[Document]:
    """
    Reciprocal Rank Fusion (RRF) algorithm for combining and re-ranking 
    documents from multiple query results.
    
    Args:
        results: List of document lists from different queries
        k: Constant for RRF formula (default 60)
    
    Returns:
        List of documents sorted by fused score
    """
    fused_scores = {}
    
    for docs in results:
        for rank, doc in enumerate(docs):
            # Serialize document for comparison (use page_content + metadata as key)
            doc_str = dumps({"content": doc.page_content, "metadata": doc.metadata})
            
            if doc_str not in fused_scores:
                fused_scores[doc_str] = {"doc": doc, "score": 0}
            
            # RRF formula: 1 / (rank + k)
            fused_scores[doc_str]["score"] += 1 / (rank + k)
    
    # Sort by fused score (descending)
    reranked_results = sorted(
        fused_scores.values(), 
        key=lambda x: x["score"], 
        reverse=True
    )
    
    # Return just the documents
    return [item["doc"] for item in reranked_results]


class RAGFusionRetriever(BaseRetriever):
    """
    Custom retriever that implements RAG-Fusion:
    1. Generates multiple related queries using an LLM
    2. Retrieves documents for each query
    3. Applies Reciprocal Rank Fusion to re-rank results
    """
    
    llm: object
    base_retriever: object
    num_queries: int = 4
    rrf_k: int = 60
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Generate queries, retrieve docs, and apply RRF."""
        
        # Step 1: Generate related queries
        query_prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are a helpful assistant that generates multiple search queries based on a single input query.
Generate {num_queries} different search queries related to: {{question}}
Provide these search queries separated by newlines.""".format(num_queries=self.num_queries),
        )
        
        generate_queries = (
            query_prompt 
            | self.llm 
            | StrOutputParser() 
            | (lambda x: x.split("\n"))
        )
        
        # Generate the queries
        queries = generate_queries.invoke({"question": query})
        # Filter out empty queries
        queries = [q.strip() for q in queries if q.strip()]
        
        logging.info(f"Generated {len(queries)} queries for RAG-Fusion")
        
        # Step 2: Retrieve documents for each query
        all_results = []
        for q in queries:
            docs = self.base_retriever.get_relevant_documents(q)
            all_results.append(docs)
            logging.info(f"Query '{q[:50]}...' retrieved {len(docs)} docs")
        
        # Step 3: Apply Reciprocal Rank Fusion
        fused_docs = reciprocal_rank_fusion(all_results, k=self.rrf_k)
        
        logging.info(f"RAG-Fusion returned {len(fused_docs)} unique documents")
        
        return fused_docs


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
    
    elif retrieval_type == "rag_fusion":
        # RAG-Fusion - generates related queries and applies RRF ranking
        base_retriever = vector_db.as_retriever()
        
        retriever = RAGFusionRetriever(
            llm=llm,
            base_retriever=base_retriever,
            num_queries=FUSION_QUERIES,
            rrf_k=RRF_K
        )
        logging.info(f"RAG-Fusion Retriever created (queries={FUSION_QUERIES}, rrf_k={RRF_K})")
    
    else:
        raise ValueError(f"Unknown retrieval type: {retrieval_type}. Use 'mmr', 'multi_query', or 'rag_fusion'")
    
    return retriever
