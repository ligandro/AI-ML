import logging
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

logging.basicConfig(level=logging.INFO)


def create_chain(retriever, llm):
    """
    Create the RAG chain with anti-hallucination prompt.
    Inspired by Epstein Files RAG project.
    """
    # Anti-hallucination prompt template
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a retrieval-based assistant. "
            "Answer ONLY using the provided context. "
            "If the answer is not present in the context, say: "
            "'I could not find this information in the document.' "
            "Do not use any external knowledge or make assumptions. "
            "Keep your answer concise and factual."
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion:\n{question}"
        )
    ])
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    logging.info("RAG chain created with anti-hallucination prompt.")
    return chain
