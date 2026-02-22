import streamlit as st
import logging
from langchain_ollama import ChatOllama

# Import configuration
from config import MODEL_NAME, LLM_TEMPERATURE, LLM_MAX_TOKENS

# Import ingest functions
from ingest.load_pdf import ingest_pdf
from ingest.embed_chunks import load_vector_db

# Import RAG functions
from rag.retriever import create_retriever
from rag.chain import create_chain

# Configure logging
logging.basicConfig(level=logging.INFO)


def main():
    st.title("📄 RAG Document Assistant 🤖")
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    retrieval_method = st.sidebar.selectbox(
        "Retrieval Method",
        ["mmr", "multi_query"],
        index=0,
        help="Choose the retrieval strategy"
    )
    
    # Display info about selected method
    if retrieval_method == "mmr":
        st.sidebar.info(
            "**MMR (Maximal Marginal Relevance)**\n\n"
            "✅ Balances relevance + diversity\n\n"
            "✅ Prevents redundant results\n\n"
            "✅ Faster (single query)\n\n"
            "✅ More consistent"
        )
    else:
        st.sidebar.info(
            "**Multi-Query Retriever**\n\n"
            "✅ Generates 5 query variations\n\n"
            "✅ Comprehensive coverage\n\n"
            "⚠️ Slower (multiple queries)\n\n"
            "⚠️ Higher token usage"
        )

    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Initialize session state for caching
        if "vector_db" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
            # Clear all session state to release database connections
            if st.session_state.get("file_name") != uploaded_file.name and st.session_state.get("file_name") is not None:
                logging.info(f"🔄 New file detected. Clearing session state...")
                st.session_state.clear()
                
                # Give time for resources to be released
                import time
                import gc
                gc.collect()
                time.sleep(0.3)
            
            with st.spinner("Processing PDF (clearing previous data)..."):
                documents, file_path = ingest_pdf(uploaded_file, clear_existing=True)
                if documents is None:
                    return

                vector_db = load_vector_db(documents, clear_existing=True)
                
                # Store in session state
                st.session_state.vector_db = vector_db
                st.session_state.file_name = uploaded_file.name
                
                logging.info(f"✅ New PDF '{uploaded_file.name}' processed successfully")
        
        # Get vector_db from session state
        vector_db = st.session_state.vector_db
        
        # Create retriever and chain (recreate if method changed)
        if ("retrieval_method" not in st.session_state or 
            st.session_state.retrieval_method != retrieval_method):
            
            with st.spinner(f"Setting up {retrieval_method.upper()} retriever..."):
                llm = ChatOllama(
                    model=MODEL_NAME, 
                    temperature=LLM_TEMPERATURE, 
                    num_predict=LLM_MAX_TOKENS
                )
                retriever = create_retriever(vector_db, llm, retrieval_type=retrieval_method)
                chain = create_chain(retriever, llm)
                
                # Store in session state
                st.session_state.retriever = retriever
                st.session_state.chain = chain
                st.session_state.retrieval_method = retrieval_method
        
        # Get chain from session state
        chain = st.session_state.chain

        st.success(f"✅ **{uploaded_file.name}** processed! Using **{retrieval_method.upper()}** retrieval.")
        st.info("💡 Upload a new PDF to automatically clear previous data.")

        # User input
        user_input = st.text_input("Enter your question:")

        if user_input:
            with st.spinner("Generating response..."):
                try:
                    response = chain.invoke(input=user_input)
                    st.markdown("**Assistant:**")
                    st.write(response)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please upload a PDF file to begin.")


if __name__ == "__main__":
    main()