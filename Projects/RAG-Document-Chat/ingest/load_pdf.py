import os
import shutil
import logging
from pathlib import Path
import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader

logging.basicConfig(level=logging.INFO)


def clear_data_directory():
    """Clear the data directory to remove old PDF files."""
    data_path = Path("data")
    if data_path.exists():
        shutil.rmtree(data_path)
        logging.info("🧹 Cleared data directory")
    os.makedirs("data", exist_ok=True)


def ingest_pdf(uploaded_file, clear_existing=True):
    """
    Load PDF document from uploaded file.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        clear_existing: If True, clears existing data directory before processing
    
    Returns:
        Tuple of (documents, file_path)
    """
    if uploaded_file is not None:
        # Clear old data to prevent accumulation
        if clear_existing:
            clear_data_directory()
        else:
            os.makedirs("data", exist_ok=True)
        
        temp_path = os.path.join("data", uploaded_file.name)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        
        loader = UnstructuredPDFLoader(file_path=temp_path)
        data = loader.load()
        logging.info("✅ PDF loaded successfully.")
        return data, temp_path
    else:
        logging.error("No file uploaded.")
        st.error("Please upload a PDF file.")
        return None, None
