import logging
import sys
import shutil
import os
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import EMBEDDING_MODEL, VECTOR_STORE_NAME, PERSIST_DIRECTORY

import ollama
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from .chunk_documents import split_documents

logging.basicConfig(level=logging.INFO)


def clear_vector_db():
    """Clear the existing vector database."""
    import time
    import gc
    
    chroma_path = Path(PERSIST_DIRECTORY)
    if chroma_path.exists():
        try:
            # Force garbage collection to release file handles
            gc.collect()
            time.sleep(0.1)  # Brief pause to ensure files are released
            
            shutil.rmtree(chroma_path)
            logging.info(f"🧹 Cleared existing vector database at {chroma_path}")
        except PermissionError as e:
            # On Windows, files might be locked. Try to work around it.
            logging.warning(f"⚠️ Could not delete all files: {e}")
            try:
                # Try removing individual files
                for root, dirs, files in os.walk(chroma_path, topdown=False):
                    for name in files:
                        file_path = os.path.join(root, name)
                        try:
                            os.chmod(file_path, 0o777)
                            os.remove(file_path)
                        except Exception:
                            pass
                    for name in dirs:
                        try:
                            os.rmdir(os.path.join(root, name))
                        except Exception:
                            pass
                try:
                    os.rmdir(chroma_path)
                    logging.info(f"🧹 Cleared vector database (with retry)")
                except Exception:
                    logging.warning("⚠️ Some files could not be deleted. They will be overwritten on next run.")
            except Exception as e2:
                logging.warning(f"⚠️ Partial cleanup only: {e2}")
    else:
        logging.info("ℹ️ No existing vector database to clear")


def load_vector_db(documents, clear_existing=True):
    """
    Create and load the vector database.
    
    Args:
        documents: List of documents to embed
        clear_existing: If True, clears existing vector DB before creating new one
    
    Returns:
        ChromaDB vector database instance
    """
    import chromadb
    import gc
    import time
    
    # Clear existing database to prevent mixing old and new data
    if clear_existing:
        # Try to delete collection via ChromaDB API first
        try:
            client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
            try:
                client.delete_collection(name=VECTOR_STORE_NAME)
                logging.info(f"🗑️ Deleted existing collection: {VECTOR_STORE_NAME}")
            except Exception:
                logging.info("ℹ️ No existing collection to delete")
            
            # Close client and force cleanup
            del client
            gc.collect()
            time.sleep(0.2)
        except Exception as e:
            logging.warning(f"⚠️ Could not delete collection via API: {e}")
        
        # Then clear the directory
        clear_vector_db()
        
        # Extra cleanup
        gc.collect()
        time.sleep(0.2)
    
    ollama.pull(EMBEDDING_MODEL)
    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    chunks = split_documents(documents)
    
    logging.info(f"📦 Creating new vector database with {len(chunks)} chunks...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        collection_name=VECTOR_STORE_NAME,
        persist_directory=PERSIST_DIRECTORY,
    )
    vector_db.persist()
    
    # Verify the database
    collection = vector_db._collection
    count = collection.count()
    logging.info(f"✅ Vector database created with {count} embeddings.")
    
    return vector_db
