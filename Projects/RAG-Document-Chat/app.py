"""Streamlit interface for local, evidence-first PDF chat."""

from __future__ import annotations

import hashlib
import logging

import streamlit as st

from config import MAX_FILES, MAX_QUESTION_CHARS
from document_chat import RagIndex, extract_pdfs

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

st.set_page_config(page_title="Document Chat", page_icon="📄", layout="wide")
st.title("Document Chat")
st.caption("Ask questions across your PDFs. Answers show the passages retrieved from the documents.")
if "upload_generation" not in st.session_state:
    st.session_state.upload_generation = 0

with st.sidebar:
    st.header("Documents")
    files = st.file_uploader(
        "Add up to five PDFs", type="pdf", accept_multiple_files=True,
        key=f"files_{st.session_state.upload_generation}",
    )
    method = st.selectbox("Search method", ["hybrid", "semantic", "keyword"],
                          help="Hybrid combines semantic and keyword search.")
    if st.button("Clear conversation and documents"):
        previous = st.session_state.get("index")
        if previous is not None:
            previous.close()
        for key in ("index", "file_signature", "messages", "warnings"):
            st.session_state.pop(key, None)
        st.session_state.upload_generation += 1
        st.rerun()
    st.caption("Runs through your local Ollama server. PDFs are held in this app session, not saved by this project.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if files:
    if len(files) > MAX_FILES:
        st.error(f"Please select no more than {MAX_FILES} PDFs.")
        st.stop()
    # Hash content as well as names so replacing a PDF with the same filename reindexes it.
    payload = [(file.name, file.getvalue()) for file in files]
    signature = hashlib.sha256(
        b"".join(name.encode() + b"\0" + hashlib.sha256(data).digest() for name, data in payload)
    ).hexdigest()
    if st.session_state.get("file_signature") != signature:
        with st.spinner("Reading pages and building the search index…"):
            try:
                chunks, warnings = extract_pdfs(payload)
                index = RagIndex.build(chunks)
            except ValueError as exc:
                st.error(str(exc))
                st.stop()
            except Exception:
                log.exception("Document indexing failed")
                st.error("Indexing failed. Check that Ollama is running and the embedding model is installed.")
                st.stop()
            previous = st.session_state.get("index")
            if previous is not None:
                try:
                    previous.close()
                except Exception:
                    log.warning("Could not release prior in-memory collection", exc_info=True)
            st.session_state.index = index
            st.session_state.file_signature = signature
            st.session_state.messages = []
            st.session_state.warnings = warnings
    st.success(f"Indexed {len(st.session_state.index.chunks)} passages from {len(files)} PDF(s).")
    for warning in st.session_state.get("warnings", []):
        st.warning(warning)
else:
    # Do not allow questions against PDFs that are no longer in the uploader.
    previous = st.session_state.get("index")
    if previous is not None:
        previous.close()
    st.session_state.pop("index", None)
    st.session_state.pop("file_signature", None)
    st.info("Upload a PDF to begin. Text-based PDFs work best; scanned pages require OCR.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Context sent to the model"):
                for source in message["sources"]:
                    st.markdown(f"**[{source['number']}] {source['name']} · page {source['page']}**")
                    st.text(source["excerpt"])

question = st.chat_input("Ask about your documents", disabled="index" not in st.session_state,
                         max_chars=MAX_QUESTION_CHARS)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching and answering…"):
            try:
                answer, hits = st.session_state.index.answer(question, method)
                sources = [
                    {"number": i, "name": hit.chunk.source, "page": hit.chunk.page,
                     "excerpt": hit.chunk.text}
                    for i, hit in enumerate(hits, 1)
                ]
                st.markdown(answer)
                with st.expander("Context sent to the model"):
                    for source in sources:
                        st.markdown(f"**[{source['number']}] {source['name']} · page {source['page']}**")
                        st.text(source["excerpt"])
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except ValueError as exc:
                st.error(str(exc))
            except Exception:
                log.exception("Question answering failed")
                st.error("Could not answer. Check that Ollama is running and the chat model is installed.")

