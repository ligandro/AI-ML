# 📚 RAG Document Chat 

A Streamlit-based AI-powered document assistant for PDF querying using LangChain, Ollama's Llama3.2, and ChromaDB.

<p align="center">
  <img width="70%" src="demo.png"> &nbsp &nbsp
</p>

---

## ⚡ Quick Demo

Upload any PDF → Ask questions → Get accurate, grounded answers

**What it does:**
- Automatically processes and chunks your PDF documents
- Embeds content into a searchable vector database
- Retrieves diverse, relevant information using MMR algorithm or Multi-Query method
- Generates answers grounded solely in document context
- Prevents hallucinations with anti-hallucination prompting

---

## 🎯 Key Features

✅ **No Hallucinations** - Answers only from document content  
✅ **Intelligent Retrieval** - MMR algorithm for diverse, relevant results or Multi-Query method
✅ **Fast Processing** - Efficient PDF chunking and embedding  
✅ **Session Management** - Auto-cleanup between different PDFs  
✅ **Interactive UI** - Streamlit interface with sidebar controls  
✅ **Local LLM** - Runs entirely on Ollama (privacy-first)  
✅ **Configurable** - Easy-to-modify settings in config.py  

---

## 🏗️ How It Works

### Three Simple Stages

**Stage 1: Document Processing**
```
PDF Upload
    ↓
Extract Text
    ↓
Smart Chunking (1200 chars, 300 overlap)
    ↓
Vector Embeddings
```

**Stage 2: Intelligent Retrieval**
```
User Question
    ↓
Find Similar Context (MMR or Multi-Query)
    ↓
Return Top Results
```

**Stage 3: Grounded Answer**
```
Context + Question
    ↓
LLaMA 3.2 (temperature=0)
    ↓
Grounded Answer (no hallucinations)
```

### Why MMR and Multi Query Instead of Similarity?

**Basic Approach:** Pure semantic similarity  
→ Returns redundant chunks from same document section

**MMR Approach:** Maximal Marginal Relevance  
→ Balances relevance + diversity for comprehensive context (fetches 60, selects top 12)

**Multi-Query Method:**  Generates multiple queries from the question to retrieve a wider range of relevant chunks, further enhancing answer quality.
---

## 📚 Project Structure

```
RAG-Document-Chat/
├── ingest/                    # Document processing pipeline
│   ├── load_pdf.py           # PDF loading & cleanup
│   ├── chunk_documents.py    # Smart chunking
│   └── embed_chunks.py       # Embedding & ChromaDB
├── rag/                       # RAG components
│   ├── retriever.py          # MMR & Multi-Query retrievers
│   └── chain.py              # LLM chain with anti-hallucination
├── config.py                  # Centralized configuration
├── app.py                     # Streamlit UI
└── README.md                  # This file
```

