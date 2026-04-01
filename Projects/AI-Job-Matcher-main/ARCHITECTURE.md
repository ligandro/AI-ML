# AI Job Matcher – Architecture

## Overview
The app is a Streamlit frontend backed by a LangGraph pipeline that orchestrates LLM-driven steps (nodes) for resume understanding, profile analysis, job matching, and cover letter generation. Ollama (OpenAI-compatible endpoint) powers all node generations.

## Data Flow
1) Upload: Streamlit saves the PDF to `uploads/` and passes `{file_path, job_description}` to the orchestrator.
2) Graph: LangGraph state (`PipelineState`) carries `file_path`, `job_description`, `raw_text`, `extracted_data`, `analysis_results`, `job_matches`, `final_recommendation`, `status`, `history`.
3) Nodes (LangGraph)
   - extract: PDF → text (pdfminer) → LLM extracts structured resume fields.
   - analyze: LLM summarizes skills/experience into a normalized schema.
   - match: LLM compares profile vs JD, returns title/match_score and reason buckets.
   - recommend: LLM drafts a cover letter tailored to the JD.
4) UI: Tabs display analysis, matching reasons, and the generated cover letter; results are persisted to `results/`.

## Agentic Design (via graph nodes)
- Coordination: LangGraph `StateGraph` encodes the directed flow extract → analyze → match → recommend.
- Determinism aids: Each node uses Pydantic models (`ExtractedData`, `AnalysisPayload`, `MatchResult`, `Recommendation`) to coerce/validate LLM JSON before updating state, with defaults on parse failure.
- Backend: Ollama (`model=llama3.2`) via OpenAI client; temperature 0.7, max_tokens 2000.

## Extensibility
- Add guards: plug in schema validation errors and retries per node.
- Branching: introduce screening/shortlisting nodes before recommendation.
- Observability: log per-node timings and decisions to `history` and `logs/`.
- Deployment: containerize; parameterize Ollama host/model via env vars.
