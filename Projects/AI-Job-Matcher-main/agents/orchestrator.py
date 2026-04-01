"""LangGraph-powered orchestrator that replaces ad-hoc agents with a graph."""

from typing import Any, Dict, List, TypedDict
from typing import Type, TypeVar
import json
from datetime import datetime

from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from pdfminer.high_level import extract_text
from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


class PipelineState(TypedDict, total=False):
    """Shared state flowing through the LangGraph pipeline."""

    file_path: str
    job_description: str
    raw_text: str
    extracted_data: Dict[str, Any]
    analysis_results: Dict[str, Any]
    job_matches: Dict[str, Any]
    final_recommendation: Dict[str, Any]
    status: str
    current_stage: str
    error: str
    history: List[str]


class OrchestratorAgent:
    """Coordinates the resume → analysis → matching → cover letter flow via LangGraph."""

    def __init__(self):
        self.llm = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # required but unused
        )
        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # Graph definition
    # ------------------------------------------------------------------
    def _build_graph(self):
        graph = StateGraph(PipelineState)
        graph.add_node("extract", self._extract_node)
        graph.add_node("analyze", self._analyze_node)
        graph.add_node("match", self._match_node)
        graph.add_node("recommend", self._recommend_node)

        graph.add_edge(START, "extract")
        graph.add_edge("extract", "analyze")
        graph.add_edge("analyze", "match")
        graph.add_edge("match", "recommend")
        graph.add_edge("recommend", END)

        return graph.compile()

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------
    async def process_application(self, resume_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the end-to-end LangGraph pipeline for a resume + job description."""

        initial_state: PipelineState = {
            "file_path": resume_data.get("file_path", ""),
            "job_description": resume_data.get("job_description", ""),
            "status": "initiated",
            "current_stage": "extract",
            "history": [
                f"received:{datetime.now().isoformat()}",
            ],
        }

        try:
            result: PipelineState = await self.graph.ainvoke(initial_state)
            result["status"] = result.get("status", "completed")
            result["current_stage"] = "completed"
            return result
        except Exception as exc:  # surface failure in state shape the UI expects
            return {
                "status": "failed",
                "current_stage": "failed",
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # LangGraph nodes
    # ------------------------------------------------------------------
    async def _extract_node(self, state: PipelineState) -> PipelineState:
        print("📄 LangGraph Extract: reading resume")
        file_path = state.get("file_path", "")

        raw_text = extract_text(file_path) if file_path else state.get("raw_text", "")
        prompt = (
            "Extract and structure information from this resume. Return JSON with keys "
            "personal_info, work_experience, education, skills, certifications.\n\n"
            f"Resume text:\n{raw_text}"
        )
        structured_text = self._call_llm(prompt)
        structured_data = self._parse_with_model(structured_text, ExtractedData)

        return {
            **state,
            "raw_text": raw_text,
            "extracted_data": structured_data,
            "current_stage": "analyze",
            "history": [*state.get("history", []), "extracted"],
        }

    async def _analyze_node(self, state: PipelineState) -> PipelineState:
        print("🔍 LangGraph Analyze: summarizing profile")
        analysis_prompt = f"""
        Analyze this structured resume data and return a JSON object with:
        {{
            "technical_skills": ["skill1", "skill2"],
            "years_of_experience": number,
            "education": {{"level": "Bachelors/Masters/PhD", "field": ""}},
            "experience_level": "Junior/Mid-level/Senior",
            "key_achievements": ["..."],
            "domain_expertise": ["..."]
        }}

        Resume data:
        {json.dumps(state.get("extracted_data", {}), indent=2)}
        """

        analysis_text = self._call_llm(analysis_prompt)
        parsed_analysis = self._parse_with_model(analysis_text, AnalysisPayload)

        return {
            **state,
            "analysis_results": {
                "skills_analysis": parsed_analysis,
                "analysis_timestamp": datetime.now().isoformat(),
            },
            "current_stage": "match",
            "history": [*state.get("history", []), "analyzed"],
        }

    async def _match_node(self, state: PipelineState) -> PipelineState:
        print("🎯 LangGraph Match: matching against JD")
        prompt = f"""
        You are an AI job matcher. Given the candidate profile and job description, return JSON:
        {{
            "title": "Job Title",
            "match_score": <0-100>,
            "analysis": {{
                "reasons_for_high_match_score": ["..."],
                "reasons_for_moderate_match_score": ["..."],
                "reasons_for_low_match_score": ["..."]
            }}
        }}

        Candidate Profile:
        {json.dumps(state.get("analysis_results", {}), indent=2)}

        Job Description:
        {state.get("job_description", "")}
        """

        match_text = self._call_llm(prompt)
        match_json = self._parse_with_model(match_text, MatchResult)

        return {
            **state,
            "job_matches": match_json,
            "current_stage": "recommend",
            "history": [*state.get("history", []), "matched"],
        }

    async def _recommend_node(self, state: PipelineState) -> PipelineState:
        print("💡 LangGraph Recommend: generating cover letter")
        prompt = f"""
        You are an AI cover letter generator. Using the candidate profile and job description, write a concise cover letter that highlights the best match points.
        Return plain text.

        Candidate Profile:
        {json.dumps(state.get("analysis_results", {}), indent=2)}

        Job Description:
        {state.get("job_description", "")}
        """

        recommendation_text = self._call_llm(prompt)

        return {
            **state,
            "final_recommendation": Recommendation(
                final_recommendation=recommendation_text,
                recommendation_timestamp=datetime.now().isoformat(),
                confidence_level="high",
            ).model_dump(),
            "status": "completed",
            "current_stage": "completed",
            "history": [*state.get("history", []), "recommended"],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _call_llm(self, prompt: str) -> str:
        """Call Ollama via the OpenAI-compatible client."""
        response = self.llm.chat.completions.create(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000,
        )
        return response.choices[0].message.content

    def _extract_json_segment(self, text: str) -> str:
        """Grab the most likely JSON object substring from LLM text."""
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
        return "{}"

    def _parse_with_model(self, text: str, model_cls: Type[T]) -> Dict[str, Any]:
        """Parse LLM text into a Pydantic model and return a dict fallback on error."""
        candidate = self._extract_json_segment(text)
        try:
            return model_cls.model_validate_json(candidate).model_dump()
        except Exception:
            return model_cls().model_dump()  # type: ignore[call-arg]


# ------------------------------------------------------------------
# Pydantic models for structured parsing
# ------------------------------------------------------------------


class ExtractedData(BaseModel):
    personal_info: Dict[str, Any] = Field(default_factory=dict)
    work_experience: List[Any] = Field(default_factory=list)
    education: List[Any] = Field(default_factory=list)
    skills: List[Any] = Field(default_factory=list)
    certifications: List[Any] = Field(default_factory=list)


class AnalysisPayload(BaseModel):
    technical_skills: List[Any] = Field(default_factory=list)
    years_of_experience: float = 0
    education: Dict[str, Any] = Field(
        default_factory=lambda: {"level": "Unknown", "field": "Unknown"}
    )
    experience_level: str = "Unknown"
    key_achievements: List[Any] = Field(default_factory=list)
    domain_expertise: List[Any] = Field(default_factory=list)


class MatchAnalysis(BaseModel):
    reasons_for_high_match_score: List[str] = Field(default_factory=list)
    reasons_for_moderate_match_score: List[str] = Field(default_factory=list)
    reasons_for_low_match_score: List[str] = Field(default_factory=list)


class MatchResult(BaseModel):
    title: str = "Untitled Role"
    match_score: float = 0
    analysis: MatchAnalysis = Field(default_factory=MatchAnalysis)


class Recommendation(BaseModel):
    final_recommendation: str = ""
    recommendation_timestamp: str = ""
    confidence_level: str = "high"


T = TypeVar("T", bound=BaseModel)
