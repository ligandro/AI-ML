"""Data structures shared by ingestion, retrieval, and answer generation."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictStr


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    page: int
    text: str


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    similarity: float


class AnswerResult(BaseModel):
    """The model's response contract; source numbers are checked separately."""

    model_config = ConfigDict(extra="forbid")
    status: Literal["answered", "insufficient_evidence"]
    answer: StrictStr
    citations: list[StrictStr]
