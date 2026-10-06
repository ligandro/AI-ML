"""Citation validation and source formatting for answers."""

import re

from pydantic import ValidationError

from document_chat.models import AnswerResult, Hit

ABSTAIN = "I could not find enough evidence in the uploaded documents to answer that."


def format_context(hit: Hit, number: int) -> str:
    """Use IDs that cannot be confused with PDF page or bibliography numbers."""
    return (
        f"SOURCE_ID=source_{number}\nFILE={hit.chunk.source}\n"
        f"PDF_PAGE={hit.chunk.page}\nTEXT={hit.chunk.text}"
    )


def render_answer(raw: str, source_count: int) -> str:
    """Reject malformed responses and citations outside the supplied context."""
    try:
        result = AnswerResult.model_validate_json(raw)
    except (ValidationError, ValueError, TypeError):
        return ABSTAIN
    if result.status == "insufficient_evidence":
        return ABSTAIN
    answer = result.answer.strip()
    if not answer or re.search(r"\[\d+\]", answer):
        return ABSTAIN
    citations = list(dict.fromkeys(result.citations))
    if not citations or any(not re.fullmatch(r"source_[1-9]\d*", citation) for citation in citations):
        return ABSTAIN
    numbers = [int(citation.removeprefix("source_")) for citation in citations]
    if any(number > source_count for number in numbers):
        return ABSTAIN
    return answer + " " + " ".join(f"[{number}]" for number in numbers)
