"""Run a small, reproducible retrieval evaluation against local PDFs.

Dataset JSONL rows: {"question": "...", "source": "file.pdf", "page": 2}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from document_chat import RagIndex, extract_pdfs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdfs", nargs="+", type=Path)
    parser.add_argument("--cases", required=True, type=Path,
                        help="JSONL with question, source filename, and one-based page")
    parser.add_argument("--method", choices=["hybrid", "semantic", "keyword"], default="hybrid")
    args = parser.parse_args()
    chunks, warnings = extract_pdfs([(path.name, path.read_bytes()) for path in args.pdfs])
    for warning in warnings:
        print("Warning:", warning)
    index = RagIndex.build(chunks)
    cases = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases:
        parser.error("No evaluation cases found.")
    successes = 0
    reciprocal = 0.0
    for case in cases:
        hits = index.search(case["question"], args.method)
        rank = next((i for i, hit in enumerate(hits, 1)
                     if hit.chunk.source == case["source"] and hit.chunk.page == case["page"]), None)
        successes += rank is not None
        reciprocal += 1 / rank if rank else 0
        print(json.dumps({"question": case["question"], "rank": rank,
                          "retrieved": [f"{h.chunk.source}:p{h.chunk.page}" for h in hits]}))
    print(json.dumps({"cases": len(cases), "hit_at_6": successes / len(cases),
                      "mrr_at_6": reciprocal / len(cases)}, indent=2))


if __name__ == "__main__":
    main()
