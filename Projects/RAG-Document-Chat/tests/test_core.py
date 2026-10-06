import unittest

from document_chat import (
    ABSTAIN, BM25, Chunk, Hit, RagIndex, chunk_page,
    format_context, reciprocal_rank_fusion, render_answer,
)


class FakeCollection:
    def query(self, **kwargs):
        return {"ids": [["a", "b"]], "distances": [[0.1, 0.8]]}


class FakeClient:
    def __init__(self, answer='{"status":"answered","answer":"It is pneumonia.","citations":["source_1"]}'):
        self.response = type("Response", (), {"message": type("Message", (), {"content": answer})()})()
        self.chat_kwargs = None

    def embed(self, **kwargs):
        return type("Embedding", (), {"embeddings": [[0.1, 0.2]]})()

    def chat(self, **kwargs):
        self.chat_kwargs = kwargs
        return self.response


class CoreTests(unittest.TestCase):
    def test_chunking_preserves_source_page_and_overlap(self):
        text = " ".join(f"word{i}" for i in range(600))
        chunks = chunk_page(text, "paper.pdf", 3, "hash")
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(c.source == "paper.pdf" and c.page == 3 for c in chunks))
        self.assertEqual(len({c.id for c in chunks}), len(chunks))
        self.assertTrue(set(chunks[0].text.split()) & set(chunks[1].text.split()))

    def test_bm25_finds_exact_term(self):
        docs = [Chunk("a", "a.pdf", 1, "radiology pneumonia finding"),
                Chunk("b", "b.pdf", 1, "football match result")]
        result = BM25(docs).search("pneumonia", 2)
        self.assertEqual(result[0][0], 0)
        self.assertEqual(len(result), 1)

    def test_rrf_rewards_agreement(self):
        scores = reciprocal_rank_fusion([["a", "b"], ["b", "c"]])
        self.assertGreater(scores["b"], scores["a"])

    def test_answer_requires_valid_citation(self):
        chunks = [Chunk("a", "a.pdf", 2, "The finding is pneumonia."),
                  Chunk("b", "b.pdf", 3, "Football match result.")]
        index = RagIndex(chunks, FakeCollection(), FakeClient(), None)
        answer, hits = index.answer("What is the pneumonia finding?", "hybrid")
        self.assertEqual(answer, "It is pneumonia. [1]")
        self.assertEqual(hits[0].chunk.page, 2)
        self.assertEqual(index.client.chat_kwargs["format"]["type"], "object")
        index.client = FakeClient('{"status":"answered","answer":"It is pneumonia.","citations":["source_99"]}')
        answer, _ = index.answer("What is the pneumonia finding?", "hybrid")
        self.assertEqual(answer, ABSTAIN)

    def test_structured_answer_abstains_on_bad_or_missing_evidence(self):
        self.assertEqual(render_answer("not json", 1), ABSTAIN)
        self.assertEqual(render_answer('{"status":"answered","answer":"Claim","citations":[]}', 1), ABSTAIN)
        self.assertEqual(render_answer('{"status":"insufficient_evidence","answer":"","citations":[]}', 1), ABSTAIN)
        self.assertEqual(render_answer('{"status":"answered","answer":"Claim","citations":["source_2"]}', 1), ABSTAIN)
        self.assertEqual(render_answer('{"status":"answered","answer":"Claim","citations":["24"]}', 1), ABSTAIN)
        self.assertEqual(render_answer('{"status":"answered","answer":"Claim","citations":[24]}', 1), ABSTAIN)

    def test_context_format_includes_source_id_filename_and_page(self):
        hit = Hit(Chunk("a", "paper.pdf", 3, "Evidence passage."), 1.0, 0.9)
        self.assertEqual(
            format_context(hit, 2),
            "SOURCE_ID=source_2\nFILE=paper.pdf\nPDF_PAGE=3\nTEXT=Evidence passage.",
        )

    def test_answer_sends_all_retrieved_hits_to_model(self):
        chunks = [
            Chunk("a", "a.pdf", 2, "The finding is pneumonia."),
            Chunk("b", "b.pdf", 3, "Pneumonia is discussed in the report."),
        ]
        index = RagIndex(chunks, FakeCollection(), FakeClient(), None)
        answer, hits = index.answer("What is the pneumonia finding?", "hybrid")
        prompt = index.client.chat_kwargs["messages"][1]["content"]
        self.assertEqual(answer, "It is pneumonia. [1]")
        self.assertEqual(len(hits), 2)
        self.assertIn("SOURCE_ID=source_2", prompt)


if __name__ == "__main__":
    unittest.main()
