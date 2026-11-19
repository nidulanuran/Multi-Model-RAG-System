"""Offline retrieval integration tests: no Groq calls or model downloads."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from Process_pipeline import HybridRetrievalPipeline
from Retrieval_Pipeline import ConversationalRAG, NO_INFORMATION


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.docs = [
            Document(id="figure", page_content="attention architecture diagram", metadata={
                "original_content": json.dumps({
                    "raw_text": "Transformer architecture",
                    "tables_html": [], "images_base64": ["original-image-payload"],
                })
            }),
            Document(id="table", page_content="BLEU benchmark results table", metadata={
                "original_content": json.dumps({
                    "raw_text": "Evaluation scores", "tables_html": ["<table>28.4</table>"],
                    "images_base64": [],
                })
            }),
            Document(id="text", page_content="training optimizer schedule", metadata={}),
            Document(id="other", page_content="translation language vocabulary", metadata={}),
        ]
        self.store = Mock()
        self.store.get.return_value = {
            "ids": [doc.id for doc in self.docs],
            "documents": [doc.page_content for doc in self.docs],
            "metadatas": [doc.metadata for doc in self.docs],
        }
        # Dense search only finds the diagram; BM25 must supply the table.
        self.store.as_retriever.return_value = RunnableLambda(lambda query: [self.docs[0]])
        self.llm = Mock()
        self.llm.with_structured_output.return_value.invoke.return_value = SimpleNamespace(
            queries=["BLEU benchmark", "translation scores", "evaluation table"]
        )
        self.reranker = Mock()
        self.reranker.compress_documents.side_effect = lambda documents, query: documents[:3]
        self.pipeline = HybridRetrievalPipeline(self.store, self.llm, self.reranker)

    def test_hybrid_search_adds_keyword_matches_and_preserves_original_assets(self):
        results = self.pipeline.chunks_reranker("BLEU")
        self.assertIn("figure", [doc.id for doc in results])
        self.assertIn("table", [doc.id for doc in results])
        for doc in results:
            original = next(item for item in self.docs if item.id == doc.id)
            self.assertEqual(doc.metadata.get("original_content"), original.metadata.get("original_content"))
        self.assertEqual(len({doc.id for doc in results}), len(results))
        self.assertEqual(self.reranker.compress_documents.call_args.kwargs["query"], "BLEU")
        self.store.as_retriever.assert_called_once_with(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.3},
        )

    def test_original_query_is_kept_and_expansion_is_bounded(self):
        self.llm.with_structured_output.return_value.invoke.return_value = SimpleNamespace(
            queries=["BLEU", "", "  scores  ", "scores", "extra"]
        )
        self.assertEqual(self.pipeline.generate_query_variations("BLEU"), ["BLEU", "scores"])

    def test_query_and_reranking_failures_fall_back(self):
        self.llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("offline")
        self.assertEqual(self.pipeline.generate_query_variations("BLEU"), ["BLEU"])
        self.reranker.compress_documents.side_effect = RuntimeError("reranker unavailable")
        results = self.pipeline.chunks_reranker("BLEU")
        self.assertTrue(results)
        self.assertLessEqual(len(results), 3)

    def test_empty_or_blank_collection_has_actionable_error(self):
        for texts in ([], ["  "]):
            self.store.get.return_value = {
                "ids": ["empty"] * len(texts), "documents": texts,
                "metadatas": [None] * len(texts),
            }
            with self.assertRaisesRegex(ValueError, "Run Ingestion_Pipeline.ipynb"):
                HybridRetrievalPipeline(self.store, self.llm, self.reranker)

    def test_same_summary_with_different_assets_is_not_deduplicated(self):
        same_text = "identical searchable summary"
        self.store.get.return_value = {
            "ids": ["one", "two"], "documents": [same_text, same_text],
            "metadatas": [{"original_content": "asset-one"}, {"original_content": "asset-two"}],
        }
        pipeline = HybridRetrievalPipeline(self.store, self.llm, self.reranker)
        self.assertEqual(
            {doc.id for doc in pipeline.hybrid_search([same_text])}, {"one", "two", "figure"}
        )

    def test_failed_retrieval_does_not_call_reranker_with_empty_candidates(self):
        self.pipeline.hybrid_retriever = Mock()
        self.pipeline.hybrid_retriever.invoke.side_effect = RuntimeError("store unavailable")
        self.assertEqual(self.pipeline.chunks_reranker("BLEU"), [])
        self.reranker.compress_documents.assert_not_called()

    def test_chroma_collection_is_read_without_changing_ingestion_data(self):
        from langchain_chroma import Chroma
        from langchain_core.embeddings import Embeddings
        from uuid import uuid4

        class OfflineEmbeddings(Embeddings):
            def embed_documents(self, texts):
                return [self.embed_query(text) for text in texts]

            def embed_query(self, text):
                return [1.0, float("BLEU" in text), float("attention" in text)]

        store = Chroma(collection_name=f"retrieval-test-{uuid4().hex}", embedding_function=OfflineEmbeddings())
        try:
            store.add_documents(self.docs)
            before = store.get(include=["documents", "metadatas"])
            pipeline = HybridRetrievalPipeline(store, self.llm, self.reranker)
            results = pipeline.chunks_reranker("BLEU")
            self.assertIn("table", [doc.id for doc in results])
            self.assertEqual(store.get(include=["documents", "metadatas"]), before)
        finally:
            store.delete_collection()


class ConversationTests(unittest.TestCase):
    def test_rewritten_question_reaches_retrieval_and_answer_with_bounded_history(self):
        model = Mock()
        pipeline = Mock()
        pipeline.chunks_reranker.return_value = [Document(page_content="BLEU is 28.4")]
        session = ConversationalRAG(pipeline, model)
        session.chat_history = [SimpleNamespace(content="previous question")] * 10
        model.invoke.side_effect = [SimpleNamespace(content="What is the Transformer BLEU score?"), SimpleNamespace(content="28.4")]
        self.assertEqual(session.ask_questions("What is its score?"), "28.4")
        pipeline.chunks_reranker.assert_called_once_with("What is the Transformer BLEU score?")
        prompt = model.invoke.call_args.args[0][-1].content
        self.assertIn("What is the Transformer BLEU score?", prompt)
        self.assertIn("BLEU is 28.4", prompt)
        self.assertEqual(len(session.chat_history), 10)

    def test_no_candidates_skips_answer_generation(self):
        model, pipeline = Mock(), Mock()
        pipeline.chunks_reranker.return_value = []
        self.assertEqual(ConversationalRAG(pipeline, model).ask_questions("anything"), NO_INFORMATION)
        model.invoke.assert_not_called()

    def test_rewrite_failure_uses_original_question_and_sessions_are_separate(self):
        model, pipeline = Mock(), Mock()
        pipeline.chunks_reranker.return_value = [Document(page_content="answer context")]
        session = ConversationalRAG(pipeline, model)
        session.chat_history = [SimpleNamespace(content="history")]
        model.invoke.side_effect = [RuntimeError("offline"), SimpleNamespace(content="answer")]
        self.assertEqual(session.ask_questions("follow up"), "answer")
        pipeline.chunks_reranker.assert_called_once_with("follow up")
        self.assertEqual(ConversationalRAG(pipeline, model).chat_history, [])


if __name__ == "__main__":
    unittest.main()
