"""Hybrid retrieval over the existing multimodal ingestion collection."""

import logging
from hashlib import sha256
import json

from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class QueryVariations(BaseModel):
    queries: list[str]


def _with_retrieval_id(doc: Document) -> Document:
    """Keep equally worded chunks with different original assets distinct."""
    identity = doc.id or sha256(
        json.dumps([doc.page_content, doc.metadata], sort_keys=True).encode()
    ).hexdigest()
    return Document(
        id=doc.id,
        page_content=doc.page_content,
        metadata={**doc.metadata, "_retrieval_id": identity},
    )


class HybridRetrievalPipeline:
    def __init__(self, vector_store, llm, reranker):
        self.vector_store = vector_store
        self.llm_with_tools = llm.with_structured_output(
            QueryVariations, method="json_mode"
        )
        self.reranker = reranker

        # Reuse the summaries/text and original_content metadata written by
        # Ingestion_Pipeline.ipynb. No second collection or re-ingestion needed.
        raw = vector_store.get(include=["documents", "metadatas"])
        documents = [
            _with_retrieval_id(Document(id=doc_id, page_content=content, metadata=meta or {}))
            for doc_id, content, meta in zip(
                raw["ids"], raw["documents"], raw["metadatas"]
            )
            if content and content.strip()
        ]
        if not documents:
            raise ValueError(
                "The multimodal collection has no searchable chunks. "
                "Run Ingestion_Pipeline.ipynb first and check CHROMA_PERSIST_DIR."
            )

        bm25 = BM25Retriever.from_documents(documents, k=3)
        self.vector_retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.3},
        )
        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[RunnableLambda(self._vector_search), bm25],
            weights=[0.7, 0.3],
            id_key="_retrieval_id",
        )

    def _vector_search(self, query: str) -> list[Document]:
        return [_with_retrieval_id(doc) for doc in self.vector_retriever.invoke(query)]

    def generate_query_variations(self, query: str) -> list[str]:
        prompt = f"""Generate exactly 3 alternative search queries for retrieving
relevant document chunks. Return only a valid JSON object with the schema
{{"queries": ["variation 1", "variation 2", "variation 3"]}}.
Original query: {query}"""
        try:
            response = self.llm_with_tools.invoke(prompt)
            variations = [item.strip() for item in response.queries if item.strip()]
        except Exception as exc:
            logger.warning("Query expansion failed; using original query: %s", exc)
            variations = []
        # Always search the original too, so expansion cannot discard exact terms.
        return list(dict.fromkeys([query, *variations[:3]]))

    def hybrid_search(self, queries: list[str]) -> list[Document]:
        candidates = []
        seen = set()
        for query in queries:
            try:
                documents = self.hybrid_retriever.invoke(query)
            except Exception as exc:
                logger.warning("Retrieval failed for %r: %s", query, exc)
                continue
            for doc in documents:
                identity = doc.metadata["_retrieval_id"]
                if identity not in seen:
                    seen.add(identity)
                    candidates.append(doc)
        return candidates

    def chunks_reranker(self, query: str) -> list[Document]:
        candidates = self.hybrid_search(self.generate_query_variations(query))
        if not candidates:
            return []
        try:
            return list(self.reranker.compress_documents(documents=candidates, query=query))
        except Exception as exc:
            logger.warning("Reranking failed; using first three candidates: %s", exc)
            return candidates[:3]


def create_pipeline(vector_store, llm) -> HybridRetrievalPipeline:
    cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=3)
    return HybridRetrievalPipeline(vector_store, llm, reranker)
