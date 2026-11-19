"""Conversational entry point shared by the CLI and retrieval notebook."""

import logging
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from Process_pipeline import create_pipeline

load_dotenv()
logger = logging.getLogger(__name__)
MAX_HISTORY_MESSAGES = 10
NO_INFORMATION = "I don't have enough information to answer that question based on the provided documents."


class ConversationalRAG:
    def __init__(self, pipeline, model):
        self.pipeline = pipeline
        self.model = model
        self.chat_history = []

    def ask_questions(self, query: str) -> str:
        user_question = query
        if self.chat_history:
            messages = [SystemMessage(content=(
                "Given the chat history, rewrite the new question to be standalone "
                "and searchable. Just return the rewritten question only."
            )), *self.chat_history, HumanMessage(content=f"New question: {query}")]
            try:
                user_question = self.model.invoke(messages).content.strip() or query
            except Exception as exc:
                logger.warning("Question rewriting failed; using original question: %s", exc)

        results = self.pipeline.chunks_reranker(user_question)
        if not results:
            return NO_INFORMATION

        context = "\n\n".join(
            f"Document {i}:\n{doc.page_content}" for i, doc in enumerate(results, 1)
        )
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=(
                f"Based on the following documents, answer this question: {user_question}\n"
                f"Documents:\n{context}\n"
                "Use only information from these documents. If you cannot find the answer, "
                f'say "{NO_INFORMATION}"'
            )),
        ]
        try:
            answer = self.model.invoke(messages).content
        except Exception as exc:
            logger.warning("Answer generation failed: %s", exc)
            return "Sorry, something went wrong while generating the answer. Please try again."

        self.chat_history.extend([HumanMessage(content=query), AIMessage(content=answer)])
        del self.chat_history[:-MAX_HISTORY_MESSAGES]
        return answer


_default_session = None


def ask_questions(query: str) -> str:
    global _default_session
    if _default_session is None:
        embedding_model = HuggingFaceEmbeddings(
            model_name=os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
        )
        vector_store = Chroma(
            persist_directory=os.getenv("CHROMA_PERSIST_DIR", "db/chroma_db"),
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"},
        )
        model = ChatGroq(model=os.getenv(
            "LLM_MODEL_NAME", "meta-llama/llama-4-scout-17b-16e-instruct"
        ))
        _default_session = ConversationalRAG(create_pipeline(vector_store, model), model)
    answer = _default_session.ask_questions(query)
    print(f"Answer:\n{answer}")
    return answer


def main():
    while True:
        query = input("Enter the question (q to quit):")
        if query.lower() == "q":
            break
        ask_questions(query)


if __name__ == "__main__":
    main()
