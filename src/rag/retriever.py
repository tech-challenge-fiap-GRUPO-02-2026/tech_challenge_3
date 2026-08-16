"""Interface única de recuperação (retrieval), abstraindo TF-IDF local ou
FAISS/LangChain, usada pelo assistente para compor contexto e citações."""
from __future__ import annotations

from dataclasses import dataclass

from src.rag.vector_store import TfidfVectorStore, build_langchain_faiss_store


@dataclass
class RetrievedPassage:
    text: str
    source: str
    section: str
    score: float


class ProtocolRetriever:
    """Retriever padrão do projeto — TF-IDF local, sem dependências
    externas. Usado por padrão para garantir reprodutibilidade e execução
    sem custos de API."""

    def __init__(self):
        self.store = TfidfVectorStore()

    def retrieve(self, query: str, k: int = 3) -> list[RetrievedPassage]:
        results = self.store.search(query, k=k)
        return [
            RetrievedPassage(text=chunk.text, source=chunk.source, section=chunk.section, score=round(score, 3))
            for chunk, score in results
        ]


class LangChainFAISSRetriever:
    """Retriever alternativo usando LangChain + FAISS (requer dependências
    extra — ver requirements-langchain.txt). Mesma interface do retriever
    padrão, para troca transparente em `medical_assistant.py`."""

    def __init__(self):
        self.vector_store = build_langchain_faiss_store()

    def retrieve(self, query: str, k: int = 3) -> list[RetrievedPassage]:
        docs_and_scores = self.vector_store.similarity_search_with_score(query, k=k)
        return [
            RetrievedPassage(
                text=doc.page_content,
                source=doc.metadata.get("source", "desconhecida"),
                section=doc.metadata.get("section", ""),
                score=round(float(score), 3),
            )
            for doc, score in docs_and_scores
        ]


def get_retriever(prefer_langchain: bool = False):
    if prefer_langchain:
        try:
            return LangChainFAISSRetriever()
        except Exception:
            pass
    return ProtocolRetriever()
