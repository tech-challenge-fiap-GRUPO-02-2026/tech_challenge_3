"""Indexação e busca semântica sobre os protocolos internos.

Duas estratégias, com a mesma interface (`.search(query, k) -> list[Chunk]`):

  - `TfidfVectorStore`: implementação leve, sem dependências pesadas
    (apenas stdlib), usada como padrão do projeto — garante que o RAG
    funcione em qualquer máquina, inclusive nos testes automatizados.
  - `build_langchain_faiss_store(...)`: constrói um índice FAISS real via
    LangChain + embeddings (OpenAI ou HuggingFace), usado quando essas
    dependências estão disponíveis (ver `src/langchain_app/medical_assistant.py`).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
PROTOCOLS_DIR = BASE_DIR / "data" / "raw" / "protocolos"

TOKEN_RE = re.compile(r"[a-zà-ú0-9]+")


@dataclass
class Chunk:
    text: str
    source: str
    section: str


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def load_protocol_chunks(protocols_dir: Path = PROTOCOLS_DIR) -> list[Chunk]:
    """Carrega e fragmenta (chunking) os protocolos por seção `## `,
    a mesma unidade usada na curadoria do dataset de fine-tuning — mantendo
    consistência entre o que o modelo aprendeu e o que o RAG recupera."""
    chunks: list[Chunk] = []
    for path in sorted(protocols_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title_match = re.search(r"^# (.+)$", text, flags=re.MULTILINE)
        title = title_match.group(1).strip() if title_match else path.stem

        current_section, buffer = None, []
        for line in text.splitlines():
            if line.startswith("## "):
                if current_section and buffer:
                    chunks.append(Chunk("\n".join(buffer).strip(), path.name, f"{title} — {current_section}"))
                current_section, buffer = line[3:].strip(), []
            elif current_section is not None:
                buffer.append(line)
        if current_section and buffer:
            chunks.append(Chunk("\n".join(buffer).strip(), path.name, f"{title} — {current_section}"))
    return chunks


class TfidfVectorStore:
    """Índice TF-IDF em memória com similaridade por cosseno, implementado
    apenas com a biblioteca padrão do Python."""

    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks = chunks if chunks is not None else load_protocol_chunks()
        self._doc_tokens = [_tokenize(c.text) for c in self.chunks]
        self._df = self._document_frequencies(self._doc_tokens)
        self._n_docs = len(self._doc_tokens)
        self._doc_vectors = [self._vectorize(tokens) for tokens in self._doc_tokens]

    @staticmethod
    def _document_frequencies(doc_tokens: list[list[str]]) -> Counter:
        df: Counter = Counter()
        for tokens in doc_tokens:
            df.update(set(tokens))
        return df

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        return math.log((1 + self._n_docs) / (1 + df)) + 1.0

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        tf = Counter(tokens)
        return {term: count * self._idf(term) for term, count in tf.items()}

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        common = set(a) & set(b)
        dot = sum(a[t] * b[t] for t in common)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        query_vec = self._vectorize(_tokenize(query))
        scored = [(chunk, self._cosine(query_vec, doc_vec)) for chunk, doc_vec in zip(self.chunks, self._doc_vectors)]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [pair for pair in scored[:k] if pair[1] > 0]


def build_langchain_faiss_store(chunks: list[Chunk] | None = None):
    """Constrói um vector store FAISS real via LangChain, quando as
    dependências (`langchain`, `faiss-cpu`, embeddings) estão instaladas.
    Usa embeddings da OpenAI se `OPENAI_API_KEY` estiver definida, caso
    contrário usa `sentence-transformers` local via HuggingFaceEmbeddings.
    """
    import os

    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    chunks = chunks if chunks is not None else load_protocol_chunks()
    docs = [Document(page_content=c.text, metadata={"source": c.source, "section": c.section}) for c in chunks]

    if os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings()
    else:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    return FAISS.from_documents(docs, embeddings)
