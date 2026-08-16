"""Formatação de citações e explicações para as respostas do assistente.

Garante o requisito de explainability: toda resposta deve indicar a fonte
da informação utilizada (protocolo, seção e paciente consultado).
"""
from __future__ import annotations

from dataclasses import dataclass

from src.rag.retriever import RetrievedPassage


@dataclass
class ExplainedAnswer:
    answer: str
    citations: list[str]
    patient_context_used: str | None


MIN_RAG_SCORE = 0.12  # abaixo disso, o trecho de protocolo é ruído — não citar


def format_citations(passages: list[RetrievedPassage], min_score: float = MIN_RAG_SCORE) -> list[str]:
    return [
        f"{p.source} — {p.section} (similaridade: {p.score})"
        for p in passages
        if p.score >= min_score
    ]


def render_answer_with_sources(
    raw_answer: str,
    passages: list[RetrievedPassage],
    patient_summary: str | None,
    primary_source: str | None = None,
) -> ExplainedAnswer:
    """Monta a resposta final com citações.

    `primary_source` é a fonte exata de onde a resposta foi tirada (ex.:
    o FAQ interno casado, ou "PubMedQA"), reportada pelo próprio provider
    (`answer_with_source`). Os trechos de RAG só entram como "contexto
    adicional" quando têm similaridade relevante (>= `MIN_RAG_SCORE`) —
    caso contrário citar um protocolo não relacionado seria enganoso.
    """
    rag_citations = format_citations(passages)
    footer_lines = []
    if primary_source:
        footer_lines.append(f"Fonte principal: {primary_source}")
    if rag_citations:
        footer_lines.append("Contexto adicional (protocolos internos):")
        footer_lines.extend(f"  - {c}" for c in rag_citations)
    if patient_summary:
        footer_lines.append(f"Contexto do paciente considerado: {patient_summary}")

    footer = ("\n\n" + "\n".join(footer_lines)) if footer_lines else ""
    all_citations = ([primary_source] if primary_source else []) + rag_citations
    return ExplainedAnswer(answer=raw_answer + footer, citations=all_citations, patient_context_used=patient_summary)
