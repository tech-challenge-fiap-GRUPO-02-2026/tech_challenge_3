"""Camada de segurança e validação do assistente médico.

Implementa os limites de atuação exigidos pelo desafio:
  - o assistente NUNCA prescreve/emite conduta definitiva sem validação
    humana explícita;
  - toda resposta clínica é marcada como sugestão sujeita a revisão;
  - um bloqueio duro (hard block) impede respostas que soem como ordem
    médica direta e não qualificada (ex.: "administre X mg agora").
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

MANDATORY_DISCLAIMER = (
    "\n\n[AVISO] Esta é uma sugestão de apoio à decisão gerada por IA, "
    "baseada nos protocolos internos citados. Não substitui o julgamento "
    "clínico. Toda conduta definitiva requer validação de um médico "
    "responsável antes da execução."
)

# Padrões que caracterizam uma ordem médica direta e não qualificada,
# que o assistente nunca deve emitir sem contexto de rascunho/validação.
DIRECT_PRESCRIPTION_PATTERNS = [
    r"\badministre\b",
    r"\bprescrevo\b",
    r"\baplique\s+\d",
    r"\btome\s+\d",
    r"\binjete\b",
]

BLOCKED_TOPICS = [
    r"\bdose\s+letal\b",
    r"\bautomedica[cç][ãa]o\s+sem\s+acompanhamento\b",
]


@dataclass
class GuardrailResult:
    allowed: bool
    reason: str | None = None
    sanitized_text: str | None = None
    flags: list[str] = field(default_factory=list)


class MedicalGuardrails:
    """Valida entrada e saída do assistente contra os limites de atuação
    definidos no protocolo de segurança do hospital."""

    def check_input(self, question: str) -> GuardrailResult:
        for pattern in BLOCKED_TOPICS:
            if re.search(pattern, question, flags=re.IGNORECASE):
                return GuardrailResult(
                    allowed=False,
                    reason="Pergunta bloqueada: fora do escopo de apoio clínico seguro.",
                    flags=["blocked_topic"],
                )
        return GuardrailResult(allowed=True)

    def review_output(self, answer: str) -> GuardrailResult:
        flags = []
        for pattern in DIRECT_PRESCRIPTION_PATTERNS:
            if re.search(pattern, answer, flags=re.IGNORECASE):
                flags.append(f"linguagem_prescritiva_direta:{pattern}")

        sanitized = answer
        if flags:
            sanitized = re.sub(
                r"|".join(DIRECT_PRESCRIPTION_PATTERNS),
                "considere (sob avaliação médica)",
                sanitized,
                flags=re.IGNORECASE,
            )

        if MANDATORY_DISCLAIMER.strip() not in sanitized:
            sanitized = sanitized.rstrip() + MANDATORY_DISCLAIMER

        return GuardrailResult(allowed=True, sanitized_text=sanitized, flags=flags)
