"""Detecção de saudações e conversa casual (small talk).

O `DeterministicMedicalProvider` só sabe responder perguntas clínicas
casadas contra a base de FAQs curadas — por design, qualquer coisa fora
disso cai no aviso de segurança genérico (`SAFE_FALLBACK`). Isso é correto
para perguntas clínicas fora de escopo, mas ruim de experiência quando o
usuário só está cumprimentando ("oi", "bom dia"). Este módulo intercepta
esses casos antes do pipeline clínico e devolve uma resposta de boas-vindas
com exemplos de perguntas que o assistente sabe responder.
"""
from __future__ import annotations

import re

GREETING_PATTERNS = [
    r"^\s*(oi+|ol[aá]+|e a[ií]|eae|opa)\s*[!.?]*\s*$",
    r"^\s*(bom\s*dia|boa\s*tarde|boa\s*noite)\s*[!.?]*\s*$",
    r"^\s*(hey|hi|hello)\s*[!.?]*\s*$",
    r"^\s*(tudo\s*bem\??|como\s*vai\??|beleza\??)\s*[!.?]*\s*$",
    r"^\s*(obrigad[oa]|valeu|vlw)\s*[!.?]*\s*$",
    r"^\s*(tchau|at[ée]\s*mais|falou)\s*[!.?]*\s*$",
    r"^\s*(quem\s*[ée]\s*voc[eê]|o\s*que\s*voc[eê]\s*faz|o\s*que\s*(voce\s*)?sabe\s*fazer)\??\s*$",
]

_COMPILED = [re.compile(p, flags=re.IGNORECASE) for p in GREETING_PATTERNS]

EXAMPLE_QUESTIONS = [
    "Quando iniciar antibiotico em suspeita de sepse?",
    "Qual a meta glicemica para paciente internado fora da UTI?",
    "Quais os criterios do qSOFA?",
    "Qual o tempo porta-balao recomendado em caso de IAM com supra de ST?",
]


def is_smalltalk(question: str) -> bool:
    return any(p.match(question) for p in _COMPILED)


def build_greeting_reply(patient_context: str | None = None) -> str:
    examples = "\n".join(f"  - {q}" for q in EXAMPLE_QUESTIONS)
    reply = (
        "Olá! Sou o assistente médico virtual de apoio à decisão clínica, "
        "treinado com os protocolos internos do hospital.\n\n"
        "Posso responder perguntas sobre condutas clínicas, exames e "
        "critérios de protocolo, sempre citando a fonte usada. Não "
        "prescrevo diretamente — toda sugestão requer validação de um "
        "médico responsável.\n\n"
        "Exemplos de perguntas que sei responder:\n" + examples
    )
    if patient_context:
        reply += f"\n\nContexto do paciente considerado: {patient_context}"
    return reply
