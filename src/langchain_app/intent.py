"""Detecção de intenção de "rodar o fluxo clínico automatizado" a partir
de texto livre.

Sem isso, uma mensagem como "Rodar fluxo clínico automatizado para
PAC-0005" digitada diretamente no chat (em vez de usar o botão dedicado)
seria tratada como pergunta clínica comum, não bateria com nada na base
curada e cairia no aviso de segurança genérico — confuso para quem só
queria executar o fluxo. Este módulo reconhece esse pedido e extrai o
`paciente_id` da própria mensagem, permitindo que `MedicalAssistant.ask()`
acione o LangGraph diretamente.
"""
from __future__ import annotations

import re

FLOW_INTENT_RE = re.compile(
    r"\b(rodar|executar|iniciar|acionar)\b.{0,40}\bfluxo\b", flags=re.IGNORECASE
)
PATIENT_ID_RE = re.compile(r"\bPAC-\d{3,}\b", flags=re.IGNORECASE)


def is_flow_request(text: str) -> bool:
    return bool(FLOW_INTENT_RE.search(text))


def extract_patient_id(text: str) -> str | None:
    match = PATIENT_ID_RE.search(text)
    if not match:
        return None
    return match.group(0).upper()
