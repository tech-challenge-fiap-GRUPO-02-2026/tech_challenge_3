"""Logging estruturado de auditoria para toda interação do assistente.

Cada evento é gravado em JSON Lines (`logs/audit_log.jsonl`), permitindo
rastrear: pergunta, paciente consultado, fontes usadas (explainability),
flags de segurança acionadas e se a resposta foi sinalizada como
rascunho sujeito a validação humana. Essencial para auditoria clínica e
para o requisito de "logging detalhado" do desafio.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
LOG_PATH = BASE_DIR / "logs" / "audit_log.jsonl"

logger = logging.getLogger("medical_assistant.audit")
logger.setLevel(logging.INFO)


class AuditLogger:
    def __init__(self, log_path: Path = LOG_PATH):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_interaction(
        self,
        *,
        question: str,
        patient_id: str | None,
        answer: str,
        sources: list[str],
        guardrail_flags: list[str],
        requires_human_validation: bool = True,
        extra: dict | None = None,
    ) -> str:
        event_id = str(uuid.uuid4())
        record = {
            "event_id": event_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "patient_id": patient_id,
            "question": question,
            "answer": answer,
            "sources": sources,
            "guardrail_flags": guardrail_flags,
            "requires_human_validation": requires_human_validation,
            "extra": extra or {},
        }
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("interaction_logged event_id=%s patient_id=%s flags=%s", event_id, patient_id, guardrail_flags)
        return event_id

    def log_flow_event(self, *, flow_name: str, node: str, patient_id: str, payload: dict) -> str:
        event_id = str(uuid.uuid4())
        record = {
            "event_id": event_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "flow_event",
            "flow_name": flow_name,
            "node": node,
            "patient_id": patient_id,
            "payload": payload,
        }
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("flow_event_logged event_id=%s flow=%s node=%s", event_id, flow_name, node)
        return event_id

    def read_recent(self, n: int = 20) -> list[dict]:
        if not self.log_path.exists():
            return []
        lines = self.log_path.read_text(encoding="utf-8").splitlines()
        return [json.loads(l) for l in lines[-n:]]
