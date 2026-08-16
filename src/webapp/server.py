"""Backend HTTP local do Assistente Médico Virtual — sem dependências
externas (usa apenas `http.server` da biblioteca padrão do Python), para
que o front-end de chat (`static/index.html`) possa perguntar direto no
navegador.

Endpoints:
    GET  /                  -> interface de chat (index.html)
    GET  /api/patients      -> lista de pacientes de exemplo
    POST /api/ask           -> {question, patient_id?} -> resposta do assistente
    POST /api/flow          -> {patient_id} -> execução do fluxo clínico (LangGraph)
    GET  /api/logs?n=10     -> últimos eventos de auditoria

Uso:
    py -m src.webapp.server [--port 8765]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.database.patient_repository import PatientRepository
from src.langchain_app.medical_assistant import MedicalAssistant
from src.langgraph_flow.clinical_flow import run_clinical_flow
from src.safety.audit_logger import AuditLogger

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Carregado uma única vez no início do processo — evita recarregar
# modelo/índice a cada pergunta.
_assistant: MedicalAssistant | None = None
_repo: PatientRepository | None = None
_audit: AuditLogger | None = None


def get_assistant() -> MedicalAssistant:
    global _assistant
    if _assistant is None:
        _assistant = MedicalAssistant()
    return _assistant


def get_repo() -> PatientRepository:
    global _repo
    if _repo is None:
        _repo = PatientRepository()
    return _repo


def get_audit() -> AuditLogger:
    global _audit
    if _audit is None:
        _audit = AuditLogger()
    return _audit


class Handler(BaseHTTPRequestHandler):
    server_version = "AssistenteMedico/1.0"

    def log_message(self, fmt, *args):  # silencia log padrão barulhento
        sys.stderr.write(f"[server] {self.address_string()} - {fmt % args}\n")

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw or b"{}")

    # ---------------------------------------------------------- GET
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            return

        if self.path.startswith("/api/patients"):
            self._send_json({"patients": _load_patients_summary()})
            return

        if self.path.startswith("/api/pubmedqa-example"):
            self._send_json({"examples": _load_pubmedqa_examples(n=3)})
            return

        if self.path.startswith("/api/logs"):
            n = 10
            if "?" in self.path:
                query = self.path.split("?", 1)[1]
                for part in query.split("&"):
                    if part.startswith("n="):
                        n = int(part[2:])
            audit = get_audit()
            self._send_json({"logs": audit.read_recent(n)})
            return

        self._send_json({"error": "not found"}, status=404)

    # ---------------------------------------------------------- POST
    def do_POST(self):
        if self.path == "/api/ask":
            body = self._read_json_body()
            question = body.get("question", "").strip()
            patient_id = body.get("patient_id") or None
            if not question:
                self._send_json({"error": "campo 'question' obrigatório"}, status=400)
                return
            assistant = get_assistant()
            response = assistant.ask(question, patient_id=patient_id)
            self._send_json(
                {
                    "answer": response.answer,
                    "citations": response.citations,
                    "patient_context": response.patient_context,
                    "guardrail_flags": response.guardrail_flags,
                    "blocked": response.blocked,
                    "event_id": response.event_id,
                    "flow_triggered": response.flow_triggered,
                }
            )
            return

        if self.path == "/api/flow":
            body = self._read_json_body()
            patient_id = body.get("patient_id", "").strip()
            if not patient_id:
                self._send_json({"error": "campo 'patient_id' obrigatório"}, status=400)
                return
            state = run_clinical_flow(patient_id, prefer_langgraph=True)
            self._send_json(dict(state))
            return

        self._send_json({"error": "not found"}, status=404)


def _load_patients_summary() -> list[dict]:
    base = Path(__file__).resolve().parents[2]
    path = base / "data" / "prontuarios" / "pacientes.csv"
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _load_pubmedqa_examples(n: int = 3) -> list[dict]:
    """Retorna as N primeiras perguntas da amostra do PubMedQA (já
    traduzidas para PT-BR por `download_pubmedqa.py`), usadas pelo
    front-end para popular os botões de pergunta rápida sem hardcoded
    text em inglês na página."""
    base = Path(__file__).resolve().parents[2]
    path = base / "data" / "raw" / "pubmedqa" / "pubmedqa_sample.jsonl"
    if not path.exists():
        return []
    examples = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
            if len(examples) >= n:
                break
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Backend do Assistente Médico Virtual")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="Não abrir o navegador automaticamente")
    args = parser.parse_args()

    # Pré-carrega o assistente (RAG + provider + prontuários) antes de
    # abrir o servidor, para a primeira pergunta do usuário já ser rápida.
    print("Carregando assistente médico (RAG + prontuários + provider)...")
    get_assistant()
    get_repo()
    get_audit()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Assistente Médico Virtual rodando em {url}")
    print("Pressione Ctrl+C para encerrar.")

    if not args.no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nEncerrando servidor...")
        server.shutdown()


if __name__ == "__main__":
    main()
