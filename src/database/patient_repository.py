"""Acesso à base estruturada de prontuários (SQLite), já anonimizada.

Simula a integração com o sistema de prontuário eletrônico do hospital
(consulta em base de dados estruturada, exigida pelo desafio). Os dados
usados aqui já passaram pelo pipeline de anonimização
(`src/data_processing/anonymize.py`) — nenhuma coluna de identificação
direta (nome, CPF, telefone, e-mail, endereço) é armazenada.
"""
from __future__ import annotations

import csv
import sqlite3
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
PACIENTES_CSV = BASE_DIR / "data" / "prontuarios" / "pacientes.csv"
EXAMES_CSV = BASE_DIR / "data" / "prontuarios" / "exames.csv"
DB_PATH = BASE_DIR / "artifacts" / "prontuarios.db"


@dataclass
class Patient:
    paciente_id: str
    idade: int
    sexo: str
    diagnostico: str
    alergias: str
    medicacoes_em_uso: str
    status: str


@dataclass
class Exam:
    paciente_id: str
    exame: str
    status: str
    resultado: str
    data_solicitacao: str


def build_database(db_path: Path = DB_PATH, force: bool = False) -> Path:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists() and not force:
        return db_path

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DROP TABLE IF EXISTS pacientes")
        conn.execute("DROP TABLE IF EXISTS exames")
        conn.execute(
            """CREATE TABLE pacientes (
                paciente_id TEXT PRIMARY KEY, idade INTEGER, sexo TEXT,
                diagnostico TEXT, alergias TEXT, medicacoes_em_uso TEXT, status TEXT
            )"""
        )
        conn.execute(
            """CREATE TABLE exames (
                paciente_id TEXT, exame TEXT, status TEXT, resultado TEXT, data_solicitacao TEXT
            )"""
        )
        with PACIENTES_CSV.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
            conn.executemany(
                "INSERT INTO pacientes VALUES (:paciente_id,:idade,:sexo,:diagnostico,:alergias,:medicacoes_em_uso,:status)",
                rows,
            )
        with EXAMES_CSV.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
            conn.executemany(
                "INSERT INTO exames VALUES (:paciente_id,:exame,:status,:resultado,:data_solicitacao)",
                rows,
            )
        conn.commit()
    finally:
        conn.close()
    return db_path


class PatientRepository:
    """Repositório de consulta somente-leitura à base de prontuários."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = build_database(db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_patient(self, paciente_id: str) -> Patient | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM pacientes WHERE paciente_id = ?", (paciente_id,)).fetchone()
            return Patient(**dict(row)) if row else None

    def get_exams(self, paciente_id: str) -> list[Exam]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM exames WHERE paciente_id = ?", (paciente_id,)).fetchall()
            return [Exam(**dict(r)) for r in rows]

    def get_pending_exams(self, paciente_id: str) -> list[Exam]:
        return [e for e in self.get_exams(paciente_id) if e.status == "pendente"]

    def patient_context_summary(self, paciente_id: str) -> str:
        """Monta um resumo textual do estado atual do paciente, usado para
        contextualizar (grounding) as respostas da LLM com informações
        atualizadas — um dos requisitos obrigatórios do desafio."""
        patient = self.get_patient(paciente_id)
        if not patient:
            return f"Paciente {paciente_id} não encontrado na base."

        exams = self.get_exams(paciente_id)
        exams_text = "; ".join(
            f"{e.exame} ({e.status}{', resultado: ' + e.resultado if e.resultado else ''})" for e in exams
        ) or "nenhum exame registrado"

        return (
            f"Paciente {patient.paciente_id}, {patient.idade} anos, sexo {patient.sexo}. "
            f"Diagnóstico atual: {patient.diagnostico}. Status: {patient.status}. "
            f"Alergias: {patient.alergias}. Medicações em uso: {patient.medicacoes_em_uso}. "
            f"Exames: {exams_text}."
        )
