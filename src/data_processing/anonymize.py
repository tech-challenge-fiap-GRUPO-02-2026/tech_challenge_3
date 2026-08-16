"""Anonimização de dados clínicos sensíveis (PII/PHI).

Remove ou mascara identificadores diretos (nome, CPF, telefone, e-mail,
endereço, convênio) de registros de prontuário, preservando apenas os
campos clínicos necessários para o assistente e para o fine-tuning.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

CPF_RE = re.compile(r"\d{3}\.\d{3}\.\d{3}-\d{2}")
PHONE_RE = re.compile(r"\(\d{2}\)\s?\d{4,5}-\d{4}")
EMAIL_RE = re.compile(r"[\w\.\-]+@[\w\-]+\.[\w\.\-]+")

# Colunas que identificam diretamente uma pessoa e nunca devem chegar
# ao dataset de fine-tuning ou à base consultada pelo assistente.
DIRECT_IDENTIFIER_COLUMNS = {
    "nome_completo",
    "cpf",
    "telefone",
    "email",
    "endereco",
    "convenio",
}

CLINICAL_COLUMNS = [
    "paciente_id",
    "idade",
    "sexo",
    "diagnostico",
    "alergias",
    "medicacoes_em_uso",
]


@dataclass
class AnonymizationReport:
    total_registros: int
    colunas_removidas: list[str]
    ocorrencias_pii_texto_livre: int


def scrub_free_text(text: str) -> str:
    """Mascara CPF, telefone e e-mail que apareçam em texto livre."""
    text = CPF_RE.sub("[CPF_REMOVIDO]", text)
    text = PHONE_RE.sub("[TELEFONE_REMOVIDO]", text)
    text = EMAIL_RE.sub("[EMAIL_REMOVIDO]", text)
    return text


def count_pii_occurrences(text: str) -> int:
    return (
        len(CPF_RE.findall(text))
        + len(PHONE_RE.findall(text))
        + len(EMAIL_RE.findall(text))
    )


def anonymize_prontuarios(raw_csv: Path, output_csv: Path) -> AnonymizationReport:
    """Lê prontuários brutos (com PII) e grava versão curada/anonimizada.

    Estratégia:
    1. Remoção estrutural de colunas identificadoras diretas.
    2. Mantém apenas o `paciente_id` (pseudônimo já presente na fonte) como
       chave de junção com a base de exames.
    3. Varredura por regex em campos de texto livre remanescentes, como
       segunda camada de defesa contra vazamento de PII.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    pii_hits = 0
    rows_out: list[dict] = []

    with raw_csv.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            total += 1
            curated = {col: row[col] for col in CLINICAL_COLUMNS}
            for key, value in curated.items():
                pii_hits += count_pii_occurrences(value)
                curated[key] = scrub_free_text(value)
            rows_out.append(curated)

    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CLINICAL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows_out)

    return AnonymizationReport(
        total_registros=total,
        colunas_removidas=sorted(DIRECT_IDENTIFIER_COLUMNS),
        ocorrencias_pii_texto_livre=pii_hits,
    )


def main() -> None:
    base = Path(__file__).resolve().parents[2]
    raw_csv = base / "data" / "raw" / "prontuarios_brutos.csv"
    output_csv = base / "data" / "processed" / "prontuarios_anonimizados.csv"

    report = anonymize_prontuarios(raw_csv, output_csv)
    print("Anonimização concluída.")
    print(f"  Registros processados: {report.total_registros}")
    print(f"  Colunas identificadoras removidas: {', '.join(report.colunas_removidas)}")
    print(f"  Ocorrências de PII mascaradas em texto livre: {report.ocorrencias_pii_texto_livre}")
    print(f"  Saída: {output_csv}")


if __name__ == "__main__":
    main()
