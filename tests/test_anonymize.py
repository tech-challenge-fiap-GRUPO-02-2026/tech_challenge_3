from pathlib import Path

from src.data_processing.anonymize import (
    DIRECT_IDENTIFIER_COLUMNS,
    anonymize_prontuarios,
    count_pii_occurrences,
    scrub_free_text,
)

BASE_DIR = Path(__file__).resolve().parents[1]


def test_scrub_free_text_masks_cpf_phone_email():
    text = "Paciente CPF 123.456.789-01, tel (11) 98765-4321, email a@b.com"
    scrubbed = scrub_free_text(text)
    assert "123.456.789-01" not in scrubbed
    assert "(11) 98765-4321" not in scrubbed
    assert "a@b.com" not in scrubbed
    assert "[CPF_REMOVIDO]" in scrubbed


def test_count_pii_occurrences():
    text = "CPF 123.456.789-01 e email x@y.com"
    assert count_pii_occurrences(text) == 2


def test_anonymize_prontuarios_removes_direct_identifiers(tmp_path):
    raw_csv = BASE_DIR / "data" / "raw" / "prontuarios_brutos.csv"
    output_csv = tmp_path / "anonimizados.csv"

    report = anonymize_prontuarios(raw_csv, output_csv)

    assert output_csv.exists()
    content = output_csv.read_text(encoding="utf-8")
    assert report.total_registros > 0
    assert set(report.colunas_removidas) == DIRECT_IDENTIFIER_COLUMNS
    for header in ("nome_completo", "cpf", "telefone", "email", "endereco"):
        assert header not in content.splitlines()[0]
    assert "@" not in content
