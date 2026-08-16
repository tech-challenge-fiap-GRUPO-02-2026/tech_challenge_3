"""Curadoria e montagem do dataset de instruction-tuning.

Combina quatro fontes em um único dataset no formato
`instruction / input / output / source`, pronto para fine-tuning
supervisionado (SFT) no formato Alpaca-like:

  1. Protocolos médicos internos do hospital (sintéticos);
  2. FAQs de médicos (sintéticas, com fonte no protocolo interno);
  3. Modelos de laudos/receitas (sintéticos);
  4. **PubMedQA** — amostra real de perguntas/respostas clínicas baseadas
     em publicações médicas (ver `download_pubmedqa.py`), sugerida no
     enunciado do desafio como fonte pública de dados.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


@dataclass
class Example:
    instruction: str
    input: str
    output: str
    source: str


def _split_sections(markdown_text: str) -> list[tuple[str, str]]:
    """Divide um protocolo markdown em seções por cabeçalho `## `."""
    sections: list[tuple[str, str]] = []
    current_title = None
    current_lines: list[str] = []
    for line in markdown_text.splitlines():
        if line.startswith("## "):
            if current_title is not None:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = line[3:].strip()
            current_lines = []
        elif current_title is not None:
            current_lines.append(line)
    if current_title is not None:
        sections.append((current_title, "\n".join(current_lines).strip()))
    return sections


def build_from_protocols() -> list[Example]:
    examples: list[Example] = []
    for path in sorted((RAW_DIR / "protocolos").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title_match = re.search(r"^# (.+)$", text, flags=re.MULTILINE)
        protocol_title = title_match.group(1).strip() if title_match else path.stem
        for section_title, section_body in _split_sections(text):
            if not section_body:
                continue
            examples.append(
                Example(
                    instruction=(
                        f"Segundo o protocolo interno \"{protocol_title}\", "
                        f"explique a seção \"{section_title}\"."
                    ),
                    input="",
                    output=section_body,
                    source=path.name,
                )
            )
    return examples


def build_from_faqs() -> list[Example]:
    examples: list[Example] = []
    faqs_path = RAW_DIR / "faqs" / "faqs_medicos.jsonl"
    with faqs_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            examples.append(
                Example(
                    instruction=item["pergunta"],
                    input="",
                    output=item["resposta"],
                    source=item.get("fonte", faqs_path.name),
                )
            )
    return examples


def build_from_laudos() -> list[Example]:
    """Exemplos de formatação — ensinam o modelo a produzir *rascunhos*
    estruturados sempre com o aviso obrigatório de validação humana."""
    laudos_path = RAW_DIR / "laudos" / "modelos_laudos.md"
    text = laudos_path.read_text(encoding="utf-8")
    blocks = re.findall(r"## (Modelo \d+ — .+?)\n\n```\n(.*?)\n```", text, flags=re.DOTALL)
    examples: list[Example] = []
    for title, template in blocks:
        examples.append(
            Example(
                instruction=f"Gere um rascunho no formato \"{title}\" para preenchimento médico.",
                input="",
                output=(
                    template.strip()
                    + "\n\n[RASCUNHO — requer validação e assinatura de médico responsável]"
                ),
                source=laudos_path.name,
            )
        )
    return examples


def build_from_pubmedqa() -> list[Example]:
    """Lê a amostra do PubMedQA já baixada e convertida por
    `download_pubmedqa.py`. Se o arquivo ainda não existir (dataset não
    baixado), retorna lista vazia — essa fonte é opcional e depende de
    conexão com a internet na primeira execução."""
    path = RAW_DIR / "pubmedqa" / "pubmedqa_sample.jsonl"
    if not path.exists():
        return []
    examples: list[Example] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            examples.append(
                Example(
                    instruction=item["pergunta"],
                    input="",
                    output=item["resposta"],
                    source=item.get("fonte", path.name),
                )
            )
    return examples


def build_dataset() -> list[Example]:
    examples = (
        build_from_protocols()
        + build_from_faqs()
        + build_from_laudos()
        + build_from_pubmedqa()
    )
    return examples


def save_dataset(examples: list[Example], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")


def main() -> None:
    examples = build_dataset()
    output_path = PROCESSED_DIR / "finetune_dataset.jsonl"
    save_dataset(examples, output_path)
    print(f"Dataset de fine-tuning gerado com {len(examples)} exemplos.")
    print(f"  Protocolos: {len(build_from_protocols())} exemplos")
    print(f"  FAQs: {len(build_from_faqs())} exemplos")
    print(f"  Modelos de laudo: {len(build_from_laudos())} exemplos")
    print(f"  PubMedQA: {len(build_from_pubmedqa())} exemplos")
    print(f"  Saída: {output_path}")


if __name__ == "__main__":
    main()
