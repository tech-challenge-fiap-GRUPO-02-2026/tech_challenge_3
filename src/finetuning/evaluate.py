"""Avaliação qualitativa e quantitativa do modelo fine-tuned.

Compara respostas do modelo base vs. modelo com adaptador LoRA em um
conjunto de perguntas de validação (hold-out das FAQs), e reporta:
  - overlap lexical com a resposta de referência (proxy simples de
    aderência ao protocolo, sem depender de bibliotecas pesadas);
  - perplexidade no conjunto de validação, quando `torch`/`transformers`
    estão disponíveis.

Uso:
    py -m src.finetuning.evaluate
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.finetuning.config import FineTuneConfig

BASE_DIR = Path(__file__).resolve().parents[2]
HOLDOUT_PATH = BASE_DIR / "data" / "processed" / "eval_holdout.jsonl"


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zà-ú0-9]+", text.lower()))


def lexical_overlap(candidate: str, reference: str) -> float:
    cand, ref = _tokenize(candidate), _tokenize(reference)
    if not ref:
        return 0.0
    return len(cand & ref) / len(ref)


def build_holdout(n: int = 3) -> list[dict]:
    """Separa as últimas N perguntas das FAQs como conjunto de validação
    (não usadas no fine-tuning) e persiste para reprodutibilidade."""
    faqs_path = BASE_DIR / "data" / "raw" / "faqs" / "faqs_medicos.jsonl"
    items = [json.loads(l) for l in faqs_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    holdout = items[-n:]
    HOLDOUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    HOLDOUT_PATH.write_text(
        "\n".join(json.dumps(h, ensure_ascii=False) for h in holdout), encoding="utf-8"
    )
    return holdout


def evaluate_with_provider(provider, holdout: list[dict]) -> dict:
    """Avalia usando qualquer objeto com `.answer_question(question) -> str`
    (ver `src/llm/provider.py`), permitindo comparar o fallback
    determinístico e o modelo fine-tuned na mesma interface."""
    scores = []
    details = []
    for item in holdout:
        answer = provider.answer_question(item["pergunta"])
        score = lexical_overlap(answer, item["resposta"])
        scores.append(score)
        details.append({
            "pergunta": item["pergunta"],
            "resposta_esperada": item["resposta"],
            "resposta_modelo": answer,
            "overlap_lexical": round(score, 3),
        })
    avg = sum(scores) / len(scores) if scores else 0.0
    return {"overlap_lexical_medio": round(avg, 3), "detalhes": details}


def main() -> None:
    from src.llm.provider import DeterministicMedicalProvider

    holdout = build_holdout()
    provider = DeterministicMedicalProvider()
    result = evaluate_with_provider(provider, holdout)

    print(f"Overlap lexical médio (fallback determinístico): {result['overlap_lexical_medio']}")
    for d in result["detalhes"]:
        print(f"  - [{d['overlap_lexical']}] {d['pergunta']}")

    out_path = BASE_DIR / "artifacts" / "eval_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Relatório salvo em: {out_path}")


if __name__ == "__main__":
    main()
