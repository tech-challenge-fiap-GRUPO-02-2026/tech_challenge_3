"""Baixa uma amostra do dataset público PubMedQA (pubmedqa.github.io),
traduz para português (PT-BR) e converte para o mesmo schema usado pelas
FAQs internas do hospital (`pergunta` / `resposta` / `fonte`), permitindo
que o assistente responda também perguntas clínicas gerais baseadas em
publicações médicas reais, além dos protocolos internos sintéticos — tudo
no mesmo idioma do restante do projeto.

PubMedQA (Jin et al., 2019) — perguntas de pesquisa biomédica derivadas de
resumos do PubMed, com resposta longa (`long_answer`) e decisão resumida
(`final_decision`: yes/no/maybe). Config `pqa_labeled`: 1.000 instâncias
rotuladas manualmente por especialistas.

A tradução usa o modelo `Helsinki-NLP/opus-mt-tc-big-en-pt` (MarianMT,
rodando localmente via `transformers`) — sem depender de API paga de
tradução. O arquivo original em inglês é preservado em
`pubmedqa_sample_en.jsonl` para auditoria/rastreabilidade da tradução.

Requer os pacotes `datasets`, `sentencepiece` e `sacremoses` (ver
`requirements-finetuning.txt`) e conexão com a internet na primeira
execução (baixa o dataset e o modelo de tradução do Hugging Face Hub).

Uso:
    py -m src.data_processing.download_pubmedqa --n 40
    py -m src.data_processing.download_pubmedqa --n 40 --no-translate  # mantém em inglês
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_PATH = BASE_DIR / "data" / "raw" / "pubmedqa" / "pubmedqa_sample.jsonl"
OUTPUT_PATH_EN = BASE_DIR / "data" / "raw" / "pubmedqa" / "pubmedqa_sample_en.jsonl"
SOURCE_LABEL = "PubMedQA (pubmedqa.github.io, traduzido)"
SOURCE_LABEL_EN = "PubMedQA (pubmedqa.github.io)"

TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-tc-big-en-pt"

DECISION_PT = {"yes": "sim", "no": "não", "maybe": "talvez"}


def download_raw(n: int = 40, seed: int = 42) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Pacote 'datasets' não instalado. Rode:\n"
            "  pip install -r requirements-finetuning.txt\n"
            f"Detalhe: {exc}"
        )

    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    dataset = dataset.shuffle(seed=seed).select(range(min(n, len(dataset))))

    examples = []
    for row in dataset:
        question = row["question"].strip()
        answer = row["long_answer"].strip()
        decision = row.get("final_decision", "").strip()
        if decision:
            answer = f"{answer} (Conclusion: {decision})"
        examples.append({
            "pergunta": question,
            "resposta": answer,
            "fonte": SOURCE_LABEL_EN,
            "pubid": row.get("pubid"),
        })
    return examples


class Translator:
    """Wrapper fino sobre o MarianMT EN->PT, com tradução em lote."""

    def __init__(self, model_name: str = TRANSLATION_MODEL):
        from transformers import MarianMTModel, MarianTokenizer

        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)

    def translate_batch(self, texts: list[str], max_length: int = 512) -> list[str]:
        batch = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        generated = self.model.generate(**batch, max_length=max_length)
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)


def translate_examples(examples: list[dict], batch_size: int = 8) -> list[dict]:
    translator = Translator()
    translated: list[dict] = []

    for i in range(0, len(examples), batch_size):
        batch = examples[i : i + batch_size]
        questions = [ex["pergunta"] for ex in batch]
        # Traduz só a parte textual da resposta; "(Conclusion: yes/no/maybe)"
        # é convertido à parte, para não depender da qualidade do modelo
        # nesse trecho estruturado.
        answers = [ex["resposta"].split(" (Conclusion:")[0] for ex in batch]

        pt_questions = translator.translate_batch(questions)
        pt_answers = translator.translate_batch(answers)

        for ex, pt_q, pt_a in zip(batch, pt_questions, pt_answers):
            decision_en = ex["resposta"].split("(Conclusion: ")[-1].rstrip(")") if "(Conclusion:" in ex["resposta"] else None
            decision_pt = DECISION_PT.get(decision_en, decision_en)
            resposta = f"{pt_a} (Conclusão: {decision_pt})" if decision_pt else pt_a
            translated.append({
                "pergunta": pt_q,
                "resposta": resposta,
                "fonte": SOURCE_LABEL,
                "pubid": ex["pubid"],
            })
    return translated


def save(examples: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")


def download_and_convert(n: int = 40, seed: int = 42, translate: bool = True) -> list[dict]:
    """Mantido por compatibilidade com `src/main.py` — baixa e, por
    padrão, já traduz para português."""
    raw = download_raw(n=n, seed=seed)
    if not translate:
        return raw
    return translate_examples(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Baixa, traduz (EN->PT) e converte uma amostra do PubMedQA")
    parser.add_argument("--n", type=int, default=40, help="Número de exemplos a baixar")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-translate", action="store_true", help="Mantém as perguntas/respostas em inglês")
    args = parser.parse_args()

    raw = download_raw(n=args.n, seed=args.seed)
    save(raw, OUTPUT_PATH_EN)
    print(f"PubMedQA (original, EN): {len(raw)} exemplos salvos em {OUTPUT_PATH_EN}")

    if args.no_translate:
        save(raw, OUTPUT_PATH)
        print(f"Tradução desativada — {OUTPUT_PATH} salvo em inglês.")
        return

    print("Traduzindo para português (Helsinki-NLP/opus-mt-tc-big-en-pt)...")
    translated = translate_examples(raw)
    save(translated, OUTPUT_PATH)
    print(f"PubMedQA (traduzido, PT-BR): {len(translated)} exemplos salvos em {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
