"""Pipeline de fine-tuning supervisionado com LoRA (PEFT) sobre um modelo
causal pequeno, usando o dataset curado em `data/processed/finetune_dataset.jsonl`.

Este script depende de `torch`, `transformers`, `peft` e `datasets`
(ver `requirements-finetuning.txt`). Essas bibliotecas são pesadas e
opcionais: o restante do projeto (LangChain, LangGraph, RAG, segurança)
funciona sem elas, usando o `LocalDeterministicProvider` como fallback
(ver `src/llm/provider.py`).

Uso:
    py -m src.finetuning.train --max-steps 30
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.finetuning.config import FineTuneConfig


def _load_examples(dataset_path: Path) -> list[dict]:
    examples = []
    with dataset_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def run_training(cfg: FineTuneConfig, max_steps: int | None = None) -> None:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:  # pragma: no cover
        print(
            "Dependências de fine-tuning não instaladas.\n"
            "Instale com: pip install -r requirements-finetuning.txt\n"
            f"Detalhe: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    examples = _load_examples(cfg.dataset_path)
    if not examples:
        raise SystemExit(f"Dataset vazio em {cfg.dataset_path}. Rode dataset_builder antes.")

    texts = [
        cfg.prompt_template.format(instruction=ex["instruction"], output=ex["output"])
        for ex in examples
    ]

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg.max_seq_length,
            padding="max_length",
        )

    dataset = Dataset.from_dict({"text": texts}).map(tokenize, batched=True)

    model = AutoModelForCausalLM.from_pretrained(cfg.base_model)
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        learning_rate=cfg.learning_rate,
        logging_steps=cfg.logging_steps,
        max_steps=max_steps if max_steps else -1,
        save_strategy="no",
        report_to=[],
        seed=cfg.seed,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()

    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    manifest = {
        "base_model": cfg.base_model,
        "lora_r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "num_examples": len(examples),
        "num_train_epochs": cfg.num_train_epochs,
        "output_dir": str(cfg.output_dir),
    }
    (cfg.output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Fine-tuning concluído. Adaptador LoRA salvo em: {cfg.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning LoRA do assistente médico")
    parser.add_argument("--base-model", default=None, help="Modelo base HuggingFace (ex.: distilgpt2)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Limita passos de treino (demo rápida)")
    args = parser.parse_args()

    cfg = FineTuneConfig()
    if args.base_model:
        cfg.base_model = args.base_model
    if args.epochs:
        cfg.num_train_epochs = args.epochs

    run_training(cfg, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
