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
    if torch.cuda.is_available():
        # gradient_checkpointing (ativado abaixo) é incompatível com o cache
        # de KV durante o treino; desativa para evitar conflito/warning.
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
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

    # Detecção de acelerador agnóstica entre NVIDIA (CUDA) e AMD (ROCm):
    # ambos os builds do PyTorch expõem a API `torch.cuda`. Em CPU, tudo
    # cai para os padrões (sem fp16/bf16, sem gradient checkpointing).
    use_gpu = torch.cuda.is_available()
    
    # bf16 é suportado em GPUs recentes (NVIDIA Ampere+ e AMD CDNA); quando
    # não houver suporte, usa fp16. Economiza VRAM (importante em placas de
    # 6 GB) e acelera o treino.
    bf16_ok = use_gpu and torch.cuda.is_bf16_supported()

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
        bf16=bf16_ok,
        fp16=use_gpu and not bf16_ok,
        gradient_checkpointing=use_gpu,
        optim="adamw_torch",
    )
    if use_gpu:
        print(
            f"GPU detectada: {torch.cuda.get_device_name(0)} | "
            f"precisão: {'bf16' if bf16_ok else 'fp16'} | "
            f"gradient_checkpointing: on"
        )
    else:
        print("Nenhuma GPU detectada — treinando em CPU (mais lento).")

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

    # Importante: o base_model precisa ser passado na CONSTRUÇÃO da config,
    # pois é o __post_init__ que deriva os target_modules do LoRA a partir
    # da arquitetura (GPT-2 usa `c_attn`; LLaMA/Mistral usam
    # `q_proj/k_proj/v_proj/o_proj`). Setar cfg.base_model depois de criada
    # não recalcula os módulos e faz o LoRA falhar em modelos LLaMA.
    cfg_kwargs = {}
    if args.base_model:
        cfg_kwargs["base_model"] = args.base_model
    if args.epochs:
        cfg_kwargs["num_train_epochs"] = args.epochs

    cfg = FineTuneConfig(**cfg_kwargs)

    run_training(cfg, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
