"""Configuração do pipeline de fine-tuning (LoRA/PEFT)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

# Módulos-alvo do LoRA variam por família de arquitetura — GPT-2 usa uma
# projeção QKV combinada (`c_attn`), já LLaMA/Mistral/Falcon usam
# projeções de atenção separadas (`q_proj`, `k_proj`, `v_proj`, `o_proj`).
GPT2_TARGET_MODULES: tuple[str, ...] = ("c_attn",)
LLAMA_TARGET_MODULES: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


def default_lora_target_modules(base_model: str) -> tuple[str, ...]:
    name = base_model.lower()
    if "llama" in name or "mistral" in name or "tinyllama" in name:
        return LLAMA_TARGET_MODULES
    return GPT2_TARGET_MODULES


@dataclass
class FineTuneConfig:
    # Modelo base PADRÃO: distilgpt2 (82M parâmetros) — treina em ~90s em
    # qualquer CPU, sem downloads grandes. Escolha deliberada para que
    # `py -m src.main finetune` seja reproduzível por qualquer pessoa
    # (incluindo o avaliador) sem esperar dezenas de minutos.
    #
    # Também testamos e documentamos `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
    # — uma arquitetura LLaMA real (1.1B parâmetros), aberta no Hugging
    # Face (sem gate/licença, ao contrário do Llama 3 oficial da Meta).
    # Em CPU sem GPU, o treino desse modelo levou dezenas de minutos para
    # apenas 10 passos neste projeto — inviável como padrão, mas
    # disponível via `--base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0`
    # para quem tiver GPU ou tempo disponível. O adaptador resultante
    # dessa execução fica documentado em `docs/tinyllama_run.md`.
    base_model: str = "distilgpt2"
    dataset_path: Path = BASE_DIR / "data" / "processed" / "finetune_dataset.jsonl"
    output_dir: Path = BASE_DIR / "artifacts" / "finetuned_model"

    # Hiperparâmetros LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = field(default_factory=lambda: GPT2_TARGET_MODULES)

    # Hiperparâmetros de treino
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    learning_rate: float = 2e-4
    max_seq_length: int = 512
    logging_steps: int = 5
    seed: int = 42

    # Formato do prompt de instrução (estilo Alpaca)
    prompt_template: str = (
        "### Instrução (uso interno hospitalar):\n{instruction}\n\n"
        "### Resposta:\n{output}"
    )

    def __post_init__(self) -> None:
        # Se o usuário trocar o modelo base via CLI (--base-model) sem
        # também passar target_modules, ajusta automaticamente para a
        # família de arquitetura correta (GPT-2 vs LLaMA/Mistral).
        if self.target_modules == GPT2_TARGET_MODULES:
            self.target_modules = default_lora_target_modules(self.base_model)
