"""Abstração de provedor de LLM.

Três implementações, com a mesma interface, para permitir demonstração
completa do projeto mesmo sem GPU/chave de API:

  - `DeterministicMedicalProvider`: sem dependências externas, responde com
    base nas FAQs curadas (fonte de verdade determinística). É o padrão do
    projeto e o que garante que os testes automatizados rodem em qualquer
    máquina.
  - `LocalFineTunedProvider`: carrega o adaptador LoRA treinado por
    `src/finetuning/train.py` (requer torch/transformers/peft instalados e
    o artefato em `artifacts/finetuned_model`).
  - `OpenAIProvider`: usa a API da OpenAI quando `OPENAI_API_KEY` está
    definida (uso opcional, útil para comparação qualitativa).

Todas as implementações expõem `generate(prompt, context="") -> str` e
`answer_question(question) -> str`.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
FAQS_PATH = BASE_DIR / "data" / "raw" / "faqs" / "faqs_medicos.jsonl"
PUBMEDQA_PATH = BASE_DIR / "data" / "raw" / "pubmedqa" / "pubmedqa_sample.jsonl"
FINETUNED_DIR = BASE_DIR / "artifacts" / "finetuned_model"

SAFE_FALLBACK = (
    "Não encontrei uma referência direta nos protocolos internos para essa "
    "pergunta. Recomendo consultar o protocolo específico ou um médico "
    "responsável antes de qualquer conduta. [Nenhuma fonte identificada]"
)


STOPWORDS = {
    # Português
    "a", "o", "os", "as", "um", "uma", "de", "da", "do", "das", "dos",
    "em", "no", "na", "nos", "nas", "para", "por", "com", "sem", "que",
    "qual", "quais", "quando", "onde", "como", "e", "ou", "se", "ao", "aos",
    "à", "é", "ser", "tem", "há",
    # English (perguntas do PubMedQA)
    "the", "a", "an", "of", "in", "on", "to", "for", "and", "or", "is",
    "are", "was", "were", "does", "do", "did", "with", "by", "as", "at",
    "be", "it", "its", "this", "that", "which",
}


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def _tokenize(text: str) -> set[str]:
    """Tokeniza sem acentos ('critérios' == 'criterios'), evitando falsos
    negativos de correspondência quando o usuário digita sem acentuação."""
    tokens = re.findall(r"[a-z0-9]+", _strip_accents(text.lower()))
    return {t for t in tokens if t not in STOPWORDS}


class DeterministicMedicalProvider:
    """Provedor sem dependências externas — casa a pergunta do usuário com
    a base de FAQs curadas (protocolos internos + amostra do PubMedQA)
    por similaridade lexical (Jaccard)."""

    def __init__(self, faqs_path: Path = FAQS_PATH, extra_sources: tuple[Path, ...] = (PUBMEDQA_PATH,)):
        self.faqs = self._load_faqs(faqs_path)
        for path in extra_sources:
            self.faqs.extend(self._load_faqs(path))

    @staticmethod
    def _load_faqs(path: Path) -> list[dict]:
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as fh:
            return [json.loads(l) for l in fh if l.strip()]

    def _best_match(self, question: str) -> tuple[dict | None, float]:
        q_tokens = _tokenize(question)
        best_item, best_score = None, 0.0
        for item in self.faqs:
            f_tokens = _tokenize(item["pergunta"])
            if not q_tokens or not f_tokens:
                continue
            score = len(q_tokens & f_tokens) / len(q_tokens | f_tokens)
            if score > best_score:
                best_item, best_score = item, score
        return best_item, best_score

    def answer_question(self, question: str, threshold: float = 0.15) -> str:
        item, score = self._best_match(question)
        if item and score >= threshold:
            return item["resposta"]
        return SAFE_FALLBACK

    def answer_with_source(self, question: str, threshold: float = 0.15) -> dict:
        item, score = self._best_match(question)
        if item and score >= threshold:
            return {"answer": item["resposta"], "source": item.get("fonte", "desconhecida"), "confidence": round(score, 3)}
        return {"answer": SAFE_FALLBACK, "source": None, "confidence": round(score, 3)}

    def generate(self, prompt: str, context: str = "") -> str:
        # Sem um modelo generativo real, o fallback determinístico responde
        # com base no melhor casamento textual disponível no prompt.
        return self.answer_question(prompt)


class LocalFineTunedProvider:
    """Carrega o modelo local ajustado via LoRA (ver src/finetuning/train.py)."""

    def __init__(self, model_dir: Path = FINETUNED_DIR, base_provider: "DeterministicMedicalProvider | None" = None):
        self.model_dir = model_dir
        self.base_provider = base_provider or DeterministicMedicalProvider()
        self._pipeline = None

    def _ensure_loaded(self):
        if self._pipeline is not None:
            return
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Nenhum modelo fine-tuned encontrado em {self.model_dir}. "
                "Rode `py -m src.finetuning.train` primeiro."
            )
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer, pipeline

        # Inferência GPU-aware e agnóstica entre NVIDIA (CUDA) e AMD (ROCm):
        # ambos expõem `torch.cuda`. Sem GPU, cai para CPU automaticamente.
        import torch

        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            model = AutoPeftModelForCausalLM.from_pretrained(
                self.model_dir, torch_dtype=dtype, device_map="auto"
            )
        else:
            model = AutoPeftModelForCausalLM.from_pretrained(self.model_dir)
        tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        # Quando device_map="auto" posiciona o modelo, o pipeline herda o
        # device do modelo; sem GPU, roda em CPU.
        self._pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

    def generate(self, prompt: str, context: str = "") -> str:
        self._ensure_loaded()
        full_prompt = f"### Instrução (uso interno hospitalar):\n{context}\n{prompt}\n\n### Resposta:\n"
        out = self._pipeline(full_prompt, max_new_tokens=200, do_sample=False)[0]["generated_text"]
        return out.split("### Resposta:")[-1].strip()

    def answer_question(self, question: str) -> str:
        try:
            return self.generate(question)
        except FileNotFoundError:
            return self.base_provider.answer_question(question)


class OpenAIProvider:
    """Provedor opcional via API da OpenAI (requer OPENAI_API_KEY)."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    def generate(self, prompt: str, context: str = "") -> str:
        from openai import OpenAI

        client = OpenAI()
        system_msg = (
            "Você é um assistente clínico de apoio à decisão. Responda apenas "
            "com base no contexto fornecido (protocolos internos). Nunca "
            "prescreva diretamente — sempre recomende validação médica. "
            "Se o contexto não tiver a resposta, diga que não sabe."
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Contexto:\n{context}\n\nPergunta: {prompt}"},
        ]
        resp = client.chat.completions.create(model=self.model, messages=messages, temperature=0.1)
        return resp.choices[0].message.content

    def answer_question(self, question: str) -> str:
        return self.generate(question)


def get_llm_provider(name: str = "auto"):
    """Fábrica de provedores.

    name: "deterministic" | "local_finetuned" | "openai" | "auto"

    "auto" tenta openai (se houver chave) -> determinístico. O modelo
    local fine-tuned NUNCA é escolhido automaticamente: em um contexto
    clínico, um modelo pouco treinado (ex.: TinyLlama-1.1B com poucos
    passos de LoRA, usado aqui por custo/tempo de CPU) pode gerar texto
    fluente porém incorreto, sem qualquer garantia de aderência aos
    protocolos. É mais seguro que a resposta padrão venha da base curada
    (determinística, sempre correta e citável) do que de um modelo
    generativo ainda não validado. Use `--provider local_finetuned`
    explicitamente para inspecionar/demonstrar o modelo treinado.
    """
    if name == "deterministic":
        return DeterministicMedicalProvider()
    if name == "local_finetuned":
        return LocalFineTunedProvider()
    if name == "openai":
        return OpenAIProvider()

    # auto — nunca escolhe local_finetuned automaticamente (ver docstring)
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIProvider()
    return DeterministicMedicalProvider()
