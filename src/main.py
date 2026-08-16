"""CLI do Assistente Médico Virtual — Tech Challenge Fase 3.

Comandos:
    py -m src.main anonymize
    py -m src.main download-pubmedqa [--n 40]
    py -m src.main build-dataset
    py -m src.main finetune [--max-steps N]
    py -m src.main evaluate
    py -m src.main ask "pergunta" [--patient-id PAC-0004]
    py -m src.main flow --patient-id PAC-0004
    py -m src.main logs [--n 10]
"""
from __future__ import annotations

import argparse
import json


def cmd_anonymize(args) -> None:
    from src.data_processing.anonymize import main as anonymize_main

    anonymize_main()


def cmd_download_pubmedqa(args) -> None:
    from src.data_processing.download_pubmedqa import (
        OUTPUT_PATH,
        OUTPUT_PATH_EN,
        download_raw,
        save,
        translate_examples,
    )

    raw = download_raw(n=args.n)
    save(raw, OUTPUT_PATH_EN)

    if args.no_translate:
        save(raw, OUTPUT_PATH)
        print(f"PubMedQA: {len(raw)} exemplos baixados (em inglês, tradução desativada).")
        return

    print("Traduzindo para português (Helsinki-NLP/opus-mt-tc-big-en-pt)...")
    translated = translate_examples(raw)
    save(translated, OUTPUT_PATH)
    print(f"PubMedQA: {len(translated)} exemplos baixados e traduzidos para PT-BR.")


def cmd_build_dataset(args) -> None:
    from src.data_processing.dataset_builder import main as build_main

    build_main()


def cmd_finetune(args) -> None:
    from src.finetuning.config import FineTuneConfig
    from src.finetuning.train import run_training

    cfg = FineTuneConfig()
    run_training(cfg, max_steps=args.max_steps)


def cmd_evaluate(args) -> None:
    from src.finetuning.evaluate import main as evaluate_main

    evaluate_main()


def cmd_ask(args) -> None:
    from src.langchain_app.medical_assistant import MedicalAssistant

    assistant = MedicalAssistant(llm_provider_name=args.provider)
    response = assistant.ask(args.question, patient_id=args.patient_id)
    print(response.answer)
    print(f"\n[bloqueado={response.blocked} | flags={response.guardrail_flags} | event_id={response.event_id}]")


def cmd_flow(args) -> None:
    from src.langgraph_flow.clinical_flow import run_clinical_flow

    state = run_clinical_flow(args.patient_id, prefer_langgraph=not args.no_langgraph)
    print(json.dumps(state, ensure_ascii=False, indent=2))


def cmd_logs(args) -> None:
    from src.safety.audit_logger import AuditLogger

    audit = AuditLogger()
    for record in audit.read_recent(args.n):
        print(json.dumps(record, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assistente Médico Virtual — Tech Challenge Fase 3")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("anonymize", help="Anonimiza os prontuários brutos").set_defaults(func=cmd_anonymize)

    p_pubmedqa = sub.add_parser("download-pubmedqa", help="Baixa e traduz (EN->PT) uma amostra do dataset público PubMedQA")
    p_pubmedqa.add_argument("--n", type=int, default=40)
    p_pubmedqa.add_argument("--no-translate", action="store_true", help="Mantém em inglês")
    p_pubmedqa.set_defaults(func=cmd_download_pubmedqa)

    sub.add_parser("build-dataset", help="Monta o dataset de fine-tuning").set_defaults(func=cmd_build_dataset)

    p_finetune = sub.add_parser("finetune", help="Executa o fine-tuning LoRA")
    p_finetune.add_argument("--max-steps", type=int, default=None)
    p_finetune.set_defaults(func=cmd_finetune)

    sub.add_parser("evaluate", help="Avalia o modelo em perguntas de validação").set_defaults(func=cmd_evaluate)

    p_ask = sub.add_parser("ask", help="Faz uma pergunta clínica ao assistente")
    p_ask.add_argument("question")
    p_ask.add_argument("--patient-id", default=None)
    p_ask.add_argument("--provider", default="auto", choices=["auto", "deterministic", "local_finetuned", "openai"])
    p_ask.set_defaults(func=cmd_ask)

    p_flow = sub.add_parser("flow", help="Executa o fluxo clínico automatizado (LangGraph)")
    p_flow.add_argument("--patient-id", required=True)
    p_flow.add_argument("--no-langgraph", action="store_true", help="Força o executor sequencial")
    p_flow.set_defaults(func=cmd_flow)

    p_logs = sub.add_parser("logs", help="Mostra os últimos eventos de auditoria")
    p_logs.add_argument("--n", type=int, default=10)
    p_logs.set_defaults(func=cmd_logs)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
