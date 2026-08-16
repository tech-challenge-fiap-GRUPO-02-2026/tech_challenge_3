"""Pipeline do Assistente Médico Virtual.

Integra, em um único fluxo:
  1. RAG sobre os protocolos internos (`src/rag/retriever.py`);
  2. Consulta à base estruturada de prontuários (`src/database/patient_repository.py`);
  3. Geração de resposta pela LLM customizada (`src/llm/provider.py`);
  4. Guardrails de segurança (`src/safety/guardrails.py`);
  5. Explainability — citação de fontes (`src/explainability/citations.py`);
  6. Logging de auditoria (`src/safety/audit_logger.py`).

`MedicalAssistant` é a implementação de referência, sem dependência
obrigatória do pacote `langchain` — o que garante que os testes e a
demonstração rodem em qualquer ambiente. Quando `langchain-core` está
instalado, `as_langchain_runnable()` expõe o mesmo pipeline como uma
`RunnableSequence` do LangChain (Construir um pipeline que integre a LLM
customizada), para uso em aplicações que já usam o ecossistema LangChain.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.database.patient_repository import PatientRepository
from src.explainability.citations import ExplainedAnswer, render_answer_with_sources
from src.langchain_app.intent import extract_patient_id, is_flow_request
from src.langchain_app.smalltalk import build_greeting_reply, is_smalltalk
from src.llm.provider import get_llm_provider
from src.rag.retriever import ProtocolRetriever, get_retriever
from src.safety.audit_logger import AuditLogger
from src.safety.guardrails import MedicalGuardrails


@dataclass
class AssistantResponse:
    answer: str
    citations: list[str]
    patient_context: str | None
    guardrail_flags: list[str]
    blocked: bool
    event_id: str | None
    flow_triggered: bool = False


class MedicalAssistant:
    """Assistente virtual médico — pipeline RAG + prontuário + LLM + segurança."""

    def __init__(
        self,
        llm_provider_name: str = "auto",
        use_langchain_retriever: bool = False,
    ):
        self.llm = get_llm_provider(llm_provider_name)
        self.retriever = get_retriever(prefer_langchain=use_langchain_retriever)
        self.patients = PatientRepository()
        self.guardrails = MedicalGuardrails()
        self.audit = AuditLogger()

    def ask(self, question: str, patient_id: str | None = None, k: int = 3) -> AssistantResponse:
        input_check = self.guardrails.check_input(question)
        if not input_check.allowed:
            event_id = self.audit.log_interaction(
                question=question,
                patient_id=patient_id,
                answer=input_check.reason or "bloqueado",
                sources=[],
                guardrail_flags=input_check.flags,
                requires_human_validation=True,
            )
            return AssistantResponse(
                answer=input_check.reason or "Pergunta bloqueada por política de segurança.",
                citations=[],
                patient_context=None,
                guardrail_flags=input_check.flags,
                blocked=True,
                event_id=event_id,
            )

        patient_summary = self.patients.patient_context_summary(patient_id) if patient_id else None

        if is_smalltalk(question):
            greeting = build_greeting_reply(patient_summary)
            event_id = self.audit.log_interaction(
                question=question,
                patient_id=patient_id,
                answer=greeting,
                sources=[],
                guardrail_flags=[],
                requires_human_validation=False,
                extra={"smalltalk": True},
            )
            return AssistantResponse(
                answer=greeting,
                citations=[],
                patient_context=patient_summary,
                guardrail_flags=[],
                blocked=False,
                event_id=event_id,
            )

        if is_flow_request(question):
            target_patient_id = extract_patient_id(question) or patient_id
            if target_patient_id:
                return self._run_flow_as_answer(question, target_patient_id)
            # Pedido de fluxo sem paciente identificável: cai para o
            # tratamento normal de pergunta (vai gerar o aviso de que
            # não há fonte, mas ao menos não finge executar um fluxo
            # sem saber para quem).

        passages = self.retriever.retrieve(question, k=k)

        primary_source = None
        if hasattr(self.llm, "answer_with_source"):
            result = self.llm.answer_with_source(question)
            raw_answer = result["answer"]
            primary_source = result.get("source")
        else:
            raw_answer = self.llm.answer_question(question)

        output_check = self.guardrails.review_output(raw_answer)
        sanitized_answer = output_check.sanitized_text or raw_answer

        explained: ExplainedAnswer = render_answer_with_sources(
            sanitized_answer, passages, patient_summary, primary_source=primary_source
        )

        event_id = self.audit.log_interaction(
            question=question,
            patient_id=patient_id,
            answer=explained.answer,
            sources=explained.citations,
            guardrail_flags=output_check.flags,
            requires_human_validation=True,
        )

        return AssistantResponse(
            answer=explained.answer,
            citations=explained.citations,
            patient_context=patient_summary,
            guardrail_flags=output_check.flags,
            blocked=False,
            event_id=event_id,
        )

    def _run_flow_as_answer(self, question: str, patient_id: str) -> AssistantResponse:
        """Executa o fluxo clínico automatizado (LangGraph) e formata o
        resultado como uma resposta de chat, para pedidos como "rodar
        fluxo automatizado para PAC-0005" digitados como texto livre."""
        from src.langgraph_flow.clinical_flow import run_clinical_flow

        state = run_clinical_flow(patient_id, prefer_langgraph=True)

        if not state.get("patient_found"):
            answer = f"Paciente {patient_id} não encontrado na base de prontuários."
            event_id = self.audit.log_interaction(
                question=question, patient_id=patient_id, answer=answer,
                sources=[], guardrail_flags=[], requires_human_validation=False,
            )
            return AssistantResponse(
                answer=answer, citations=[], patient_context=None,
                guardrail_flags=[], blocked=False, event_id=event_id, flow_triggered=True,
            )

        lines = [
            f"Fluxo clínico automatizado executado para {patient_id}.",
            "",
            f"Diagnóstico: {state.get('diagnosis', '-')}",
            f"Exames pendentes: {', '.join(state.get('pending_exams') or []) or 'nenhum'}",
            "",
            "Sugestão:",
            state.get("suggestion", "-"),
        ]
        if state.get("alerts"):
            lines.append("")
            lines.append("Alertas para a equipe:")
            lines.extend(f"  - {a}" for a in state["alerts"])
        lines.append("")
        lines.append(f"Nós executados: {' -> '.join(state.get('log') or [])}")
        answer = "\n".join(lines)

        event_id = self.audit.log_interaction(
            question=question,
            patient_id=patient_id,
            answer=answer,
            sources=state.get("protocol_guidance") or [],
            guardrail_flags=state.get("guardrail_flags") or [],
            requires_human_validation=True,
            extra={"flow_triggered": True},
        )

        return AssistantResponse(
            answer=answer,
            citations=state.get("protocol_guidance") or [],
            patient_context=self.patients.patient_context_summary(patient_id),
            guardrail_flags=state.get("guardrail_flags") or [],
            blocked=False,
            event_id=event_id,
            flow_triggered=True,
        )

    def as_langchain_runnable(self):
        """Expõe o pipeline como `Runnable` do LangChain (langchain-core),
        para integração com aplicações que já orquestram via LCEL.
        Requer `pip install langchain-core`.
        """
        from langchain_core.runnables import RunnableLambda

        def _invoke(payload: dict) -> dict:
            response = self.ask(payload["question"], payload.get("patient_id"))
            return {
                "answer": response.answer,
                "citations": response.citations,
                "blocked": response.blocked,
            }

        return RunnableLambda(_invoke)
