"""Fluxo de decisão clínica automatizado, coordenado com LangGraph.

Ao receber o identificador de um paciente, o fluxo executa as etapas:

    intake -> verificar_exames_pendentes -> consultar_protocolo
           -> sugerir_conduta -> revisao_seguranca -> emitir_alertas -> fim

Cada nó grava um evento de auditoria (`AuditLogger.log_flow_event`). O nó
`sugerir_conduta` nunca produz uma prescrição definitiva — apenas um
rascunho de sugestão marcado como pendente de validação médica, e
`revisao_seguranca` é o portão (gate) que sanitiza a saída antes de
qualquer alerta ser emitido para a equipe.

A implementação principal usa `langgraph.graph.StateGraph` quando o pacote
`langgraph` está instalado. Um executor sequencial equivalente
(`_run_sequential`) garante que o fluxo funcione e seja testável mesmo sem
essa dependência opcional instalada.
"""
from __future__ import annotations

from typing import TypedDict

from src.database.patient_repository import PatientRepository
from src.llm.provider import get_llm_provider
from src.rag.retriever import get_retriever
from src.safety.audit_logger import AuditLogger
from src.safety.guardrails import MedicalGuardrails

FLOW_NAME = "fluxo_clinico_automatizado"

HIGH_RISK_DIAGNOSIS_KEYWORDS = ["sepse", "infarto", "iam", "cetoacidose"]


class ClinicalFlowState(TypedDict, total=False):
    patient_id: str
    patient_found: bool
    diagnosis: str
    pending_exams: list[str]
    protocol_guidance: list[str]
    suggestion: str
    guardrail_flags: list[str]
    alerts: list[str]
    requires_human_validation: bool
    log: list[str]


def _node_intake(state: ClinicalFlowState, *, repo: PatientRepository, audit: AuditLogger) -> ClinicalFlowState:
    patient = repo.get_patient(state["patient_id"])
    found = patient is not None
    state["patient_found"] = found
    state["diagnosis"] = patient.diagnostico if patient else ""
    state.setdefault("log", []).append("intake")
    audit.log_flow_event(flow_name=FLOW_NAME, node="intake", patient_id=state["patient_id"], payload={"patient_found": found})
    return state


def _node_check_pending_exams(state: ClinicalFlowState, *, repo: PatientRepository, audit: AuditLogger) -> ClinicalFlowState:
    if not state.get("patient_found"):
        state["pending_exams"] = []
        return state
    pending = [e.exame for e in repo.get_pending_exams(state["patient_id"])]
    state["pending_exams"] = pending
    state["log"].append("verificar_exames_pendentes")
    audit.log_flow_event(
        flow_name=FLOW_NAME, node="verificar_exames_pendentes", patient_id=state["patient_id"],
        payload={"pending_exams": pending},
    )
    return state


def _node_consult_protocol(state: ClinicalFlowState, *, retriever, audit: AuditLogger) -> ClinicalFlowState:
    if not state.get("patient_found"):
        state["protocol_guidance"] = []
        return state
    passages = retriever.retrieve(state["diagnosis"], k=2)
    guidance = [f"{p.source} — {p.section}" for p in passages]
    state["protocol_guidance"] = guidance
    state["log"].append("consultar_protocolo")
    audit.log_flow_event(
        flow_name=FLOW_NAME, node="consultar_protocolo", patient_id=state["patient_id"],
        payload={"guidance_sources": guidance},
    )
    return state


def _node_suggest_treatment(state: ClinicalFlowState, *, llm, retriever, audit: AuditLogger) -> ClinicalFlowState:
    if not state.get("patient_found"):
        state["suggestion"] = "Paciente não encontrado na base — nenhuma sugestão gerada."
        return state

    # Tenta primeiro casar o diagnóstico direto com uma FAQ curada
    # (mais confiável); se não houver match, usa o trecho de protocolo
    # mais relevante recuperado pelo RAG como base da sugestão — em vez
    # de arriscar uma pergunta livre longa, que tende a não casar bem
    # com a similaridade lexical da base determinística.
    result = llm.answer_with_source(state["diagnosis"]) if hasattr(llm, "answer_with_source") else None
    if result and result.get("source"):
        suggestion = result["answer"]
    else:
        passages = retriever.retrieve(state["diagnosis"], k=1)
        if passages:
            suggestion = f"Conforme {passages[0].source} — {passages[0].section}:\n{passages[0].text}"
        else:
            suggestion = llm.answer_question(state["diagnosis"])

    state["suggestion"] = suggestion
    state["log"].append("sugerir_conduta")
    audit.log_flow_event(
        flow_name=FLOW_NAME, node="sugerir_conduta", patient_id=state["patient_id"],
        payload={"suggestion": suggestion},
    )
    return state


def _node_safety_review(state: ClinicalFlowState, *, guardrails: MedicalGuardrails, audit: AuditLogger) -> ClinicalFlowState:
    result = guardrails.review_output(state.get("suggestion", ""))
    state["suggestion"] = result.sanitized_text or state.get("suggestion", "")
    state["guardrail_flags"] = result.flags
    state["requires_human_validation"] = True
    state["log"].append("revisao_seguranca")
    audit.log_flow_event(
        flow_name=FLOW_NAME, node="revisao_seguranca", patient_id=state["patient_id"],
        payload={"flags": result.flags},
    )
    return state


def _node_emit_alerts(state: ClinicalFlowState, *, audit: AuditLogger) -> ClinicalFlowState:
    alerts = []
    diagnosis_lower = state.get("diagnosis", "").lower()
    if any(kw in diagnosis_lower for kw in HIGH_RISK_DIAGNOSIS_KEYWORDS):
        alerts.append(f"ALERTA equipe médica: diagnóstico de alto risco ({state['diagnosis']}) — priorizar avaliação.")
    if state.get("pending_exams"):
        alerts.append(f"ALERTA: exames pendentes que podem impactar a conduta: {', '.join(state['pending_exams'])}.")
    state["alerts"] = alerts
    state["log"].append("emitir_alertas")
    audit.log_flow_event(
        flow_name=FLOW_NAME, node="emitir_alertas", patient_id=state["patient_id"],
        payload={"alerts": alerts},
    )
    return state


def _run_sequential(patient_id: str) -> ClinicalFlowState:
    repo = PatientRepository()
    retriever = get_retriever()
    llm = get_llm_provider("deterministic")
    guardrails = MedicalGuardrails()
    audit = AuditLogger()

    state: ClinicalFlowState = {"patient_id": patient_id, "log": []}
    state = _node_intake(state, repo=repo, audit=audit)
    state = _node_check_pending_exams(state, repo=repo, audit=audit)
    state = _node_consult_protocol(state, retriever=retriever, audit=audit)
    state = _node_suggest_treatment(state, llm=llm, retriever=retriever, audit=audit)
    state = _node_safety_review(state, guardrails=guardrails, audit=audit)
    state = _node_emit_alerts(state, audit=audit)
    return state


def _run_langgraph(patient_id: str) -> ClinicalFlowState:
    from langgraph.graph import END, StateGraph

    repo = PatientRepository()
    retriever = get_retriever()
    llm = get_llm_provider("deterministic")
    guardrails = MedicalGuardrails()
    audit = AuditLogger()

    graph = StateGraph(ClinicalFlowState)
    graph.add_node("intake", lambda s: _node_intake(s, repo=repo, audit=audit))
    graph.add_node("verificar_exames_pendentes", lambda s: _node_check_pending_exams(s, repo=repo, audit=audit))
    graph.add_node("consultar_protocolo", lambda s: _node_consult_protocol(s, retriever=retriever, audit=audit))
    graph.add_node("sugerir_conduta", lambda s: _node_suggest_treatment(s, llm=llm, retriever=retriever, audit=audit))
    graph.add_node("revisao_seguranca", lambda s: _node_safety_review(s, guardrails=guardrails, audit=audit))
    graph.add_node("emitir_alertas", lambda s: _node_emit_alerts(s, audit=audit))

    graph.set_entry_point("intake")
    graph.add_edge("intake", "verificar_exames_pendentes")
    graph.add_edge("verificar_exames_pendentes", "consultar_protocolo")
    graph.add_edge("consultar_protocolo", "sugerir_conduta")
    graph.add_edge("sugerir_conduta", "revisao_seguranca")
    graph.add_edge("revisao_seguranca", "emitir_alertas")
    graph.add_edge("emitir_alertas", END)

    app = graph.compile()
    return app.invoke({"patient_id": patient_id, "log": []})


def run_clinical_flow(patient_id: str, prefer_langgraph: bool = True) -> ClinicalFlowState:
    """Executa o fluxo clínico automatizado para um paciente.

    Usa `langgraph` quando disponível (`prefer_langgraph=True`, padrão);
    caso contrário, ou se a dependência não estiver instalada, cai para o
    executor sequencial equivalente — mesma lógica de nós, sem a
    dependência externa.
    """
    if prefer_langgraph:
        try:
            return _run_langgraph(patient_id)
        except ImportError:
            pass
    return _run_sequential(patient_id)
