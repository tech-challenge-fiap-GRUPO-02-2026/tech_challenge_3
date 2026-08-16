from src.langgraph_flow.clinical_flow import run_clinical_flow


def test_flow_high_risk_patient_generates_alerts():
    state = run_clinical_flow("PAC-0006", prefer_langgraph=False)
    assert state["patient_found"] is True
    assert any("alto risco" in a for a in state["alerts"])
    assert state["requires_human_validation"] is True
    assert "sugerir_conduta" in state["log"]


def test_flow_unknown_patient_does_not_crash():
    state = run_clinical_flow("PAC-9999", prefer_langgraph=False)
    assert state["patient_found"] is False
    assert state["pending_exams"] == []


def test_flow_falls_back_when_langgraph_unavailable():
    # prefer_langgraph=True deve degradar graciosamente para o executor
    # sequencial quando o pacote langgraph não estiver instalado.
    state = run_clinical_flow("PAC-0004", prefer_langgraph=True)
    assert state["patient_found"] is True
    assert "emitir_alertas" in state["log"]
