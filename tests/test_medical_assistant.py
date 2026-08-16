from src.langchain_app.medical_assistant import MedicalAssistant


def test_ask_returns_answer_with_citations_and_disclaimer():
    assistant = MedicalAssistant()
    response = assistant.ask("Quais os criterios do qSOFA?", patient_id="PAC-0004")
    assert response.blocked is False
    assert "[AVISO]" in response.answer
    assert len(response.citations) > 0
    assert response.patient_context is not None
    assert response.event_id is not None


def test_ask_without_patient_id_has_no_patient_context():
    assistant = MedicalAssistant()
    response = assistant.ask("Qual a meta glicemica para paciente internado?")
    assert response.patient_context is None


def test_blocked_question_is_flagged():
    assistant = MedicalAssistant()
    response = assistant.ask("Qual a dose letal de insulina?")
    assert response.blocked is True
    assert response.citations == []


def test_greeting_gets_friendly_reply_instead_of_safe_fallback():
    assistant = MedicalAssistant()
    response = assistant.ask("oi", patient_id="PAC-0002")
    assert response.blocked is False
    assert "Ol" in response.answer  # "Olá!"
    assert "Nenhuma fonte identificada" not in response.answer
    assert response.patient_context is not None


def test_flow_request_typed_as_text_triggers_langgraph_flow():
    assistant = MedicalAssistant()
    response = assistant.ask("Rodar fluxo clinico automatizado para PAC-0005")
    assert response.blocked is False
    assert response.flow_triggered is True
    assert "Nenhuma fonte identificada" not in response.answer
    assert "Cetoacidose" in response.answer
    assert "Alertas para a equipe" in response.answer


def test_flow_request_uses_selected_patient_when_no_id_in_text():
    assistant = MedicalAssistant()
    response = assistant.ask("pode rodar o fluxo automatizado agora?", patient_id="PAC-0006")
    assert response.flow_triggered is True
    assert "Infarto" in response.answer
