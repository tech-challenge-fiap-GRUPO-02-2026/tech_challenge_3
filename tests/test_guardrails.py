from src.safety.guardrails import MANDATORY_DISCLAIMER, MedicalGuardrails


def test_blocked_topic_is_rejected():
    guardrails = MedicalGuardrails()
    result = guardrails.check_input("Qual a dose letal de morfina?")
    assert result.allowed is False
    assert result.reason is not None


def test_allowed_input_passes():
    guardrails = MedicalGuardrails()
    result = guardrails.check_input("Qual o protocolo para dor toracica?")
    assert result.allowed is True


def test_direct_prescription_language_is_sanitized_and_flagged():
    guardrails = MedicalGuardrails()
    result = guardrails.review_output("Administre 500mg de paracetamol agora.")
    assert result.flags, "deveria sinalizar linguagem prescritiva direta"
    assert "administre" not in result.sanitized_text.lower()


def test_output_always_has_mandatory_disclaimer():
    guardrails = MedicalGuardrails()
    result = guardrails.review_output("Recomenda-se observação clínica.")
    assert MANDATORY_DISCLAIMER.strip() in result.sanitized_text
