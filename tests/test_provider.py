from src.llm.provider import DeterministicMedicalProvider, SAFE_FALLBACK


def test_provider_matches_known_faq():
    provider = DeterministicMedicalProvider()
    answer = provider.answer_question("Quando iniciar antibiotico em suspeita de sepse?")
    assert "1 hora" in answer


def test_provider_falls_back_when_no_match():
    provider = DeterministicMedicalProvider()
    answer = provider.answer_question("Qual a previsão do tempo em Marte amanhã?")
    assert answer == SAFE_FALLBACK


def test_answer_with_source_returns_confidence_and_source():
    provider = DeterministicMedicalProvider()
    result = provider.answer_with_source("Qual a meta glicemica para paciente internado fora da UTI?")
    assert result["source"] == "protocolo_diabetes.md"
    assert result["confidence"] > 0
