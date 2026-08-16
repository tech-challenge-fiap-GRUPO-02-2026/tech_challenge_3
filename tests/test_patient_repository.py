from src.database.patient_repository import PatientRepository


def test_get_patient_returns_known_patient():
    repo = PatientRepository()
    patient = repo.get_patient("PAC-0004")
    assert patient is not None
    assert patient.diagnostico == "Sepse de foco pulmonar"


def test_get_patient_unknown_returns_none():
    repo = PatientRepository()
    assert repo.get_patient("PAC-9999") is None


def test_get_pending_exams():
    repo = PatientRepository()
    pending = repo.get_pending_exams("PAC-0006")
    assert any(e.exame == "Troponina" for e in pending)


def test_patient_context_summary_mentions_diagnosis():
    repo = PatientRepository()
    summary = repo.patient_context_summary("PAC-0001")
    assert "Insuficiencia cardiaca congestiva" in summary
