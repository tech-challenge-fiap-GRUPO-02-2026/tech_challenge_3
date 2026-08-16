from src.rag.retriever import ProtocolRetriever


def test_retriever_returns_relevant_passages():
    retriever = ProtocolRetriever()
    results = retriever.retrieve("meta glicemica paciente internado", k=3)
    assert len(results) > 0
    assert any("diabetes" in r.source for r in results)


def test_retriever_scores_are_between_0_and_1():
    retriever = ProtocolRetriever()
    results = retriever.retrieve("sepse antibiotico", k=3)
    for r in results:
        assert 0.0 <= r.score <= 1.0
