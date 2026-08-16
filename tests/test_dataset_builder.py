from src.data_processing.dataset_builder import (
    build_from_faqs,
    build_from_laudos,
    build_from_protocols,
)


def test_build_from_protocols_returns_sectioned_examples():
    examples = build_from_protocols()
    assert len(examples) > 0
    assert all(ex.instruction and ex.output for ex in examples)
    assert all(ex.source.endswith(".md") for ex in examples)


def test_build_from_faqs_matches_source_dataset():
    examples = build_from_faqs()
    assert len(examples) == 10
    assert all(ex.instruction and ex.output for ex in examples)


def test_build_from_laudos_includes_human_validation_notice():
    examples = build_from_laudos()
    assert len(examples) == 4
    assert all("validação" in ex.output.lower() for ex in examples)
