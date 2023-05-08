"""Test code examples for evaluation of the Extraction AI in the documentation."""
import sys
import unittest


@unittest.skipIf(sys.version_info[:2] != (3, 8), 'This AI can only be loaded on Python 3.8.')
def test_evaluate_extraction_ai():
    """Test evaluation of the Extraction AI."""
    from konfuzio_sdk.data import Project
    from tests.variables import OFFLINE_PROJECT, TEST_DOCUMENT_ID
    from konfuzio_sdk.trainer.information_extraction import load_model

    MODEL_PATH = 'tests/trainer/2023-04-28-12-10-45_lohnabrechnung_rfextractionai_.pkl'
    pipeline = load_model(MODEL_PATH)

    project = Project(id_=None, project_folder=OFFLINE_PROJECT)
    test_document = project.get_document_by_id(TEST_DOCUMENT_ID)
    pipeline.test_documents = [test_document]

    # To get the evaluation of the full pipeline
    evaluation = pipeline.evaluate_full()
    assert evaluation.f1() > 0.65
    print(f"Full evaluation F1 score: {evaluation.f1()}")
    assert evaluation.recall() > 0.8
    print(f"Full evaluation recall: {evaluation.recall()}")
    assert evaluation.precision() > 0.65
    print(f"Full evaluation precision: {evaluation.precision()}")

    # To get the evaluation of the Tokenizer alone
    evaluation = pipeline.evaluate_tokenizer()
    assert evaluation.tokenizer_f1() > 0.1  # Whitespace tokenizer creates a lot of false positives
    print(f"Tokenizer evaluation F1 score: {evaluation.tokenizer_f1()}")

    # To get the evaluation of the Label classifier given perfect tokenization
    evaluation = pipeline.evaluate_clf()
    assert evaluation.clf_f1() > 0.95
    print(f"Label classifier evaluation F1 score: {evaluation.clf_f1()}")

    # To get the evaluation of the LabelSet given perfect Label classification
    evaluation = pipeline.evaluate_clf()
    assert evaluation.f1() > 0.9
    print(f"Label Set evaluation F1 score: {evaluation.f1()}")
