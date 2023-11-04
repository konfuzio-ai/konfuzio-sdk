"""Test code examples for evaluation of the Extraction AI in the documentation."""
import pytest
import sys
import unittest

from konfuzio_sdk.settings_importer import is_dependency_installed

MODEL_PATH = 'tests/trainer/2023-05-11-15-44-10_lohnabrechnung_rfextractionai_.pkl'


@pytest.mark.skipif(
    not is_dependency_installed('torch'),
    reason='Required dependencies not installed.',
)
@unittest.skipIf(sys.version_info[:2] != (3, 8), 'This AI can only be loaded on Python 3.8.')
def test_evaluate_extraction_ai():
    """Test evaluation of the Extraction AI."""
    from tests.variables import OFFLINE_PROJECT, TEST_DOCUMENT_ID
    from konfuzio_sdk.trainer.information_extraction import RFExtractionAI
    from konfuzio_sdk.data import Project

    project = Project(id_=None, project_folder=OFFLINE_PROJECT)
    test_document = project.get_document_by_id(TEST_DOCUMENT_ID)

    # start init

    pipeline = RFExtractionAI.load_model(MODEL_PATH)
    # end init

    pipeline.test_documents = [test_document]

    # start scores
    # To get the evaluation of the full pipeline
    evaluation = pipeline.evaluate_full()
    print(f"Full evaluation F1 score: {evaluation.f1()}")
    print(f"Full evaluation recall: {evaluation.recall()}")
    print(f"Full evaluation precision: {evaluation.precision()}")

    # To get the evaluation of the Tokenizer alone
    evaluation = pipeline.evaluate_tokenizer()
    print(f"Tokenizer evaluation F1 score: {evaluation.tokenizer_f1()}")

    # To get the evaluation of the Label classifier given perfect tokenization
    evaluation = pipeline.evaluate_clf()
    print(f"Label classifier evaluation F1 score: {evaluation.clf_f1()}")

    # To get the evaluation of the LabelSet given perfect Label classification
    evaluation = pipeline.evaluate_clf()
    print(f"Label Set evaluation F1 score: {evaluation.f1()}")
    # end scores

    # start scores_tokenizer
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
    # end scores_tokenizer
