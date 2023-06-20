"""Test creating word-level Bboxes."""
import pytest


def test_word_bboxes():
    """Test creation of the word-level Bboxes."""
    # start import
    from copy import deepcopy
    from konfuzio_sdk.data import Project
    from konfuzio_sdk.tokenizer.regex import WhitespaceTokenizer

    # end import
    from tests.variables import TEST_PROJECT_ID, TEST_DOCUMENT_ID

    YOUR_PROJECT_ID, YOUR_DOCUMENT_ID = TEST_PROJECT_ID, TEST_DOCUMENT_ID
    # start project
    project = Project(id_=YOUR_PROJECT_ID)
    # end project

    # start document
    document = project.get_document_by_id(YOUR_DOCUMENT_ID)
    # end document
    # start copy
    document = deepcopy(document)
    # end copy
    # start tokenize
    tokenizer = WhitespaceTokenizer()
    document = tokenizer.tokenize(document)
    # end tokenize
    assert len(document.spans()) == 309
    # start image
    document.get_page_by_index(0).get_annotations_image(display_all=True)
    # end image
    # start spans
    span_bboxes = [span.bbox() for span in document.spans()]
    # end spans
    assert len(span_bboxes) == 309


@pytest.mark.skip(reason="Copy of test above with uninterrupted code lines.")
def test_word_bboxes_uninterrupted():
    """Copy of test above with uninterrupted code lines."""
    from tests.variables import TEST_PROJECT_ID, TEST_DOCUMENT_ID

    YOUR_PROJECT_ID, YOUR_DOCUMENT_ID = TEST_PROJECT_ID, TEST_DOCUMENT_ID

    # start full word_bboxes
    from copy import deepcopy
    from konfuzio_sdk.data import Project
    from konfuzio_sdk.tokenizer.regex import WhitespaceTokenizer

    project = Project(id_=YOUR_PROJECT_ID)

    document = project.get_document_by_id(YOUR_DOCUMENT_ID)

    document = deepcopy(document)

    tokenizer = WhitespaceTokenizer()
    document = tokenizer.tokenize(document)

    document.get_page_by_index(0).get_annotations_image(display_all=True)

    span_bboxes = [span.bbox() for span in document.spans()]
    # end full word_bboxes
    assert len(span_bboxes) == 309
