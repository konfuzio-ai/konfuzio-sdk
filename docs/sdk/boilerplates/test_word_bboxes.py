"""Test creating word-level Bboxes."""


def test_word_bboxes():
    """Test creation of the word-level Bboxes."""
    from tests.variables import TEST_DOCUMENT_ID, TEST_PROJECT_ID

    YOUR_PROJECT_ID, YOUR_DOCUMENT_ID = TEST_PROJECT_ID, TEST_DOCUMENT_ID
    # start full word_bboxes
    # start import
    from copy import deepcopy

    from konfuzio_sdk.data import Project
    from konfuzio_sdk.tokenizer.regex import WhitespaceTokenizer

    # end import

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
    # end full word_bboxes
    assert len(span_bboxes) == 309
