"""Test creating word-level Bboxes."""


def test_word_bboxes():
    """Test creation of the word-level Bboxes."""
    from copy import deepcopy
    from konfuzio_sdk.data import Project
    from konfuzio_sdk.tokenizer.regex import WhitespaceTokenizer

    from tests.variables import TEST_PROJECT_ID, TEST_DOCUMENT_ID

    YOUR_PROJECT_ID, YOUR_DOCUMENT_ID = TEST_PROJECT_ID, TEST_DOCUMENT_ID
    project = Project(id_=YOUR_PROJECT_ID)
    project = Project(id_=YOUR_PROJECT_ID, strict_data_validation=False)
    document = project.get_document_by_id(YOUR_DOCUMENT_ID)
    document = deepcopy(document)
    tokenizer = WhitespaceTokenizer()
    document = tokenizer.tokenize(document)
    assert len(document.spans()) == 309
    document.get_page_by_index(0).get_annotations_image(display_all=True)
    span_bboxes = [span.bbox() for span in document.spans()]
    assert len(span_bboxes) == 309
