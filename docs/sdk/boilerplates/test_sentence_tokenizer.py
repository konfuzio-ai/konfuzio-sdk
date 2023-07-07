"""Test creating Paragraph Annotations."""


def test_sentence_tokenizer():
    """Test sentence tokenizer."""
    from tests.variables import TEST_PROJECT_ID, TEST_DOCUMENT_ID

    YOUR_PROJECT_ID, YOUR_DOCUMENT_ID = TEST_PROJECT_ID, TEST_DOCUMENT_ID

    # start import
    from konfuzio_sdk.data import Project
    from konfuzio_sdk.tokenizer.paragraph_and_sentence import SentenceTokenizer

    # initialize a Project and fetch a Document to tokenize
    project = Project(id_=YOUR_PROJECT_ID)
    project = Project(id_=YOUR_PROJECT_ID, strict_data_validation=False)

    document = project.get_document_by_id(YOUR_DOCUMENT_ID)

    # create the SentenceTokenizer and tokenize the Document

    tokenizer = SentenceTokenizer(mode='detectron')

    document = tokenizer(document)

    document.get_page_by_index(0).get_annotations_image(display_all=True)
    # end import
