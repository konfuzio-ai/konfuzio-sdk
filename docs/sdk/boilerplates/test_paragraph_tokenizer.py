"""Test creating Paragraph Annotations."""


def test_paragraph_tokenizer():
    """Test Paragraph Tokenizer."""
    from tests.variables import TEST_PROJECT_ID, TEST_DOCUMENT_ID

    YOUR_PROJECT_ID, YOUR_DOCUMENT_ID = TEST_PROJECT_ID, TEST_DOCUMENT_ID

    # start init1
    from konfuzio_sdk.data import Project
    from konfuzio_sdk.tokenizer.paragraph_and_sentence import ParagraphTokenizer

    # initialize a Project and fetch a Document to tokenize
    project = Project(id_=YOUR_PROJECT_ID)

    document = project.get_document_by_id(YOUR_DOCUMENT_ID)

    # create the ParagraphTokenizer and tokenize the Document

    tokenizer = ParagraphTokenizer(mode='detectron', create_detectron_labels=True)

    document = tokenizer(document)

    document.get_page_by_index(0).get_annotations_image()
    # end init1


def test_paragraph_tokenizer_line_distance_mode():
    """Test Paragraph Tokenizer in line_distance mode."""
    from tests.variables import TEST_PROJECT_ID, TEST_DOCUMENT_ID

    YOUR_PROJECT_ID, YOUR_DOCUMENT_ID = TEST_PROJECT_ID, TEST_DOCUMENT_ID

    # start init2
    from konfuzio_sdk.data import Project
    from konfuzio_sdk.tokenizer.paragraph_and_sentence import ParagraphTokenizer

    # initialize a Project and fetch a Document to tokenize
    project = Project(id_=YOUR_PROJECT_ID)

    document = project.get_document_by_id(YOUR_DOCUMENT_ID)

    # create the ParagraphTokenizer and tokenize the Document

    tokenizer = ParagraphTokenizer(mode='line_distance')

    document = tokenizer(document)

    document.get_page_by_index(0).get_annotations_image(display_all=True)  # display_all to show NO_LABEL Annotations
    # end init2
