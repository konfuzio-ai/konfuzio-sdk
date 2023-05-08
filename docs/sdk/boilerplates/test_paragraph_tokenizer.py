"""Test creating Paragraph Annotations."""
from konfuzio_sdk.data import Project
from konfuzio_sdk.tokenizer.paragraph_and_sentence import ParagraphTokenizer
from tests.variables import TEST_PROJECT_ID, TEST_DOCUMENT_ID

YOUR_PROJECT_ID, YOUR_DOCUMENT_ID = TEST_PROJECT_ID, TEST_DOCUMENT_ID
# initialize a Project and fetch a Document to tokenize


def test_paragraph_tokenizer():
    """Test Paragraph Tokenizer."""
    project = Project(id_=YOUR_PROJECT_ID)
    document = project.get_document_by_id(YOUR_DOCUMENT_ID)

    project = Project(id_=458)
    document = project.get_document_by_id(601418)

    # create the ParagraphTokenizer and tokenize the Document

    tokenizer = ParagraphTokenizer(mode='detectron', create_detectron_labels=True)

    document = tokenizer(document)
