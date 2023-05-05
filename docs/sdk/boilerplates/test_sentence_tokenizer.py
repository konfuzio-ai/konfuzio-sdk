"""Test creating Paragraph Annotations."""
from konfuzio_sdk.data import Project
from konfuzio_sdk.tokenizer.paragraph_and_sentence import SentenceTokenizer
from tests.variables import TEST_PROJECT_ID, TEST_DOCUMENT_ID

YOUR_PROJECT_ID, YOUR_DOCUMENT_ID = TEST_PROJECT_ID, TEST_DOCUMENT_ID
# initialize a Project and fetch a Document to tokenize

project = Project(id_=YOUR_PROJECT_ID)
document = project.get_document_by_id(YOUR_DOCUMENT_ID)

project = Project(id_=458)
document = project.get_document_by_id(601418)

# create the SentenceTokenizer and tokenize the Document

tokenizer = SentenceTokenizer(mode='line_distance')

document = tokenizer(document)
