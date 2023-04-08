"""Test creating Paragraph Annotations."""
from konfuzio_sdk.data import Project
from konfuzio_sdk.tokenizer.paragraph_and_sentence import ParagraphTokenizer

project = Project(id_=458)

document = project.get_document_by_id(601418)

tokenizer = ParagraphTokenizer(mode='detectron', create_detectron_labels=True)

document = tokenizer(document)
