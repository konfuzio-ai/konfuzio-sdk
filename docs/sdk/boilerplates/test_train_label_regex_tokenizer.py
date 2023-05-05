"""Test code examples for training a Label regex Tokenizer."""
from konfuzio_sdk.data import Project
from konfuzio_sdk.tokenizer.regex import RegexTokenizer
from konfuzio_sdk.tokenizer.base import ListTokenizer

from tests.variables import TEST_PROJECT_ID, TEST_DOCUMENT_ID

YOUR_PROJECT_ID, YOUR_CATEGORY_ID, YOUR_DOCUMENT_ID = TEST_PROJECT_ID, 63, TEST_DOCUMENT_ID

my_project = Project(id_=YOUR_PROJECT_ID)
category = my_project.get_category_by_id(id_=YOUR_CATEGORY_ID)

tokenizer = ListTokenizer(tokenizers=[])

label = my_project.get_label_by_name("Lohnart")

for regex in label.find_regex(category=category):
    regex_tokenizer = RegexTokenizer(regex=regex)
    tokenizer.tokenizers.append(regex_tokenizer)

# You can then use it to create an Annotation for every matching string in a Document.
document = my_project.get_document_by_id(YOUR_DOCUMENT_ID)
tokenizer.tokenize(document)
assert len(document.spans()) == 179
