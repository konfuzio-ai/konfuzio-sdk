"""Test code examples for finding Spans of a Label not found by the Tokenizer."""
from konfuzio_sdk.data import Project
from konfuzio_sdk.tokenizer.regex import WhitespaceTokenizer

from variables import YOUR_PROJECT_ID

my_project = Project(id_=YOUR_PROJECT_ID)
category = my_project.categories[0]

tokenizer = WhitespaceTokenizer()

label = my_project.get_label_by_name('Austellungsdatum')

spans_not_found = label.spans_not_found_by_tokenizer(tokenizer, categories=[category])
assert len(spans_not_found) == 1

for span in spans_not_found:
    print(f"{span}: {span.offset_string}")
