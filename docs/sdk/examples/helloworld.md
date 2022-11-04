## Create Regex-based Annotations

Let's see a simple example of how can we use the `konfuzio_sdk` package to get information on a project and to post annotations.

You can follow the example below to post annotations of a certain word or expression in the first document uploaded.

```python
import re

from konfuzio_sdk.data import Project, Annotation, Label

my_project = Project(id_=YOUR_PROJECT_ID)

# Word/expression to annotate in the document
# should match an existing one in your document
input_expression = "John Smith"

# Label for the annotation
label_name = "Name"
# Creation of the Label in the project default label set
my_label = Label(my_project, text=label_name)
# Saving it online
my_label.save()

# Label Set where label belongs
label_set_id = my_label.label_sets[0].id_

# First document in the project
document = my_project.documents[0]

# Matches of the word/expression in the document
matches_locations = [(m.start(0), m.end(0)) for m in re.finditer(input_expression, document.text)]

# List to save the links to the annotations created
new_annotations_links = []

# Create annotation for each match
for offsets in matches_locations:
    annotation_obj = Annotation(
        document=document,
        document_id=document.id_,
        start_offset=offsets[0],
        end_offset=offsets[1],
        label=my_label,
        label_set_id=label_set_id,
        accuracy=1.0,
    )
    new_annotation_added = annotation_obj.save()
    if new_annotation_added:
        new_annotations_links.append(annotation_obj.get_link())

print(new_annotations_links)

```


### Finding Spans of a Label Not Found by a Tokenizer

Here is an example of how to use the `Label.spans_not_found_by_tokenizer` method. This will allow you to determine if a RegexTokenizer is suitable at finding the Spans of a Label, or what Spans might have been annotated wrong. Say, you have a number of annotations assigned to the `IBAN` Label and want to know which Spans would not be found when using the WhiteSpace Tokenizer. You can follow this example to find all the relevant Spans.

```python
from konfuzio_sdk.data import Project, Annotation, Label
from konfuzio_sdk.tokenizer.regex import WhitespaceTokenizer

my_project = Project(id_=YOUR_PROJECT_ID)
category = Project.categories[0]

tokenizer = WhitespaceTokenizer()

iban_label = project.get_label_by_name('IBAN')

spans_not_found = iban_label.spans_not_found_by_tokenizer(tokenizer, categories=[category])

for span in spans_not_found:
    print(f"{span}: {span.offset_string}")

```
