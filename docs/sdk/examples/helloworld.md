## Create Regex-based Annotations

Let's see a simple example of how we can use the `konfuzio_sdk` package to get information on a project and to post annotations.

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
# Getting the Label from the project
my_label = my_project.get_label_by_name(label_name)

# LabelSet to which the Label belongs
label_set = my_label.label_sets[0]

# First document in the project
document = my_project.documents[0]

# Matches of the word/expression in the document
matches_locations = [(m.start(0), m.end(0)) for m in re.finditer(input_expression, document.text)]

# List to save the links to the annotations created
new_annotations_links = []

# Create annotation for each match
for offsets in matches_locations:
    span = Span(start_offset=offsets[0], end_offset=offsets[1])
    annotation_obj = Annotation(
        document=document,
        label=my_label,
        label_set=label_set,
        confidence=1.0,
        spans=[span],
        is_correct=True
    )
    new_annotation_added = annotation_obj.save()
    if new_annotation_added:
        new_annotations_links.append(annotation_obj.get_link())

print(new_annotations_links)

```

## Train Label Regex Tokenizer

You can use the `konfuzio_sdk` package to train a custom Regex tokenizer. 

In this example, you will see how to find regex expressions that match with occurences of the "IBAN" Label in the training data. 

```python
from konfuzio_sdk.data import Project
from konfuzio_sdk.tokenizer.regex import RegexTokenizer
from konfuzio_sdk.tokenizer.base import ListTokenizer

my_project = Project(id_=YOUR_PROJECT_ID)
category = project.get_category_by_id(id_=CATEGORY_ID)

tokenizer = ListTokenizer(tokenizers=[])

iban_label = my_project.get_label_by_name("IBAN")

for regex in iban_label.find_regex(category=category):
    regex_tokenizer = RegexTokenizer(regex=regex)
    tokenizer.tokenizers.append(regex_tokenizer)

# You can then use it to create an Annotation for every matching string in a document.
document = project.get_document_by_id(DOCUMENT_ID)
tokenizer.tokenize(document)

```

## Finding Spans of a Label Not Found by a Tokenizer

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

## Evaluate a Trained Extraction AI Model

In this example we will see how we can evaluate a trained `RFExtractionAI` model. We will assume that we have a trained pickled model available. See [here](https://dev.konfuzio.com/sdk/examples/examples.html#train-a-konfuzio-sdk-model-to-extract-information-from-payslip-documents) for how to train such a model, and check out the [Evaluation](https://dev.konfuzio.com/sdk/sourcecode.html#evaluation) documentation for more details.

```python
from konfuzio_sdk.data import Project
from konfuzio_sdk.trainer.information_extraction import load_model

pipeline = load_model(MODEL_PATH)

# To get the evaluation of the full pipeline
evaluation = pipeline.evaluate_full()
print(f"Full evaluation F1 score: {evaluation.f1()}")
print(f"Full evaluation recall: {evaluation.recall()}")
print(f"Full evaluation precision: {evaluation.precision()}")

# To get the evaluation of the tokenizer alone
evaluation = pipeline.evaluate_tokenizer()
print(f"Tokenizer evaluation F1 score: {evaluation.tokenizer_f1()}")

# To get the evaluation of the Label classifier given perfect tokenization
evaluation = pipeline.evaluate_clf()
print(f"Label classifier evaluation F1 score: {evaluation.clf_f1()}")

# To get the evaluation of the LabelSet given perfect Label classification
evaluation = pipeline.evaluate_label_set_clf()
print(f"Label Set evaluation F1 score: {evaluation.f1()}")

```