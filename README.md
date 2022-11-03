# Konfuzio SDK

![Downloads](https://pepy.tech/badge/konfuzio-sdk)

The Konfuzio Software Development Kit (Konfuzio SDK) provides a
[Python API](https://dev.konfuzio.com/sdk/sourcecode.html) to interact with the
[Konfuzio Server](https://dev.konfuzio.com/index.html#konfuzio-server).

## Features

The SDK allows you to retrieve visual and text features to build your own document models. Konfuzio Server serves as an
UI to define the data structure, manage training/test data and to deploy your models as API.

Function               | Public Host Free*                         | On-Site (Paid)      |
:--------------------- | :---------------------------------------- | :-------------------|
OCR Text               | :heavy_check_mark:                        |  :heavy_check_mark: |
OCR Handwriting        | :heavy_check_mark:                        |  :heavy_check_mark: |
Text Annotation        | :heavy_check_mark:                        |  :heavy_check_mark: |
PDF Annotation         | :heavy_check_mark:                        |  :heavy_check_mark: |
Image Annotation       | :heavy_check_mark:                        |  :heavy_check_mark: |
Table Annotation       | :heavy_check_mark:                        |  :heavy_check_mark: |
Download HOCR          | :heavy_check_mark:                        |  :heavy_check_mark: |
Download Images        | :heavy_check_mark:                        |  :heavy_check_mark: |
Download PDF with OCR  | :heavy_check_mark:                        |  :heavy_check_mark: |
Deploy AI models       | :heavy_multiplication_x:                  |  :heavy_check_mark: |

`*` Under fair use policy: We will impose 10 pages/hour throttling eventually.

## Installation

As developer register on our [public HOST for free: https://app.konfuzio.com](https://app.konfuzio.com/accounts/signup/)

Then you can use pip to install Konfuzio SDK and run init:

    pip install konfuzio_sdk

    konfuzio_sdk init

The init will create a Token to connect to the Konfuzio Server. This will create variables `KONFUZIO_USER`,
`KONFUZIO_TOKEN` and `KONFUZIO_HOST` in an `.env` file in your working directory.

Find the full installation guide [here](https://dev.konfuzio.com/sdk/configuration_reference.html)
or setup PyCharm as described [here](https://dev.konfuzio.com/sdk/quickstart_pycharm.html).

## Basics

 ```python
from konfuzio_sdk.data import Project, Document

# Initialize the project:
my_project = Project(id_='YOUR_PROJECT_ID')

# Get any project online
doc: Document = my_project.get_document_by_id('DOCUMENT_ID_ONLNIE')

# Get the Annotations in a Document
doc.annotations()

# Filter Annotations by Label
label = my_project.get_label_by_name('MY_OWN_LABEL_NAME')
doc.annotations(label=label)

# Or get all Annotations that belong to one Label
label.annotations

# Force a project update. To save time documents will only be updated if they have changed.
my_project.update()
```

Find more explanations in the [Examples](https://dev.konfuzio.com/sdk/examples/examples.html).

## Regex

**Pro Tip**: Read our technical blog post [Automated Regex](https://helm-nagel.com/Automated-Regex-Generation-based-on-examples) to find out how we use Regex to detect outliers in our annotated data.

```python
from konfuzio_sdk.regex import suggest_regex_for_string
from konfuzio_sdk.data import Project, Label

my_project = Project(id_='YOUR_PROJECT_ID')
label: Label = my_project.get_label_by_name('MY_OWN_LABEL_NAME')

# Get Regex tokens to capture (nearly) all annotations of this Label
tokens = label.tokens()
assert tokens == [
    "(?P<GesamtBrutto_N_4420363_2111>\\d\\.\\d\\d\\d\\,\\d\\d)",
    "(?P<GesamtBrutto_N_9812334_1498>\\d\\d\\d\\,\\d\\d)"
]

# Get optimize regex (Can be multiple if multiple create a higher accuracy than a single one)
label_regex = label.regex()
assert label_regex == [
    "[ ]+(?:(?P<GesamtBrutto_N_4420363_2111>\\d\\.\\d\\d\\d\\,\\d\\d)|(?P<GesamtBrutto_N_9812334_1498>\\d\\d\\d\\,\\d\\d))\n"
]

# Suggest a RegEx for a string without optimization
regex = suggest_regex_for_string('Date: 20.05.2022')
assert regex == 'Date:[ ]+\\d\\d\\.\\d\\d\\.\\d\\d\\d\\d'
```

## Tokenizer

Create a Tokenizer based on a Regex and evaluate it on a Document level.

```python
from konfuzio_sdk.data import Project
from konfuzio_sdk.tokenizer.regex import RegexTokenizer

my_project = Project(id_='YOUR_PROJECT_ID')
document = my_project.get_document_by_id(document_id='YOUR_DOCUMENT_ID')

# Define the Regex expression
regex = r'[^ \n\t\f]+'

# Build a Tokenizer based on Regex 
tokenizer = RegexTokenizer(regex=regex)
assert tokenizer.regex == regex

# Evaluate the Tokenizer in a Document
evaluation = tokenizer.evaluate(document)

# Ratio of correct Spans found by the Tokenizer in the Document
ratio_of_spans_found = evaluation.is_found_by_tokenizer.sum() / evaluation.is_correct.sum()

```

## Add visual features to text

Calculate the bounding box of a Span using the start and end character.

```python
from pprint import pprint

from konfuzio_sdk.data import Project, LabelSet, Label, AnnotationSet, Annotation, Span
import os

OFFLINE_PROJECT = os.path.join("tests", "example_project_data")

my_project = Project(id_=None, project_folder=OFFLINE_PROJECT)  # use offline data and don't connect to Server
document = my_project.get_document_by_id(44823)
label = Label(project=my_project)
label_set = LabelSet(project=my_project, categories=[document.category])
annotation_set = AnnotationSet(document=document, label_set=label_set)

span = Span(start_offset=60, end_offset=65)

annotation = Annotation(
    label=label,
    annotation_set=annotation_set,
    label_set=label_set,
    document=document,
    spans=[span],
)

span_with_bbox_information = span.bbox()

pprint(span_with_bbox_information.__dict__)

```

```python
{'_line_index': None,
 '_page_index': 0,
 'annotation': Annotation (None) None (60, 65),
 'bottom': 32.849,
 'end_offset': 65,
 'id_local': 74,
 'start_offset': 60,
 'top': 23.849,
 'x0': 426.0,
 'x1': 442.8,
 'y0': 808.831,
 'y1': 817.831}
````


## CLI

We provide the basic function to create a new Project via CLI:

`konfuzio_sdk create_project YOUR_PROJECT_NAME`

You will see "Project `{YOUR_PROJECT_NAME}` (ID `{YOUR_PROJECT_ID}`) was created successfully!" printed.

And download any project via the id:

`konfuzio_sdk download_data YOUR_PROJECT_ID`

## Tutorials

- [Train a Konfuzio SDK Model to Extract Information From Payslip Documents](https://dev.konfuzio.com/sdk/examples/examples.html#train-a-konfuzio-sdk-model-to-extract-information-from-payslip-documents) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/konfuzio-ai/document-ai-python-sdk/blob/master/docs/sdk/examples/RFExtractionAI%20Demo.ipynb):
  An example of how Konfuzio SDK package can be used in a pipeline to have an easy feedback workflow can be seen in this
  tutorial
- [Automate Annotations with Regex](https://dev.konfuzio.com/sdk/examples/examples.html#create-regex-based-annotations)
  : An example of how to create regex-based annotations in a Konfuzio project.
- [Retrain Flair NER-Ontonotes-Fast with Human Revised Annotations](https://dev.konfuzio.com/sdk/examples/examples.html#retrain-flair-ner-ontonotes-fast-with-human-revised-annotations) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/konfuzio-ai/document-ai-python-sdk/blob/master/docs/sdk/examples/human_in_the_loop.ipynb):
  An example of how Konfuzio SDK package can be used in a pipeline to have an easy feedback workflow can be seen in this
  tutorial
- [Count Relevant Expressions in Annual Reports](https://dev.konfuzio.com/sdk/examples/examples.html#count-relevant-expressions-in-annual-reports) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/konfuzio-ai/document-ai-python-sdk/blob/master/docs/sdk/examples/word_count.ipynb):
  An example of how to retrieve structured and organized information from documents.

The Konfuzio Server Tutorial:

[![Watch the video](https://img.youtube.com/vi/KJC48LMvM2I/maxresdefault.jpg)](https://youtu.be/KJC48LMvM2I)

## References

- [Konfuzio SDK Python API - Source Code](https://dev.konfuzio.com/sdk/sourcecode.html)
- [Konfuzio Server REST API](https://app.konfuzio.com/v2/swagger/)
- [How to Contribute](https://dev.konfuzio.com/sdk/contribution.html)
- [Issue Tracker](https://github.com/konfuzio-ai/document-ai-python-sdk/issues)
- [MIT License](https://github.com/konfuzio-ai/document-ai-python-sdk/blob/master/LICENSE.md)
- [Konfuzio Homepage](https://www.konfuzio.com/en/)

# Supported CRUD Operations

| data structure | Create/Upload | Edit | Update (sync) | Delete     |
|----------------|---------------|------|---------------|------------|
| Project        | yes           | x    | yes           | x          |
| Document       | yes           | yes  | yes           | only local |
| Label          | yes           | x    | x             | x          |
| Annotation     | yes           | x    | x             | yes        |
| Label set      | x             | x    | x             | x          |
| Annotation set | x             | x    | x             | x          |
| Category       | x             | x    | x             | x          |
