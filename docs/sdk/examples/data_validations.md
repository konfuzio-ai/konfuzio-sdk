## Data Validation Rules

Konfuzio automatically applies a set of rules for validating data within a [Project](https://dev.konfuzio.com/sdk/sourcecode.html#project). 
Data validation ensures that Training and Test data is consistent and well formed for training an
[Extraction AI](https://dev.konfuzio.com/sdk/sourcecode.html#extraction-ai) with Konfuzio.

If a Document fails one of the following checks, it will not be possible to train an Extraction AI with that Document.

### Initializing a Project with data validations enabled

By default, any [Project](https://dev.konfuzio.com/sdk/sourcecode.html#project) has data validation enabled, so nothing 
special needs to be done to enable it.

```python
from konfuzio_sdk.data import Project

project = Project(id_=YOUR_PROJECT_ID)  # all the data in this project will be validated
```

### Document Validation Rules

A [Document](https://dev.konfuzio.com/sdk/sourcecode.html#document) passes the data validation rules only if all the
contained Annotations, Spans and Bboxes pass the data validation rules.
If at least one Annotation, Span, or Bbox within a Document fails one of the following checks, the entire Document is 
marked as unsuitable for training an Extraction AI.

### Annotation Validation Rules

An [Annotation](https://dev.konfuzio.com/sdk/sourcecode.html#annotation) passes the data validation rules only if:

1. The Annotation is not from a Category different from the Document's Category
2. The Annotation is not entirely overlapping with another Annotation (partial overlaps are allowed)
3. The Annotation has at least one Span (in other words, it should contain text and not be an empty Annotation)

### Span Validation Rules

A [Span](https://dev.konfuzio.com/sdk/sourcecode.html#span) passes the data validation rules only if:

1. The Span contains non-empty text (the start offset must be strictly greater than the end offset)
2. The Span is contained within a single line of text (must not be distributed across multiple lines)

### Bbox Validation Rules

A [Bbox](https://dev.konfuzio.com/sdk/sourcecode.html#bbox) passes the data validation rules only if:

1. The Bbox has positive width and height
2. The Bbox is entirely contained within the bounds of a Page
3. The text of the Bbox must correspond to the text in the Document (please note that, in Konfuzio, each Bbox within a Document is associated to a single character from the Document's text by default)

### Initializing a Project with data validations disabled

By default, any [Project](https://dev.konfuzio.com/sdk/sourcecode.html#project) has data validation enabled.

A possible reason for choosing to disable the data validations that come with the Konfuzio SDK, is that an expert user
wants to define a custom data structure or training pipeline which violates some assumptions normally present in Konfuzio
Extraction AIs and pipelines.
If you don't want to validate your data, you should initialize the Project with `strict_data_validation=False`.

We highly recommend to keep data validation enabled at all times, as it ensures that Training and Test data is consistent
for training an Extraction AI. Disabling data validation and training an 
[Extraction AI](https://dev.konfuzio.com/sdk/sourcecode.html#extraction-ai) with potentially duplicated, malformed,
or inconsistent data can **decrease the quality of an Extraction AI**. Only disable it if you know what you are doing.

```python
from konfuzio_sdk.data import Project

project = Project(id_=YOUR_PROJECT_ID, strict_data_validation=False)
```
