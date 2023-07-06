## Data Validation Rules

Konfuzio automatically applies a set of rules for validating data within a [Project](https://dev.konfuzio.com/sdk/sourcecode.html#project). 
Data Validation Rules ensure that Training and Test data is consistent and well formed for training an
[Extraction AI](https://dev.konfuzio.com/sdk/sourcecode.html#extraction-ai) with Konfuzio.

In general, if a Document fails any of the checks described in the next sections, it will not be possible to train an 
AI with that Document.

More specifically:
- If a Document fails any of the checks described in the [Bbox Validation Rules](#bbox-validation) section, it 
will not be possible to initialize the Project as a Python object (such as with 
`project = Project(YOUR_PROJECT_ID)`), and a `ValueError` will be raised. All other Documents in the Project will be 
able to be initialized.
- If a Document fails any of the checks described in the sections 
[Annotation Validation Rules](#annotation-validation) and [Span Validation Rules](#span-validation), it 
will not be possible to retrieve the Annotations (including their Spans) that fail the specific checks (such as with 
`annotation = document.get_annotation_by_id(YOUR_ANNOTATION_ID)`), and a `ValueError` will be raised. All other 
Annotations in the Document will be retrievable.

### Initializing a Project with the Data Validation Rules enabled

By default, any [Project](https://dev.konfuzio.com/sdk/sourcecode.html#project) has the Data Validation Rules enabled, so nothing 
special needs to be done to enable it.

.. literalinclude:: /sdk/boilerplates/test_data_validation.py
   :language: python
   :start-after: Start initialization
   :end-before: End initialization
   :dedent: 4

### Document Validation Rules

A [Document](https://dev.konfuzio.com/sdk/sourcecode.html#document) passes the Data Validation Rules only if all the
contained Annotations, Spans and Bboxes pass the Data Validation Rules.
If at least one Annotation, Span, or Bbox within a Document fails one of the following checks, the entire Document is 
marked as unsuitable for training an Extraction AI.

.. _annotation-validation:

### Annotation Validation Rules

An [Annotation](https://dev.konfuzio.com/sdk/sourcecode.html#annotation) passes the Data Validation Rules only if:

1. The Annotation is not from a Category different from the Document's Category
2. The Annotation is not entirely overlapping with another Annotation with the same Label
    - It implies that partial overlaps with same Labels are allowed
    - It implies that full overlaps with different Labels are allowed
3. The Annotation has at least one Span

Please note that the Annotation Validation Rules are indifferent about the values of `Annotation.is_correct` or `Annotation.revised`.
For more information about what these boolean values mean, see [Konfuzio Server - Annotations](https://help.konfuzio.com/modules/annotations/index.html).

.. _span-validation:

### Span Validation Rules

A [Span](https://dev.konfuzio.com/sdk/sourcecode.html#span) passes the Data Validation Rules only if:

1. The Span contains non-empty text (the start offset must be strictly greater than the end offset)
2. The Span is contained within a single line of text (must not be distributed across multiple lines)

.. _bbox-validation:

### Bbox Validation Rules

A [Bbox](https://dev.konfuzio.com/sdk/sourcecode.html#bbox) passes the Data Validation Rules only if:

1. The Bbox has non-negative width and height (zero is allowed for compatibility reasons with many OCR engines)
2. The Bbox is entirely contained within the bounds of a Page
3. The character that is mapped by the Bbox must correspond to the text in the Document

### Initializing a Project with the Data Validation Rules disabled

By default, any [Project](https://dev.konfuzio.com/sdk/sourcecode.html#project) has the Data Validation Rules enabled.

A possible reason for choosing to disable the Data Validation Rules that come with the Konfuzio SDK, is that an expert user
wants to define a custom data structure or training pipeline which violates some assumptions normally present in Konfuzio 
Extraction AIs and pipelines.
If you don't want to validate your data, you should initialize the Project with `strict_data_validation=False`.

We highly recommend to keep the Data Validation Rules enabled at all times, as it ensures that Training and Test data 
is consistent for training an Extraction AI. Disabling the Data Validation Rules and training an 
[Extraction AI](https://dev.konfuzio.com/sdk/sourcecode.html#extraction-ai) with potentially duplicated, malformed,
or inconsistent data can **decrease the quality of an Extraction AI**. Only disable them if you know what you are doing.

.. literalinclude:: /sdk/boilerplates/test_data_validation.py
   :language: python
   :start-after: Start no val
   :end-before: End no val
   :dedent: 4
