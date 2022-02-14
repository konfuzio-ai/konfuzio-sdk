# Konfuzio SDK

![Downloads](https://pepy.tech/badge/konfuzio-sdk)

The Konfuzio Software Development Kit (Konfuzio SDK) provides a 
[Python API](https://dev.konfuzio.com/sdk/sourcecode.html) to interact with the
[Konfuzio Server](https://dev.konfuzio.com/index.html#konfuzio-server). The SDK allows you to retrieve visual and text
features to build your own document models. Konfuzio Server serves as an UI to define the data structure, manage
training/test data and to deploy your models as API.


Function               | Public Host (Free)       | On-Site (Paid)      |
:--------------------- | :------------------------| :-------------------|
OCR Text               | :heavy_check_mark:       |  :heavy_check_mark: |
OCR Handwriting        | :heavy_check_mark:       |  :heavy_check_mark: |
Text Annotation        | :heavy_check_mark:       |  :heavy_check_mark: |
PDF Annotation         | :heavy_check_mark:       |  :heavy_check_mark: |
Image Annotation       | :heavy_check_mark:       |  :heavy_check_mark: |
Table Annotation       | :heavy_check_mark:       |  :heavy_check_mark: |
Download HOCR          | :heavy_check_mark:       |  :heavy_check_mark: |
Download Images        | :heavy_check_mark:       |  :heavy_check_mark: |
Download PDF with OCR  | :heavy_check_mark:       |  :heavy_check_mark: |
Deploy AI models       | :heavy_multiplication_x: |  :heavy_check_mark: |

## Installation

As developer register on our [public HOST for free: https://app.konfuzio.com](https://app.konfuzio.com/accounts/signup/)

Then you can use pip to install Konfuzio SDK:  `pip install konfuzio_sdk`

Use `konfuzio_sdk init` to create a Token to connect to the Konfuzio Server. This will create variables `KONFUZIO_USER`,
`KONFUZIO_TOKEN` and `KONFUZIO_HOST` in an `.env` file in your working directory.

Find the full installation guide [here](https://dev.konfuzio.com/sdk/configuration_reference.html)
or setup PyCharm as described [here](https://dev.konfuzio.com/sdk/quickstart_pycharm.html).

## Python

 ```python
from konfuzio_sdk.data import Project, Document, LabelSet, Label, AnnotationSet, Annotation, Span

# Initialize the project:
my_project = Project(id_=YOUR_PROJECT_ID)

# Get any project online
doc: Document = my_project.get_document_by_id(DOCUMENT_ID_ONLNIE)

# Get the Annotations in a Document
doc.annotations()

# Filter Annotations by Label
label = my_project.get_label_by_name('MY_OWN_LABEL_NAME')
doc.annotations(label=label)

# Or get all Annotations that belong to one Label
label.annotations

# Force a project update. To save time documents will only be updated if they have changed.
my_project.update()

# Calculate the Bounding Box of a Text Offset.
label = Label(project=my_project)
label_set = LabelSet(project=my_project)
annotation_set = AnnotationSet(document=doc, label_set=label_set)
annotation = Annotation(label=label, annotation_set=annotation_set, label_set=label_set, document=doc)
span_with_bbox_information = Span(annotation=annotation, start_offset=10, end_offset=50).bbox()
```

## CLI

We provide the basic function to create a new Project via CLI:

  `konfuzio_sdk create_project YOUR_PROJECT_NAME`

You will see "Project `{YOUR_PROJECT_NAME}` (ID `{YOUR_PROJECT_ID}`) was created successfully!" printed.

And download any project via the id:

  `konfuzio_sdk download_data YOUR_PROJECT_ID`


## Tutorials

- [Example Usage](https://dev.konfuzio.com/sdk/examples/examples.html): Some examples of the basic Konfuzio SDK functionalities.
- [Create Regex-based Annotations](https://dev.konfuzio.com/sdk/examples/examples.html#create-regex-based-annotations)
: An example of how to create regex-based annotations in a Konfuzio project.
- [Retrain Flair NER-Ontonotes-Fast with Human Revised Annotations](https://dev.konfuzio.com/sdk/examples/examples.html#retrain-flair-ner-ontonotes-fast-with-human-revised-annotations) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/konfuzio-ai/document-ai-python-sdk/blob/master/docs/sdk/examples/human_in_the_loop.ipynb): An example of how Konfuzio SDK package can be used in a pipeline to have an easy feedback workflow can be seen in this tutorial
- [Count Relevant Expressions in Annual Reports](https://dev.konfuzio.com/sdk/examples/examples.html#count-relevant-expressions-in-annual-reports) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/konfuzio-ai/document-ai-python-sdk/blob/master/docs/sdk/examples/word_count.ipynb): An example of how to retrieve structured and organized information from documents.

The Konfuzio Server Tutorial:

[![Watch the video](https://img.youtube.com/vi/KJC48LMvM2I/maxresdefault.jpg)](https://youtu.be/KJC48LMvM2I)

## References

- [Konfuzio SDK Python API - Source Code](https://dev.konfuzio.com/sdk/sourcecode.html)
- [Konfuzio Server REST API](https://app.konfuzio.com/v2/swagger/)
- [How to Contribute](https://dev.konfuzio.com/sdk/contribution.html)
- [Issue Tracker](https://github.com/konfuzio-ai/document-ai-python-sdk/issues)
- [MIT License](https://github.com/konfuzio-ai/document-ai-python-sdk/blob/master/LICENSE.md)
- [Konfuzio Homepage](https://www.konfuzio.com/en/)
