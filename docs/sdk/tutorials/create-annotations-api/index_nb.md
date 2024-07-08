---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

## Create, change and delete Annotations using the wrapped API call

---

**Prerequisites:**
- Data Layer concepts of Konfuzio: Project, Document, Annotation, Span, Bounding Box, Label, Label Set, Annotation Set
- Understanding of concepts of REST API

**Difficulty:** Medium

**Goal:** Explain how to create different types of Annotation (textual, visual) using the methods from the SDK listed in
`konfuzio_sdk.api`, how to change or delete them.

---

### Environment
You need to install the Konfuzio SDK before diving into the tutorial. \
To get up and running quickly, you can use our Colab Quick Start notebook. \
<a href="https://colab.research.google.com/github/konfuzio-ai/konfuzio-sdk/blob/master/notebooks/Quick_start_template_for_Konfuzio_SDK.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

As an alternative you can follow the [installation section](../get_started.html#install-sdk) to install and initialize the Konfuzio SDK locally or on an environment of your choice.

### Introduction

There are several ways to create an Annotation in a Document: using the SmartView or DVUI on Konfuzio's app or an 
on-prem installation, via the `Annotation` class in the SDK or via the direct call to the API endpoint 
`api/v3/annotations`. In Server documentation, we already provide an [instruction](https://dev.konfuzio.com/web/api-v3.html#create-an-annotation) 
on creating an Annotation via the POST request using `curl`; in this tutorial, we will explain how to make this 
request using the methods from `konfuzio_sdk.api` which serves as a wrapper around the calls to the API.

Let's start by making necessary imports:
```python tags=["remove-cell"]
import logging
from konfuzio_sdk.api import restore_snapshot
from konfuzio_sdk.data import Project

logging.getLogger("konfuzio_sdk").setLevel(logging.ERROR)
YOUR_PROJECT_ID = restore_snapshot(snapshot_id=65)
project = Project(id_=YOUR_PROJECT_ID)
original_document_text = Project(id_=46).get_document_by_id(44823).text
YOUR_DOCUMENT_ID = [document for document in project.documents if document.text == original_document_text][0].id_
YOUR_LABEL_ID = project.get_label_by_name('Steuer-Brutto').id_
YOUR_LABEL_SET_ID = project.get_label_set_by_name('Lohnabrechnung').id_
NEW_LABEL_ID = YOUR_LABEL_ID

project.get_document_by_id(YOUR_DOCUMENT_ID).get_bbox()
```
```python
import json 

from konfuzio_sdk.api import post_document_annotation, delete_document_annotation
from konfuzio_sdk.data import Span, Project
```

To create any Annotation, it is necessary to provide several fields to `post_document_annotation`:

- `document_id`: ID of a Document in which Annotation is created
- `label`: ID of a Label assigned to the Annotation
- `spans`: Coordinates of Bounding Boxes representing the position of the Annotation in the Document. 
- either `label_set` or `annotation_set`: provide an ID of a Label Set if you want to create Annotation within a new 
Annotation Set, or provide an ID of an existing Annotation Set if you want to add Annotation into it without creating a
new Annotation Set.

### Creating a textual Annotation

To create an Annotation that is based on existing text of a Document, let's firstly define the test Document and the 
Span that will be passed as the `spans` argument. You can define one or more Spans.
```python
test_document = Project(id_=YOUR_PROJECT_ID).get_document_by_id(YOUR_DOCUMENT_ID)
spans = [Span(document=test_document, start_offset=3067, end_offset=3074)]
```

Next, let's specify arguments for a POST request that creates Annotations and send it to the server. We want to create
an Annotation within a new Annotation Set so we specify `label_set_id`.
```python
response = post_document_annotation(document_id=YOUR_DOCUMENT_ID, spans=spans, label_id=YOUR_LABEL_ID, confidence=100.0,
                                    label_set_id=YOUR_LABEL_SET_ID)
```
Let's check if an Annotation has been created successfully and has a Span coinciding with the one created by us above:
```python
response = json.loads(response.text)
print(response['span'])
```
```python tags=['remove-cell']
negative_id = delete_document_annotation(response['id'])
assert delete_document_annotation(negative_id, delete_from_database=True).status_code == 204
```

### Creating a visual Annotation

To create an Annotation that is based on Bounding Boxes' coordinates, let's create a dictionary of coordinates that will
be passed as the `spans` argument. You can define one or more Bounding Boxes. Note that you don't need to specify 
offsets, only the `page_index` is needed.
```python tags=['remove-cell']
YOUR_DOCUMENT_ID = YOUR_DOCUMENT_ID + 11
YOUR_LABEL_ID = project.get_label_by_name('Bezeichnung')
```
```python
bboxes = [
        {'page_index': 0, 'x0': 457, 'x1': 480, 'y0': 290, 'y1': 303},
        {'page_index': 0, 'x0': 452.16, 'x1': 482.64, 'y0': 306, 'y1': 313,}
]
```
Next, we specify arguments for a POST request to create an Annotation and send it to the server. We want to create
an Annotation within a new Annotation Set, so we specify `label_set_id`.
```python
response = post_document_annotation(document_id=YOUR_DOCUMENT_ID, spans=bboxes, label_id=YOUR_LABEL_ID, confidence=100.0,
                                    label_set_id=YOUR_LABEL_SET_ID)
```
Let's check if an Annotation has been created successfully:
```python
response = json.loads(response.text)
print(response['span'])
```
```python tags=['remove-cell']
assert delete_document_annotation(response['id'], delete_from_database=True)
YOUR_ANNOTATION_ID = test_document.annotations(label=project.get_label_by_id(NEW_LABEL_ID))[0].id_
```

### Change an Annotation
To update details of an Annotation, use `change_document_annotation` method from `konfuzio_sdk.api`. You can specify 
a Label, a Label Set, an Annotation Set, `is_correct` and `revised` statuses, Span list and selection Bbox to be 
updated to a new value.
```python 
from konfuzio_sdk.api import change_document_annotation

response = change_document_annotation(annotation_id=YOUR_ANNOTATION_ID, label=NEW_LABEL_ID)
```
Let's check if an Annotation's Label was changed successfully:
```python
print(response.json()['label'])
```

### Delete an Annotation
To delete an Annotation, use `delete_document_annotation` method from `konfuzio_sdk.api`. This method runs in two modes:
soft deletion (does not delete from the database, just deletes from approved Annotations viewed in the Document, 
creating a negative Annotation instead) and hard deletion (deletes Annotations permanently from the database). For AI 
training purposes, we recommend setting `delete_from_database` to False if you don't want to remove an Annotation 
permanently.
```python tags=["skip-execution", "nbval-skip"]
from konfuzio_sdk.api import delete_document_annotation

# soft-delete and create a negative Annotation
negative_id = delete_document_annotation(annotation_id=YOUR_ANNOTATION_ID)
# hard-delete and remove a negative Annotation from DB permanently
assert delete_document_annotation(negative_id, delete_from_database=True).status_code == 204
```

### Conclusion

In this tutorial, we have explained how to create different types of Annotations using native Konfuzio SDK's wrappers
around the API calls to the server. Here is the full code for the tutorial:
```python tags=['skip-execution', 'nbval-skip']
import json 

from konfuzio_sdk.api import post_document_annotation, delete_document_annotation, change_document_annotation
from konfuzio_sdk.data import Span, Project

test_document = Project(id_=YOUR_PROJECT_ID).get_document_by_id(YOUR_DOCUMENT_ID)
spans = [Span(document=test_document, start_offset=3067, end_offset=3074)]
response = post_document_annotation(document_id=YOUR_DOCUMENT_ID, spans=spans, label_id=YOUR_LABEL_ID, confidence=100.0,
                                    label_set_id=YOUR_LABEL_SET_ID)
response = json.loads(response.text)
print(response['span'])

bboxes = [
        {'page_index': 0, 'x0': 457, 'x1': 480, 'y0': 290, 'y1': 303},
        {'page_index': 0, 'x0': 452.16, 'x1': 482.64, 'y0': 306, 'y1': 313,}
]
response = post_document_annotation(document_id=YOUR_DOCUMENT_ID, spans=bboxes, label_id=YOUR_LABEL_ID, confidence=100.0,
                                    label_set_id=YOUR_LABEL_SET_ID)
response = json.loads(response.text)
print(response['span'])

response = change_document_annotation(annotation_id=YOUR_ANNOTATION_ID, label=NEW_LABEL_ID)
print(response.json()['label'])

# soft-delete and create a negative Annotation
negative_id = delete_document_annotation(annotation_id=YOUR_ANNOTATION_ID)
# hard-delete and remove a negative Annotation from DB permanently
assert delete_document_annotation(negative_id, delete_from_database=True).status_code == 204
```

### What's next?

- [Learn more about Konfuzio's REST API and its possibilities](https://dev.konfuzio.com/web/api-v3.html)