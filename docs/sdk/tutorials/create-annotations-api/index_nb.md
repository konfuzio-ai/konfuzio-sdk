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

## Create Annotations using the wrapped API call

---

**Prerequisites:**
- Data Layer concepts of Konfuzio: Project, Document, Annotation, Span, Bounding Box, Label, Label Set, Annotation Set
- Understanding of concepts of REST API

**Difficulty:** Medium

**Goal:** Explain how to create different types of Annotation (textual, visual) using the methods from the SDK listed in
`konfuzio_sdk.api`

---

### Introduction

There are several ways to create an Annotation in a Document: using the SmartView or DVUI on Konfuzio's app or an 
on-prem installation, via the `Annotation` class in the SDK or via the direct call to the API endpoint 
`api/v3/annotations`. In Server documentation, we already provide an [instruction](https://dev.konfuzio.com/web/api-v3.html#create-an-annotation) 
on creating an Annotation via the POST request using `curl`; in this tutorial, we will explain how to make this 
request using the methods from `konfuzio_sdk.api` which serves as a wrapper around the calls to the API.

Let's start by making necessary imports:
```python tags=["remove-cell"]
import logging
from konfuzio_sdk.api import get_document_annotations
from konfuzio_sdk.data import Project

logging.getLogger("konfuzio_sdk").setLevel(logging.ERROR)
YOUR_DOCUMENT_ID = 44823
YOUR_LABEL_ID = 12503
YOUR_LABEL_SET_ID = 63
YOUR_PROJECT_ID = 46
project = Project(id_=YOUR_PROJECT_ID)
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
spans = [Span(document=test_document, start_offset=3056, end_offset=3064)]
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
annotations = get_document_annotations(YOUR_DOCUMENT_ID)['results']
negative_id = delete_document_annotation(response['id'])
assert delete_document_annotation(negative_id, delete_from_database=True).status_code == 204
```

### Creating a visual Annotation

To create an Annotation that is based on Bounding Boxes' coordinates, let's create a dictionary of coordinates that will
be passed as the `spans` argument. You can define one or more Bounding Boxes. Note that you don't need to specify 
offsets, only the `page_index` is needed.
```python tags=['remove-cell']
YOUR_DOCUMENT_ID = YOUR_DOCUMENT_ID + 11
YOUR_LABEL_ID = 862
```
```python
bboxes = [
        {'page_index': 0, 'x0': 198, 'x1': 300, 'y0': 508, 'y1': 517},
        {'page_index': 0, 'x0': 197.76, 'x1': 233, 'y0': 495, 'y1': 508},
]
```
Next, we specify arguments for a POST request to create an Annotation and send it to the server. We want to create
an Annotation within a new Annotation Set so we specify `label_set_id`.
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
assert delete_document_annotation(response['id'])
```
### Conclusion

In this tutorial, we have explained how to create different types of Annotations using native Konfuzio SDK's wrappers
around the API calls to the server. Here is the full code for the tutorial:
```python tags=['skip-execution', 'nbval-skip']
import json 

from konfuzio_sdk.api import post_document_annotation, delete_document_annotation
from konfuzio_sdk.data import Span, Project

test_document = Project(id_=YOUR_PROJECT_ID).get_document_by_id(YOUR_DOCUMENT_ID)
spans = [Span(document=test_document, start_offset=3056, end_offset=3064)]
response = post_document_annotation(document_id=YOUR_DOCUMENT_ID, spans=spans, label_id=YOUR_LABEL_ID, confidence=100.0,
                                    label_set_id=YOUR_LABEL_SET_ID)
response = json.loads(response.text)
print(response['span'])

bboxes = [
        {'page_index': 0, 'x0': 198, 'x1': 300, 'y0': 508, 'y1': 517},
        {'page_index': 0, 'x0': 197.76, 'x1': 233, 'y0': 495, 'y1': 508},
]
response = post_document_annotation(document_id=YOUR_DOCUMENT_ID, spans=bboxes, label_id=YOUR_LABEL_ID, confidence=100.0,
                                    label_set_id=YOUR_LABEL_SET_ID)
response = json.loads(response.text)
print(response['span'])
```

### What's next?

- [Learn more about Konfuzio's REST API and its possibilities](https://dev.konfuzio.com/web/api-v3.html)