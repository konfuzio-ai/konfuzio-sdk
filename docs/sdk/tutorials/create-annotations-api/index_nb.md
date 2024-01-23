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

logging.getLogger("konfuzio_sdk").setLevel(logging.ERROR)
```
```python
from konfuzio_sdk.api import post_document_annotation
```

To create any Annotation, it is necessary to provide several fields to `post_document_annotation`:

- `document`: ID of a Document in which Annotation is created
- `label`: ID of a Label assigned to the Annotation
- `span`: Coordinates of Bounding Boxes representing the position of the Annotation in the Document. Despite the name,
can contain multiple Bounding Boxes.

### Creating a textual Annotation