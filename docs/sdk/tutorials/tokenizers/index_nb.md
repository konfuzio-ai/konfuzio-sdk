---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: konfuzio
    language: python
    name: python3
---

## Tokenization

---

**Prerequisites:**
- Install the Konfuzio SDK.
- Have access to a Project on the Konfuzio platform.
- Be familiar with the following concepts:
    - [Document](ADD-LINK)
    - [Bounding Box](ADD-LINK)
    - [Spans](ADD-LINK)
    - [Label](ADD-LINK)

**Difficulty:** This tutorial is suitable for beginners in NLP and the Konfuzio SDK.

**Goal:** Be familiar with the concept of tokenization and master how different tokenization approaches can be used with Konfuzio.

---

### Introduction
In this tutorial, we will explore the concept of tokenization and the various tokenization strategies available in the Konfuzio SDK. Tokenization is a foundational tool in natural language processing (NLP) that involves breaking text into smaller units called tokens. We will focus on the `WhitespaceTokenizer`, `Label-Specific Regex Tokenizer`, `ParagraphTokenizer`, and `SentenceTokenizer` as different tools for different tokenization tasks.



### Whitespace Tokenization
The `WhitespaceTokenizer`, part of the Konfuzio SDK, is a simple yet effective tool for basic tokenization tasks. It segments text into tokens using white spaces as natural delimiters.

#### Use case: Retrieving the Word Bounding Box for a Document
In this section, we will walk through how to use the `WhitespaceTokenizer` to extract word-level [Bounding Boxes](ADD-LINK) for a Document.

We will use the Konfuzio SDK to tokenize the Document and identify word-level [Spans](ADD-LINK), which can then be visualized or used to extract bounding box information.


##### Steps
1. Import necessary modules

```python tags=["remove-cell"]
# This is necessary to make sure we can import from 'tests'
import sys
sys.path.insert(0, '../../../../')
```

```python tags=["remove-cell"]
from tests.variables import TEST_PROJECT_ID, TEST_DOCUMENT_ID
```

```python
from copy import deepcopy
from konfuzio_sdk.data import Project
from konfuzio_sdk.tokenizer.regex import WhitespaceTokenizer
```

2. Initialize a Project and a Document instance

```python tags=["remove-output"]
project = Project(id_=TEST_PROJECT_ID)
document = project.get_document_by_id(TEST_DOCUMENT_ID)

# We create a copy of the document object to make sure it contains no Annotations
document = deepcopy(document)

```

3. Tokenize the Document
This process involves splitting the Document into word-level Spans using the WhitespaceTokenizer.

```python tags=["remove-output"]
tokenizer = WhitespaceTokenizer()
tokenized_spans = tokenizer.tokenize(document)
```

4. Visualize word-level Annotations
We now visually check that the Bounding Boxes are correctly assigned.

```python
document.get_page_by_index(0).get_annotations_image(display_all=True)
```

Observe how each individual word is enclosed in a Bounding Box. Also note that these Bounding Boxes have no [Label](ADD-LINK) associated, thereby the placeholder 'NO_LABEL' is shown above each Bounding Box.


5. Retrieving Bounding Boxes

Each Bounding Box corresponds is associated to a specific word and is defined by four coordinates:
- x0 and y0 specify the coordinates of the bottom left corner;
- x1 and y1 specify the coordinates of the top right corner

Which allow to determine the size and position of the Box on the page.

All Bounding Boxes calculated after tokenization occurred can be obtained as follows:

```python
span_bboxes = [span.bbox() for span in document.spans()]
```

Let us inspect the first 10 Bounding Boxes' coordinates to verify that each comprises 4 coordinate points.

```python
span_bboxes[:10]
```

### Label-Specific Regex Tokenization


### Paragraph Tokenization


### Sentence Tokenization


### Choosing the Right Tokenizer

```python tags=["remove-output"]

```

```python tags=["remove-cell"]

```

### Conclusion
In this tutorial, we have walked through the essential steps for [...] Below is the full code to accomplish this task:

```python tags=["skip-execution"]

```

### What's next?

- ...
- ...

