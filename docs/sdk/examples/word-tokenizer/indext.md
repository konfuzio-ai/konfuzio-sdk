# Tutorial: Getting Word Bounding Box (BBox) for a Document

In this tutorial, we will walk through how to extract the bounding box (BBox) for words in a document, rather than for individual characters, using the Konfuzio SDK. This process involves the use of the `WhitespaceTokenizer` from the Konfuzio SDK to tokenize the document and identify word-level spans, which can then be visualized or used to extract BBox information.

## Prerequisites

- You will need to have the Konfuzio SDK installed.
- You should have access to a project on the Konfuzio platform.

## Preview of Result

Click on image to enlarge.

<img src="https://github.com/konfuzio-ai/konfuzio-sdk/assets/2879188/5f7a8501-cd89-487d-a332-0703f3c35fc8" data-canonical-src="https://github.com/konfuzio-ai/konfuzio-sdk/assets/2879188/5f7a8501-cd89-487d-a332-0703f3c35fc8" width="200" height="400" />

## Steps

1. **Import necessary modules**:

```python
from konfuzio_sdk.data import Project
from konfuzio_sdk.tokenizer.regex import WhitespaceTokenizer
from copy import deepcopy
```

2. **Initialize your project**:

This involves creating a project instance with the appropriate ID.

```python
project = Project(id_=...)
```

Replace `...` with your project's ID.

3. **Retrieve a document from your project**:

```python
document = project.get_document_by_id(...)
```

Replace `...` with your document's ID.

4. **Create a copy of your document without annotations**:

```python
document = deepcopy(document)
```

5. **Tokenize the document**:

This process involves splitting the document into word-level spans using the WhitespaceTokenizer.

```python
tokenizer = WhitespaceTokenizer()
document = tokenizer(document)
```

6. **Visualize all word-level annotations**:

After getting the bounding box for all spans, you might want to visually check the results to make sure the bounding boxes are correctly assigned. Here's how you can do it:

```python
document.get_page_by_index(0).get_annotations_image(display_all=True)
```

<img src="https://github.com/konfuzio-ai/konfuzio-sdk/assets/2879188/5f7a8501-cd89-487d-a332-0703f3c35fc8" data-canonical-src="https://github.com/konfuzio-ai/konfuzio-sdk/assets/2879188/5f7a8501-cd89-487d-a332-0703f3c35fc8" width="200" height="400" />

This will display an image of the document with all word-level annotations. The image may look a bit messy with all the labels.

7. **Get bounding box for all spans**:

You can retrieve bounding boxes for all word-level spans using the following code:

```python
span_bboxes = [span.bbox() for span in document.spans()]
```

Each bounding box consists of the coordinates of the top-left corner of the box, as well as its width and height. The list `span_bboxes` will contain these bounding box coordinates for each word in the document.

**NOTE:** As per the conversation above, as of now, there are no specific methods in the Konfuzio SDK to retrieve all projects/label sets in a project/document details for other entities. If needed, custom code would have to be written for this purpose. Also, the API does not currently support a solution for this task.
