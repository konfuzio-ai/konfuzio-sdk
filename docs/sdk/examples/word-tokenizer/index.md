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

.. literalinclude:: /sdk/boilerplates/test_word_bboxes.py
   :language: python
   :lines: 6-9
   :dedent: 4

2. **Initialize your project**:

This involves creating a project instance with the appropriate ID.

.. literalinclude:: /sdk/boilerplates/test_word_bboxes.py
   :language: python
   :lines: 13
   :dedent: 4

3. **Retrieve a document from your project**:

.. literalinclude:: /sdk/boilerplates/test_word_bboxes.py
   :language: python
   :lines: 15
   :dedent: 4

4. **Create a copy of your document without annotations**:

.. literalinclude:: /sdk/boilerplates/test_word_bboxes.py
   :language: python
   :lines: 16
   :dedent: 4

5. **Tokenize the document**:

This process involves splitting the document into word-level spans using the WhitespaceTokenizer.

.. literalinclude:: /sdk/boilerplates/test_word_bboxes.py
   :language: python
   :lines: 17-18
   :dedent: 4

6. **Visualize all word-level annotations**:

After getting the bounding box for all spans, you might want to visually check the results to make sure the bounding boxes are correctly assigned. Here's how you can do it:

.. literalinclude:: /sdk/boilerplates/test_word_bboxes.py
   :language: python
   :lines: 20
   :dedent: 4

<img src="https://github.com/konfuzio-ai/konfuzio-sdk/assets/2879188/5f7a8501-cd89-487d-a332-0703f3c35fc8" data-canonical-src="https://github.com/konfuzio-ai/konfuzio-sdk/assets/2879188/5f7a8501-cd89-487d-a332-0703f3c35fc8" width="200" height="400" />

This will display an image of the document with all word-level annotations. The image may look a bit messy with all the labels.

7. **Get bounding box for all spans**:

You can retrieve bounding boxes for all word-level spans using the following code:

.. literalinclude:: /sdk/boilerplates/test_word_bboxes.py
   :language: python
   :lines: 21
   :dedent: 4

Each bounding box consists of the coordinates of the top-left corner of the box, as well as its width and height. The list `span_bboxes` will contain these bounding box coordinates for each word in the document.

**NOTE:** As per the conversation above, as of now, there are no specific methods in the Konfuzio SDK to retrieve all projects/label sets in a project/document details for other entities. If needed, custom code would have to be written for this purpose. Also, the API does not currently support a solution for this task.
