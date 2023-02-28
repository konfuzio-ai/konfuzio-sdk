## Document Categorization

### Name-based Categorization AI

Use the name of the category as an effective fallback logic to categorize documents when no Categorization AI is available:

.. literalinclude:: /sdk/boilerplates/test_document_categorization.py
   :language: python
   :lines: 2-4,7-9,11-26,28-33

### Working with the Category of a Document and its individual Pages

You can initialize a Document with a Category, which will count as if a human manually revised it.

.. literalinclude:: /sdk/boilerplates/test_document_categorization.py
   :language: python
   :lines: 9,11,35-38,40

If a Document is initialized with no Category, it can be manually set later.

.. literalinclude:: /sdk/boilerplates/test_document_categorization.py
   :language: python
   :lines: 42-50


If you use a Categorization AI to automatically assign a Category to a Document (such as the 
[NameBasedCategorizationAI](tutorials.html#categorization-fallback-logic)), each Page will be assigned a 
Category Annotation with predicted confidence information, and the following properties will be accessible. You can 
also find these documented under [API Reference - Document](sourcecode.html#document), 
[API Reference - Page](sourcecode.html#page) and 
[API Reference - Category Annotation](sourcecode.html#category-annotation).

| Property                     | Description                                                                                                                                                                                                                       |
|-------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `CategoryAnnotation.category`    | The AI predicted Category of this Category<br>Annotation.                                                                                                                                                                         |
| `CategoryAnnotation.confidence`  | The AI predicted confidence of this Category<br>Annotation.                                                                                                                                                                       |
| `Document.category_annotations`   | List of predicted Category Annotations at the<br>Document level.                                                                                                                                                                  |
| `Document.maximum_confidence_category_annotation`   | Get the maximum confidence predicted Category<br>Annotation, or the human revised one if present.                                                                                                                                 |
| `Document.maximum_confidence_category`   | Get the maximum confidence predicted Category<br>or the human revised one if present.                                                                                                                                             |
| `Document.category`  | Returns a Category only if all Pages have same<br>Category, otherwise None. In that case, it hints<br>to the fact that the Document should probably<br>be revised or split into Documents with<br>consistently categorized Pages. |
| `Page.category_annotations`   | List of predicted Category Annotations at the<br>Page level.                                                                                                                                                                      |
| `Page.maximum_confidence_category_annotation`   | Get the maximum confidence predicted Category<br>Annotation or the one revised by the user for this<br>Page.                                                                                                                      |
| `Page.category`  | Get the maximum confidence predicted Category<br>or the one revised by user for this Page.                                                                                                                                        |
