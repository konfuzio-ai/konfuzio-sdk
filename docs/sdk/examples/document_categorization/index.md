## Document Categorization

### Name-based categorization

Use the name of the category as an effective fallback logic to categorize documents when no Categorization AI is available:

```python
from konfuzio_sdk.data import Project
from konfuzio_sdk.trainer.document_categorization import NameBasedCategorizationAI

# Set up your project.
project = Project(id_=YOUR_PROJECT_ID)

# Initialize the categorization model.
categorization_model = NameBasedCategorizationAI(project)
categorization_model.categories = project.categories

# Retrieve a document to categorize.
test_document = project.get_document_by_id(YOUR_DOCUMENT_ID)

# The categorization model returns a copy of the SDK Document with category attribute
# (use inplace=True to maintain the original document instead).
# If the input document is already categorized, the already present category is used
# (use recategorize=True if you want to force a recategorization).
result_doc = categorization_model.categorize(document=test_document)

# Each page is categorized individually.
for page in result_doc.pages():
    print(f"Found category {page.category} for {page}")

# The category of the document is defined when all pages' categories are equal.
# If the document contains mixed categories, only the page level category will be defined,
# and the document level category will be None.
print(f"Found category {result_doc.category} for {result_doc}")
```

### Working with the Category of a Document and its individual Pages

You can initialize a Document with a Category, which will count as if a human manually revised it.

```python
MY_PROJECT_ID: int
my_project = Project(id_=MY_PROJECT_ID)

MY_CATEGORY_ID: int
my_category = my_project.get_category_by_id(MY_CATEGORY_ID)

my_document = Document(text="My text.", category=my_category)
assert my_document.category == my_category
assert my_document.category_is_revised == True
```

If a Document is initialized with no Category, it can be manually set later.

```python
DOCUMENT_ID: int
document = my_project.get_document_by_id(DOCUMENT_ID)
assert document.category is None
document.set_category(my_category)
assert document.category == my_category
assert document.category_is_revised == True
# This will set it for all of its pages as well.
for page in document.pages():
    assert page.category == my_category
```

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
