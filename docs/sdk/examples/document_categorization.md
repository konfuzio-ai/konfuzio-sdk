## Document Categorization

### Categorization Fallback Logic

Use the name of the category as an effective fallback logic to categorize documents when no categorization AI is available:

```python
from konfuzio_sdk.data import Project
from konfuzio_sdk.trainer.document_categorization import FallbackCategorizationModel

# Set up your project.
project = Project(id_=YOUR_PROJECT_ID)

# Initialize the categorization model.
categorization_model = FallbackCategorizationModel(project)
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
