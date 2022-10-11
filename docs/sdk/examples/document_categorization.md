## Document Categorization

### Categorization Fallback Logic

Use the name of the category as an effective fallback logic to categorize documents when no categorization AI is available:

```python
from konfuzio_sdk.trainer.document_categorization import FallbackCategorizationModel

project = Project(id_=YOUR_PROJECT_ID)
categorization_model = FallbackCategorizationModel(project)
categorization_model.categories = project.categories

test_document = project.get_document_by_id(YOUR_DOCUMENT_ID)

# returns virtual SDK Document with category attribute
result_doc = categorization_model.categorize(document=test_document)

# if the input document is already categorized, the already present category is used
# unless recategorize is True
result_doc = categorization_model.categorize(document=test_document, recategorize=True)

print(f"Found category {result_doc.category} for {result_doc}")

# option to modify the provided document in place
categorization_model.categorize(document=test_document, inplace=True)

print(f"Found category {test_document.category} for {test_document}")
```