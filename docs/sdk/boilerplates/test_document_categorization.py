"""Test Document Categorization code examples from the documentation."""
from konfuzio_sdk.data import Project, Document
from konfuzio_sdk.trainer.document_categorization import FallbackCategorizationModel

YOUR_PROJECT_ID = 46
YOUR_DOCUMENT_ID = 44865

# Set up your project.
project = Project(id_=YOUR_PROJECT_ID)
YOUR_CATEGORY_ID = project.categories[0]

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

YOUR_PROJECT_ID: int
my_project = Project(id_=YOUR_PROJECT_ID)

YOUR_CATEGORY_ID: int
my_category = my_project.get_category_by_id(YOUR_CATEGORY_ID)

my_document = Document(text="My text.", category=my_category)
assert my_document.category == my_category
assert my_document.category_is_revised is True

YOUR_DOCUMENT_ID: int
document = my_project.get_document_by_id(YOUR_DOCUMENT_ID)
document.category = None
assert document.category is None
document.set_category(my_category)
assert document.category == my_category
assert document.category_is_revised is True
# This will set it for all of its pages as well.
for page in document.pages():
    assert page.category == my_category
