"""Test Document Categorization code examples from the documentation."""
from konfuzio_sdk.data import Project, Document
from konfuzio_sdk.trainer.document_categorization import NameBasedCategorizationAI

from variables import YOUR_PROJECT_ID

YOUR_DOCUMENT_ID = 44865

# Set up your project.
project = Project(id_=YOUR_PROJECT_ID)
YOUR_CATEGORY_ID = project.categories[0].id_

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
    assert page.category == project.categories[0]
    print(f"Found category {page.category} for {page}")

# The category of the document is defined when all pages' categories are equal.
# If the document contains mixed categories, only the page level category will be defined,
# and the document level category will be None.
print(f"Found category {result_doc.category} for {result_doc}")

my_category = project.get_category_by_id(YOUR_CATEGORY_ID)

my_document = Document(text="My text.", project=project, category=my_category)
assert my_document.category == my_category
my_document.category_is_revised = True
assert my_document.category_is_revised is True

document = project.get_document_by_id(YOUR_DOCUMENT_ID)
document.set_category(None)
assert document.category == project.no_category
document.set_category(my_category)
assert document.category == my_category
assert document.category_is_revised is True
# This will set it for all of its pages as well.
for page in document.pages():
    assert page.category == my_category
my_document.delete()
