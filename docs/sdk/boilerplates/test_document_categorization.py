"""Test Document Categorization code examples from the documentation."""
from konfuzio_sdk.data import Project, Document
from konfuzio_sdk.trainer.document_categorization import NameBasedCategorizationAI
from konfuzio_sdk.trainer.information_extraction import load_model
from konfuzio_sdk.trainer.document_categorization import build_categorization_ai_pipeline
from konfuzio_sdk.trainer.document_categorization import ImageModel, TextModel

from variables import YOUR_PROJECT_ID

YOUR_DOCUMENT_ID = 44865

# Set up your Project.
project = Project(id_=YOUR_PROJECT_ID)
YOUR_CATEGORY_ID = project.categories[0].id_

# Initialize the Categorization Model.
categorization_model = NameBasedCategorizationAI(project.categories)


# Retrieve a Document to categorize.
test_document = project.get_document_by_id(YOUR_DOCUMENT_ID)

# The Categorization Model returns a copy of the SDK Document with Category attribute
# (use inplace=True to maintain the original Document instead).
# If the input Document is already categorized, the already present Category is used
# (use recategorize=True if you want to force a recategorization).
result_doc = categorization_model.categorize(document=test_document)

# Each Page is categorized individually.
for page in result_doc.pages():
    assert page.category == project.categories[0]
    print(f"Found category {page.category} for {page}")

# The Category of the Document is defined when all pages' Categories are equal.
# If the Document contains mixed Categories, only the Page level Category will be defined,
# and the Document level Category will be None.
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
# This will set it for all of its Pages as well.
for page in document.pages():
    assert page.category == my_category
my_document.delete()


# Build the Categorization AI architecture using a template
# of pre-built Image and Text classification Models.
categorization_pipeline = build_categorization_ai_pipeline(
    categories=project.categories,
    documents=project.documents,
    test_documents=project.test_documents,
    image_model=ImageModel.EfficientNetB0,
    text_model=TextModel.NBOWSelfAttention,
)

# Train the AI.
categorization_pipeline.fit(
    document_training_config={
        'batch_size': 1,
        'max_len': None,
        'n_epochs': 1,
        'optimizer': {'name': 'Adam'},
    }
)

# Evaluate the AI
data_quality = categorization_pipeline.evaluate(use_training_docs=True)
assert data_quality.f1(categorization_pipeline.categories[0]) == 1.0
assert data_quality.f1(categorization_pipeline.categories[1]) == 1.0
assert data_quality.f1(None) == 1.0
ai_quality = categorization_pipeline.evaluate()
assert ai_quality.f1(categorization_pipeline.categories[0]) == 1.0
assert ai_quality.f1(categorization_pipeline.categories[1]) == 1.0
assert ai_quality.f1(None) == 1.0

# Categorize a Document
document = project.get_document_by_id(YOUR_DOCUMENT_ID)
categorization_result = categorization_pipeline.categorize(document=document)
assert isinstance(categorization_result, Document)
for page in categorization_result.pages():
    print(f"Found category {page.category} for {page}")

# Save and load a pickle file for the AI
pickle_ai_path = categorization_pipeline.save()

categorization_pipeline_loaded = load_model(categorization_pipeline.pipeline_path)
result = categorization_pipeline.categorize(document=document)
assert isinstance(result, Document)


# Image Models
ImageModel.VGG11
ImageModel.VGG13
ImageModel.VGG16
ImageModel.VGG19
ImageModel.EfficientNetB0
ImageModel.EfficientNetB1
ImageModel.EfficientNetB2
ImageModel.EfficientNetB3
ImageModel.EfficientNetB4
ImageModel.EfficientNetB5
ImageModel.EfficientNetB6
ImageModel.EfficientNetB7
ImageModel.EfficientNetB8

# Text Models
TextModel.NBOW
TextModel.NBOWSelfAttention
TextModel.LSTM
TextModel.BERT
