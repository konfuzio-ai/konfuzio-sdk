---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region editable=true slideshow={"slide_type": ""} -->
## Categorize a Document using Categorization AI

---

**Prerequisites:**
- Data Layer concepts of Konfuzio SDK
- AI concepts of Konfuzio SDK

**Difficulty:** Medium

**Goal:** Learn how to categorize a Document using one of Categorization AIs pre-constructed by Konfuzio

---

### Introduction

To categorize a Document with a Categorization AI constructed by Konfuzio, there are two main options: the Name-based Categorization AI and the more complex Model-based Categorization AI.

### Name-based Categorization AI

The name-based Categorization AI is a simple logic that checks if a name of the Category appears in the Document. It can be used to categorize Documents when no model-based Categorization AI is available.
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""} tags=["remove-cell"] vscode={"languageId": "plaintext"}
YOUR_PROJECT_ID = 46
YOUR_CATEGORY_ID = 63
YOUR_DOCUMENT_ID = 44865
```

```python editable=true slideshow={"slide_type": ""} tags=["remove-output"] vscode={"languageId": "plaintext"}
from konfuzio_sdk.data import Project
from konfuzio_sdk.trainer.document_categorization import NameBasedCategorizationAI

# Set up your Project.
project = Project(id_=YOUR_PROJECT_ID)

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
# and the Document level Category will be NO_CATEGORY.
print(f"Found category {result_doc.category} for {result_doc}")
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
### Model-based Categorization AI

For better results you can build, train and test a Categorization AI using Image Models and Text Models to classify the image and text of each Page.
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""}
from konfuzio_sdk.data import Project, Document
from konfuzio_sdk.trainer.document_categorization import build_categorization_ai_pipeline
from konfuzio_sdk.trainer.document_categorization import ImageModel, TextModel, CategorizationAI

# Set up your Project.
project = Project(id_=YOUR_PROJECT_ID)
```

```python editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
for doc in project.documents + project.test_documents:
    doc.get_images()
for document in project.documents[3:] + project.test_documents[1:]:
    document.dataset_status = 4 
project.get_document_by_id(44864).dataset_status = 4
```

```python editable=true slideshow={"slide_type": ""} tags=["remove-output"]
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
categorization_pipeline.fit(n_epochs=1, optimizer={'name': 'Adam'})

# Evaluate the AI
data_quality = categorization_pipeline.evaluate(use_training_docs=True)
ai_quality = categorization_pipeline.evaluate()
assert data_quality.f1(None) == 1.0
assert ai_quality.f1(None) == 1.0

# Categorize a Document
document = project.get_document_by_id(YOUR_DOCUMENT_ID)
categorization_result = categorization_pipeline.categorize(document=document)
assert isinstance(categorization_result, Document)
for page in categorization_result.pages():
    print(f"Found category {page.category} for {page}")

# Save and load a pickle file for the AI
pickle_ai_path = categorization_pipeline.save()
categorization_pipeline = CategorizationAI.load_model(pickle_ai_path)
```

```python editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
os.remove(pickle_ai_path)
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
To prepare the data for training and testing your AI, follow the [data preparation tutorial](https://dev.konfuzio.com/sdk/tutorials/data-preparation/index.html).

For a list of available Models see all the available [Categorization Models](#categorization-ai-models) below.

### Categorization AI Models

When using `build_categorization_ai_pipeline`, you can select which Image Module and/or Text Module to use for 
classification. At least one between the Image Model or the Text Model must be specified. Both can also be used 
at the same time.

The list of available Categorization Models is implemented as an Enum containing the following elements:
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""} tags=["skip-execution"]
from konfuzio_sdk.trainer.document_categorization import ImageModel, TextModel

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
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
See more details about these Categorization Models under [API Reference - Categorization AI](ADD LINK).

### Possible configurations

The following configurations of Categorization AI are tested. Tokenizer, text and image processors can be specified 
when building Categorization pipeline locally; text and image processors can be specified when building the pipeline 
either locally or on app/on-prem installation.

Each line stands for a single configuration. If a field is None, it requires specifying it as None; otherwise, the 
default value will be applied.

You can find more information on how to use these configurations, what are default values and where to specify
them [here](https://help.konfuzio.com/modules/projects/index.html?highlight=efficientnet#categorization-ai-parameters).

| Tokenizer | Text processor    | Image processor | Image processing version |
|-----------|-------------------|-----------------|--------------------------|
| WhitespaceTokenizer | NBOWSelfAttention | EfficientNet | efficientnet_b0          |
| WhitespaceTokenizer | NBOWSelfAttention | EfficientNet | efficientnet_b3          |
| WhitespaceTokenizer | NBOW              | VGG | vgg11                    |
| WhitespaceTokenizer | LSTM              | VGG | vgg13                    |
| ConnectedTextTokenizer | NBOW              | VGG | vgg11                    |
| ConnectedTextTokenizer | LSTM              | VGG | vgg13                    |
| None | None | EfficientNet | efficientnet_b0          |
| None | None | EfficientNet | efficientnet_b3          |
| None | None | VGG | vgg11                    |
| None | None | VGG | vgg13                    |
| None | None | VGG | vgg16                    |
| None | None | VGG | vgg19                    |

<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} -->
### Conclusion

In this tutorial, we presented two different ways to categorize a Document using AIs constructed by Konfuzio and provided possible configurations that can be used in model-based Categorization.
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} -->
### What's next?

- Create your custom Categorization AI

<!-- #endregion -->