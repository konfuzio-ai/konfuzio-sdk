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

## Categorize a Document using Categorization AI

---

**Prerequisites:**
- Data Layer concepts of Konfuzio SDK: Document, Category, Project, Page
- AI concepts of Konfuzio SDK: Extraction
- Understanding of ML concepts: train-validation loop, optimizer, epochs

**Difficulty:** Medium

**Goal:** Learn how to categorize a Document using one of Categorization AIs pre-constructed by Konfuzio

---

### Environment
You need to install the Konfuzio SDK before diving into the tutorial. \
To get up and running quickly, you can use our Colab Quick Start notebook. \
<a href="https://colab.research.google.com/github/konfuzio-ai/konfuzio-sdk/blob/master/notebooks/Quick_start_template_for_Konfuzio_SDK.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

As an alternative you can follow the [installation section](../get_started.html#install-sdk) to install and initialize the Konfuzio SDK locally or on an environment of your choice.

### Introduction

To categorize a Document with a Categorization AI constructed by Konfuzio, there are two main options: the Name-based Categorization AI and the more complex Model-based Categorization AI.

### Name-based Categorization AI

The name-based Categorization AI is a simple logic that checks if a name of the Category appears in the Document. It can be used to categorize Documents when no model-based Categorization AI is available.

Let's begin with making imports, initializing the Categorization model and calling the Document to categorize.
```python editable=true slideshow={"slide_type": ""} tags=["remove-cell"] vscode={"languageId": "plaintext"}
import logging
import os
import konfuzio_sdk
from konfuzio_sdk.api import get_project_list
from konfuzio_sdk.data import Project

logging.getLogger("konfuzio_sdk").setLevel(logging.ERROR)
projects = get_project_list()
YOUR_PROJECT_ID = None
while not TEST_PROJECT_ID:
    for project in reversed(projects['results']):
        if 'ZGF0YV8xNDM5Mi02Ni56aXA=' in project['name']:
            TEST_PROJECT_ID = project['id']
            break
project = Project(id_=YOUR_PROJECT_ID)
YOUR_CATEGORY_ID = project.get_category_by_name('Lohnabrechnung').id_
original_document_text = Project(id_=46).get_document_by_id(44825).text
YOUR_DOCUMENT_ID = [document for document in project.documents if document.text == original_document_text][0].id_
```

```python editable=true slideshow={"slide_type": ""} tags=["remove-output"] vscode={"languageId": "plaintext"}
from konfuzio_sdk.data import Project
from konfuzio_sdk.trainer.document_categorization import NameBasedCategorizationAI

project = Project(id_=YOUR_PROJECT_ID)
categorization_model = NameBasedCategorizationAI(project.categories)
test_document = project.get_document_by_id(YOUR_DOCUMENT_ID)
```

Then, we categorize the Document. The Categorization Model returns a copy of the SDK Document with Category attribute (use inplace=True to maintain the original Document instead).
If the input Document is already categorized, the already present Category is used (use recategorize=True if you want to force a recategorization). Each Page is categorized individually.
```python
result_doc = categorization_model.categorize(document=test_document)

for page in result_doc.pages():
    assert page.category == project.categories[0]
    print(f"Found category {page.category} for {page}")
```

The Category of the Document is defined when all pages' Categories are equal. If the Document contains mixed Categories, only the Page level Category will be defined, and the Document level Category will be NO_CATEGORY.
```python
print(f"Found category {result_doc.category} for {result_doc}")
```

### Model-based Categorization AI

For better results you can build, train and test a Categorization AI using Image Models and Text Models to classify the image and text of each Page.

Let's start with the imports and initializing the Project.
```python editable=true slideshow={"slide_type": ""}
from konfuzio_sdk.data import Project, Document
from konfuzio_sdk.trainer.document_categorization import build_categorization_ai_pipeline
from konfuzio_sdk.trainer.document_categorization import ImageModel, TextModel, CategorizationAI

project = Project(id_=YOUR_PROJECT_ID)
```
```python editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
logging.getLogger("konfuzio_sdk").setLevel(logging.CRITICAL)
logging.getLogger("timm").setLevel(logging.CRITICAL)
for doc in project.documents + project.test_documents:
    doc.get_images()
for document in project.documents[3:] + project.test_documents[1:]:
    document.dataset_status = 4
original_document_text = Project(id_=46).get_document_by_id(44864).text
cur_document_id = [document for document in project.documents if document.text == original_document_text][0].id_
project.get_document_by_id(cur_document_id).dataset_status = 4
```

Build the Categorization AI architecture using a template of pre-built Image and Text classification Models. In this tutorial, we use `EfficientNetB0` and `NBOWSelfAttention` together.

```python editable=true slideshow={"slide_type": ""}
categorization_pipeline = build_categorization_ai_pipeline(
    categories=project.categories,
    documents=project.documents,
    test_documents=project.test_documents,
    image_model=ImageModel.EfficientNetB0,
    text_model=TextModel.NBOWSelfAttention,
)
```

Train and evaluate the AI. You can specify parameters for training, for example, number of epochs and an optimizer.
```python tags=["remove-output"]
categorization_pipeline.fit(n_epochs=1, optimizer={'name': 'Adam'})
data_quality = categorization_pipeline.evaluate(use_training_docs=True)
ai_quality = categorization_pipeline.evaluate()
assert data_quality.f1(None) == 1.0
assert ai_quality.f1(None) == 1.0
```

Categorize a Document using the newly trained model.
```python
document = project.get_document_by_id(YOUR_DOCUMENT_ID)
categorization_result = categorization_pipeline.categorize(document=document)
assert isinstance(categorization_result, Document)
for page in categorization_result.pages():
    print(f"Found category {page.category} for {page}")
```

Save the model and check that it can be loaded after that to ensure it could be uploaded to the Konfuzio app or to an on-prem installation.
```python
pickle_ai_path = categorization_pipeline.save()
categorization_pipeline = CategorizationAI.load_model(pickle_ai_path)
```

```python editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
os.remove(pickle_ai_path)
```

To prepare the data for training and testing your AI, follow the [data preparation tutorial](https://dev.konfuzio.com/sdk/tutorials/data-preparation/index.html).

For a list of available Models see all the available [Categorization Models](#categorization-ai-models) below.

### Categorization AI Models

When using `build_categorization_ai_pipeline`, you can select which Image Module and/or Text Module to use for 
classification. At least one between the Image Model or the Text Model must be specified. Both can also be used 
at the same time.

The list of available Categorization Models is implemented as an Enum containing the following elements:

```python editable=true slideshow={"slide_type": ""} tags=["remove-output"]
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

See more details about these Categorization Models under [API Reference - Categorization AI](https://dev.konfuzio.com/sdk/sourcecode.html#categorization-ai).

### Possible configurations

The following configurations of Categorization AI are tested. Tokenizer, text and image processors can be specified 
when building Categorization pipeline locally; text and image processors can be specified when building the pipeline 
either locally or on app/on-prem installation.

Each line stands for a single configuration. If a field is None, it requires specifying it as None; otherwise, the 
default value will be applied.

You can find more information on how to use these configurations, what are default values and where to specify
them [here](https://help.konfuzio.com/modules/projects/index.html?highlight=efficientnet#categorization-ai-parameters).

| Tokenizer               | Text model class  | Text model name          | Image model class | Image model name  |
|-------------------------|-------------------|--------------------------|-------------------|-------------------|
| WhitespaceTokenizer     | NBOWSelfAttention | `nbowselfattention`      | EfficientNet      | `efficientnet_b0` |
| WhitespaceTokenizer     | NBOWSelfAttention | `nbowselfattention`      | EfficientNet      | `efficientnet_b3` |
| WhitespaceTokenizer     | NBOW              | `nbow`                   | VGG               | `vgg11`           |
| WhitespaceTokenizer     | LSTM              | `lstm`                   | VGG               | `vgg13`           |
| ConnectedTextTokenizer  | NBOW              | `nbow`                   | VGG               | `vgg11`           |
| ConnectedTextTokenizer  | LSTM              | `lstm`                   | VGG               | `vgg13`           |
| None                    | None              | None                     | EfficientNet      | `efficientnet_b0` |
| None                    | None              | None                     | EfficientNet      | `efficientnet_b3` |
| None                    | None              | None                     | VGG               | `vgg11`           |
| None                    | None              | None                     | VGG               | `vgg13`           |
| None                    | None              | None                     | VGG               | `vgg16`           |
| None                    | None              | None                     | VGG               | `vgg19`           |
| TransformersTokenizer   | *BERT* *          | `bert-base-german-cased` | None              | None              |

***Note**: In this table, we list a single BERT-based model (`bert-base-german-cased`). The following table lists the
possible values for text model versions that can be passed as `name` argument when configuring BERT model for 
Categorization.

### Models compatible with BERT class

| Name                                           | Embeddings dimension | Language | Number of parameters |
|------------------------------------------------|----------------------|----------|----------------------|
| `bert-base-uncased`                            | 768                  | English  | 110 million          |
| `distilbert-base-uncased`                      | 768                  | English  | 66 million           |
| `google/mobilebert-uncased`                    | 512                  | English  | 25 million           |
| `albert-base-v2`                               | 768                  | English  | 12 million           |
| `german-nlp-group/electra-base-german-uncased` | 768                  | German   | 111 million          |
| `bert-base-german-cased`                       | 768                  | German   | 110 million          |
| `bert-base-german-uncased`                     | 768                  | German   | 110 million          |
| `distilbert-base-german-cased`                 | 768                  | German   | 66 million           |    
| `bert-base-multilingual-cased`                 | 768                  | Multiple | 110 million          |

**Note:** This list is not exhaustive. We only list the models that are fully tested. However, you can use the 
[Huggingface hub](https://huggingface.co/models) to find other models that best suit your needs. To ensure a model is 
compatible with Categorization AI, initialize it with the SDK's `TransformersTokenizer` class as presented in an example
below, replacing the value of `name` to the name of your model of choice. If a model is compatible, the initialization 
will be successful; otherwise, an error about incompatibility will appear.

```python tags=["skip-execution", "nbval-skip"]
from konfuzio_sdk.trainer.tokenization import TransformersTokenizer

tokenizer = TransformersTokenizer(name='bert-base-chinese')
```

### Configurable parameters

Every group of models/configuration you decide to use has manually configurable parameters. Follow this section to find 
out what parameters are configurable and which models accept them.

Some of the parameters are universally accepted for training regardless of the model. 

- `n_epochs` - number of times the entire training dataset is passed through the model during training. BERT models
require lower values like 3-5, other models can require higher number, like 20+. Default value is 20.
- `patience` - number of epochs to wait before early stopping if the model's performance on the validation set does 
not improve. Default value is 3.
- `optimizer` -  algorithm used to update the model's parameters during training. Default value is `AdamW` with learning
rate of `1e-4`.
- `lr_decay` - rate at which the learning rate is reduced over time during training to help the model maximize training
efficiency. Default value is 0.999.

Other parameters are configurable only for some of the models and might not have a unified default value.

- `input_dim` - dimensionality of the input data, which represents the number of features or variables in the input.
- `dropout_rate` - fraction of the input units to randomly set to 0 during training to prevent overfitting.
- `emb_dim` - dimensionality of the embeddings (vector representation). Default value is 64.
- `n_heads` - number of attention heads in multi-head attention mechanisms which enable the model to attend to different
parts of the input simultaneously. Note that `n_heads` must be a factor of `emb_dim`, i.e. `emb_dim % n_heads == 0`.
- `hid_dim` - dimensionality of the hidden states in the model. Default value is 256.
- `n_layers` - number of layers in the model. Default value is 2.
- `bidirectional` - whether to use bidirectional processing in LSTM, enabling the model to consider both past and future
context. Default value is True.
- `name` - a name or identifier for the model.
- `freeze` - whether to freeze the weights of certain layers or parameters during training, preventing them from being 
updated.

| Model             | `input_dim` | `dropout_rate` | `emb_dim` | `n_heads` | `hid_dim` | `n_layers` | `bidirectional` | `name` | `freeze` |
|-------------------|-------------|----------------|-----------|-----------|-----------|------------|-----------------|--------|----------|
| NBOW              | ✔           | ✔              | ✔         | ✘         | ✘         | ✘          | ✘               | ✘      | ✘        |
| NBOWSelfAttention | ✔           | ✘              | ✔         | ✔         | ✘         | ✘          | ✘               | ✘      | ✘        |
| LSTM              | ✔           | ✔              | ✔         | ✘         | ✔         | ✔          | ✔               | ✘      | ✘        |
| BERT              | ✘           | ✘              | ✘         | ✘         | ✘         | ✘          | ✘               | ✔      | ✔        |
| VGG               | ✘           | ✘              | ✘         | ✘         | ✘         | ✘          | ✘               | ✔      | ✔        |
| EfficientNet      | ✘           | ✘              | ✘         | ✘         | ✘         | ✘          | ✘               | ✔      | ✔        |


### Conclusion

In this tutorial, we presented two different ways to categorize a Document using AIs constructed by Konfuzio and provided possible configurations that can be used in model-based Categorization.

### What's next?

- [Create your own custom Categorization AI](https://dev.konfuzio.com/sdk/tutorials/create-custom-categorization-ai/index.html)
