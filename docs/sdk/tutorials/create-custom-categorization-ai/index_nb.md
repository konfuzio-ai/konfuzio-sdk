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

## Create and save a custom Categorization AI

---

**Prerequisites:**
- Data Layer concepts of Konfuzio: Project, Document, Category, Page
- AI concepts of Konfuzio: Categorization
- Understanding of OOP: Classes, inheritance

**Difficulty:** Medium

**Goal:** Learn how to create a custom Categorization AI with manually defined architecture and save it in a Bento containerized format.

---

### Environment
You need to install the Konfuzio SDK before diving into the tutorial. \
To get up and running quickly, you can use our Colab Quick Start notebook. \
<a href="https://colab.research.google.com/github/konfuzio-ai/konfuzio-sdk/blob/master/notebooks/Quick_start_template_for_Konfuzio_SDK.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

As an alternative you can follow the [installation section](../get_started.html#install-sdk) to install and initialize the Konfuzio SDK locally or on an environment of your choice.

### Introduction

This tutorial explains how to build and train a custom Categorization AI locally, how to save it and upload it to the Konfuzio 
Server. If you run this tutorial in Colab and experience any version compatibility issues when working with the SDK, restart the
runtime and initialize the SDK once again; this will resolve the issue.

**Note**: you don't necessarily need to create the AI from scratch if you already have some Document-processing architecture.
You just need to wrap it into the class that corresponds to our Categorization AI structure. Follow the steps in this 
tutorial to find out what are the requirements for that.

By default, any [Categorization AI](https://dev.konfuzio.com/sdk/tutorials/document_categorization/index.html) class should derive from the `AbstractCategorizationModel` class and implement methods `__init__`, `fit`, `_categorize_page`, `build_bento`, `bento_metadata`, and `entrypoint_methods` methods.

Let's make necessary imports and define the class. In `__init__`, you can either not make any changes or initialize key variables required by your custom AI.

```python tags=["remove-cell"]
import logging

logging.getLogger("konfuzio_sdk").setLevel(logging.ERROR)
logging.getLogger("timm").setLevel(logging.CRITICAL)
```

```python editable=true slideshow={"slide_type": ""} tags=["skip-execution", "nbval-skip"]
import bentoml
import lz4
import os
import pathlib
import torch
from konfuzio_sdk.trainer.document_categorization import AbstractCategorizationAI
from konfuzio_sdk.data import Page, Category, Project
from typing import List

class CustomCategorizationAI(AbstractCategorizationAI):
    """Define a custom Categorization AI."""

    # specify if an AI uses text, images 
    requires_text = True
    requires_images = False
    requires_segmentation = False # always false because it is only required for existing segmentation AIs

    def __init__(self, categories: List[Category], *args, **kwargs) -> None:
        """
        Initialize a class.

        categories: A list of Categories between which the AI is going to distinguish.
        """
        # a list of Categories between which the AI will differentiate
        super().__init__(categories)
```

Then we need to define `fit` method. It can contain any custom architecture, for instance, a multi-layered perceptron, or some hardcoded logic like [Name-based Categorization](https://dev.konfuzio.com/sdk/tutorials/document_categorization/index.html#name-based-categorization-ai). This method does not return anything; rather, it modifies the `self.model` if you provide this attribute.

This method is allowed to be implemented as a no-op if you provide the trained model in other ways.

```python editable=true slideshow={"slide_type": ""} tags=["skip-execution", "nbval-skip"]
    def fit(self) -> None:
        """Fit the classifier on training Documents."""
        self.classifier_iterator = build_document_classifier_iterator(
                    self.documents,
                    self.train_transforms,
                    use_image = True,
                    use_text = False,
                    device='cpu',
                )
        self.classifier._fit_classifier(self.classifier_iterator, **kwargs)
```

Next, we need to define how the model assigns a Category to a Page inside a `_categorize_page` method. **NB:** The result of extraction must be the input Page with added Categorization attribute `Page.category`.

```python editable=true slideshow={"slide_type": ""} tags=["skip-execution", "nbval-skip"]
    def _categorize_page(self, page: Page) -> Page:
        """
        Assign Category Annotations to the Document's Page.
        
        page: A Page to which Category Annotations are assigned.
        """
        page_image = page.get_image()
        predicted_category_id, predicted_confidence = self._predict(page_image)
        
        for category in self.categories:
            if category.id_ == predicted_category_id:
                _ = CategoryAnnotation(category=category, confidence=predicted_confidence, page=page)
        
        return page
```

[BentoML](https://docs.bentoml.com/en/latest/?_gl=1*vms2y5*_gcl_au*OTYwNzkwODIzLjE3MTk5NDQwOTk.) allows to containerize the AI models and run them independently. A custom Categorization AI class needs three methods to support usage with BentoML:
- `build_bento()` method which allows building the Bento archive that can later be uploaded to Konfuzio app or an on-prem installation, as well as served and used locally;
- `entrypoint_methods()`, a property that defines what methods will be exposed in a resulting Bento model (typically `categorize` is enough);
- `bento_metadata()` which defines what metadata will be saved in the model: whether it requires usage of images, text and/or segmentation, and also the formats of an expected request and response in Pydantic format.

```python editable=true slideshow={"slide_type": ""} tags=["skip-execution", "nbval-skip"]
    @property
    def entrypoint_methods(self) -> dict:
        """List the model's methods to make it accessible in a containerized instance of the AI."""
        return {
            'categorize': {'batchable': False}
        }

    @property
    def bento_metadata(self) -> dict:
        """List if the AI requires processing of images/text and segmentation, and specify formats of request and response. Needed for server support of the AI. """
        return {
            'requires_images': getattr(self, 'requires_images', False),
            'requires_segmentation': getattr(self, 'requires_segmentation', False),
            'requires_text': getattr(self, 'requires_text', False),
            'request': 'CategorizeRequest20240729', 
            'response': 'CategorizeResponse20240729', 
        }

    def build_bento(self, bento_model) -> Bento:
        """Build a archived instance of the AI."""
        bento_module_dir = 'konfuzio-sdk/konfuzio_sdk/bento/categorization' # specify your own path if the root folder where konfuzio_sdk is stored has a different name
        dict_metadata = self.project.create_project_metadata_dict()

        #  create a temporary directory for a future bento archive
        with tempfile.TemporaryDirectory() as temp_dir:
            # copy bento_module_dir to temp_dir
            shutil.copytree(bento_module_dir, temp_dir + '/categorization')
            # include metadata
            with open(f'{temp_dir}/categories_and_label_data.json5', 'w') as f:
                json.dump(dict_metadata, f, indent=2, sort_keys=True)
            # include the AI model name so the service can load it correctly
            with open(f'{temp_dir}/AI_MODEL_NAME', 'w') as f:
                f.write(self._pkl_name)

            built_bento = bentoml.bentos.build(
                name=f"categorization_{self.category.id_ if self.category else '0'}",
                service=f'categorization/categorizationai_service.py:CategorizationService',
                include=[
                    'categorization/*.py',
                    'categories_and_label_data.json5',
                    'AI_MODEL_NAME',
                ],
                labels=self.bento_metadata,
                python={
                    'packages': [
                        'https://github.com/konfuzio-ai/konfuzio-sdk/archive/refs/heads/master.zip#egg=konfuzio-sdk'
                    ],
                    'lock_packages': True,
                },
                build_ctx=temp_dir,
                models=[str(bento_model.tag)],
            )

        return built_bento
```

After building the class, we need to test it to ensure it works. Let's make necessary imports, initialize the Project
and the AI. You can run the AI over a small subset of Documents so that it does not take too much time.


```python editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
YOUR_PROJECT_ID = 46
YOUR_DOCUMENT_ID = 44823
import os
from konfuzio_sdk.trainer.document_categorization import CategorizationAI

CustomCategorizationAI = CategorizationAI
```

```python editable=true slideshow={"slide_type": ""}
from konfuzio_sdk.data import Project
from konfuzio_sdk.trainer.document_categorization import (
        EfficientNet,
        PageImageCategorizationModel,
    )

project = Project(id_=YOUR_PROJECT_ID)
categorization_pipeline = CustomCategorizationAI(project.categories)

categorization_pipeline.documents = [
        document for category in categorization_pipeline.categories for document in category.documents()
    ][:5]
categorization_pipeline.test_documents = [
    document for category in categorization_pipeline.categories for document in category.test_documents()
][:5]
```

Then, define all necessary components of the AI, train and evaluate it.
```python tags=["remove-output"]
categorization_pipeline.category_vocab = categorization_pipeline.build_template_category_vocab()
image_model = EfficientNet(name='efficientnet_b0')
categorization_pipeline.classifier = PageImageCategorizationModel(
        image_model=image_model,
        output_dim=len(categorization_pipeline.category_vocab),
    )
categorization_pipeline.build_preprocessing_pipeline(use_image=True)
categorization_pipeline.fit(n_epochs=1)
```
```python
data_quality = categorization_pipeline.evaluate(use_training_docs=True)
ai_quality = categorization_pipeline.evaluate(use_training_docs=False)
print(data_quality.f1(category=project.categories[0]))
print(ai_quality.f1(category=project.categories[0]))
```

Now you can categorize a Document.
```python
document = project.get_document_by_id(YOUR_DOCUMENT_ID)
categorization_result = categorization_pipeline.categorize(document=document)
for page in categorization_result.pages():
    print(f"Found category {page.category} for {page}")
print(f"Found category {categorization_result.category} for {categorization_result}")
```

Finally, save the model and check that it is loadable. Saving can be done in two different ways: as a compressed model and as a Bento containerized instance. 
**Note:** Now both of methods are supported by Konfuzio Server, but we will depreciate the former method when Bento components are developed for all AI types. 

Saving and loading a model via compression:
```python
pickle_model_path = categorization_pipeline.save(reduce_weight=False)
categorization_pipeline_loaded = CustomCategorizationAI.load_model(pickle_model_path)
```

Saving a model as a Bento instance:
```python
bento, path_to_bento = categorization_pipeline.save_bento(output_dir=project.model_folder)
```

To test that a model works, you can find out the model's name in BentoML's local registry and serve it via the following command:
```python tags=["skip-execution", "nbval-skip"]
bento_name = bento.tag.name + ':' + bento.tag.version
```

```bash tags=["skip-execution", "nbval-skip"]
bentoml serve extraction_0:gmu2lrbyugahyasc # replace the name to your value of bento_name variable
```

After you have trained and saved your custom AI, you can upload it using the steps from the [tutorial](https://help.konfuzio.com/tutorials/migrate-trained-ai-to-an-new-project-to-annotate-documents-faster/index.html#upload-extraction-or-category-ai-to-target-instance) or using the method 
`upload_ai_model()` as described in [Upload your AI](https://dev.konfuzio.com/sdk/tutorials/upload-your-ai/index.html), provided that you have the Superuser rights.

### Conclusion
In this tutorial, we have walked through the construction, training, testing and preparation to uploading of the custom Categorization AI. Below is the full code to accomplish this task. Note that this class is completely demonstrative, and to make it functional you would need to replace contents of defined methods with your own code.

```python editable=true slideshow={"slide_type": ""} tags=["skip-execution", "nbval-skip"] vscode={"languageId": "plaintext"}
import bentoml
import lz4
import os
import pathlib
import torch
from konfuzio_sdk.trainer.document_categorization import AbstractCategorizationAI
from konfuzio_sdk.data import Page, Category, Project
from typing import List

class CustomCategorizationAI(AbstractCategorizationAI):
    """Define a custom Categorization AI."""

    # specify if an AI uses text, images 
    requires_text = True
    requires_images = False
    requires_segmentation = False # always false because it is only required for existing segmentation AIs

    def __init__(self, categories: List[Category], *args, **kwargs) -> None:
        """
        Initialize a class.

        categories: A list of Categories between which the AI is going to distinguish.
        """
        # a list of Categories between which the AI will differentiate
        super().__init__(categories)
    
    def fit(self) -> None:
        """Fit the classifier on training Documents."""
        self.classifier_iterator = build_document_classifier_iterator(
                    self.documents,
                    self.train_transforms,
                    use_image = True,
                    use_text = False,
                    device='cpu',
                )
        self.classifier._fit_classifier(self.classifier_iterator, **kwargs)
    
    def _categorize_page(self, page: Page) -> Page:
        """
        Assign Category Annotations to the Document's Page.
        
        page: A Page to which Category Annotations are assigned.
        """
        page_image = page.get_image()
        predicted_category_id, predicted_confidence = self._predict(page_image)
        
        for category in self.categories:
            if category.id_ == predicted_category_id:
                _ = CategoryAnnotation(category=category, confidence=predicted_confidence, page=page)
        
        return page
    
    @property
    def entrypoint_methods(self) -> dict:
        """List the model's methods to make it accessible in a containerized instance of the AI."""
        return {
            'categorize': {'batchable': False}
        }

    @property
    def bento_metadata(self) -> dict:
        """List if the AI requires processing of images/text and segmentation, and specify formats of request and response. Needed for server support of the AI. """
        return {
            'requires_images': getattr(self, 'requires_images', False),
            'requires_segmentation': getattr(self, 'requires_segmentation', False),
            'requires_text': getattr(self, 'requires_text', False),
            'request': 'CategorizeRequest20240729', 
            'response': 'CategorizeResponse20240729', 
        }

    def build_bento(self, bento_model) -> Bento:
        """Build a archived instance of the AI."""
        bento_module_dir = 'konfuzio-sdk/konfuzio_sdk/bento/categorization' # specify your own path if the root folder where konfuzio_sdk is stored has a different name
        dict_metadata = self.project.create_project_metadata_dict()

        #  create a temporary directory for a future bento archive
        with tempfile.TemporaryDirectory() as temp_dir:
            # copy bento_module_dir to temp_dir
            shutil.copytree(bento_module_dir, temp_dir + '/categorization')
            # include metadata
            with open(f'{temp_dir}/categories_and_label_data.json5', 'w') as f:
                json.dump(dict_metadata, f, indent=2, sort_keys=True)
            # include the AI model name so the service can load it correctly
            with open(f'{temp_dir}/AI_MODEL_NAME', 'w') as f:
                f.write(self._pkl_name)

            built_bento = bentoml.bentos.build(
                name=f"categorization_{self.category.id_ if self.category else '0'}",
                service=f'categorization/categorizationai_service.py:CategorizationService',
                include=[
                    'categorization/*.py',
                    'categories_and_label_data.json5',
                    'AI_MODEL_NAME',
                ],
                labels=self.bento_metadata,
                python={
                    'packages': [
                        'https://github.com/konfuzio-ai/konfuzio-sdk/archive/refs/heads/master.zip#egg=konfuzio-sdk'
                    ],
                    'lock_packages': True,
                },
                build_ctx=temp_dir,
                models=[str(bento_model.tag)],
            )

        return built_bento

project = Project(id_=YOUR_PROJECT_ID)
categorization_pipeline = CustomCategorizationAI(project.categories)

categorization_pipeline.documents = [
        document for category in categorization_pipeline.categories for document in category.documents()
    ][:5]
categorization_pipeline.test_documents = [
    document for category in categorization_pipeline.categories for document in category.test_documents()
][:5]

categorization_pipeline.category_vocab = categorization_pipeline.build_template_category_vocab()
image_model = EfficientNet(name='efficientnet_b0')
categorization_pipeline.classifier = PageImageCategorizationModel(
        image_model=image_model,
        output_dim=len(categorization_pipeline.category_vocab),
    )
categorization_pipeline.build_preprocessing_pipeline(use_image=True)
categorization_pipeline.fit(n_epochs=1)

data_quality = categorization_pipeline.evaluate(use_training_docs=True)
ai_quality = categorization_pipeline.evaluate(use_training_docs=False)
print(data_quality.f1(category=project.categories[0]))
print(ai_quality.f1(category=project.categories[0]))

document = project.get_document_by_id(YOUR_DOCUMENT_ID)
categorization_result = categorization_pipeline.categorize(document=document)
for page in categorization_result.pages():
    print(f"Found category {page.category} for {page}")
print(f"Found category {categorization_result.category} for {categorization_result}")

pickle_model_path = categorization_pipeline.save_bento()
categorization_pipeline_loaded = CustomCategorizationAI.load_model(pickle_model_path)
bento, path_to_bento = categorization_pipeline.save_bento(output_dir=project.model_folder)
bento_name = bento.tag.name + ':' + bento.tag.version
```

```python tags=["remove-cell"]
os.remove(pickle_model_path)
```

### What's next?

- [Learn how to upload a trained custom AI to the Konfuzio app or an on-prem installation](https://dev.konfuzio.com/sdk/tutorials/upload-your-ai/index.html)


