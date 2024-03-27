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

## Create a custom Categorization AI

---

**Prerequisites:**
- Data Layer concepts of Konfuzio: Project, Document, Category, Page
- AI concepts of Konfuzio: Categorization
- Understanding of OOP: Classes, inheritance

**Difficulty:** Medium

**Goal:** Learn how to create a custom Categorization AI with manually defined architecture.

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

**Note**: currently, the Server supports AI models created using `python 3.8`.

By default, any [Categorization AI](https://dev.konfuzio.com/sdk/tutorials/document_categorization/index.html) class should derive from the `AbstractCategorizationModel` class and implement methods `__init__`, `fit`, `_categorize_page` and `save`.

Let's make necessary imports and define the class. In `__init__`, you can either not make any changes or initialize key variables required by your custom AI.
```python tags=["remove-cell"]
import logging

logging.getLogger("konfuzio_sdk").setLevel(logging.ERROR)
logging.getLogger("timm").setLevel(logging.CRITICAL)
```

```python editable=true slideshow={"slide_type": ""} tags=["skip-execution", "nbval-skip"]
import lz4
import os
import pathlib
import torch
from konfuzio_sdk.trainer.document_categorization import AbstractCategorizationAI
from konfuzio_sdk.data import Page, Category, Project
from typing import List

class CustomCategorizationAI(AbstractCategorizationAI):
    def __init__(self, categories: List[Category], *args, **kwargs):
        # a list of Categories between which the AI will differentiate
        super().__init__(categories)
```

Then we need to define `fit` method. It can contain any custom architecture, for instance, a multi-layered perceptron, or some hardcoded logic like [Name-based Categorization](https://dev.konfuzio.com/sdk/tutorials/document_categorization/index.html#name-based-categorization-ai). This method does not return anything; rather, it modifies the `self.model` if you provide this attribute.

This method is allowed to be implemented as a no-op if you provide the trained model in other ways.

```python editable=true slideshow={"slide_type": ""} tags=["skip-execution", "nbval-skip"]
    def fit(self):
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
        page_image = page.get_image()
        predicted_category_id, predicted_confidence = self._predict(page_image)
        
        for category in self.categories:
            if category.id_ == predicted_category_id:
                _ = CategoryAnnotation(category=category, confidence=predicted_confidence, page=page)
        
        return page
```

Lastly, we define saving method for the new AI. It needs to be compressed into .lz4 format to be compatible with Konfuzio when uploaded to the server or an on-prem installation later.
For example, saving can be defined similarly to the way it is defined in `CategorizationAI` class. Make sure to check that the save path exists using `pathlib`, 
assign paths for temporary .pt file and final compressed file (this is needed for compressing the initially saved model),
create dictionary to save all necessary model data needed for categorization/inference. If you use torch-based model, 
save it using `torch.save()` and then compress the resulting file via lz4, removing the temporary file afterward.

```python editable=true slideshow={"slide_type": ""} tags=["skip-execution", "nbval-skip"] vscode={"languageId": "plaintext"}
    def save(self, output_dir: str=None):

        if not output_dir:
            self.output_dir = self.project.model_folder
        else:
            self.output_dir = output_dir
        pathlib.Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        temp_pt_file_path = self.temp_pt_file_path
        compressed_file_path = self.compressed_file_path

        data_to_save = {
            "classifier": self.classifier,
            "categories": self.categories,
            "model_type": "CustomCategorizationAI",
        }

        # save all necessary model data
        torch.save(data_to_save, temp_pt_file_path)
        with open(temp_pt_file_path, 'rb') as f_in:
            with open(compressed_file_path, 'wb') as f_out:
                compressed = lz4.frame.compress(f_in.read())
                f_out.write(compressed)
        self.pipeline_path = compressed_file_path
        os.remove(temp_pt_file_path)
        return self.pipeline_path
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

Finally, save the model and check that it is loadable.
```python
pickle_model_path = categorization_pipeline.save(reduce_weight=False)
categorization_pipeline_loaded = CustomCategorizationAI.load_model(pickle_model_path)
```

After you have trained and saved your custom AI, you can upload it using the steps from the [tutorial](https://help.konfuzio.com/tutorials/migrate-trained-ai-to-an-new-project-to-annotate-documents-faster/index.html#upload-extraction-or-category-ai-to-target-instance) or using the method 
`upload_ai_model()` as described in [Upload your AI](https://dev.konfuzio.com/sdk/tutorials/upload-your-ai/index.html), provided that you have the Superuser rights.

### Conclusion
In this tutorial, we have walked through the construction, training, testing and preparation to uploading of the custom Categorization AI. Below is the full code to accomplish this task. Note that this class is completely demonstrative, and to make it functional you would need to replace contents of defined methods with your own code.

```python editable=true slideshow={"slide_type": ""} tags=["skip-execution", "nbval-skip"] vscode={"languageId": "plaintext"}
import os

from konfuzio_sdk.trainer.document_categorization import AbstractCategorizationAI, EfficientNet, PageImageCategorizationModel
from konfuzio_sdk.data import Page, Category, Project
from typing import List

class CustomCategorizationAI(AbstractCategorizationAI):
    def __init__(self, categories: List[Category], *args, **kwargs):
        super().__init__(categories)

    def fit(self, n_epochs=1) -> None:
        self.classifier_iterator = self.build_document_classifier_iterator(
                    self.documents,
                    self.train_transforms,
                    use_image = True,
                    use_text = False,
                    device='cpu',
                )
        self.classifier, training_metrics = self._fit_classifier(train_iterator, **kwargs)

    def _categorize_page(self, page: Page) -> Page:
        page_image = page.get_image()
        predicted_category_id, predicted_confidence = self._predict(page_image)
        
        for category in self.categories:
            if category.id_ == predicted_category_id:
                _ = CategoryAnnotation(category=category, confidence=predicted_confidence, page=page)
        
        return page

    def save(self, output_dir: str=None):
        if not output_dir:
            self.output_dir = self.project.model_folder
        else:
            self.output_dir = output_dir

        pathlib.Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        temp_pt_file_path = self.temp_pt_file_path
        compressed_file_path = self.compressed_file_path

        if self.categories:
            self.categories[0].project.lose_weight()

        data_to_save = {
            "classifier": self.classifier,
            "categories": self.categories,
            "model_type": "CustomCategorizationAI",
        }

        torch.save(data_to_save, temp_pt_file_path)
        with open(temp_pt_file_path, 'rb') as f_in:
            with open(compressed_file_path, 'wb') as f_out:
                compressed = lz4.frame.compress(f_in.read())
                f_out.write(compressed)
        self.pipeline_path = compressed_file_path
        os.remove(temp_pt_file_path)
        return self.pipeline_path

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

pickle_model_path = categorization_pipeline.save(reduce_weight=False)
categorization_pipeline_loaded = CustomCategorizationAI.load_model(pickle_model_path)
```

```python tags=["remove-cell"]
os.remove(pickle_model_path)
```

### What's next?

- [Learn how to upload a trained custom AI to the Konfuzio app or an on-prem installation](https://dev.konfuzio.com/sdk/tutorials/upload-your-ai/index.html)


