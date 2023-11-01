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
- Data Layer concepts of Konfuzio
- AI concepts of Konfuzio

**Difficulty:** Medium

**Goal:** Learn how to create a custom Categorization AI with manually defined architecture.

---

### Introduction

This tutorial explains how to build and train a custom Categorization AI locally, how to save it and upload it to the Konfuzio 
Server. If you run this tutorial in Colab and experience any version compatibility issues when working with the SDK, restart the
runtime and initialize the SDK once again; this will resolve the issue.

**Note**: you don't necessarily need to create the AI from scratch if you already have some document-processing architecture.
You just need to wrap it into the class that corresponds to our Categorization AI structure. Follow the steps in this 
tutorial to find out what are the requirements for that.

**Note**: currently, the Server supports AI models created using `torch<2.0.0`.

By default, any [Categorization AI](ADD LINK) class should derive from the `AbstractCategorizationModel` class and implement methods `__init__`, `fit`, `_categorize_page` and `save`.

<!-- #region editable=true slideshow={"slide_type": ""} -->
Let's make necessary imports and define the class. In `__init__`, you can either not make any changes or initialize key variables required by your custom AI.
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""} tags=["skip-execution", "nbval-skip"]
import lz4
import torch
from konfuzio_sdk.trainer.document_categorization import AbstractCategorizationAI
from konfuzio_sdk.data import Page, Category, Project
from typing import List

class CustomCategorizationAI(AbstractCategorizationAI):
    def __init__(self, categories: List[Category], *args, **kwargs):
        # a list of Categories between which the AI will differentiate
        super().__init__(categories)
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
Then we need to define `fit` method. It can contain any custom architecture, for instance, a multi-layered perceptron, or some hardcoded logic like [Name-based Categorization](ADD LINK). his method does not return anything; rather, it modifies the `self.model` if you provide this attribute.

This method is allowed to be implemented as a no-op if you provide the trained model in other ways.
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""} tags=["skip-execution", "nbval-skip"]
    def fit(self):
        # Define architecture and training that the model undergoes, i.e. a NN architecture or a custom hardcoded logic
        # for instance:
        
        self.classifier_iterator = build_document_classifier_iterator(
                    self.documents,
                    self.train_transforms,
                    use_image = True,
                    use_text = False,
                    device='cpu',
                )
        self.classifier._fit_classifier(self.classifier_iterator, **kwargs)
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
Next, we need to define how the model assigns a Category to a Page inside of a `_categorize_page` method. **NB:** The result of extraction must be the input Page with added Categorization attribute `Page.category`.
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""} tags=["skip-execution", "nbval-skip"]
    def _categorize_page(self, page: Page) -> Page:
        # define how the model assigns a Category to a Page.
        # for instance:

        page_image = page.get_image()
        predicted_category_id, predicted_confidence = self._predict(page_image)
        
        for category in self.categories:
            if category.id_ == predicted_category_id:
                _ = CategoryAnnotation(category=category, confidence=predicted_confidence, page=page)
        
        return page
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
Lastly, we define saving method for the new AI. It needs to be compressed into .lz4 format to be compatible with Konfuzio when uploaded to the server or an on-prem installation later.
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""} tags=["skip-execution", "nbval-skip"] vscode={"languageId": "plaintext"}
    def save(self, output_dir: str=None):

        # define how to save a model – for example, in a way it's defined in the CategorizationAI
        if not output_dir:
            self.output_dir = self.project.model_folder
        else:
            self.output_dir = output_dir

        # make sure output dir exists
        pathlib.Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # temp_pt_file_path is needed to save an intermediate .pt file that later will be compressed and deleted.
        temp_pt_file_path = self.temp_pt_file_path
        compressed_file_path = self.compressed_file_path

        if self.categories:
            self.categories[0].project.lose_weight()

        # create dictionary to save all necessary model data. all attributes are arbitrary, include those that your custom AI has.
        # save only the necessary parts of the model for extraction/inference.
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

```python editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
import lz4
import torch
from konfuzio_sdk.trainer.document_categorization import AbstractCategorizationAI
from konfuzio_sdk.data import Page, Category
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
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
After building the class, we need to test it to ensure it works. You can run it over a small subset of documents so that it does not take too much time. 
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
YOUR_PROJECT_ID = 46
YOUR_DOCUMENT_ID = 44823
from konfuzio_sdk.trainer.document_categorization import CategorizationAI

CustomCategorizationAI = CategorizationAI
```

```python editable=true slideshow={"slide_type": ""} tags=["remove-output"]
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
categorization_pipeline.category_vocab = categorization_pipeline.build_template_category_vocab()
image_model = EfficientNet(name='efficientnet_b0')
categorization_pipeline.classifier = PageImageCategorizationModel(
        image_model=image_model,
        output_dim=len(categorization_pipeline.category_vocab),
    )
categorization_pipeline.build_preprocessing_pipeline(use_image=True)
categorization_pipeline.fit(n_epochs=1)

# evaluate the AI
data_quality = categorization_pipeline.evaluate(use_training_docs=True)
ai_quality = categorization_pipeline.evaluate(use_training_docs=False)

# Categorize a Document
document = project.get_document_by_id(YOUR_DOCUMENT_ID)
categorization_result = categorization_pipeline.categorize(document=document)
for page in categorization_result.pages():
    print(f"Found category {page.category} for {page}")
print(f"Found category {categorization_result.category} for {categorization_result}")

# Save and load a pickle file for the model
pickle_model_path = categorization_pipeline.save(reduce_weight=False)
categorization_pipeline_loaded = CustomCategorizationAI.load_model(pickle_model_path)
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
After you have trained and saved your custom AI, you can upload it using the steps from the [tutorial](ADD LINK) or using the method 
`upload_ai_model()` as described in [Upload your AI](ADD LINK), provided that you have the Superuser rights.
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} -->
### Conclusion
In this tutorial, we have walked through the construction, training, testing and preparation to uploading of the custom Categorization AI. Below is the full code to accomplish this task. Note that this class is completely demonstrative, and to make it functional you would need to replace contents of defined methods with your own code.
<!-- #endregion -->

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

# evaluate the AI
data_quality = categorization_pipeline.evaluate(use_training_docs=True)
ai_quality = categorization_pipeline.evaluate(use_training_docs=False)

# Categorize a Document
document = project.get_document_by_id(YOUR_DOCUMENT_ID)
categorization_result = categorization_pipeline.categorize(document=document)
for page in categorization_result.pages():
    print(f"Found category {page.category} for {page}")
print(f"Found category {categorization_result.category} for {categorization_result}")

# Save and load a pickle file for the model
pickle_model_path = categorization_pipeline.save(reduce_weight=False)
categorization_pipeline_loaded = CustomCategorizationAI.load_model(pickle_model_path)
```

```python tags=["remove-cell"]
os.remove(pickle_model_path)
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
### What's next?

- Learn how to upload a trained custom AI to the Konfuzio app or an on-prem installation.

<!-- #endregion -->
