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

## Create a custom File Splitting AI

---

**Prerequisites:** 

- Data Layer concepts of Konfuzio: Category, Page
- AI concepts of Konfuzio: File Splitting

**Difficulty:** Medium

**Goal:** Learn how to build your own File Splitting AI and how to use it on Konfuzio app or in an on-prem installation.

---

### Environment
You need to install the Konfuzio SDK before diving into the tutorial. \
To get up and running quickly, you can use our Colab Quick Start notebook. \
<a href="https://colab.research.google.com/github/konfuzio-ai/konfuzio-sdk/blob/master/notebooks/Quick_start_template_for_Konfuzio_SDK.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

As an alternative you can follow the [installation section](../get_started.html#install-sdk) to install and initialize the Konfuzio SDK locally or on an environment of your choice.

### Introduction

This tutorial explains how to train a custom File Splitting AI locally, how to save it and upload it to the Konfuzio 
Server. If you run this tutorial in Colab and experience any version compatibility issues when working with the SDK, restart the
runtime and initialize the SDK once again; this will resolve the issue.

Note: you don't necessarily need to create the AI from scratch if you already have some Document-processing architecture.
You just need to wrap it into the class that corresponds to our File Splitting AI structure. Follow the steps in this 
tutorial to find out what are the requirements for that.

### Necessary methods for the Custom AI

By default, any [File Splitting AI](sourcecode.html#file-splitting-ai) class should derive from the `AbstractFileSplittingModel` class and implement the following methods:

- `__init__()` to initialize the class;
- `fit()` to train the model if needed;
- `predict()` to run prediction over the Pages;
- `check_is_ready()` to check if all components of the class needed for prediction are in place.

Let's begin with necessary imports and `__init__`. We need to pass Categories as the input because we define the split points based off on the Documents within certain Categories. This method can also be used to initialize key variables required by the custom AI, e.g. a pre-trained model.

```python editable=true slideshow={"slide_type": ""} tags=["skip-execution", "nbval-skip"] vscode={"languageId": "plaintext"}
import transformers

from konfuzio_sdk.trainer.file_splitting import AbstractFileSplittingModel
from konfuzio_sdk.data import Page, Category
from typing import List

class CustomFileSplittingModel(AbstractFileSplittingModel):
    def __init__(self, categories: List[Category], *args, **kwargs):
        super().__init__(categories)
        self.model = transformers.AutoModelForClassification("your-model-name-here")
```

Next, we need to define `fit()` method. It can be used to define architecture and training that the model undergoes, i.e. a NN architecture or a custom hardcoded logic. This method does not return anything; rather, it modifies the self.model if you provide this attribute. It also can be left empty if the AI uses a pre-trained model.

```python editable=true slideshow={"slide_type": ""} tags=["skip-execution", "nbval-skip"]
    def fit(self):
        self.model.train(self.documents)
```

Then, we define `predict()` which describes how the model determines a split point for a Page.

```python editable=true slideshow={"slide_type": ""} tags=["skip-execution", "nbval-skip"]
    def predict(self, page: Page) -> Page:
        predicted_page = self.model.predict(page)
        return predicted_page
```

Lastly, we define the method `check_is_ready()` which checks if all components needed for training/prediction are set, for instance, is self.tokenizer set or are all Categories non-empty – containing training and testing Documents.

```python editable=true slideshow={"slide_type": ""} tags=["nbval-skip", "skip-execution"]
    def check_is_ready(self) -> bool:
        if not self.model:
            raise ValueError('Cannot run without the correctly set model.')
        return True
```

### Conclusion

In this tutorial, we have walked through the steps for building a custom File Splitting AI. Below is the full code of the class; note that in order for it to run, you need to fill the methods with meaningful code instead of pseudocode.

```python editable=true slideshow={"slide_type": ""} tags=["skip-execution", "nbval-skip"] vscode={"languageId": "plaintext"}
import transformers

from konfuzio_sdk.trainer.file_splitting import AbstractFileSplittingModel
from konfuzio_sdk.data import Page, Category
from typing import List

class CustomFileSplittingModel(AbstractFileSplittingModel):
    def __init__(self, categories: List[Category], *args, **kwargs):
        super().__init__(categories)
        self.model = transformers.AutoModelForClassification("your-model-name-here")

    def fit(self):
        self.model.train(self.documents)

    def predict(self, page: Page) -> Page:
        predicted_page = self.model.predict(page)
        return predicted_page

    def check_is_ready(self) -> bool:
        if not self.model:
            raise ValueError('Cannot run without the correctly set model.')
        return True
```

### What's next?

- [Learn how to upload the custom AI you've just created](https://dev.konfuzio.com/sdk/tutorials/upload-your-ai/index.html)
