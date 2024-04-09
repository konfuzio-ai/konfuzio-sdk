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

## Train a Context-Aware File Splitting AI

---

**Prerequisites:**

- Data Layer concepts of Konfuzio: Project, Category, Page, Document, Span
- AI concepts of Konfuzio: File Splitting

**Difficulty:** Medium

**Goal:** Learn how to train a Context-Aware File Splitting AI and use it to split Documents.

---

### Environment
You need to install the Konfuzio SDK before diving into the tutorial. \
To get up and running quickly, you can use our Colab Quick Start notebook. \
<a href="https://colab.research.google.com/github/konfuzio-ai/konfuzio-sdk/blob/master/notebooks/Quick_start_template_for_Konfuzio_SDK.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

As an alternative you can follow the [installation section](../get_started.html#install-sdk) to install and initialize the Konfuzio SDK locally or on an environment of your choice.

### Introduction

Konfuzio SDK offers several approaches for automatically splitting a multi-Document file into several Documents. One of them is Context-Aware File Splitting AI that uses a context-aware logic. By context-aware we mean a rule-based approach that looks for common strings between the first Pages of all Category's Documents. Upon predicting whether a Page is a potential splitting point (meaning whether it is 
first or not), we compare Page's contents to these common first-page strings; if there is occurrence of at least one 
such string, we mark a Page to be first (thus meaning it is a splitting point).

#### Initialize and train Context-Aware File Splitting AI

In this tutorial we will be using pre-built classes `ContextAwareFileSplittingModel` and `SplittingAI`. Let's start with making necessary imports, initializing the Project and fetching the test Document.

```python editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
import logging
import os
from konfuzio_sdk.samples import LocalTextProject
logging.getLogger("konfuzio_sdk").setLevel(logging.ERROR)
logging.getLogger("tqdm").setLevel(logging.ERROR)
YOUR_PROJECT_ID = 46
YOUR_DOCUMENT_ID = 44865
```

```python editable=true slideshow={"slide_type": ""} tags=["remove-output"] vscode={"languageId": "plaintext"}
from konfuzio_sdk.data import Page, Category, Project
from konfuzio_sdk.trainer.file_splitting import SplittingAI, ContextAwareFileSplittingModel
from konfuzio_sdk.tokenizer.regex import ConnectedTextTokenizer

project = Project(id_=YOUR_PROJECT_ID)
test_document = project.get_document_by_id(YOUR_DOCUMENT_ID)
```
```python tags=["remove-cell"]
from copy import deepcopy

project = LocalTextProject()
test_document = ConnectedTextTokenizer().tokenize(deepcopy(project.get_document_by_id(9)))
project.categories = [project.get_category_by_id(3), project.get_category_by_id(4)]
```

Then, initialize a Context-Aware File Splitting Model and "fit" it on the Project's Categories. Tokenizer is needed to split the texts of the Documents in the Categories into the groups among which the algorhythm will search for the intersections.

`allow_empty_categories` parameter allows to have Categories that have Documents so diverse that there has not been any intersections found for them.

```python editable=true slideshow={"slide_type": ""}
file_splitting_model = ContextAwareFileSplittingModel(
    categories=project.categories, tokenizer=ConnectedTextTokenizer()
)

file_splitting_model.fit(allow_empty_categories=True)
```

Save the model:

```python editable=true slideshow={"slide_type": ""}
save_path = file_splitting_model.save(include_konfuzio=True)
```

Run the prediction to ensure it is able to predict the split points (first Pages) correctly:

```python editable=true slideshow={"slide_type": ""}
for page in test_document.pages():
    pred = file_splitting_model.predict(page)
    if pred.is_first_page:
        print(
            'Page {} is predicted as the first. Confidence: {}.'.format(page.number, page.is_first_page_confidence)
        )
    else:
        print('Page {} is predicted as the non-first.'.format(page.number))
```

#### Use the model with the Splitting AI

Splitting AI is a more high-level interface to Context Aware File Splitting Model and any other models that can be developed for File Splitting purposes. It takes a Document as an input, rather than individual Pages, because it utilizes page-level prediction of possible split points and returns Document or Documents with changes depending on the prediction mode.

You can load a pre-saved model or pass an initialized instance as the input. In this example, we load a previously saved one.

```python editable=true slideshow={"slide_type": ""} vscode={"languageId": "plaintext"}
model = ContextAwareFileSplittingModel.load_model(save_path)

splitting_ai = SplittingAI(model)
```

Splitting AI can be run in two modes: returning a list of Sub-Documents as the result of the input Document splitting or returning a copy of the input Document with Pages predicted as first having an attribute `is_first_page`. The flag `return_pages` has to be True for the latter; we will use it for an example.

```python editable=true slideshow={"slide_type": ""}
new_document = splitting_ai.propose_split_documents(test_document, return_pages=True)

for page in new_document[0].pages():
    if page.is_first_page:
        print(
            'Page {} is predicted as the first. Confidence: {}.'.format(page.number, page.is_first_page_confidence)
        )
    else:
        print('Page {} is predicted as the non-first.'.format(page.number))
```

### Conclusion

In this tutorial, we have walked through the essential steps for training and using Context-Aware File Splitting Model. Below is the full code to accomplish this task:

```python editable=true slideshow={"slide_type": ""} tags=["skip-execution", "nbval-skip"]
from konfuzio_sdk.data import Page, Category, Project
from konfuzio_sdk.trainer.file_splitting import SplittingAI, ContextAwareFileSplittingModel
from konfuzio_sdk.tokenizer.regex import ConnectedTextTokenizer

project = Project(id_=YOUR_PROJECT_ID)
test_document = project.get_document_by_id(YOUR_DOCUMENT_ID)
file_splitting_model = ContextAwareFileSplittingModel(
    categories=project.categories, tokenizer=ConnectedTextTokenizer()
)
file_splitting_model.documents = file_splitting_model.documents
file_splitting_model.fit(allow_empty_categories=True)
save_path = file_splitting_model.save(include_konfuzio=True)
for page in test_document.pages():
    pred = file_splitting_model.predict(page)
    if pred.is_first_page:
        print(
            'Page {} is predicted as the first. Confidence: {}.'.format(page.number, page.is_first_page_confidence)
        )
    else:
        print('Page {} is predicted as the non-first.'.format(page.number))
model = ContextAwareFileSplittingModel.load_model(save_path)
splitting_ai = SplittingAI(model)
new_document = splitting_ai.propose_split_documents(test_document, return_pages=True)
for page in new_document[0].pages():
    if page.is_first_page:
        print(
            'Page {} is predicted as the first. Confidence: {}.'.format(page.number, page.is_first_page_confidence)
        )
    else:
        print('Page {} is predicted as the non-first.'.format(page.number))
```

```python editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
os.remove(save_path)
```

### What's next?

- [Learn how to create a custom File Splitting AI](https://dev.konfuzio.com/sdk/tutorials/create-custom-splitting-ai/index.html)
- [Find out how to evaluate a File Splitting AI's performance](https://dev.konfuzio.com/sdk/tutorials/file-splitting-evaluation/index.html)
