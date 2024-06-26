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

## Upload your AI

---

**Prerequisites:**

- Data Layer concepts of Konfuzio: Project
- AI concepts of Konfuzio: Information Extraction, Document Categorization, File Splitting
- Konfuzio API
- Superuser status

**Difficulty:** Medium 

**Goal:** Learn how to upload a previously saved instance of any AI - Extraction, Categorization or File Splitting.

---

### Environment
You need to install the Konfuzio SDK before diving into the tutorial. \
To get up and running quickly, you can use our Colab Quick Start notebook. \
<a href="https://colab.research.google.com/github/konfuzio-ai/konfuzio-sdk/blob/master/notebooks/Quick_start_template_for_Konfuzio_SDK.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

As an alternative you can follow the [installation section](../get_started.html#install-sdk) to install and initialize the Konfuzio SDK locally or on an environment of your choice.

### Introduction

If you want to upload a model that you've built locally to the Server, you can use one of the two options, provided that you have the Superuser rights.

First option is manual, using the steps from the [tutorial](https://help.konfuzio.com/tutorials/migrate-trained-ai-to-an-new-project-to-annotate-documents-faster/index.html#upload-extraction-or-category-ai-to-target-instance).

Second is using the method `upload_ai_model()` from `konfuzio_sdk.api`. Arguments are different for different types of 
AI. The method returns the model's ID that can later be used to update or delete the model.

```python editable=true slideshow={"slide_type": ""} tags=["skip-execution", "nbval-skip"] vscode={"languageId": "plaintext"}
from konfuzio_sdk.api import upload_ai_model

extraction_model_id = upload_ai_model(pickle_model_path, ai_type='extraction', category_id=YOUR_CATEGORY_ID)
categorization_model_id = upload_ai_model(pickle_model_path, ai_type='categorization', project_id=YOUR_PROJECT_ID)
splitting_model_id = upload_ai_model(pickle_model_path, ai_type='filesplitting', project_id=YOUR_PROJECT_ID)
```

### Update model's metadata

You can update an uploaded model via the `update_ai_model()` method. The information you can change is model's name and 
description.


```python editable=true slideshow={"slide_type": ""} tags=["skip-execution", "nbval-skip"]
from konfuzio_sdk.api import update_ai_model

update_ai_model(YOUR_MODEL_ID, ai_type='extraction')
update_ai_model(YOUR_MODEL_ID, ai_type='categorization')
update_ai_model(YOUR_MODEL_ID, ai_type='filesplitting')
```

### Remove a model

You can also remove an uploaded model by using `delete_ai_model()`.

```python editable=true slideshow={"slide_type": ""} tags=["skip-execution", "nbval-skip"]
from konfuzio_sdk.api import delete_ai_model

delete_ai_model(YOUR_MODEL_ID, ai_type='extraction')
delete_ai_model(YOUR_MODEL_ID, ai_type='categorization')
delete_ai_model(YOUR_MODEL_ID, ai_type='filesplitting')
```

### What's next?

- [Get to know more about the Konfuzio API](https://dev.konfuzio.com/web/api-v3.html)
