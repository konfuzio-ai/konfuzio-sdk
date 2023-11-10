---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: konfuzio
    language: python
    name: python3
---

## Set the Category manually

---

**Prerequisites:** 
- Data Layer concepts of Konfuzio SDK: Project, Category, Document

**Difficulty:** Easy

**Goal:** Learn how to set, change and remove Category of a Document and its Pages manually.

---

### Introduction

When creating a new Document, the first step is to assign a Category to it. In this tutorial you will find out how to do it manually.

You can initialize a Document with a specific Category:

```python tags=["remove-cell"]
import logging
import konfuzio_sdk

logging.getLogger("konfuzio_sdk").setLevel(logging.ERROR)

YOUR_PROJECT_ID = 46
YOUR_CATEGORY_ID = 63
YOUR_DOCUMENT_ID = 44865
```

```python
from konfuzio_sdk.data import Project, Document

project = Project(id_=YOUR_PROJECT_ID)
my_category = project.get_category_by_id(YOUR_CATEGORY_ID)
my_document = Document(text="My text.", project=project, category=my_category)
assert my_document.category == my_category
```

You can also use `Document.set_category` to set a Document’s Category after it has been initialized. This will count as if a human manually revised it.

*Note:* a Document’s Category can be changed via set_category only if the original Category has been set to no_category. Otherwise, an attempt to change a Category will cause an error.

```python
document = project.get_document_by_id(YOUR_DOCUMENT_ID)
document.set_category(None)
assert document.category == project.no_category
document.set_category(my_category)
assert document.category == my_category
assert document.category_is_revised is True
```

Each Page's Category will also be changed to a Category set to this Document.

```python
for page in document.pages():
    assert page.category == my_category
```

If a Document is initialized with no Category, it will automatically be set to NO_CATEGORY. Another Category can be manually set later.


### Conclusion
In this tutorial, we walked you through the steps of manually setting and changing the Category of a Document and its Pages. Below is the full code to accomplish this task:

```python tags=["skip-execution"]
project = Project(id_=YOUR_PROJECT_ID)
my_category = project.get_category_by_id(YOUR_CATEGORY_ID)

my_document = Document(text="My text.", project=project, category=my_category)
assert my_document.category == my_category

document = project.get_document_by_id(YOUR_DOCUMENT_ID)
document.set_category(None)
assert document.category == project.no_category
document.set_category(my_category)
assert document.category == my_category
assert document.category_is_revised is True

for page in document.pages():
    assert page.category == my_category
```

### What's next?

- [Learn how to categorize Documents automatically using Categorization AI](https://dev.konfuzio.com/sdk/tutorials/document_categorization/index.html)
- [Create your own custom Categorization AI](https://dev.konfuzio.com/sdk/tutorials/create-custom-categorization-ai/index.html)
