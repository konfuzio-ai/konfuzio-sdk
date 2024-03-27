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

## Create Regex-based Annotations

---

**Prerequisites:** 

- Data Layer concepts of Konfuzio: Project, Document, Annotation, Label, Annotation Set, Label Set
- Regular expressions

**Difficulty:** Easy

**Goal:** Learn how to create Annotations based on simple regular expression-based logic.

---

### Environment
You need to install the Konfuzio SDK before diving into the tutorial. \
To get up and running quickly, you can use our Colab Quick Start notebook. \
<a href="https://colab.research.google.com/github/konfuzio-ai/konfuzio-sdk/blob/master/notebooks/Quick_start_template_for_Konfuzio_SDK.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

As an alternative you can follow the [installation section](../get_started.html#install-sdk) to install and initialize the Konfuzio SDK locally or on an environment of your choice.

### Introduction

In this guide, we'll show you how to use Python and regular expressions (regex) to automatically identify and annotate specific text patterns within a Document. 

### Initialize a Project, define searched term and a Label

Let's say we have a Document, and we want to highlight every instance of the term "Musterstraße", which might represent a specific street name or location. Our task is to find this term, label it as "Lohnart", and associate it with the "Brutto-Bezug" Label Set.

```python editable=true slideshow={"slide_type": ""} tags=["remove-cell"] vscode={"languageId": "plaintext"}
import logging

logging.getLogger("konfuzio_sdk").setLevel(logging.ERROR)
YOUR_PROJECT_ID = 46
```

```python editable=true slideshow={"slide_type": ""} vscode={"languageId": "plaintext"}
import re
from konfuzio_sdk.data import Project, Annotation, Span, AnnotationSet

my_project = Project(id_=YOUR_PROJECT_ID)
input_expression = "Musterstraße"
label_name = "Lohnart"

my_label = my_project.get_label_by_name(label_name)
label_set = my_label.label_sets[0]
print(my_label)
print(label_set)
```

### Get a Document and find matches of a string in it

We fetch the first Document in the Project and search for the matches of the word/expression in the Document.

```python editable=true slideshow={"slide_type": ""}
document = my_project.documents[0]

matches_locations = [(m.start(0), m.end(0)) for m in re.finditer(input_expression, document.text)]
print(matches_locations)
```

### Create the Annotations

For each found match we create an Annotation. Note that no Annotation can exist outside the Annotation Set and every Annotation Set has to contain at least one Annotation.
By using `Annotation.save()` we ensure that each Annotation is saved online.

```python editable=true slideshow={"slide_type": ""} tags=["remove-output"]
new_annotations_links = []

for offsets in matches_locations:
    span = Span(start_offset=offsets[0], end_offset=offsets[1])
    annotation_set = AnnotationSet(document=document, label_set=label_set)
    annotation_obj = Annotation(
        document=document,
        annotation_set=annotation_set,
        label=my_label,
        label_set=label_set,
        confidence=1.0,
        spans=[span],
        is_correct=True,
    )

    new_annotation_added = annotation_obj.save()
    if new_annotation_added:
        new_annotations_links.append(annotation_obj.get_link())
    # if you want to remove the Annotation and ensure it's deleted online, you can use the following:
    annotation_obj.delete(delete_online=True)
```

### Conclusion
In this tutorial, we have walked through the essential steps for creating regex-based Annotations. Below is the full code to accomplish this task:

```python editable=true slideshow={"slide_type": ""} tags=["skip-execution", "skip-nbeval"] vscode={"languageId": "plaintext"}
import re
from konfuzio_sdk.data import Project, Annotation, Span, AnnotationSet

my_project = Project(id_=YOUR_PROJECT_ID)

input_expression = "Musterstraße"
label_name = "Lohnart"

my_label = my_project.get_label_by_name(label_name)

label_set = my_label.label_sets[0]

document = my_project.documents[0]

matches_locations = [(m.start(0), m.end(0)) for m in re.finditer(input_expression, document.text)]

new_annotations_links = []

for offsets in matches_locations:
    span = Span(start_offset=offsets[0], end_offset=offsets[1])
    annotation_set = AnnotationSet(document=document, label_set=label_set)
    annotation_obj = Annotation(
        document=document,
        annotation_set=annotation_set,
        label=my_label,
        label_set=label_set,
        confidence=1.0,
        spans=[span],
        is_correct=True,
    )

    new_annotation_added = annotation_obj.save()
    if new_annotation_added:
        new_annotations_links.append(annotation_obj.get_link())
    # if you want to remove the Annotation and ensure it's deleted online, you can use the following:
    annotation_obj.delete(delete_online=True)
```

### What's next?

- [Learn how to create Annotations automatically using Extraction AI](https://dev.konfuzio.com/sdk/tutorials/information_extraction/index.html)
- [Get to know how to visualize created Annotations](https://dev.konfuzio.com//sdk/explanations.html#coordinates-system)