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

- Data Layer concepts of Konfuzio

**Difficulty:** Easy

**Goal:** Learn how to create Annotations based on simple regular expression-based logic.

---

### Introduction

In this guide, we'll show you how to use Python and regular expressions (regex) to automatically identify and annotate specific text patterns within a Document. 

### Initialize a Project, define searched term and a Label

Let's say we have a Document, and we want to highlight every instance of the term "Musterstraße", which might represent a specific street name or location. Our task is to find this term, label it as "Lohnart", and associate it with the "Brutto-Bezug" Label Set.

```python editable=true slideshow={"slide_type": ""} tags=["remove-cell"] vscode={"languageId": "plaintext"}
YOUR_PROJECT_ID = 46
```

```python editable=true slideshow={"slide_type": ""} tags=["remove-output"] vscode={"languageId": "plaintext"}
import re
from konfuzio_sdk.data import Project, Annotation, Span, AnnotationSet

my_project = Project(id_=YOUR_PROJECT_ID)
# Word/expression to annotate in the Document should match an existing one in your Document
input_expression = "Musterstraße"

# Label for the Annotation
label_name = "Lohnart"

# Getting the Label from the Project
my_label = my_project.get_label_by_name(label_name)

# LabelSet to which the Label belongs
label_set = my_label.label_sets[0]
```

### Get a Document and find matches of a string in it

```python editable=true slideshow={"slide_type": ""}
# First Document in the Project
document = my_project.documents[0]

# Matches of the word/expression in the Document
matches_locations = [(m.start(0), m.end(0)) for m in re.finditer(input_expression, document.text)]
```

### Create the Annotations

```python editable=true slideshow={"slide_type": ""}
# List to save the links to the Annotations created
new_annotations_links = []

# Create Annotation for each match
for offsets in matches_locations:
    span = Span(start_offset=offsets[0], end_offset=offsets[1])
    annotation_set = AnnotationSet(document=document, label_set=label_set)
    # note that no Annotation can exist outside the Annotation Set and every Annotation Set has to contain at least one Annotation
    annotation_obj = Annotation(
        document=document,
        annotation_set=annotation_set,
        label=my_label,
        label_set=label_set,
        confidence=1.0,
        spans=[span],
        is_correct=True,
    )
    # ensure that the Annotation is saved online
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
# Word/expression to annotate in the Document should match an existing one in your Document
input_expression = "Musterstraße"

# Label for the Annotation
label_name = "Lohnart"

# Getting the Label from the Project
my_label = my_project.get_label_by_name(label_name)

# LabelSet to which the Label belongs
label_set = my_label.label_sets[0]

# First Document in the Project
document = my_project.documents[0]

# Matches of the word/expression in the Document
matches_locations = [(m.start(0), m.end(0)) for m in re.finditer(input_expression, document.text)]

# List to save the links to the Annotations created
new_annotations_links = []

# Create Annotation for each match
for offsets in matches_locations:
    span = Span(start_offset=offsets[0], end_offset=offsets[1])
    annotation_set = AnnotationSet(document=document, label_set=label_set)
    # note that no Annotation can exist outside the Annotation Set and every Annotation Set has to contain at least one Annotation
    annotation_obj = Annotation(
        document=document,
        annotation_set=annotation_set,
        label=my_label,
        label_set=label_set,
        confidence=1.0,
        spans=[span],
        is_correct=True,
    )
    # ensure that the Annotation is saved online
    new_annotation_added = annotation_obj.save()
    if new_annotation_added:
        new_annotations_links.append(annotation_obj.get_link())
    # if you want to remove the Annotation and ensure it's deleted online, you can use the following:
    annotation_obj.delete(delete_online=True)
```

### What's next?

- [Learn how to create Annotations automatically using Extraction AI](https://dev.konfuzio.com/sdk/tutorials/information_extraction/index.html)
- [Get to know how to visualize created Annotations](https://dev.konfuzio.com//sdk/explanations.html#coordinates-system)