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

## Understanding Konfuzio Data Validation Rules

---

**Prerequisites:**
- Data Layer concepts of Konfuzio: Project, Document, Annotation, Span, Bbox, Category

**Difficulty:** Easy

**Goal:** Learn how Konfuzio's Data Validation Rules ensure consistent and well-formed training data for Extraction AI.

---

### Environment
You need to install the Konfuzio SDK before diving into the tutorial. \
To get up and running quickly, you can use our Colab Quick Start notebook. \
<a href="https://colab.research.google.com/github/konfuzio-ai/konfuzio-sdk/blob/master/notebooks/Quick_start_template_for_Konfuzio_SDK.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

As an alternative you can follow the [installation section](../get_started.html#install-sdk) to install and initialize the Konfuzio SDK locally or on an environment of your choice.

### Introduction

Konfuzio automatically applies a set of rules to validate data within a Project. These rules play a crucial role in ensuring that training and test data meet the necessary criteria for training an Extraction AI with Konfuzio.

In this tutorial, we will explore the different types of validation rules and understand how they impact the suitability of a Document for training an Extraction AI.

### Data validation rules overview

Konfuzio applies the following [data validation rules](https://dev.konfuzio.com/sdk/explanations.html#data-validation-rules):

1. **Document validation rules:**
   A Document passes the data validation rules only if all the contained Annotations, Spans, and Bboxes pass the checks. If any Annotation, Span, or Bbox within a Document fails, the entire Document is marked as unsuitable for training an Extraction AI.

2. **Annotation validation rules:**
   - The Annotation must be from the same Category as the Document.
   - The Annotation must not entirely overlap with another Annotation with the same Label (partial overlaps are allowed).
   - Full overlaps with different Labels are allowed.
   - The Annotation must have at least one Span.
   - Annotation validation rules are indifferent about the values of `Annotation.is_correct` or `Annotation.revised`. For more information about what these boolean values mean, see <a href="https://help.konfuzio.com/modules/annotations/index.html">Konfuzio Server - Annotations</a>.

3. **Span validation rules:**
   - The Span must contain non-empty text (start offset must be strictly lesser than the end offset).
   - The Span must be contained within a single line of text (not distributed across multiple lines).

4. **Bbox validation rules:**
   - The Bbox must have non-negative width and height (zero is allowed for compatibility reasons with many OCR engines).
   - The Bbox must be entirely contained within the bounds of a Page.
   - The character mapped by the Bbox must correspond to the text in the Document.

### Initializing a Project with data validation rules

By default, any Project has the data validation rules enabled, so nothing special needs to be done to enable it.

```python tags=["remove-cell"]
from tests.variables import TEST_PROJECT_ID, TEST_PAYSLIPS_CATEGORY_ID, TEST_DOCUMENT_ID
```

```python tags=["remove-output"]
from konfuzio_sdk.data import Project

project = Project(id_=TEST_PROJECT_ID)
```

### Initializing a Project with data validation rules disabled

In some cases, you may want to disable the data validation rules to define a custom data structure or training pipeline that violates some assumptions normally present in Konfuzio Extraction AIs and pipelines. If you don’t want to validate your data, you should initialize the Project with `strict_data_validation=False`.

```python tags=["remove-output"]
project = Project(id_=TEST_PROJECT_ID, strict_data_validation=False)
```

Note: We highly recommend keeping the data validation rules enabled at all times, as they ensure that training and test data is consistent for training an Extraction AI. Disabling the data validation rules and training an Extraction AI with potentially duplicated, malformed, or inconsistent data can decrease the quality of an Extraction AI. Only disable them if you know what you are doing.


### Conclusion
In this tutorial, you have learned about the important data validation rules in Konfuzio and how they play a vital role in ensuring the quality and consistency of training data for Extraction AI. It is recommended to always keep these rules enabled to maintain the highest level of accuracy in your AI models.


### What's Next?

- <a href="/sdk/tutorials/information_extraction">Explore Extraction AI methods</a> 

