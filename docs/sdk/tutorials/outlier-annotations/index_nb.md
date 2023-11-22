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

## Find possible outliers among the ground-truth Annotations

---

**Prerequisites:**

- Data Layer concepts of Konfuzio: Annotation, Label, Document, Project
- Regular expressions

**Difficulty:** Medium

**Goal:** Learn how to spot potentially wrong Annotations of a particular Label after your Documents have been processed via Information Extraction.

---

### Introduction

If you want to ensure that Annotations of a Label are consistent and check for possible outliers, you can use one of the `Label` class's methods. There are three of them available.

#### Outliers by regex

`Label.get_outliers_by_regex` looks for the "worst" regexes used to find the Annotations under a given Label. "Worst" is determined by
the number of True Positives (correctly extracted Annotations) calculated when evaluating the regexes' performance. The method returns Annotations predicted by the regexes with the least amount of True Positives. By default, the method returns Annotations retrieved by the regex that performs on the level of 10% in comparison to the best one.

```python editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
import logging

logging.getLogger("konfuzio_sdk").setLevel(logging.ERROR)
YOUR_PROJECT_ID = 46
YOUR_LABEL_NAME = 'Bank inkl. IBAN'
TOP_WORST = 1.0
```

Initialize the Project, select the Label you want to assess and run the method, passing all Categories that are referring to the Label Set of a given Label as an input. TOP_WORST is threshold for determining what percentage of the worst regexes' output to return and can be also modified manually; by default it is 0.1.

```python editable=true slideshow={"slide_type": ""} vscode={"languageId": "plaintext"}
from konfuzio_sdk.data import Project

project = Project(id_=YOUR_PROJECT_ID)
label = project.get_label_by_name(YOUR_LABEL_NAME)
outliers = label.get_probable_outliers_by_regex(project.categories, top_worst_percentage=TOP_WORST)
```

#### Outliers by confidence

`Label.get_probable_outliers_by_confidence` looks for the Annotations with the least confidence level, provided it is lower
than the specified threshold (the default threshold is 0.5). The method accepts an instance of `EvaluationExtraction` class as an input and uses confidence predictions from there.

```python editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
YOUR_LABEL_NAME = 'Austellungsdatum'
```

Initialize the Project and select the Label you want to assess.

```python editable=true slideshow={"slide_type": ""}
from konfuzio_sdk.data import Project

project = Project(id_=YOUR_PROJECT_ID)
label = project.get_label_by_name(YOUR_LABEL_NAME)
```

```python editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
from konfuzio_sdk.trainer.information_extraction import RFExtractionAI
from konfuzio_sdk.tokenizer.base import ListTokenizer
from konfuzio_sdk.tokenizer.regex import RegexTokenizer

pipeline = RFExtractionAI()
pipeline.tokenizer = ListTokenizer(tokenizers=[])
pipeline.category = label.project.get_category_by_id(id_=63)
train_doc_ids = {44823, 44839, 44840, 44841}
pipeline.documents = [doc for doc in pipeline.category.documents() if doc.id_ in train_doc_ids]
for cur_label in pipeline.category.labels:
    for regex in cur_label.find_regex(category=pipeline.category):
        pipeline.tokenizer.tokenizers.append(RegexTokenizer(regex=regex))
pipeline.test_documents = pipeline.category.test_documents()
pipeline.df_train, pipeline.label_feature_list = pipeline.feature_function(
    documents=pipeline.documents, require_revised_annotations=False
)
pipeline.fit()
predictions = []
for doc in pipeline.documents:
    predicted_doc = pipeline.extract(document=doc)
    predictions.append(predicted_doc)
GROUND_TRUTHS = pipeline.documents
PREDICTIONS = predictions
```

Pass a list of ground-truth Documents and a list of their processed counterparts into the `EvaluationExtraction` class, then use `get_probable_outliers_by_confidence` with evaluation results as the input.

```python editable=true slideshow={"slide_type": ""} tags=["remove-cell"]
CONFIDENCE = 0.9
GROUND_TRUTH_DOCUMENTS = pipeline.documents
PREDICTED_DOCUMENTS = predictions
```

```python editable=true slideshow={"slide_type": ""}
from konfuzio_sdk.evaluate import ExtractionEvaluation

evaluation = ExtractionEvaluation(documents=list(zip(GROUND_TRUTH_DOCUMENTS, PREDICTED_DOCUMENTS)), strict=False)
outliers = label.get_probable_outliers_by_confidence(evaluation, confidence=CONFIDENCE)
```

#### Outliers by normalization

`Label.get_probable_outliers_by_normalization` looks for the Annotations that are unable to pass normalization by the data
type of the given Label, meaning that they are not of the same data type themselves, thus outliers. For instance, if a Label with the data type "Date" is assigned to the line "Example st. 1", it will be returned by this method, because this line does not qualify as a date.

Initialize the Project and the Label you want to assess, then run `get_probable_outliers_by_normalization` passing all Categories that are referring to the Label Set of a given Label as an input.
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""}
from konfuzio_sdk.data import Project

project = Project(id_=YOUR_PROJECT_ID)
label = project.get_label_by_name(YOUR_LABEL_NAME)
outliers = label.get_probable_outliers_by_normalization(project.categories)
```

### Conclusion
In this tutorial, we have walked through the essential steps for finding potential outliers amongst the Annotations. Below is the full code to accomplish this task.
Note that you need to replace placeholders with respective values for the tutorial to run.

```python editable=true slideshow={"slide_type": ""} tags=["skip-execution", "nbval-skip"] vscode={"languageId": "plaintext"}
from konfuzio_sdk.data import Project
from konfuzio_sdk.evaluate import ExtractionEvaluation

project = Project(id_=YOUR_PROJECT_ID, strict_data_validation=False)

label = project.get_label_by_name(YOUR_LABEL_NAME)

# get outliers by regex
outliers = label.get_probable_outliers_by_regex(project.categories, top_worst_percentage=TOP_WORST)

# get outliers by confidence
evaluation = ExtractionEvaluation(documents=list(zip(GROUND_TRUTH_DOCUMENTS, PREDICTED_DOCUMENTS)), strict=False)
outliers = label.get_probable_outliers_by_confidence(evaluation, confidence=CONFIDENCE)

# get outliers by normalization
outliers = label.get_probable_outliers_by_normalization(project.categories)
```

### What's next?

- [Learn how to create regex-based Annotations](https://dev.konfuzio.com/sdk/tutorials/regex_based_annotations/index.html)
- [Get to know how to create a custom Extraction AI](https://dev.konfuzio.com/sdk/tutorials/information_extraction/index.html#train-a-custom-date-extraction-ai)

