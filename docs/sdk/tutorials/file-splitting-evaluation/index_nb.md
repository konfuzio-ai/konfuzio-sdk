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

## Evaluate a Splitting AI

---

**Prerequisites:** 

- Data Layer concepts of Konfuzio: Document, Project, Category, Page
- AI Layer concepts of Konfuzio: File Splitting
- Understanding of metrics for evaluating an AI's performance

**Difficulty:** Medium

**Goal:** Introduce the `FileSplittingEvaluation` class and explain how to measure Splitting AIs' performances using it.

---

### Environment
You need to install the Konfuzio SDK before diving into the tutorial. \
To get up and running quickly, you can use our Colab Quick Start notebook. \
<a href="https://colab.research.google.com/github/konfuzio-ai/konfuzio-sdk/blob/master/notebooks/Quick_start_template_for_Konfuzio_SDK.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

As an alternative you can follow the [installation section](../get_started.html#install-sdk) to install and initialize the Konfuzio SDK locally or on an environment of your choice.

### Introduction

`FileSplittingEvaluation` class can be used to evaluate performance of Splitting AIs, returning a 
set of metrics that includes precision, recall, F1 measure, True Positives, False Positives, True Negatives, and False 
Negatives. 

The class's methods `calculate()` and `calculate_by_category()` are run at initialization. The class receives two lists 
of Documents as an input – first list consists of ground-truth Documents where all first Pages are marked as such, 
second is of Documents on Pages of which File Splitting Model ran a prediction of them being first or non-first. 

### FileSplittingEvaluation class

Let's initialize the class:

```python editable=true slideshow={"slide_type": ""} tags=["skip-execution", "nbval-skip"] vscode={"languageId": "plaintext"}
evaluation = FileSplittingEvaluation(
        ground_truth_documents=YOUR_GROUND_TRUTH_LIST, prediction_documents=YOUR_PREDICTION_LIST
    )
```

The class compares each pair of Pages. If a Page is labeled as first and the model also predicted it as first, it is 
considered a True Positive. If a Page is labeled as first but the model predicted it as non-first, it is considered a 
False Negative. If a Page is labeled as non-first but the model predicted it as first, it is considered a False 
Positive. If a Page is labeled as non-first and the model also predicted it as non-first, it is considered a True 
Negative.

|  | predicted correctly | predicted incorrectly |
| ------ | ------ | ------ |
|    first Page    |    TP    | FN |
|    non-first Page    |   TN     | FP |

### Example of evaluation input and output 

Suppose in our test dataset we have 2 Documents of 2 Categories: one 3-paged, consisting of a single file (-> it has 
only one ground-truth first Page) of a first Category, and one 5-paged, consisting of three files: two 2-paged and one 
1-paged (-> it has three ground-truth first Pages), of a second Category.

![Document 1](document_example_1.png)

_First document_

![Document 2](document_example_2.png)

_Second document_

Let's create these mock Documents. This example builds the Documents from scratch and without uploading a supported file;
if you uploaded your Documents to the Konfuzio Server, you can retrieve and use them.
```python tags=["remove-cell"]
from konfuzio_sdk.samples import LocalTextProject
YOUR_PROJECT = LocalTextProject()
YOUR_CATEGORY_1 = YOUR_PROJECT.get_category_by_id(3)
YOUR_CATEGORY_2 = YOUR_PROJECT.get_category_by_id(4)
```

```python editable=true tags=["remove-output"] slideshow={"slide_type": ""}
from konfuzio_sdk.data import Document, Page
from konfuzio_sdk.evaluate import FileSplittingEvaluation, EvaluationCalculator
from konfuzio_sdk.trainer.file_splitting import SplittingAI

text_1 = "Hi all,\nI like bread.\nI hope to get everything done soon.\nHave you seen it?"
document_1 = Document(id_=20, project=YOUR_PROJECT, category=YOUR_CATEGORY_1, text=text_1, dataset_status=3)
_ = Page(
    id_=None, original_size=(320, 240), document=document_1, start_offset=0, end_offset=21, number=1, copy_of_id=29
)

_ = Page(
    id_=None, original_size=(320, 240), document=document_1, start_offset=22, end_offset=57, number=2, copy_of_id=30
)

_ = Page(
    id_=None, original_size=(320, 240), document=document_1, start_offset=58, end_offset=75, number=3, copy_of_id=31
)

text_2 = "Evening,\nthank you for coming.\nI like fish.\nI need it.\nEvening."
document_2 = Document(id_=21, project=YOUR_PROJECT, category=YOUR_CATEGORY_2, text=text_2, dataset_status=3)
_ = Page(
    id_=None, original_size=(320, 240), document=document_2, start_offset=0, end_offset=8, number=1, copy_of_id=32
)
_ = Page(
    id_=None, original_size=(320, 240), document=document_2, start_offset=9, end_offset=30, number=2, copy_of_id=33
)
_ = Page(
    id_=None, original_size=(320, 240), document=document_2, start_offset=31, end_offset=43, number=3, copy_of_id=34
)
_.is_first_page = True
_ = Page(
    id_=None, original_size=(320, 240), document=document_2, start_offset=44, end_offset=54, number=4, copy_of_id=35
)
_ = Page(
    id_=None, original_size=(320, 240), document=document_2, start_offset=55, end_offset=63, number=5, copy_of_id=36
)
_.is_first_page = True
```

### Running Evaluation on predicted Documents' Pages

We need to pass two lists of Documents into the `FileSplittingEvaluation` class. So, before that, we need to run each 
Page of the Documents through the model's prediction.

Let's say the evaluation gave good results, with only one first Page being predicted as non-first and all the other 
Pages being predicted correctly. An example of how the evaluation would be implemented would be:

```python tags=["remove-cell"]
import logging
from konfuzio_sdk.trainer.file_splitting import ContextAwareFileSplittingModel
from konfuzio_sdk.tokenizer.regex import ConnectedTextTokenizer
YOUR_MODEL = ContextAwareFileSplittingModel(categories=[YOUR_CATEGORY_1, YOUR_CATEGORY_2], tokenizer=ConnectedTextTokenizer())
YOUR_MODEL.fit(allow_empty_categories=True)
logging.getLogger("konfuzio_sdk").setLevel(logging.CRITICAL)
```

```python editable=true slideshow={"slide_type": ""}
splitting_ai = SplittingAI(YOUR_MODEL)
pred_1: Document = splitting_ai.propose_split_documents(document_1, return_pages=True)[0]
pred_2: Document = splitting_ai.propose_split_documents(document_2, return_pages=True)[0]

evaluation = FileSplittingEvaluation(
    ground_truth_documents=[document_1, document_2], prediction_documents=[pred_1, pred_2]
)
print('True positives: ' + str(evaluation.tp()))
print('True negatives: ' + str(evaluation.tn()))
print('False positives: ' + str(evaluation.fp()))
print('False negatives: ' + str(evaluation.fn()))
print('Precision: ' + str(evaluation.precision()))
print('Recall: ' + str(evaluation.recall()))
print('F1 score: ' + str(evaluation.f1()))
```

Our results could be reflected in a following table:

| TPs | TNs | FPs | FNs | precision | recall | F1    |
| ---- |-----|-----| ----- | ---- | ---- |-------|
| 3 | 4   |  0  | 1 | 1 | 0.75 | 0.85  |

If we want to see evaluation results by Category, the implementation of the Evaluation would look like this:
```python editable=true slideshow={"slide_type": ""}
print('True positives for Category 1: ' + str(evaluation.tp(search=YOUR_CATEGORY_1)) + ', for Category 2: ' + str(evaluation.tp(search=YOUR_CATEGORY_2)))
print('True negatives for Category 1: ' + str(evaluation.tn(search=YOUR_CATEGORY_1)) + ', for Category 2: ' +  str(evaluation.tn(search=YOUR_CATEGORY_2)))
print('False positives for Category 1: ' + str(evaluation.fp(search=YOUR_CATEGORY_1))+ ', for Category 2: ' + str(evaluation.fp(search=YOUR_CATEGORY_2)))
print('False negatives for Category 1: ' + str(evaluation.fn(search=YOUR_CATEGORY_1)) + ', for Category 2: ' + str(evaluation.fn(search=YOUR_CATEGORY_2)))
print('Precision for Category 1: ' + str(evaluation.precision(search=YOUR_CATEGORY_1)) + ', for Category 2: ' + str(evaluation.precision(search=YOUR_CATEGORY_2)))
print('Recall for Category 1: ' + str(evaluation.recall(search=YOUR_CATEGORY_1)) + ', for Category 2: ' + str(evaluation.recall(search=YOUR_CATEGORY_2)))
print('F1 score for Category 1: ' + str(evaluation.f1(search=YOUR_CATEGORY_1)) + ', for Category 2: ' + str(evaluation.f1(search=YOUR_CATEGORY_2)))
```

The output could be reflected in a following table:

| Category | TPs | TNs | FPs | FNs | precision | recall | F1  |
| ---- |-----|-----|-----|-----| ---- |--------|-----|
| Category 1 | 1   | 2   | 0   | 0   | 1 | 1      | 1   |
| Category 2 | 2   | 2   | 0   | 1   | 1 | 0.66   | 0.8 |

To log metrics after evaluation, you can call `EvaluationCalculator`'s method `metrics_logging` (you would need to 
specify the metrics accordingly at the class's initialization). Example usage:

```python tags=["remove-cell"]
logging.getLogger("konfuzio_sdk").setLevel(logging.INFO)
```

```python editable=true slideshow={"slide_type": ""}
EvaluationCalculator(tp=3, fp=0, fn=1, tn=4).metrics_logging()
```

### Conclusion
In this tutorial, we have walked through the essential steps for evaluating the performance of File Splitting AI using FileSplittingEvaluation class. Below is the full code to accomplish this task:

```python editable=true slideshow={"slide_type": ""} tags=["skip-execution", "nbval-skip"] vscode={"languageId": "plaintext"}
from konfuzio_sdk.data import Document, Page
from konfuzio_sdk.evaluate import FileSplittingEvaluation, EvaluationCalculator
from konfuzio_sdk.trainer.file_splitting import SplittingAI

text_1 = "Hi all,\nI like bread.\nI hope to get everything done soon.\nHave you seen it?"
document_1 = Document(id_=20, project=YOUR_PROJECT, category=YOUR_CATEGORY_1, text=text_1, dataset_status=3)
_ = Page(
    id_=None, original_size=(320, 240), document=document_1, start_offset=0, end_offset=21, number=1, copy_of_id=29
)

_ = Page(
    id_=None, original_size=(320, 240), document=document_1, start_offset=22, end_offset=57, number=2, copy_of_id=30
)

_ = Page(
    id_=None, original_size=(320, 240), document=document_1, start_offset=58, end_offset=75, number=3, copy_of_id=31
)

text_2 = "Evening,\nthank you for coming.\nI like fish.\nI need it.\nEvening."
document_2 = Document(id_=21, project=YOUR_PROJECT, category=YOUR_CATEGORY_2, text=text_2, dataset_status=3)
_ = Page(
    id_=None, original_size=(320, 240), document=document_2, start_offset=0, end_offset=8, number=1, copy_of_id=32
)
_ = Page(
    id_=None, original_size=(320, 240), document=document_2, start_offset=9, end_offset=30, number=2, copy_of_id=33
)
_ = Page(
    id_=None, original_size=(320, 240), document=document_2, start_offset=31, end_offset=43, number=3, copy_of_id=34
)
_.is_first_page = True
_ = Page(
    id_=None, original_size=(320, 240), document=document_2, start_offset=44, end_offset=54, number=4, copy_of_id=35
)
_ = Page(
    id_=None, original_size=(320, 240), document=document_2, start_offset=55, end_offset=63, number=5, copy_of_id=36
)
_.is_first_page = True

splitting_ai = SplittingAI(YOUR_MODEL)
pred_1: Document = splitting_ai.propose_split_documents(document_1, return_pages=True)[0]
pred_2: Document = splitting_ai.propose_split_documents(document_2, return_pages=True)[0]

YOUR_GROUND_TRUTH_LIST = [document_1, document_2]
YOUR_PREDICTION_LIST = [pred_1, pred_2]

evaluation = FileSplittingEvaluation(
        ground_truth_documents=YOUR_GROUND_TRUTH_LIST, prediction_documents=YOUR_PREDICTION_LIST
    )

print('True positives: ' + str(evaluation.tp()))
print('True negatives: ' + str(evaluation.tn()))
print('False positives: ' + str(evaluation.fp()))
print('False negatives: ' + str(evaluation.fn()))
print('Precision: ' + str(evaluation.precision()))
print('Recall: ' + str(evaluation.recall()))
print('F1 score: ' + str(evaluation.

print('True positives for Category 1: ' + str(evaluation.tp(search=YOUR_CATEGORY_1)) + ', for Category 2: ' + str(evaluation.tp(search=YOUR_CATEGORY_2)))
print('True negatives for Category 1: ' + str(evaluation.tn(search=YOUR_CATEGORY_1)) + ', for Category 2: ' +  str(evaluation.tn(search=YOUR_CATEGORY_2)))
print('False positives for Category 1: ' + str(evaluation.fp(search=YOUR_CATEGORY_1))+ ', for Category 2: ' + str(evaluation.fp(search=YOUR_CATEGORY_2)))
print('False negatives for Category 1: ' + str(evaluation.fn(search=YOUR_CATEGORY_1)) + ', for Category 2: ' + str(evaluation.fn(search=YOUR_CATEGORY_2)))
print('Precision for Category 1: ' + str(evaluation.precision(search=YOUR_CATEGORY_1)) + ', for Category 2: ' + str(evaluation.precision(search=YOUR_CATEGORY_2)))
print('Recall for Category 1: ' + str(evaluation.recall(search=YOUR_CATEGORY_1)) + ', for Category 2: ' + str(evaluation.recall(search=YOUR_CATEGORY_2)))
print('F1 score for Category 1: ' + str(evaluation.f1(search=YOUR_CATEGORY_1)) + ', for Category 2: ' + str(evaluation.f1(search=YOUR_CATEGORY_2)))

EvaluationCalculator(tp=3, fp=0, fn=1, tn=4).metrics_logging()
```

### What's next?

- [Upload an evaluated AI to Konfuzio app or an on-prem installation](https://dev.konfuzio.com/sdk/tutorials/upload-your-ai/index.html)
