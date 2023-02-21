## FileSplittingEvaluation class

`FileSplittingEvaluation` class can be used to evaluate performance of Context-Aware File Splitting Model, returning a 
set of metrics that includes precision, recall, f1 measure, True Positives, False Positives, True Negatives, and False 
Negatives. 

The class's methods `calculate()` and `calculate_by_category()` are run at initialization. The class receives two lists 
of Documents as an input – first list consists of ground-truth Documents where all first Pages are marked as such, 
second is of Documents on Pages of which File Splitting Model ran a prediction of them being first or non-first. 

The initialization would look like this:
```python
evaluation = FileSplittingEvaluation(ground_truth_documents=YOUR_GROUND_TRUTH_LIST, 
                                     prediction_documents=YOUR_PREDICTION_LIST)
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

After iterating through all Pages of all Documents, precision, recall and f1 measure are calculated. If you wish to set 
metrics to `None` in case there has been an attempt of zero division, set `allow_zero=True` at the initialization.

To see a certain metric after the class has been initialized, you can call a metric's method:
```
print(evaluation.fn())
```

It is also possible to look at the metrics calculated by each Category independently. For this, pass 
`search=YOUR_CATEGORY_HERE` when calling the wanted metric's method: 
```
print(evaluation.fn(search=YOUR_CATEGORY_HERE))
``` 

For more details, see the [Python API Documentation](https://dev.konfuzio.com/sdk/sourcecode.html#evaluation) on 
Evaluation.

### Example of evaluation input and output 

Suppose in our test dataset we have 2 Documents of 2 Categories: one 3-paged, consisting of a single file (-> it has 
only one ground-truth first Page) of a first Category, and one 5-paged, consisting of three files: two 2-paged and one 
1-paged (-> it has three ground-truth first Pages), of a second Category.

.. image:: /sdk/examples/evaluation/document_example_1.png

_First document_

.. image:: /sdk/examples/evaluation/document_example_2.png

_Second document_

```python
# generate the test Documents

from konfuzio_sdk.data import Category, Project, Document, Page
from konfuzio_sdk.evaluate import FileSplittingEvaluation
from konfuzio_sdk.trainer.file_splitting import SplittingAI

# This example builds the Documents from scratch and without uploading a Supported File.
# If you uploaded your Document to the Konfuzio Server, you can just retrieve it with:
# document_1 = project.get_document_by_id(YOUR_DOCUMENT_ID)
text_1 = "Hi all,\nI like bread.\nI hope to get everything done soon.\nHave you seen it?"
document_1 = Document(id_=None, project=YOUR_PROJECT, category=YOUR_CATEGORY_1, text=text_1, dataset_status=3)
_ = Page(
        id_=None,
        original_size=(320, 240),
        document=document_1,
        start_offset=0,
        end_offset=21,
        number=1,
    )
_ = Page(
    id_=None,
    original_size=(320, 240),
    document=document_1,
    start_offset=22,
    end_offset=57,
    number=2,
)

_ = Page(
    id_=None,
    original_size=(320, 240),
    document=document_1,
    start_offset=58,
    end_offset=75,
    number=3,
)

# As with the previous example Document, you can just retrieve an online Document with
# document_2 = project.get_document_by_id(YOUR_DOCUMENT_ID)
text_2 = "Good evening,\nthank you for coming.\nCan you give me that?\nI need it.\nSend it to me."
document_2 = Document(id_=None, project=YOUR_PROJECT, category=YOUR_CATEGORY_2, text=text_2, dataset_status=3)
_ = Page(
    id_=None,
    original_size=(320, 240),
    document=document_2,
    start_offset=0,
    end_offset=12,
    number=1
)
_ = Page(
    id_=None,
    original_size=(320, 240),
    document=document_2,
    start_offset=13,
    end_offset=34,
    number=2
)
_ = Page(
    id_=None,
    original_size=(320, 240),
    document=document_2,
    start_offset=35,
    end_offset=56,
    number=3
)
_.is_first_page = True
_ = Page(
    id_=None,
    original_size=(320, 240),
    document=document_2,
    start_offset=57,
    end_offset=67,
    number=4
)
_ = Page(
    id_=None,
    original_size=(320, 240),
    document=document_2,
    start_offset=68,
    end_offset=82,
    number=5
)
_.is_first_page = True
```

We need to pass two lists of Documents into the `FileSplittingEvaluation` class. So, before that, we need to run each 
Page of the Documents through the model's prediction.

Let's say the evaluation gave good results, with only one first Page being predicted as non-first and all the other 
Pages being predicted correctly. An example of how the evaluation would be implemented would be:
```python
splitting_ai = SplittingAI(YOUR_MODEL_HERE)
pred_1: Document = splitting_ai.propose_split_documents(document_1, return_pages=True)[0] 
pred_2: Document = splitting_ai.propose_split_documents(document_2, return_pages=True)[0]
evaluation = FileSplittingEvaluation(ground_truth_documents=[document_1, document_2], 
                                     prediction_documents=[pred_1, pred_2])
print(evaluation.tp()) # returns: 3
print(evaluation.tn()) # returns: 4
print(evaluation.fp()) # returns: 0
print(evaluation.fn()) # returns: 1
print(evaluation.precision()) # returns: 1
print(evaluation.recall()) # returns: 0.75
print(evaluation.f1()) # returns: 0.85
```

Our results could be reflected in a following table:

| TPs | TNs | FPs | FNs | precision | recall | F1    |
| ---- |-----|-----| ----- | ---- | ---- |-------|
| 3 | 4   |  0  | 1 | 1 | 0.75 | 0.85  |

If we want to see evaluation results by Category, the implementation of the Evaluation would look like this:
```python
print(evaluation.tp(search=CATEGORY_1), evaluation.tp(search=CATEGORY_2)) # returns: 1 2
print(evaluation.tn(search=CATEGORY_1), evaluation.tn(search=CATEGORY_2)) # returns: 2 2 
print(evaluation.fp(search=CATEGORY_1), evaluation.fp(search=CATEGORY_2)) # returns: 0 0
print(evaluation.fn(search=CATEGORY_1), evaluation.fn(search=CATEGORY_2)) # returns: 0 1
print(evaluation.precision(search=CATEGORY_1), evaluation.precision(search=CATEGORY_2)) # returns: 1 1
print(evaluation.recall(search=CATEGORY_1), evaluation.recall(search=CATEGORY_2)) # returns: 1 0.66
print(evaluation.f1(search=CATEGORY_1), evaluation.f1(search=CATEGORY_2)) # returns: 1 0.79
```

the output could be reflected in a following table:

| Category | TPs | TNs | FPs | FNs | precision | recall | F1   |
| ---- |-----|-----|-----|-----| ---- |--------|------|
| Category 1 | 1   | 2   | 0   | 0   | 1 | 1      | 1    |
| Category 2 | 2   | 2   | 0   | 1   | 1 | 0.66   | 0.79 |

To log metrics after evaluation, you can call `EvaluationCalculator`'s method `metrics_logging` (you would need to 
specify the metrics accordingly at the class's initialization). Example usage:
```python
EvaluationCalculator(tp=3, fp=0, fn=1, tn=4).metrics_logging()
```