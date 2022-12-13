### FileSplittingEvaluation class

FileSplittingEvaluation class can be used to evaluate performance of ContextAwareFileSplittingModel, returning a set of 
metrics that includes precision, recall, f1 measure, true positives, false positives and false negatives. 

The class's methods `calculate()` and `calculate_by_category()` are ran at initialization. The class receives pairs of 
Documents as an input – first Document is ground-truth where all first Pages are marked as such, second is Document on 
Pages of which FileSplittingModel ran a prediction of them being first or non-first. 

Suppose `gt_doc_1`, `gt_doc_2` and the following are ground truth Documents, and `pred_doc_1`, `pred_doc2` and the 
following are the versions of these Documents that underwent the prediction step. Then the initialization would look 
like this:
```python
evaluation = FileSplittingEvaluation([(gt_doc_1, pred_doc_1), (gt_doc_2, pred_doc_2), ...])
```

Pages of each pair are compared; if a Page is first and predicted as such, it is a true positive; if it is first and 
predicted as non-first, it is a false negative; if it is non-first and predicted as first, it is a false positive; if 
it is non-first and predicted as such, it is true negative. 

|  | predicted correctly | predicted incorrectly |
| ------ | ------ | ------ |
|    first Page    |    TP    | FN |
|    non-first Page    |   TN     | FP |

After iterating through all Pages of all Documents, precision, recall and f1 measure are calculated. If you wish to set 
metrics to `None` in case there has been an attempt of zero division, set `allow_zero=True` at the initialization.


To see a certain metric after the class has been initialized, you can call a metric's method. 
```
print(evaluation.fn())
```

It is also possible to look at the metrics calculated by each Category independently. For this, pass `search=YOUR_CATEGORY_HERE` when calling the wanted metric's method: 
```
print(evaluation.fn(search=YOUR_CATEGORY_HERE))
``` 

### Example of evaluation input and output 

Suppose in our test dataset we have 2 Documents of 2 Categories: one 3-paged, consisting of a single file (-> it has only one ground-truth first Page) of a first Category, and one 5-paged, consisting of three files: two 2-paged and one 1-paged (-> it has three ground-truth first Pages), of a second Category.

We need to pass a list of tuples of `(ground_truth_doc, predicted_doc)` into the `FileSplittingEvaluation` class, so we have to run both Documents' Pages through prediction before that and create a list for the input – `ground_truth_docs` would be original Documents and `predicted_docs` would be their copies with Pages predicted to be first or non-first.

Suppose we ran the evaluation and received high results: only one first Page was predicted as non-first and the rest were predicted first/non-first correctly. Our results could be reflected in a following table:

| TPs | FPs | FNs | precision | recall | F1    |
| ---- | ---- | ----- | ---- | ---- |-------|
| 3 | 0 | 1 | 1 | 0.75 | 0.85  |

If we want to see evaluation results by Category, the output could be reflected in a following table:

| Category | TPs | FPs | FNs | precision | recall | F1    |
| ---- | ---- | ---- | ---- | ---- | ---- |-------|
| Category 1 | 2 | 0 | 1 | 1 | 0.66 | 0.79  |
| Category 2 | 1 | 0 | 0 | 1 | 1 | 1     |