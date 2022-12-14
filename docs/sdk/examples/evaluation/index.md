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

![first_document_example](image_1.png)  

_First document_

![second_document_example](image_2.png)

_Second document_

We need to pass a list of tuples of `(ground_truth_doc, predicted_doc)` into the `FileSplittingEvaluation` class, so we have to run both Documents' Pages through prediction before that and create a list for the input – `ground_truth_docs` would be original Documents and `predicted_docs` would be their copies with Pages predicted to be first or non-first.

Suppose we ran the evaluation and received high results: only one first Page was predicted as non-first and the rest 
were predicted first/non-first correctly. The calculation would be done by the following method of the 
`FileSplittingEvaluation` class:
```python
    def calculate(self):
        """Calculate metrics for the filesplitting logic."""
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for ground_truth, prediction in self.documents:
            for page_gt, page_pr in zip(ground_truth.pages(), prediction.pages()):
                if page_gt.is_first_page and page_pr.is_first_page:
                    tp += 1
                elif not page_gt.is_first_page and page_pr.is_first_page:
                    fp += 1
                elif page_gt.is_first_page and not page_pr.is_first_page:
                    fn += 1
                elif not page_gt.is_first_page and not page_pr.is_first_page:
                    tn += 1
        if tp + fp != 0:
            precision = tp / (tp + fp)
        else:
            if self.allow_zero:
                precision = None
            else:
                raise ZeroDivisionError(
                    "TP and FP are zero, please specify allow_zero=True if you want precision to be None."
                )
        if tp + fn != 0:
            recall = tp / (tp + fn)
        else:
            if self.allow_zero:
                recall = None
            else:
                raise ZeroDivisionError(
                    "TP and FN are zero, please specify allow_zero=True if you want recall to be None."
                )
        if precision + recall != 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            if self.allow_zero:
                f1 = None
            else:
                raise ZeroDivisionError("FP and FN are zero, please specify allow_zero=True if you want F1 to be None.")
        self.project = self.documents[0][0].project
        self.evaluation_results = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
```
`self.allow_zero` is an attribute that determines whether we return `None` in case there's a zero division attempt or 
throw an error.

Our results could be reflected in a following table:

| TPs | FPs | FNs | precision | recall | F1    |
| ---- | ---- | ----- | ---- | ---- |-------|
| 3 | 0 | 1 | 1 | 0.75 | 0.85  |

If we want to see evaluation results by Category, the calculation would be done by the following method of the 
`FileSplittingEvaluation` class:
```python
    def calculate_metrics_by_category(self):
        """Calculate metrics by Category independently."""
        categories = list(set([doc_pair[0].category for doc_pair in self.documents]))
        self.evaluation_results_by_category = {
            'tp': {},
            'fp': {},
            'fn': {},
            'tn': {},
            'precision': {},
            'recall': {},
            'f1': {},
        }
        for category in categories:
            tp = 0
            fp = 0
            fn = 0
            tn = 0
            for ground_truth, prediction in [
                [document_1, document_2]
                for document_1, document_2 in self.documents
                if document_1.category and document_1.category.id_ == category.id_
            ]:
                for page_gt, page_pr in zip(ground_truth.pages(), prediction.pages()):
                    if page_gt.is_first_page and page_pr.is_first_page:
                        tp += 1
                    elif not page_gt.is_first_page and page_pr.is_first_page:
                        fp += 1
                    elif page_gt.is_first_page and not page_pr.is_first_page:
                        fn += 1
                    elif not page_gt.is_first_page and not page_pr.is_first_page:
                        tn += 1
            if tp + fp != 0:
                precision = tp / (tp + fp)
            else:
                if self.allow_zero:
                    precision = None
                else:
                    raise ZeroDivisionError(
                        "TP and FP are zero, please specify allow_zero=True if you want precision to be None."
                    )
            if tp + fn != 0:
                recall = tp / (tp + fn)
            else:
                if self.allow_zero:
                    recall = None
                else:
                    raise ZeroDivisionError(
                        "TP and FN are zero, please specify allow_zero=True if you want recall to be None."
                    )
            if precision + recall != 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                if self.allow_zero:
                    f1 = None
                else:
                    raise ZeroDivisionError(
                        "FP and FN are zero, please specify allow_zero=True if you want F1 to be None."
                    )
            self.evaluation_results_by_category['tp'][category.id_] = tp
            self.evaluation_results_by_category['fp'][category.id_] = fp
            self.evaluation_results_by_category['fn'][category.id_] = fn
            self.evaluation_results_by_category['tn'][category.id_] = tn
            self.evaluation_results_by_category['precision'][category.id_] = precision
            self.evaluation_results_by_category['recall'][category.id_] = recall
            self.evaluation_results_by_category['f1'][category.id_] = f1
```

the output could be reflected in a following table:

| Category | TPs | FPs | FNs | precision | recall | F1    |
| ---- | ---- | ---- | ---- | ---- | ---- |-------|
| Category 1 | 2 | 0 | 1 | 1 | 0.66 | 0.79  |
| Category 2 | 1 | 0 | 0 | 1 | 1 | 1     |