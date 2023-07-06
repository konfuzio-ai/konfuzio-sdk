### Evaluate a File Splitting AI

`FileSplittingEvaluation` class can be used to evaluate performance of Context-Aware File Splitting Model, returning a 
set of metrics that includes precision, recall, f1 measure, True Positives, False Positives, True Negatives, and False 
Negatives. 

The class's methods `calculate()` and `calculate_by_category()` are run at initialization. The class receives two lists 
of Documents as an input – first list consists of ground-truth Documents where all first Pages are marked as such, 
second is of Documents on Pages of which File Splitting Model ran a prediction of them being first or non-first. 

The initialization would look like this:

.. literalinclude:: /sdk/boilerplates/test_file_splitting_evaluation.py
   :language: python
   :start-after: start eval_example
   :end-before: end eval_example
   :dedent: 4

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

.. literalinclude:: /sdk/boilerplates/test_file_splitting_evaluation.py
   :language: python
   :start-after: start single_metric
   :end-before: end single_metric
   :dedent: 4

It is also possible to look at the metrics calculated by each Category independently. For this, pass 
`search=YOUR_CATEGORY_HERE` when calling the wanted metric's method: 

.. literalinclude:: /sdk/boilerplates/test_file_splitting_evaluation.py
   :language: python
   :start-after: start metric_category
   :end-before: end metric_category
   :dedent: 4

For more details, see the [Python API Documentation](https://dev.konfuzio.com/sdk/sourcecode.html#ai-evaluation) on 
Evaluation.

#### Example of evaluation input and output 

Suppose in our test dataset we have 2 Documents of 2 Categories: one 3-paged, consisting of a single file (-> it has 
only one ground-truth first Page) of a first Category, and one 5-paged, consisting of three files: two 2-paged and one 
1-paged (-> it has three ground-truth first Pages), of a second Category.

.. image:: /sdk/examples/evaluation/document_example_1.png

_First document_

.. image:: /sdk/examples/evaluation/document_example_2.png

_Second document_

.. literalinclude:: /sdk/boilerplates/test_file_splitting_evaluation.py
   :language: python
   :start-after: start document creation
   :end-before: end document creation
   :dedent: 4

We need to pass two lists of Documents into the `FileSplittingEvaluation` class. So, before that, we need to run each 
Page of the Documents through the model's prediction.

Let's say the evaluation gave good results, with only one first Page being predicted as non-first and all the other 
Pages being predicted correctly. An example of how the evaluation would be implemented would be:

.. literalinclude:: /sdk/boilerplates/test_file_splitting_evaluation.py
   :language: python
   :start-after: start splitting
   :end-before: end splitting
   :dedent: 4
.. literalinclude:: /sdk/boilerplates/test_file_splitting_evaluation.py
   :language: python
   :start-after: start scores
   :end-before: end scores
   :dedent: 4

Our results could be reflected in a following table:

| TPs | TNs | FPs | FNs | precision | recall | F1    |
| ---- |-----|-----| ----- | ---- | ---- |-------|
| 3 | 4   |  0  | 1 | 1 | 0.75 | 0.85  |

If we want to see evaluation results by Category, the implementation of the Evaluation would look like this:

.. literalinclude:: /sdk/boilerplates/test_file_splitting_evaluation.py
   :language: python
   :start-after: start scores_category
   :end-before: end scores_category
   :dedent: 4

the output could be reflected in a following table:

| Category | TPs | TNs | FPs | FNs | precision | recall | F1  |
| ---- |-----|-----|-----|-----| ---- |--------|-----|
| Category 1 | 1   | 2   | 0   | 0   | 1 | 1      | 1   |
| Category 2 | 2   | 2   | 0   | 1   | 1 | 0.66   | 0.8 |

To log metrics after evaluation, you can call `EvaluationCalculator`'s method `metrics_logging` (you would need to 
specify the metrics accordingly at the class's initialization). Example usage:

.. literalinclude:: /sdk/boilerplates/test_file_splitting_evaluation.py
   :language: python
   :start-after: start calculator
   :end-before: end calculator
   :dedent: 4