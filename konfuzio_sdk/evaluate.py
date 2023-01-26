"""Calculate the accuracy on any level in a  Document."""
import logging
from typing import Tuple, List, Optional, Union

import pandas
import numpy
from sklearn.utils.extmath import weighted_mode

from konfuzio_sdk.utils import sdk_isinstance, memory_size_of
from konfuzio_sdk.data import Document, Category

logger = logging.getLogger(__name__)


RELEVANT_FOR_EVALUATION = [
    "is_matched",  # needed to group spans in Annotations
    "id_local",  # needed to group spans in Annotations
    "id_",  # even we won't care of the id_, as the ID is defined by the start and end span
    # "confidence", we don't care about the confidence of doc_a
    "start_offset",  # only relevant for the merge but allows to track multiple sequences per annotation
    "end_offset",  # only relevant for the merge but allows to track multiple sequences per annotation
    "is_correct",  # we care if it is correct, humans create Annotations without confidence
    "label_id",
    "label_threshold",
    "above_predicted_threshold",
    "revised",  # we need it to filter feedback required Annotations
    "annotation_set_id",
    "label_set_id",
    "document_id",
    "document_id_local",
    "category_id",  # Identify the Category to be able to run an evaluation across categories
    # "id__predicted", we don't care of the id_ see "id_"
    "id_local_predicted",
    "confidence_predicted",  # we care about the confidence of the prediction
    "start_offset_predicted",
    "end_offset_predicted",
    # "is_correct_predicted", # it's a prediction so we don't know if it is correct
    "label_id_predicted",
    "label_threshold_predicted",  # we keep a flexibility to be able to predict the threshold
    # "revised_predicted",  # it's a prediction so we ignore if it is revised
    "annotation_set_id_predicted",
    "label_set_id_predicted",
    "document_id_predicted",
    "document_id_local_predicted",
    "is_correct_label",
    "is_correct_label_set",
    "is_correct_annotation_set_id",
    "is_correct_id_",
    "duplicated",
    "duplicated_predicted",
]

logger = logging.getLogger(__name__)


def grouped(group, target: str):
    """Define which of the correct element in the predicted group defines the "correct" group id_."""
    verbose_validation_column_name = f"defined_to_be_correct_{target}"
    # all rows where is_correct is nan relate to an element which has no correct element partner
    eligible_to_vote = group['above_predicted_threshold'].fillna(False) & group['is_matched'].fillna(False)
    if not eligible_to_vote.any():  # no Spans provide confidence above Threshold of Label
        if not group['confidence_predicted'].any() or not group[target].any():
            group[verbose_validation_column_name] = group[target].mode(dropna=False)[0]
        else:
            group[verbose_validation_column_name] = int(weighted_mode(group[target], group['confidence_predicted'])[0])
    elif not group.loc[eligible_to_vote][target].any():  # no Spans should be predicted
        group[verbose_validation_column_name] = None
    else:  # get the most frequent annotation_set_id from the high confidence Spans in this group
        if not group.loc[eligible_to_vote]['confidence_predicted'].any():  # CLF does not provide a confidence score
            # get the most frequent annotation_set_id from the *correct* Annotations in this group
            group[verbose_validation_column_name] = group.loc[eligible_to_vote][target].mode(dropna=False)[0]
        else:
            group[verbose_validation_column_name] = int(
                weighted_mode(group.loc[eligible_to_vote][target], group.loc[eligible_to_vote]['confidence_predicted'])[
                    0
                ]
            )
    validation_column_name = f"is_correct_{target}"
    group[validation_column_name] = group[target] == group[verbose_validation_column_name]
    return group


def compare(doc_a, doc_b, only_use_correct=False, strict=True) -> pandas.DataFrame:
    """Compare the Annotations of two potentially empty Documents wrt. to **all** Annotations.

    :param doc_a: Document which is assumed to be correct
    :param doc_b: Document which needs to be evaluated
    :param only_use_correct: Unrevised feedback in doc_a is assumed to be correct.
    :param strict: Evaluate on a Character exact level without any postprocessing, an amount Span "5,55 " will not be
     exact with "5,55"
    :raises ValueError: When the Category differs.
    :return: Evaluation DataFrame
    """
    df_a = pandas.DataFrame(doc_a.eval_dict(use_correct=only_use_correct))
    df_b = pandas.DataFrame(doc_b.eval_dict(use_correct=False))
    if doc_a.category != doc_b.category:
        raise ValueError(f'Categories of {doc_a} with {doc_a.category} and {doc_b} with {doc_a.category} do not match.')
    if strict:  # many to many inner join to keep all Spans of both Documents
        spans = pandas.merge(df_a, df_b, how="outer", on=["start_offset", "end_offset"], suffixes=('', '_predicted'))
        # add criteria to evaluate Spans
        spans["is_matched"] = spans['id_local'].notna()  # start and end offset are identical
        spans["start_offset_predicted"] = spans['start_offset']  # start and end offset are identical
        spans["end_offset_predicted"] = spans['end_offset']  # start and end offset are identical

        spans["above_predicted_threshold"] = spans["confidence_predicted"] >= spans["label_threshold_predicted"]

        spans["is_correct_label"] = spans["label_id"] == spans["label_id_predicted"]
        spans["is_correct_label_set"] = spans["label_set_id"] == spans["label_set_id_predicted"]
        spans['duplicated'] = False
        spans['duplicated_predicted'] = False

        # add check to evaluate multiline Annotations
        spans = spans.groupby("id_local", dropna=False).apply(lambda group: grouped(group, "id_"))
        # add check to evaluate Annotation Sets
        spans = spans.groupby("annotation_set_id_predicted", dropna=False).apply(
            lambda group: grouped(group, "annotation_set_id")
        )
    else:
        # allows  start_offset_predicted <= end_offset and end_offset_predicted >= start_offset
        spans = pandas.merge(df_a, df_b, how="outer", on=["label_id", "label_set_id"], suffixes=('', '_predicted'))
        # add criteria to evaluate Spans
        spans["is_matched"] = (spans["start_offset_predicted"] <= spans["end_offset"]) & (
            spans["end_offset_predicted"] >= spans["start_offset"]
        )
        spans["above_predicted_threshold"] = spans["confidence_predicted"] >= spans["label_threshold_predicted"]
        spans["is_correct_label"] = True
        spans["is_correct_label_set"] = True
        spans["label_id_predicted"] = spans["label_id"]
        spans["label_set_id_predicted"] = spans["label_set_id"]

        spans = spans.sort_values(by='is_matched', ascending=False)
        spans['duplicated'] = spans.duplicated(subset=['id_local'], keep='first')
        spans['duplicated_predicted'] = spans.duplicated(subset=['id_local_predicted'], keep='first')
        spans = spans.drop(spans[(spans['duplicated']) & (spans['duplicated_predicted'])].index)
        # add check to evaluate multiline Annotations
        spans = spans.groupby("id_local", dropna=False).apply(lambda group: grouped(group, "id_"))
        # add check to evaluate Annotation Sets
        spans = spans.groupby("annotation_set_id_predicted", dropna=False).apply(
            lambda group: grouped(group, "annotation_set_id")
        )
    spans = spans[RELEVANT_FOR_EVALUATION]

    assert not spans.empty  # this function must be able to evaluate any two docs even without annotations

    spans["tokenizer_true_positive"] = (
        (spans["is_correct"])
        & (spans["is_matched"])
        & (spans["start_offset_predicted"] == spans['start_offset'])
        & (spans["end_offset_predicted"] == spans['end_offset'])
        & (spans["document_id_local_predicted"].notna())
    )

    spans["tokenizer_false_negative"] = (
        (spans["is_correct"]) & (spans["is_matched"]) & (spans["document_id_local_predicted"].isna())
    )

    spans["tokenizer_false_positive"] = (
        (~spans["tokenizer_false_negative"])
        & (~spans["tokenizer_true_positive"])
        & (spans["document_id_local_predicted"].notna())
        & (spans["end_offset"] != 0)  # ignore placeholder
    )

    spans["clf_true_positive"] = (
        (spans["is_correct"])
        & (spans["is_matched"])
        & (spans["document_id_local_predicted"].notna())
        & (spans["above_predicted_threshold"])
        & (spans["is_correct_label"])
    )

    spans["clf_false_negative"] = (
        (spans["is_correct"])
        & (spans["is_matched"])
        & (spans["document_id_local_predicted"].notna())
        & (~spans["above_predicted_threshold"])
        & (spans["is_correct_label"])
    )

    spans["clf_false_positive"] = (
        (spans["is_correct"])
        & (spans["is_matched"])
        & (spans["document_id_local_predicted"].notna())
        & (~spans["is_correct_label"])
    )

    # Evaluate which **spans** are TN, TP, FP and keep RELEVANT_FOR_MAPPING to allow grouping of confidence measures
    spans["true_positive"] = (
        (spans["is_matched"])
        & (spans["is_correct"])
        & (spans["above_predicted_threshold"])
        & (~spans["duplicated"])
        & (  # Everything is correct
            (spans["is_correct_label"])
            & (spans["is_correct_label_set"])
            & (spans["is_correct_annotation_set_id"])
            & (spans["is_correct_id_"])
        )
    )

    spans["false_negative"] = (
        (spans["is_correct"])
        & (~spans["duplicated"])
        & ((~spans["is_matched"]) | (~spans["above_predicted_threshold"]) | (spans["label_id_predicted"].isna()))
    )

    spans["false_positive"] = (  # commented out on purpose (spans["is_correct"]) &
        (spans["above_predicted_threshold"])
        & (~spans["false_negative"])
        & (~spans["true_positive"])
        & (~spans["duplicated_predicted"])
        & (  # Something is wrong
            (~spans["is_correct_label"])
            | (~spans["is_correct_label_set"])
            | (~spans["is_correct_annotation_set_id"])
            | (~spans["is_correct_id_"])
            | (~spans["is_matched"])
        )
    )
    spans = spans.replace({numpy.nan: None})
    # one Span must not be defined as TP or FP or FN more than once
    quality = (spans[['true_positive', 'false_positive', 'false_negative']].sum(axis=1) <= 1).all()
    assert quality
    return spans


class EvaluationCalculator:
    """Calculate precision, recall, f1, based on TP, FP, FN."""

    def __init__(self, tp: int = 0, fp: int = 0, fn: int = 0, tn: int = 0, allow_zero: bool = True):
        """
        Store evaluation information.

        :param tp: True Positives.
        :param fp: False Positives.
        :param fn: False Negatives.
        :param tn: True Negatives.
        :param allow_zero: If true, will calculate None for precision and recall when the straightforward application
        of the formula would otherwise result in 0/0. Raises ZeroDivisionError otherwise.
        """
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.tn = tn
        self._valid(allow_zero)

    def _valid(self, allow_zero: bool = True) -> None:
        """Check for 0/0 on precision, recall, and F1 calculation."""
        if allow_zero:
            return
        if self.fp + self.fn == 0:
            raise ZeroDivisionError("FP and FN are zero, please specify allow_zero=True if you want F1 to be None.")
        if self.tp + self.fp == 0:
            raise ZeroDivisionError(
                "TP and FP are zero, please specify allow_zero=True if you want precision to be None."
            )
        if self.tp + self.fn == 0:
            raise ZeroDivisionError("TP and FN are zero, please specify allow_zero=True if you want recall to be None.")

    @property
    def precision(self) -> Optional[float]:
        """Apply precision formula. Returns None if TP+FP=0."""
        return None if (self.tp + self.fp == 0) else self.tp / (self.tp + self.fp)

    @property
    def recall(self) -> Optional[float]:
        """Apply recall formula. Returns None if TP+FN=0."""
        return None if (self.tp + self.fn == 0) else self.tp / (self.tp + self.fn)

    @property
    def f1(self) -> Optional[float]:
        """Apply F1-score formula. Returns None if precision and recall are both None."""
        return None if (self.tp + 0.5 * (self.fp + self.fn) == 0) else self.tp / (self.tp + 0.5 * (self.fp + self.fn))

    def metrics_logging(self):
        """Log metrics."""
        logger.info(f"true positives: {self.tp}")
        logger.info(f"false negatives: {self.fn}")
        logger.info(f"true negatives: {self.tn}")
        logger.info(f"false positives: {self.fp}")
        logger.info(f"precision: {self.precision}")
        logger.info(f"recall: {self.recall}")
        logger.info(f"F1: {self.f1}")


class Evaluation:
    """Calculated accuracy measures by using the detailed comparison on Span Level."""

    def __init__(self, documents: List[Tuple[Document, Document]], strict: bool = True):
        """
        Relate to the two document instances.

        :param documents: A list of tuple Documents that should be compared.
        :param strict: A boolean passed to the `compare` function.
        """
        logger.info(f"Initializing Evaluation object with {len(documents)} documents. Evaluation mode {strict=}.")
        self.documents = documents
        self.strict = strict
        self.only_use_correct = True
        self.data = None
        self.calculate()
        logger.info(f"Size of evaluation DataFrame: {memory_size_of(self.data)/1000} KB.")

    def calculate(self):
        """Calculate and update the data stored within this Evaluation."""
        evaluations = []  # start anew, the configuration of the Evaluation might have changed.
        for ground_truth, predicted in self.documents:
            evaluation = compare(
                doc_a=ground_truth, doc_b=predicted, only_use_correct=self.only_use_correct, strict=self.strict
            )
            evaluations.append(evaluation)

        self.data = pandas.concat(evaluations)

    def _query(self, search=None):
        """Query the comparison data.

        :param search: use a search query in pandas
        """
        from konfuzio_sdk.data import Label, Document, LabelSet

        if search is None:
            return self.data
        elif sdk_isinstance(search, Label):
            assert search.id_ is not None, f'{search} must have a ID'
            query = f'label_id == {search.id_} | (label_id_predicted == {search.id_})'
        elif sdk_isinstance(search, Document):
            assert search.id_ is not None, f'{search} must have a ID.'
            query = f'document_id == {search.id_} | (document_id_predicted == {search.id_})'
        elif sdk_isinstance(search, LabelSet):
            assert search.id_ is not None, f'{search} must have a ID.'
            query = f'label_set_id == {search.id_} | (label_set_id_predicted == {search.id_})'
        else:
            raise NotImplementedError
        return self.data.query(query)

    def tp(self, search=None) -> int:
        """Return the True Positives of all Spans."""
        return self._query(search=search)["true_positive"].sum()

    def fp(self, search=None) -> int:
        """Return the False Positives of all Spans."""
        return self._query(search=search)["false_positive"].sum()

    def fn(self, search=None) -> int:
        """Return the False Negatives of all Spans."""
        return self._query(search=search)["false_negative"].sum()

    def tn(self, search=None) -> int:
        """Return the True Negatives of all Spans."""
        return (
            len(self._query(search=search)) - self.tp(search=search) - self.fn(search=search) - self.fp(search=search)
        )

    def tokenizer_tp(self, search=None) -> int:
        """Return the tokenizer True Positives of all Spans."""
        return self._query(search=search)["tokenizer_true_positive"].sum()

    def tokenizer_fp(self, search=None) -> int:
        """Return the tokenizer False Positives of all Spans."""
        return self._query(search=search)["tokenizer_false_positive"].sum()

    def tokenizer_fn(self, search=None) -> int:
        """Return the tokenizer False Negatives of all Spans."""
        return self._query(search=search)["tokenizer_false_negative"].sum()

    def clf_tp(self, search=None) -> int:
        """Return the Label classifier True Positives of all Spans."""
        return self._query(search=search)["clf_true_positive"].sum()

    def clf_fp(self, search=None) -> int:
        """Return the Label classifier False Positives of all Spans."""
        return self._query(search=search)["clf_false_positive"].sum()

    def clf_fn(self, search=None) -> int:
        """Return the Label classifier False Negatives of all Spans."""
        return self._query(search=search)["clf_false_negative"].sum()

    def get_evaluation_data(self, search, allow_zero: bool = True) -> EvaluationCalculator:
        """Get precision, recall, f1, based on TP, FP, FN."""
        return EvaluationCalculator(
            tp=self.tp(search), fp=self.fp(search), fn=self.fn(search), tn=self.tn(search), allow_zero=allow_zero
        )

    def precision(self, search=None) -> Optional[float]:
        """Calculate the Precision and see f1 to calculate imbalanced classes."""
        return EvaluationCalculator(tp=self.tp(search=search), fp=self.fp(search=search)).precision

    def recall(self, search=None) -> Optional[float]:
        """Calculate the Recall and see f1 to calculate imbalanced classes."""
        return EvaluationCalculator(tp=self.tp(search=search), fn=self.fn(search=search)).recall

    def f1(self, search=None) -> Optional[float]:
        """Calculate the F1 Score of one class.

        Please note: As suggested by Opitz et al. (2021) use the arithmetic mean over individual F1 scores.

        "F1 is often used with the intention to assign equal weight to frequent and infrequent classes, we recommend
        evaluating classifiers with F1 (the arithmetic mean over individual F1 scores), which is significantly more
        robust towards the error type distribution."

        Opitz, Juri, and Sebastian Burst. “Macro F1 and Macro F1.” arXiv preprint arXiv:1911.03347 (2021).
        https://arxiv.org/pdf/1911.03347.pdf

        :param search: Parameter used to calculate the value for one class.

        Example:
            1. If you have three Documents, calculate the F-1 Score per Document and use the arithmetic mean.
            2. If you have three Labels, calculate the F-1 Score per Label and use the arithmetic mean.
            3. If you have three Labels and three documents, calculate six F-1 Scores and use the arithmetic mean.

        """
        return EvaluationCalculator(tp=self.tp(search=search), fp=self.fp(search=search), fn=self.fn(search=search)).f1

    def tokenizer_f1(self, search=None) -> Optional[float]:
        """
        Calculate the F1 Score of one the tokenizer.

        :param search: Parameter used to calculate the value for one Data object.
        """
        return EvaluationCalculator(
            tp=self.tokenizer_tp(search=search),
            fp=self.tokenizer_fp(search=search),
            fn=self.tokenizer_fn(search=search),
        ).f1

    def clf_f1(self, search=None) -> Optional[float]:
        """
        Calculate the F1 Score of one the Label classifier.

        :param search: Parameter used to calculate the value for one Data object.
        """
        return EvaluationCalculator(
            tp=self.clf_tp(search=search),
            fp=self.clf_fp(search=search),
            fn=self.clf_fn(search=search),
        ).f1

    def _apply(self, group, issue_name):
        """Vertical merge error methods helper method."""
        if len(group) < 2:
            group[issue_name] = False
            return group
        if len(set(group['id_local'])) > 1 or len(set(group['id_local_predicted'])) > 1:
            group[issue_name] = True
        return True

    def get_missing_vertical_merge(self):
        """Return Spans that should have been merged."""
        self.data.groupby('id_local').apply(lambda group: self._apply(group, 'missing_merge'))

        return self.data[self.data['missing_merge']]

    def get_wrong_vertical_merge(self):
        """Return Spans that were wrongly merged vertically."""
        self.data.groupby('id_local_predicted').apply(lambda group: self._apply(group, 'wrong_merge'))
        return self.data[self.data['wrong_merge']]


class FileSplittingEvaluation:
    """Evaluate the quality of the filesplitting logic."""

    def __init__(self, ground_truth_documents: List[Document], prediction_documents: List[Document]):
        """
        Initialize and run the metrics calculation.

        :param ground_truth_documents: A list of original unchanged Documents.
        :type ground_truth_documents: list
        :param prediction_documents: A list of Documents with Pages newly predicted to be first or non-first.
        :type prediction_documents: list
        :raises ValueError: When ground_truth_documents and prediction_documents are not the same length.
        :raises ValueError: When a Page does not have a value of is_first_page.
        :raises ValueError: When an original Document and prediction are not referring to the same Document.
        """
        if len(ground_truth_documents) != len(prediction_documents):
            raise ValueError("ground_truth_documents and prediction_documents must be same length.")
        for document in ground_truth_documents:
            for page in document.pages():
                if page.is_first_page is None:
                    raise ValueError(f'Page {page.number} of {document} does not have a value of is_first_page.')
        for document in prediction_documents:
            for page in document.pages():
                if page.is_first_page is None:
                    raise ValueError(
                        f'Page {page.number} of prediction of {document.copy_of_id} does not have a value '
                        f'of is_first_page.'
                    )
        for ground_truth, prediction in zip(ground_truth_documents, prediction_documents):
            if ground_truth.id_ != prediction.copy_of_id:
                raise ValueError(
                    f"Incorrect prediction passed for {ground_truth}. Prediction has to be a copy of a "
                    f"ground truth Document."
                )
        projects = list(set([document.project for document in ground_truth_documents]))
        if len(projects) > 1:
            raise ValueError("All Documents have to belong to the same Project.")
        self.document_pairs = [
            [document[0], document[1]] for document in zip(ground_truth_documents, prediction_documents)
        ]
        self.project = projects[0]  # because we check that exactly one Project exists across the Documents
        self.evaluation_results = None
        self.evaluation_results_by_category = None
        self.calculate()
        self.calculate_metrics_by_category()

    def _metric_calculations(self, category=None):
        """
        Calculate metrics for a single category.

        :param category: A Category to calculate metrics for.
        :type category: Category
        :returns: Seven metrics.
        """
        tp, fp, fn, tn = 0, 0, 0, 0
        if category:
            evaluation_documents = [
                [document_1, document_2]
                for document_1, document_2 in self.document_pairs
                if document_1.category and document_1.category.id_ == category.id_
            ]
        else:
            evaluation_documents = self.document_pairs
        for ground_truth, prediction in evaluation_documents:
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
            precision = EvaluationCalculator(tp=tp, fp=fp, fn=fn, tn=tn).precision
        else:
            raise ZeroDivisionError("TP and FP are zero.")
        if tp + fn != 0:
            recall = EvaluationCalculator(tp=tp, fp=fp, fn=fn, tn=tn).recall
        else:
            raise ZeroDivisionError("TP and FN are zero.")
        if precision + recall != 0:
            f1 = EvaluationCalculator(tp=tp, fp=fp, fn=fn, tn=tn).f1
        else:
            raise ZeroDivisionError("FP and FN are zero.")
        return tp, fp, fn, tn, precision, recall, f1

    def calculate(self):
        """Calculate metrics for the filesplitting logic."""
        tp, fp, fn, tn, precision, recall, f1 = self._metric_calculations()
        self.evaluation_results = {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    def calculate_metrics_by_category(self):
        """Calculate metrics by Category independently."""
        categories = list(set([doc_pair[0].category for doc_pair in self.document_pairs]))
        self.evaluation_results_by_category = {}
        for category in categories:
            self.evaluation_results_by_category[category.id_] = {}
            tp, fp, fn, tn, precision, recall, f1 = self._metric_calculations(category)
            self.evaluation_results_by_category[category.id_]['true_positives'] = tp
            self.evaluation_results_by_category[category.id_]['false_positives'] = fp
            self.evaluation_results_by_category[category.id_]['false_negatives'] = fn
            self.evaluation_results_by_category[category.id_]['true_negatives'] = tn
            self.evaluation_results_by_category[category.id_]['precision'] = precision
            self.evaluation_results_by_category[category.id_]['recall'] = recall
            self.evaluation_results_by_category[category.id_]['f1'] = f1

    def _query(self, metric: str, search: Category = None) -> Union[int, float, None]:
        """
        Get a specific metric for a given category or get all metrics for all categories.

        :param metric: The name of the metric to get.
        :type metric: str
        :param search: The Category to get the metric for, if not provided will return all metrics for all Categories.
        :type search: Category
        :returns: A metric or a dictionary of metrics.
        :raises KeyError: If the given Category is not present in the project the evaluation is running on.
        """
        if search:
            if search.id_ not in self.evaluation_results_by_category:
                raise KeyError(
                    f'{search} is not present in {self.project}. Only Categories within a Project can be used for '
                    f'viewing metrics.'
                )
            return self.evaluation_results_by_category[search.id_][metric]
        return self.evaluation_results[metric]

    def get_evaluation_data(self, search: Category = None, allow_zero: bool = True) -> EvaluationCalculator:
        """
        Get precision, recall, f1, based on TP, TN, FP, FN.

        :param search: display true positives within a certain Category.
        :type search: Category
        :param allow_zero: If true, will calculate None for precision and recall when the straightforward application
        of the formula would otherwise result in 0/0. Raises ZeroDivisionError otherwise.
        :type allow_zero: bool
        """
        return EvaluationCalculator(
            tp=self.tp(search), fp=self.fp(search), fn=self.fn(search), tn=self.tn(search), allow_zero=allow_zero
        )

    def tp(self, search: Category = None) -> int:
        """
        Return correctly predicted first Pages.

        :param search: display true positives within a certain Category.
        :type search: Category
        :raises KeyError: When the Category in search is not present in the Project from which the Documents are.
        """
        return self._query('true_positives', search)

    def fp(self, search: Category = None) -> int:
        """
        Return non-first Pages incorrectly predicted as first.

        :param search: display false positives within a certain Category.
        :type search: Category
        :raises KeyError: When the Category in search is not present in the Project from which the Documents are.
        """
        return self._query('false_positives', search)

    def fn(self, search: Category = None) -> int:
        """
        Return first Pages incorrectly predicted as non-first.

        :param search: display false negatives within a certain Category.
        :type search: Category
        :raises KeyError: When the Category in search is not present in the Project from which the Documents are.
        """
        return self._query('false_negatives', search)

    def tn(self, search: Category = None) -> int:
        """
        Return non-first Pages predicted as non-first.

        :param search: display true negatives within a certain Category.
        :type search: Category
        :raises KeyError: When the Category in search is not present in the Project from which the Documents are.
        """
        return self._query('true_negatives', search)

    def precision(self, search: Category = None) -> float:
        """
        Return precision.

        :param search: display precision within a certain Category.
        :type search: Category
        :raises KeyError: When the Category in search is not present in the Project from which the Documents are.
        """
        return self._query('precision', search)

    def recall(self, search: Category = None) -> float:
        """
        Return recall.

        :param search: display recall within a certain Category.
        :type search: Category
        :raises KeyError: When the Category in search is not present in the Project from which the Documents are.
        """
        return self._query('recall', search)

    def f1(self, search: Category = None) -> float:
        """
        Return F1-measure.

        :param search: display F1 measure within a certain Category.
        :type search: Category
        :raises KeyError: When the Category in search is not present in the Project from which the Documents are.
        """
        return self._query('f1', search)
