"""Calculate the accuracy on any level in a  Document."""
from typing import Dict, Tuple, List, Optional, Union

import pandas
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from sklearn.utils.extmath import weighted_mode

from konfuzio_sdk.utils import sdk_isinstance
from konfuzio_sdk.data import Category, Document


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
    spans = spans.replace({np.nan: None})
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


class ExtractionEvaluation:
    """Calculated accuracy measures by using the detailed comparison on Span Level."""

    def __init__(self, documents: List[Tuple[Document, Document]], strict: bool = True):
        """
        Relate to the two document instances.

        :param documents: A list of tuple Documents that should be compared.
        :param strict: A boolean passed to the `compare` function.
        """
        self.documents = documents
        self.strict = strict
        self.only_use_correct = True
        self.data = None
        self.calculate()

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


class CategorizationEvaluation:
    """Calculated evaluation measures for the classification task of Document categorization."""

    def __init__(self, categories: List[Category], documents: List[Tuple[Document, Document]]):
        """
        Relate to the two document instances.

        :param project: The project containing the Documents and Categories to be evaluated.
        :param documents: A list of tuple Documents that should be compared.
        """
        self.categories = categories
        self.documents = documents
        self.evaluation_results = None
        self._clf_report = None
        self.calculate()

    @property
    def labels(self) -> List[int]:
        """List of category ids as class labels."""
        return [category.id_ for category in self.categories]

    @property
    def labels_names(self) -> List[str]:
        """List of category names as class names."""
        return [category.name for category in self.categories]

    @property
    def actual_classes(self) -> List[int]:
        """List of ground truth category ids."""
        return [ground_truth.category.id_ for ground_truth, predicted in self.documents]

    @property
    def predicted_classes(self) -> List[int]:
        """List of predicted category ids."""
        return [
            predicted.category.id_ if predicted.category is not None else -1
            for ground_truth, predicted in self.documents
        ]

    def confusion_matrix(self) -> pandas.DataFrame:
        """Confusion matrix."""
        return confusion_matrix(self.actual_classes, self.predicted_classes, labels=self.labels + [-1])

    def _get_tp_tn_fp_fn_per_label(self) -> Dict:
        """
        Get the tp, fp, tn and fn for each label.

        The label for which the evaluation is being done is considered the positive class. All others are considered as
        negative class.

        Follows the logic:
        tpi = cii (value in the diagonal of the cm for the respective label)
        fpi = ∑nl=1 cli − tpi (sum of the column of the cm - except tp)
        fni = ∑nl=1 cil − tpi (sum of the row of the cm - except tp)
        tni = ∑nl=1 ∑nk=1 clk − tpi − fpi − fni (all other values not considered above)

        cm = [[1, 1, 0],
            [0, 2, 1],
            [1, 2, 3]]

        For label '1':
        tp = 2
        fp = 1 + 2 = 3
        fn = 1 + 0 = 1
        tn = 11 - 2 - 3 - 1 = 5

        :return: dictionary with the results per label
        """
        confusion_matrix = self.confusion_matrix()
        sum_columns = np.sum(confusion_matrix, axis=0)
        sum_rows = np.sum(confusion_matrix, axis=1)
        sum_all = np.sum(confusion_matrix)

        results = {}

        for ind, category_id in enumerate(self.labels):
            tp = confusion_matrix[ind, ind]
            fp = sum_columns[ind] - tp
            fn = sum_rows[ind] - tp
            tn = sum_all - fn - fp - tp

            results[category_id] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}

        return results

    def calculate(self):
        """Calculate and update the data stored within this Evolution."""
        self.evaluation_results = self._get_tp_tn_fp_fn_per_label()
        self._clf_report = classification_report(
            y_true=self.actual_classes,
            y_pred=self.predicted_classes,
            labels=self.labels,
            target_names=self.labels_names,
            output_dict=True,
        )

    def _search_category(self, category: Optional[Category] = None) -> Optional[int]:
        """Return the category id to filter for."""
        if sdk_isinstance(category, Category):
            return category.id_
        elif isinstance(category, int):
            return category
        elif category is None:
            return None
        else:
            raise NotImplementedError

    def tp(self, category: Optional[Category] = None) -> int:
        """Return the True Positives of all Documents."""
        search_category_id = self._search_category(category)
        return sum(
            [
                evaluation["tp"]
                for category_id, evaluation in self.evaluation_results.items()
                if (search_category_id is None) or (category_id == search_category_id)
            ]
        )

    def fp(self, category: Optional[Category] = None) -> int:
        """Return the False Positives of all Documents."""
        search_category_id = self._search_category(category)
        return sum(
            [
                evaluation["fp"]
                for category_id, evaluation in self.evaluation_results.items()
                if (search_category_id is None) or (category_id == search_category_id)
            ]
        )

    def fn(self, category: Optional[Category] = None) -> int:
        """Return the False Negatives of all Documents."""
        search_category_id = self._search_category(category)
        return sum(
            [
                evaluation["fn"]
                for category_id, evaluation in self.evaluation_results.items()
                if (search_category_id is None) or (category_id == search_category_id)
            ]
        )

    def tn(self, category: Optional[Category] = None) -> int:
        """Return the True Negatives of all Documents."""
        search_category_id = self._search_category(category)
        return sum(
            [
                evaluation["tn"]
                for category_id, evaluation in self.evaluation_results.items()
                if (search_category_id is None) or (category_id == search_category_id)
            ]
        )

    def precision(self, category: Optional[Category]) -> Optional[float]:
        """Calculate the Precision and see f1 to calculate imbalanced classes."""
        if category is None:
            return self._clf_report['weighted avg']['precision']
        else:
            return EvaluationCalculator(tp=self.tp(category=category), fp=self.fp(category=category)).precision

    def recall(self, category: Optional[Category]) -> Optional[float]:
        """Calculate the Recall and see f1 to calculate imbalanced classes."""
        if category is None:
            return self._clf_report['weighted avg']['recall']
        else:
            return EvaluationCalculator(tp=self.tp(category=category), fn=self.fn(category=category)).recall

    def f1(self, category: Optional[Category]) -> Optional[float]:
        """Calculate the F1 Score of one class."""
        if category is None:
            return self._clf_report['weighted avg']['f1-score']
        else:
            return EvaluationCalculator(
                tp=self.tp(category=category), fp=self.fp(category=category), fn=self.fn(category=category)
            ).f1

    def update_names_and_indexes(self, names: Union[List[str], None] = None, indexes: Union[List[str], None] = None):
        """Update the lists of labels names and classes indexes to consider a None prediction."""
        if names is not None and indexes is not None:
            # if we already have the names and indexes, we check for 'NO_LABEL'
            if 'NO_LABEL' not in names:
                # if 'NO_LABEL' not in the names, we add it and also the index 0
                if 0 in indexes:
                    # if classes_indexes already has a 0, we cannot add NO_LABEL
                    print('A prediction is NoneType and not possible to add NO_LABEL.')
                    return None

                names.append('NO_LABEL')
                indexes.append(0)

            if names.index('NO_LABEL') != indexes.index(0):
                print('Index of "NO_LABEL" is not 0.')
                return None

        elif names is not None and indexes is None:
            # if we only have labels_names, we add 'NO_LABEL'
            if 'NO_LABEL' not in names:
                names.append('NO_LABEL')

        elif names is None and indexes is not None:
            if 0 in indexes:
                print('Using 0 as class index for None prediction.')
            else:
                indexes.append(0)

        return names, indexes

    def get_metrics_per_label(self) -> List[dict]:
        """
        Get metrics per label.

        These metrics are obtained directly from sklearn:

        - precision - the ability of the classifier not to label as positive a sample that is negative.
            tp / (tp + fp)

        - recall - the ability of the classifier to find all the positive samples
            tp / (tp + fn)

        - f score - weighted harmonic mean of the precision and recall
            beta = 1.0 (default value)
            beta2 = beta ** 2
            denom = beta2 * precision + recall
            f_score = (1 + beta2) * precision * recall / denom

        - support - number of occurrences of each class in actual_classes

        These metrics are calculated based on the results (tp, fp, tn, fn) per label
        - accuracy - measures how often the algorithm classifies a data point correctly
             (tp + tn) / (tp + tn + fp + fn)

        - balanced accuracy - avoids inflated performance estimates on imbalanced datasets. For balanced datasets, the
        score is equal to accuracy. (formula from from
        https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score)
            (recall + specificity) / 2

        :return: metrics per label
        """
        predicted_classes = self.predicted_classes
        labels_names = self.labels_names
        classes_indexes = self.labels
        actual_classes = self.actual_classes

        if classes_indexes is None:
            classes_indexes = [int(i) for i in set(actual_classes + predicted_classes)]

        result_per_label = self._get_tp_tn_fp_fn_per_label()

        precision, recall, fscore, support = precision_recall_fscore_support(
            actual_classes, predicted_classes, labels=classes_indexes
        )

        # store results for each label
        results_labels = []

        for i, label in enumerate(labels_names):
            r_label = result_per_label[i + 1]

            tp = r_label['tp']
            fp = r_label['fp']
            tn = r_label['tn']
            fn = r_label['fn']

            specificity = tn / (tn + fp + 1e-10)

            results = {
                'label': label,
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn,
                'accuracy': (tp + tn) / (tp + tn + fp + fn),
                'balanced accuracy': (recall[i] + specificity) / 2,
                'f1-score': fscore[i],
                'precision': precision[i],
                'recall': recall[i],
                'support': support[i],
            }

            results_labels.append(results)

        return results_labels

    def get_general_metrics(self) -> Dict[str, float]:
        """
        Get general metrics.

        The classification report from sklearn returns macro averaged metrics and weighted averaged metrics.
        We are returning the weighted averaged metrics which result from averaging the support-weighted mean per label.

        Gets general accuracy, balanced accuracy and f1-score over all labels.
        """
        predicted_classes = self.predicted_classes
        labels_names = self.labels_names
        classes_indexes = self.labels
        actual_classes = self.actual_classes

        if labels_names is None:
            labels_names = [str(i) for i in set(actual_classes + predicted_classes)]

        # if 'NO_LABEL' in labels_names:
        #    if labels_names.index('NO_LABEL') not in actual_classes:
        #        labels_names.remove('NO_LABEL')

        if classes_indexes is None:
            classes_indexes = [int(i) for i in set(actual_classes + predicted_classes)]

        try:
            clf_report = classification_report(
                y_true=actual_classes,
                y_pred=predicted_classes,
                labels=classes_indexes,
                target_names=labels_names,
                output_dict=True,
            )
            f1_score = clf_report['weighted avg']['f1-score']
            precision = clf_report['weighted avg']['precision']
            recall = clf_report['weighted avg']['recall']
            support = clf_report['weighted avg']['support']
        except Exception:
            f1_score = None
            precision = None
            recall = None
            support = None

        try:
            cm = confusion_matrix(actual_classes, predicted_classes, labels=classes_indexes)
            tp = np.sum(cm.diagonal())
        except Exception:
            tp = 0

        results_general = {
            'label': 'general/all annotations',
            'tp': tp,
            'fp': len(actual_classes) - tp,
            'accuracy': accuracy_score(actual_classes, predicted_classes),
            'balanced accuracy': balanced_accuracy_score(actual_classes, predicted_classes),
            'f1-score': f1_score,
            'precision': precision,
            'recall': recall,
            'support': support,
        }

        return results_general
