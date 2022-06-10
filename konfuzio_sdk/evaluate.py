"""Calculate the accuracy on any level in a  Document."""
import logging
from typing import Dict

import pandas as pd

from dataclasses import dataclass

from tabulate import tabulate

logger = logging.getLogger(__name__)

RELEVANT_FOR_EVALUATION = [
    "id_local",  # needed to group spans in Annotations
    "id_",  # even we won't care of the id_, as the ID is defined by the start and end span
    # "confidence", we don't care about the confidence of doc_a
    "start_offset",  # only relevant for the merge but allows to track multiple sequences per annotation
    "end_offset",  # only relevant for the merge but allows to track multiple sequences per annotation
    "offset_string",
    "is_correct",  # we care if it is correct, humans create Annotations without confidence
    "label_id",
    "label_name",
    "label_threshold",
    "revised",  # we need it to filter feedback required Annotations
    "annotation_set_id",
    "label_set_id",
    "label_set_name",
    "document_id",
    "document_dataset_status",
    "document_id_local",
    "category_id",  # Identify the Category to be able to run an evaluation across categories
    # "id__predicted", we don't care of the id_ see "id_"
    "confidence_predicted",  # we care about the confidence of the prediction
    # "start_offset_predicted", only relevant for the merge
    # "end_offset_predicted", only relevant for the merge
    # "is_correct_predicted", # it's a prediction so we don't know if it is correct
    "label_id_predicted",
    "label_threshold_predicted",  # we keep a flexibility to be able to predict the threshold
    # "revised_predicted",  # it's a prediction so we ignore if it is revised
    "annotation_set_id_predicted",
    "label_set_id_predicted",
    "document_id_predicted",
    "document_id_local_predicted",
]


def grouped(group, target: str):
    """Define which of the correct element in the predicted group defines the "correct" group id_."""
    verbose_validation_column_name = f"defined_to_be_correct_{target}"
    # all rows where is_correct is nan relate to an element which has no correct element partner
    correct = group["is_correct"].fillna(False)  # so fill nan with False as .loc will need boolean

    if len(group.loc[correct][target]) == 0:  # no "correct" element in the group, but the predicted grouping is correct
        group[verbose_validation_column_name] = group[target].mode(dropna=False)[0]
    else:  # get the most frequent annotation_set_id from the *correct* Annotations in this group
        group[verbose_validation_column_name] = group.loc[correct][target].mode(dropna=False)[0]

    validation_column_name = f"is_correct_{target}"
    group[validation_column_name] = group[target] == group[verbose_validation_column_name]
    return group


def compare(doc_a, doc_b, only_use_correct=False) -> pd.DataFrame:
    """Compare the Annotations of two potentially empty Documents wrt. to **all** Annotations.

    :param doc_a: Document which is assumed to be correct
    :param doc_b: Document which needs to be evaluated
    :param only_use_correct: Unrevised feedback in doc_a is assumed to be correct.
    :return: Evaluation DataFrame
    """
    if doc_a.category != doc_b.category:
        raise ValueError(f'Categories of {doc_a} with {doc_a.category} and {doc_b} with {doc_a.category} do not match.')
    df_a = pd.DataFrame(doc_a.eval_dict(use_correct=only_use_correct))
    df_b = pd.DataFrame(doc_b.eval_dict(use_correct=False))

    # many to many inner join to keep all **spans** of doc_a and doc_b
    spans = pd.merge(df_a, df_b, how="outer", on=["start_offset", "end_offset"], suffixes=('', '_predicted'))
    spans = spans[RELEVANT_FOR_EVALUATION]

    # add criteria to evaluate **spans**
    spans["above_predicted_threshold"] = spans["confidence_predicted"] >= spans["label_threshold_predicted"]
    spans["is_correct_label"] = spans["label_id"] == spans["label_id_predicted"]
    spans["is_correct_label_set"] = spans["label_set_id"] == spans["label_set_id_predicted"]
    # add check to evaluate multiline Annotations
    spans = spans.groupby("id_local", dropna=False).apply(lambda group: grouped(group, "id_"))
    # add check to evaluate Annotation Sets
    spans = spans.groupby("annotation_set_id_predicted", dropna=False).apply(
        lambda group: grouped(group, "annotation_set_id")
    )

    assert not spans.empty  # this function must be able to evaluate any two docs even without annotations

    # Evaluate which **spans** are TN, TP, FP and keep RELEVANT_FOR_MAPPING to allow grouping of confidence measures
    spans["true_positive"] = 1 * (
        (spans["is_correct"])
        & (spans["above_predicted_threshold"])
        & (  # Everything is correct
            (spans["is_correct_label"])
            & (spans["is_correct_label_set"])
            & (spans["is_correct_annotation_set_id"])
            & (spans["is_correct_id_"])
        )
    )

    spans["false_positive"] = 1 * (  # commented out on purpose (spans["is_correct"]) &
        spans["above_predicted_threshold"]
        & (  # Something is wrong
            (~spans["is_correct_label"])
            | (~spans["is_correct_label_set"])
            | (~spans["is_correct_annotation_set_id"])
            | (~spans["is_correct_id_"])
        )
    )

    spans["false_negative"] = 1 * ((spans["is_correct"]) & (~spans["above_predicted_threshold"]))

    spans["is_found_by_tokenizer"] = 1 * (spans["is_correct"] & spans["document_id_local_predicted"].notna())

    # one **span** cannot be assigned to more than one group, however can be a True Negative
    quality = (spans[['true_positive', 'false_positive', 'false_negative']].sum(axis=1) <= 1).all()
    assert quality
    return spans


@dataclass
class Evaluation:
    tp: float
    tn: float
    fp: float
    fn: float

    def __init__(self, evaluation: 'DataFrame'):
        self.evaluation = evaluation

    # adding two Evaluations
    def __add__(self, other):
        return Evaluation(self.tp + other.tp, self.tn + other.tn, self.fp + other.fp, self.fp + other.fn)

    @property
    def tn(self):
        return 0

    @property
    def tp(self):
        return self.evaluation['true_positive'].sum()

    @property
    def fn(self):
        return self.evaluation['false_negative'].sum()

    @property
    def fp(self):
        return self.evaluation['false_positive'].sum()

    @property
    def accuracy(self) -> float:
        try:
            return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
        except ZeroDivisionError:
            return 0.0
        except TypeError:
            return 0.0

    def accuracy_display(self):
        return '{0:.2%}'.format(self.accuracy)

    @property
    def precision(self):
        try:
            return self.tp / (self.tp + self.fp)
        except ZeroDivisionError:
            return 0.0

    def precision_display(self):
        return '{0:.2}'.format(self.precision)

    @property
    def recall(self):
        try:
            return self.tp / (self.tp + self.fn)
        except ZeroDivisionError:
            return 0.0

    def recall_display(self):
        return '{0:.2}'.format(self.recall)

    @property
    def f1_score(self):
        try:
            return 2 * ((self.precision * self.recall) / (self.precision + self.recall))
        except ZeroDivisionError:
            return 0.0

    def f1_score_display(self):
        return '{0:.2}'.format(self.f1_score)

    def fields_display(self) -> list:
        return [
            self.accuracy_display(),
            int(self.tp),
            int(self.fp),
            int(self.tn),
            int(self.fn),
            self.precision_display(),
            self.recall_display(),
            self.f1_score_display(),
        ]

    def to_dict(self) -> Dict:
        r = {
            'accuracy': self.accuracy,
            'f1_score': self.f1_score,
            'precision': self.precision,
            'recall': self.recall,
            'tp': self.tp,
            'fp': self.fp,
            'fn': self.fn
        }
        return r

    def label_evaluations(self, dataset_status=None) -> pd.DataFrame:
        df_list = []
        if self.evaluation.empty:
            return pd.DataFrame()

        if dataset_status:
            evaluation = self.evaluation[self.evaluation['document_dataset_status'].isin(dataset_status)]
        else:
            evaluation = self.evaluation
        for label_name, label_df in evaluation.groupby('label_name'):
            df_list.append({'label_name': label_name, **Evaluation(label_df).to_dict()})

        df = pd.DataFrame(df_list)
        logger.info('\n' + tabulate(df, floatfmt=".2%", headers="keys", tablefmt="pipe") + '\n')
        return df