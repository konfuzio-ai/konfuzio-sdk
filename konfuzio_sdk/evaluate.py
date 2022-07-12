"""Calculate the accuracy on any level in a  Document."""
from typing import Tuple, List

import pandas
from sklearn.utils.extmath import weighted_mode


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
        # add check to evaluate multiline Annotations
        spans = spans.groupby("id_local", dropna=False).apply(lambda group: grouped(group, "id_"))
        # add check to evaluate Annotation Sets
        spans = spans.groupby("annotation_set_id_predicted", dropna=False).apply(
            lambda group: grouped(group, "annotation_set_id")
        )
    spans = spans[RELEVANT_FOR_EVALUATION]

    assert not spans.empty  # this function must be able to evaluate any two docs even without annotations

    # Evaluate which **spans** are TN, TP, FP and keep RELEVANT_FOR_MAPPING to allow grouping of confidence measures
    spans["true_positive"] = 1 * (
        (spans["is_matched"])
        & (spans["is_correct"])
        & (spans["above_predicted_threshold"])
        & (  # Everything is correct
            (spans["is_correct_label"])
            & (spans["is_correct_label_set"])
            & (spans["is_correct_annotation_set_id"])
            & (spans["is_correct_id_"])
        )
    )

    spans["false_negative"] = 1 * (
        (spans["is_correct"]) & ((~spans["is_matched"]) | (~spans["above_predicted_threshold"]))
    )

    spans["false_positive"] = 1 * (  # commented out on purpose (spans["is_correct"]) &
        (spans["above_predicted_threshold"])
        & (~spans["false_negative"])
        & (~spans["true_positive"])
        & (  # Something is wrong
            (~spans["is_correct_label"])
            | (~spans["is_correct_label_set"])
            | (~spans["is_correct_annotation_set_id"])
            | (~spans["is_correct_id_"])
        )
    )

    spans["is_found_by_tokenizer"] = 1 * (
        (spans["start_offset"] == spans["start_offset_predicted"])
        & (spans["end_offset"] == spans["end_offset_predicted"])
        & (spans["is_correct"])
        & (spans["document_id_local_predicted"].notna())
    )

    # one Span must not be defined as TP or FP or FN more than once
    quality = (spans[['true_positive', 'false_positive', 'false_negative']].sum(axis=1) <= 1).all()
    assert quality
    return spans


class Evaluation:
    """Calculated accuracy measures by using the detailed comparison on Span Level."""

    from konfuzio_sdk.data import Document

    def __init__(self, documents: List[Tuple[Document, Document]]):
        """
        Relate to the two document instances.

        :param documents: A list of tuple Documents that should be compared.
        """
        self.documents = documents
        self.strict = True
        self.only_use_correct = True
        self.data = None
        self.calculate()

    def calculate(self):
        """Calculate and update the data stored within this Evolution."""
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
        elif isinstance(search, Label):
            assert search.id_ is not None, f'{search} must have a ID'
            query = f'label_id == {search.id_} | (label_id_predicted == {search.id_})'
        elif isinstance(search, Document):
            assert search.id_ is not None, f'{search} must have a ID.'
            query = f'document_id == {search.id_} | (document_id_predicted == {search.id_})'
        elif isinstance(search, LabelSet):
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

    def tokenizer(self, search=None) -> int:
        """Return the of all Spans that are found by the Tokenizer."""
        return self._query(search=search)["is_found_by_tokenizer"].sum()

    def precision(self, search) -> float:
        """Calculate the Precision and see f1 to calculate imbalanced classes."""
        return self.tp(search=search) / (self.tp(search=search) + self.fp(search=search))

    def recall(self, search) -> float:
        """Calculate the Recall and see f1 to calculate imbalanced classes."""
        return self.tp(search=search) / (self.tp(search=search) + self.fn(search=search))

    def f1(self, search) -> float:
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
        return self.tp(search=search) / (
            self.tp(search=search) + 0.5 * (self.fp(search=search) + self.fn(search=search))
        )
