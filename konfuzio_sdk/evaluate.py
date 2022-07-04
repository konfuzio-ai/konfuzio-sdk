"""Calculate the accuracy on any level in a  Document."""
from warnings import warn

import pandas as pd

RELEVANT_FOR_EVALUATION = [
    "id_local",  # needed to group spans in Annotations
    "id_",  # even we won't care of the id_, as the ID is defined by the start and end span
    # "confidence", we don't care about the confidence of doc_a
    "start_offset",  # only relevant for the merge but allows to track multiple sequences per annotation
    "end_offset",  # only relevant for the merge but allows to track multiple sequences per annotation
    "is_correct",  # we care if it is correct, humans create Annotations without confidence
    "label_id",
    "label_threshold",
    "revised",  # we need it to filter feedback required Annotations
    "annotation_set_id",
    "label_set_id",
    "document_id",
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
    eligible_to_vote = group['above_predicted_threshold'].fillna(False)
    if not len(group.loc[eligible_to_vote][target]):  # fallback if none of the Spans provide confidence
        group[verbose_validation_column_name] = group[target].mode(dropna=False)[0]
    else:  # get the most frequent annotation_set_id from the high confidence Spans in this group
        group[verbose_validation_column_name] = group.loc[eligible_to_vote][target].mode(dropna=False)[0]

    validation_column_name = f"is_correct_{target}"
    group[validation_column_name] = group[target] == group[verbose_validation_column_name]
    return group


def compare(doc_a, doc_b, only_use_correct=False, strict=True) -> pd.DataFrame:
    """Compare the Annotations of two potentially empty Documents wrt. to **all** Annotations.

    :param doc_a: Document which is assumed to be correct
    :param doc_b: Document which needs to be evaluated
    :param only_use_correct: Unrevised feedback in doc_a is assumed to be correct.
    :param strict: Evaluate on a Character exact level without any postprocessing, an amount Span "5,55 " will not be
     exact with "5,55"
    :return: Evaluation DataFrame
    """
    if not strict:
        warn('This method is WIP: https://gitlab.com/konfuzio/objectives/-/issues/9332', FutureWarning, stacklevel=2)
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
