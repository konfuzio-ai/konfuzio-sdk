"""Calculate the accuracy on any level in a document."""
import itertools

import pandas as pd

RELEVANT_FOR_EVALUATION = [
    "id_local",  # needed to group offsets in annotations
    "id",  # even we don't care of the id, as the ID is defined by the start and end offset
    # "accuracy", we don't care about the accuracy of doc_a  # todo rename to confidence
    "start_offset",  # only relevant for the merge but allows to track multiple sequences per annotation
    "end_offset",  # only relevant for the merge but allows to track multiple sequences per annotation
    "is_correct",  # we care if it is correct, humans create annotations without accuracy
    "label_id",
    "label_threshold",
    "revised",  # we need it to filter feedback required annotations
    "annotation_set_id",
    "label_set_id",
    # "id_predicted", we don't care of the id see "id"
    "accuracy_predicted",  # we care about the accuracy of the prediction # todo rename to confidence
    # "start_offset_predicted", only relevant for the merge
    # "end_offset_predicted", only relevant for the merge
    # "is_correct_predicted", # it's a prediction so we don't know if it is correct
    "label_id_predicted",
    "label_threshold_predicted",  # we keep a flexibility to be able to predict the threshold
    # "revised_predicted",  # it's a prediction so we ignore if it is revised
    "annotation_set_id_predicted",
    "label_set_id_predicted",
]


def grouped(group, target: str):
    """Define which of the correct element in the predicted group defines the "correct" group id."""
    verbose_validation_column_name = f"defined_to_be_correct_{target}"
    # all rows where is_correct is nan relate to an element which has no correct element partner
    correct = group["is_correct"].fillna(False)  # so fill nan with False as .loc will need boolean

    # TODO: check if the group selection follows the logic of text annotation
    if correct.isnull().all():
        # there is no correct element we can map on, all nan values
        group[verbose_validation_column_name] = 0
    elif len(group.loc[correct][target]) == 0:
        # there is no "correct" element in the group, however the predicted grouping is correct
        # todo support for mode if it is not math. defined, i.e. [1, 1, 2, 2]
        group[verbose_validation_column_name] = group[target].mode(dropna=False)[0]
    else:
        # get the most frequent annotation_set_id from the *correct* annotations in this group
        # todo support for mode if it is not math. defined, i.e. [1, 1, 2, 2]
        group[verbose_validation_column_name] = group.loc[correct][target].mode(dropna=False)[0]

    validation_column_name = f"is_correct_{target}"
    group[validation_column_name] = group[target] == group[verbose_validation_column_name]
    return group


def compare(doc_a, doc_b):
    """Compare the annotations of two potentially empty documents wrt. to **all** annotations."""
    from konfuzio_sdk.data import Annotation, Label, Project, Document  # prevent circular reference

    # TODO:
    #  make use_correct a variable because if we have feedback required annotations in the document,
    #  we ignore them if predicted
    #  This should be an option in the project for the user. (could be confusing)

    # TODO: we cannot add annotations without label therefore we cannot add an "empty" Annotation
    if not doc_a.annotations(use_correct=False):
        doc_a.add_annotation(Annotation(label=Label(project=Project()), document=Document()))

    if not doc_b.annotations(use_correct=False):
        doc_b.add_annotation(Annotation(label=Label(project=Project()), document=Document()))

    # As one Annotation could have more than one **offset**, we unpack the list per annotation with itertools
    # We can encapsulate this code more, however the aim was to present any transformation in a very transparent way
    doc_a = pd.DataFrame(list(itertools.chain(*[anno.eval_dict for anno in doc_a.annotations(use_correct=False)])))
    doc_b = pd.DataFrame(list(itertools.chain(*[anno.eval_dict for anno in doc_b.annotations(use_correct=False)])))

    # many to many inner join to keep all **offsets** of doc_a and doc_b
    offsets = pd.merge(doc_a, doc_b, how="outer", on=["start_offset", "end_offset"], suffixes=('', '_predicted'))
    offsets = offsets[RELEVANT_FOR_EVALUATION]

    # add criteria to evaluate **offsets**
    offsets["above_threshold"] = offsets["accuracy_predicted"] > offsets["label_threshold_predicted"]
    offsets["is_correct_label"] = offsets["label_id"] == offsets["label_id_predicted"]
    offsets["is_correct_label_set"] = offsets["label_set_id"] == offsets["label_set_id_predicted"]
    # add check to evaluate multiline Annotations
    offsets = offsets.groupby("id_local", dropna=False).apply(lambda group: grouped(group, "id"))
    # add check to evaluate annotation sets
    offsets = offsets.groupby("annotation_set_id_predicted", dropna=False).apply(
        lambda group: grouped(group, "annotation_set_id")
    )

    assert not offsets.empty  # this function must be able to evaluate any two docs even without annotations

    # Evaluate which **offsets** are TN, TP, FP and keep RELEVANT_FOR_MAPPING to allow grouping of accuracy measures
    # TODO: exclude feedback required already in the document
    offsets["true_positive"] = 1 * (
        (offsets["is_correct"])
        & (offsets["above_threshold"])
        & (  # Everything is correct
            (offsets["is_correct_label"])
            & (offsets["is_correct_label_set"])
            & (offsets["is_correct_annotation_set_id"])
            & (offsets["is_correct_id"])
        )
    )

    offsets["false_positive"] = 1 * (  # commented out on purpose (offsets["is_correct"]) &
        offsets["above_threshold"]
        & (  # Something is wrong
            (~offsets["is_correct_label"])
            | (~offsets["is_correct_label_set"])
            | (~offsets["is_correct_annotation_set_id"])
            | (~offsets["is_correct_id"])
        )
    )

    offsets["false_negative"] = 1 * ((offsets["is_correct"]) & (~offsets["above_threshold"]))

    # TODO: consider feedback required
    # To calculate f1 score range:
    # f1 score min with all feedback required considered as fp and f1 score max with all feedback requires as tp
    # offsets["is_correct"].fillna(False)
    # offsets["revised"].fillna(False)

    # feedback required already existing in the document (from previous AI)
    # filter out annotations that are not mapped (not existing in the document)
    # offsets["feedback_required"] = 1 * (
    #     (~offsets["is_correct"])
    #     & (~offsets["revised"])
    #     & (offsets["above_threshold"])
    #     & (  # Everything is correct
    #         (offsets["is_correct_label"])
    #         & (offsets["is_correct_label_set"])
    #         & (offsets["is_correct_annotation_set_id"])
    #         & (offsets["is_correct_id"])
    #     )
    # )

    # one **offset** cannot be assigned to more than one group, however can be a True Negative
    quality = (offsets[['true_positive', 'false_positive', 'false_negative']].sum(axis=1) <= 1).all()
    assert quality
    return offsets
