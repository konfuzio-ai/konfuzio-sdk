"""Script to export evaluation filters."""
import argparse
import os
from typing import Optional
from warnings import warn

import pandas as pd

from konfuzio_sdk import KONFUZIO_HOST
from konfuzio_sdk.data import Project
from konfuzio_sdk.trainer.information_extraction import load_model

warn(
    'This module is WIP: https://gitlab.com/konfuzio/objectives/-/issues/9603#note_1134286927',
    FutureWarning,
    stacklevel=2,
)

parser = argparse.ArgumentParser()
parser.add_argument('--project_id', default=None, type=int)
parser.add_argument('--project_folder', default=None, type=str)
parser.add_argument('--update_project', default=True, type=bool)
parser.add_argument('--export_folder', default=None, type=str)
parser.add_argument('--model_path', type=str)
args = parser.parse_args()


def export_evaluation_filters(project: Project, model_path: str, export_folder: Optional[str] = None) -> None:
    """Export evaluation filters."""
    pipeline = load_model(model_path)
    data_quality = pipeline.data_quality()

    if export_folder is None:
        export_folder = f"./evaluation_{project.id_}"
    os.makedirs(export_folder, exist_ok=True)
    # for each of the following, export a excel file

    # sort docs data quality by increasing F1 score
    sorted_docs_by_f1 = sorted([(doc, data_quality.f1(doc)) for doc in pipeline.documents], key=lambda x: x[1])
    # [(doc, f1_score)]
    # excel with just doc link and score sorted in increasing order
    data = {}
    data['doc_link'] = [doc.link for doc, _, in sorted_docs_by_f1]
    data['f1'] = [f1 for _, f1 in sorted_docs_by_f1]
    df = pd.DataFrame.from_dict(data)
    df.to_excel(os.path.join(export_folder, f"{project.id_}_documents_by_f1score.xlsx"))

    # sort docs by increasing number of annotations in the labelset which contains least in that doc
    sorted_docs_by_anns_in_smallest_label_set = sorted(
        [
            sorted(
                [
                    (
                        docx,
                        target_label_set,
                        min([len([a for a in docx.annotations(use_correct=True) if a.label_set == target_label_set])]),
                    )
                    for target_label_set in pipeline.category.label_sets
                    if target_label_set.name != "NO_LABEL_SET"
                ],
                key=lambda x: x[1],
            )[0]
            for docx in pipeline.documents
        ],
        key=lambda x: x[-1],
    )
    # [(doc, labelset, n_anns)]
    # excel with just doc link and ann count sorted in increasing order
    data = {}
    data['doc_link'] = [doc.link for doc, _, _ in sorted_docs_by_anns_in_smallest_label_set]
    data['label_set'] = [label_set.name for _, label_set, _ in sorted_docs_by_anns_in_smallest_label_set]
    data['annotations'] = [annotations_count for _, _, annotations_count in sorted_docs_by_anns_in_smallest_label_set]
    df = pd.DataFrame.from_dict(data)
    df.to_excel(os.path.join(export_folder, f"{project.id_}_documents_by_annotations_count.xlsx"))

    # use resulting dataframe of compare function to sort all tp spans by increasing confidence
    sorted_correct_predictions_by_confidence = data_quality.data[data_quality.data["true_positive"] == 1].sort_values(
        by=['confidence_predicted']
    )[['id_', 'confidence_predicted']]
    # dataframe of spans
    # excel with just annotation link corresponding to the predicted span and
    # confidence percentage sorted in increasing order
    sorted_correct_predictions_by_confidence['id_'] = sorted_correct_predictions_by_confidence['id_'].apply(
        lambda x: f"{KONFUZIO_HOST}/a/" + str(int(x))
    )
    sorted_correct_predictions_by_confidence.to_excel(
        os.path.join(export_folder, f"{project.id_}_annotations_by_confidence.xlsx")
    )

    # sort labels data quality by increasing F1 score
    sorted_labels_by_f1 = sorted(
        [
            (label, data_quality.f1(label) if data_quality.f1(label) is not None else 0)
            for label in pipeline.category.labels
            if label.name != "NO_LABEL"
        ],
        key=lambda x: x[1],
    )
    # [(label, f1_score)]
    # excel with evaluation table sorted in increasing order
    data = {}
    data['label'] = [label.name for label, _ in sorted_labels_by_f1]
    data['f1'] = [f1 for _, f1 in sorted_labels_by_f1]
    df = pd.DataFrame.from_dict(data)
    df.to_excel(os.path.join(export_folder, f"{project.id_}_labels_by_f1score.xlsx"))

    # sort labelsets data quality by increasing F1 score
    sorted_label_sets_by_f1 = sorted(
        [
            (label_set, data_quality.f1(label_set) if data_quality.f1(label_set) is not None else 0)
            for label_set in pipeline.category.label_sets
            if label_set.name != "NO_LABEL_SET"
        ],
        key=lambda x: x[1],
    )
    # [(labelset, f1_score)]
    # excel with evaluation table sorted in increasing order
    data = {}
    data['label_set'] = [label_set.name for label_set, _ in sorted_label_sets_by_f1]
    data['f1'] = [f1 for _, f1 in sorted_label_sets_by_f1]
    df = pd.DataFrame.from_dict(data)
    df.to_excel(os.path.join(export_folder, f"{project.id_}_label_sets_by_f1score.xlsx"))


if __name__ == '__main__':
    project_id = args.project_id
    project_folder = args.project_folder
    update_project = args.update_project
    export_folder = args.export_folder
    model_path = args.model_path

    if model_path is None:
        raise ValueError(
            'You need to provide a model path. '
            'Example: "python export_evaluation_filters.py --model_path ./PATH/model.pkl"'
        )

    if project_id is None and project_folder is None:
        raise ValueError(
            'You need to provide at least a Project ID or a Project folder. '
            'Example: "python export_evaluation_filters.py --project_id 46" '
            'Example: "python export_evaluation_filters.py --project_folder ./PATH/data_46"'
        )

    p = Project(id_=project_id, project_folder=project_folder)
    export_evaluation_filters(p, model_path, export_folder)
