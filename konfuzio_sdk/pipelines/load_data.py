"""Load ai models in pickle (pkl) or pytorch (pt) format."""
import os
import logging
import cloudpickle
import datetime
import glob

from typing import Optional
import pandas as pd
from konfuzio_sdk import BASE_DIR
from konfuzio_sdk.data import Annotation, Document, LabelSet, AnnotationSet, Label

logger = logging.getLogger(__name__)


def get_latest_document_model(model_name, folder_path=None) -> Optional[str]:
    """
    Get path to first in folder_path or BASE_DIR (descending order) of matching file.

    :param model_name: unix style pathname pattern expansion "*document_model.pkl"
    :param folder_path: search this path if specified
    :return: path of the first file in BASE_DIR, sorted in descending order
    """
    if not folder_path:
        folder_path = BASE_DIR

    def get_datetime(document_name):
        """Convert datetime to string."""
        try:
            datetime_string = os.path.basename(document_name).split('_')[0]
            return datetime.datetime.strptime(datetime_string, '%Y-%m-%d-%H-%M-%S')
        except Exception as e:
            logger.error(f'Could not find model name including a valid timestamp: Error: {e}')
            return datetime.datetime.min

    file_names = glob.glob(os.path.join(folder_path, model_name))
    if not file_names:
        file_names = glob.glob(model_name)
        if not file_names:
            raise FileNotFoundError
    file_names.sort(key=get_datetime)
    return file_names[-1]


def add_extractions_as_annotations(
        extractions: pd.DataFrame, document: Document,
        label: Label, label_set: LabelSet, annotation_set: AnnotationSet
):
    """Add the extraction of a model to the document."""
    if not extractions.empty:
        # TODO: define required fields
        required_fields = ['start_offset', 'end_offset', 'confidence']
        if not set(required_fields).issubset(extractions.columns):
            raise ValueError(f'Extraction do not contain all required fields: {required_fields}.'
                             f' Extraction columns: {extractions.columns.to_list()}')

        annotations = extractions[
            ['start_offset', 'end_offset', 'confidence']  # 'page_index', 'x0', 'x1', 'y0', 'y1']
        ].sort_values(by='confidence', ascending=False)

        for annotation in annotations.to_dict('records'):  # todo ask Ana: are Start and End always ints
            _ = Annotation(
                document=document,
                label=label,
                confidence=annotation['confidence'],
                label_set=label_set,
                annotation_set=annotation_set,
                bboxes=[annotation],
            )


def extraction_result_to_document(document: Document, extraction_result: dict) -> Document:
    """Return a virtual Document annotated with AI Model output."""
    virtual_doc = Document(
        project=document.category.project,
        text=document.text,
        bbox=document.get_bbox(),
        category=document.category,
        pages=document.pages,
    )
    virtual_annotation_set_id = 0  # counter for across mult. Annotation Set groups of a Label Set

    # define Annotation Set for the Category Label Set: todo: this is unclear from API side
    # default Annotation Set will be always added even if there are no predictions for it
    category_label_set = document.category.project.get_label_set_by_id(document.category.id_)
    virtual_default_annotation_set = AnnotationSet(
        document=virtual_doc, label_set=category_label_set, id_=virtual_annotation_set_id
    )

    for label_or_label_set_name, information in extraction_result.items():
        if isinstance(information, pd.DataFrame) and not information.empty:
            # annotations belong to the default Annotation Set
            label = document.category.project.get_label_by_name(label_or_label_set_name)
            add_extractions_as_annotations(
                document=virtual_doc,
                extractions=information,
                label=label,
                label_set=category_label_set,
                annotation_set=virtual_default_annotation_set,
            )

        elif isinstance(information, list) or isinstance(information, dict):
            # process multi Annotation Sets that are not part of the category Label Set
            label_set = document.category.project.get_label_set_by_name(label_or_label_set_name)

            if not isinstance(information, list):
                information = [information]

            for entry in information:  # represents one of pot. multiple annotation-sets belonging of one LabelSet
                virtual_annotation_set_id += 1
                virtual_annotation_set = AnnotationSet(
                    document=virtual_doc, label_set=label_set, id_=virtual_annotation_set_id
                )

                for label_name, extractions in entry.items():
                    label = document.category.project.get_label_by_name(label_name)
                    add_extractions_as_annotations(
                        document=virtual_doc,
                        extractions=extractions,
                        label=label,
                        label_set=label_set,
                        annotation_set=virtual_annotation_set,
                    )
    return virtual_doc


def load_pickle(pickle_name: str):
    """
    Load a pkl file or a pt (pytorch) file.

    First check if the .pkl file exists at ./konfuzio.MODEL_ROOT/pickle_name, if not then assumes it is at ./pickle_name
    Then, it assumes the .pkl file is compressed with bz2 and tries to extract and load it. If the pickle file is not
    compressed with bz2 then it will throw an OSError and we then try and load the .pkl file will dill. This will then
    throw an UnpicklingError if the file is not a pickle file, as expected.

    :param pickle_name:
    :return:
    """
    # https://stackoverflow.com/a/43006034/5344492
    import dill
    dill._dill._reverse_typemap['ClassType'] = type
    pickle_path = os.path.join(BASE_DIR, pickle_name)
    if not os.path.isfile(pickle_path):
        pickle_path = pickle_name

    if pickle_name.endswith('.pt'):
        import torch
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'

        with open(pickle_path, 'rb') as f:
            file_data = torch.load(pickle_path, map_location=torch.device(device))

        if isinstance(file_data, dict):
            from konfuzio.default_models import load_default_model
            from konfuzio.default_models.parameters_config import MODEL_PARAMETERS_TO_SAVE

            # verification of str in path can be removed after all models being updated with the model_type
            possible_names = ['_LabelAnnotationSetModel', '_DocumentModel', '_ParagraphModel', '_CustomDocumentModel',
                              '_SentenceModel']
            if ('model_type' in file_data.keys() and file_data['model_type'] in MODEL_PARAMETERS_TO_SAVE.keys()) or \
                    any([n in pickle_name for n in possible_names]):
                file_data = load_default_model(pickle_name)

            else:
                raise NameError("Model type not recognized.")

        else:
            with open(pickle_path, 'rb') as f:
                file_data = torch.load(f, map_location=torch.device(device))
    else:
        import bz2
        try:
            with bz2.open(pickle_path, 'rb') as f:
                file_data = cloudpickle.load(f)
                logger.info(f'{pickle_name} decompressed and loaded via cloudpickle.')
        except OSError:
            with open(pickle_path, 'rb') as f:
                file_data = cloudpickle.load(f)
                logger.info(f'{pickle_name} loaded via cloudpickle.')

    return file_data
