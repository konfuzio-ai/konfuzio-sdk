"""Handle data from the API."""

import json
import logging
import os
import pathlib
import shutil
import time
from copy import deepcopy
from datetime import tzinfo
from typing import Dict, Optional, List, Union, Tuple

import dateutil.parser
from konfuzio_sdk import KONFUZIO_HOST, DATA_ROOT, KONFUZIO_PROJECT_ID, FILE_ROOT
from konfuzio_sdk.api import (
    get_document_details,
    konfuzio_session,
    download_file_konfuzio_api,
    get_meta_of_files,
    get_project_labels,
    post_document_annotation,
    get_project_label_sets,
    retry_get,
    delete_document_annotation,
    upload_file_konfuzio_api,
    create_label,
    update_file_konfuzio_api,
)
from konfuzio_sdk.utils import is_file, convert_to_bio_scheme

logger = logging.getLogger(__name__)


class Data(object):
    """Collect general functionality to work with data from API."""

    id = None

    def __eq__(self, other):
        """Compare any point of data with their ID, overwrite if needed."""
        return hasattr(other, 'id') and self.id and other.id and self.id == other.id

    def __hash__(self):
        """Return hash(self)."""
        return hash(str(self.id))


class AnnotationSet(Data):
    """Represent an Annotation Set - group of annotations."""

    def __init__(self, id, document, label_set, annotations, **kwargs):
        """
        Create an Annotation Set.

        :param id: ID of the annotation set
        :param document: Document where the annotation set belongs
        :param label_set: Label set where the annotation set belongs
        :param annotations: Annotations of the annotation set
        """
        self.id = id
        self.document = document
        self.label_set = label_set
        self.annotations = annotations
        _annotations = [x for x in annotations if x.start_offset and x.end_offset]
        if len(_annotations) > 0:
            self.start_offset = min(x.start_offset for x in _annotations)
            self.end_offset = max(x.end_offset for x in _annotations)
        else:
            self.start_offset = None
            self.end_offset = None


class LabelSet(Data):
    """A Label Set is a group of labels."""

    def __init__(
        self,
        project,
        id: int,
        name: str,
        name_clean: str,
        labels: List[int],
        is_default=False,
        categories: List["Category"] = [],
        has_multiple_annotation_sets=False,
        **kwargs,
    ):
        """
        Create a named Label Set.

        :param project: Project where the Label Set belongs
        :param id: ID of the Label Set
        :param name: Name of Label Set
        :param name_clean: Normalized name of the Label Set
        :param labels: Labels that belong to the Label Set (IDs)
        :param is_default: Bool for the Label Set to be the default one in the project
        :param categories: Categories to which the LabelSet belongs
        :param has_multiple_annotation_sets: Bool to allow the Label Set to have different annotation sets in a document
        """
        self.id = id
        self.name = name
        self.name_clean = name_clean
        self.is_default = is_default
        if 'default_label_sets' in kwargs:
            self.categories = kwargs['default_label_sets']
        elif 'default_section_labels' in kwargs:
            self.categories = kwargs['default_section_labels']
        else:
            self.categories = categories
        self.has_multiple_annotation_sets = has_multiple_annotation_sets

        if 'has_multiple_sections' in kwargs:
            self.has_multiple_annotation_sets = kwargs['has_multiple_sections']

        self.project: Project = project

        self.labels: List[Label] = []
        project.add_label_set(self)

        for label in labels:
            if isinstance(label, int):
                label = self.project.get_label_by_id(id=label)
            self.add_label(label)

    def __repr__(self):
        """Return string representation of the Label Set."""
        return f'{self.name} ({self.id})'

    def add_label(self, label):
        """
        Add label to Label Set, if it does not exist.

        :param label: Label ID to be added
        """
        if label not in self.labels:
            self.labels.append(label)


class Category(LabelSet):
    """A Category is used to group documents."""

    def __init__(self, *args, **kwargs):
        """
        Create a named Category.

        A Category is also a Label Set but cannot have other categories associated to it.
        """
        LabelSet.__init__(self, *args, **kwargs)

        self.is_default = True
        self.has_multiple_annotation_sets = False
        self.categories = []
        self.project.add_category(self)


class Label(Data):
    """A label is the name of a group of individual pieces of information annotated in a type of document."""

    def __init__(
        self,
        project,
        id: Union[int, None] = None,
        text: str = None,
        get_data_type_display: str = None,
        text_clean: str = None,
        description: str = None,
        label_sets: List[LabelSet] = [],
        has_multiple_top_candidates: bool = False,
        *initial_data,
        **kwargs,
    ):
        """
        Create a named Label.

        :param project: Project where the label belongs
        :param id: ID of the label
        :param text: Name of the label
        :param get_data_type_display: Data type of the label
        :param text_clean: Normalized name of the label
        :param description: Description of the label
        :param label_sets: Label sets that use this label
        """
        self.id = id
        self.name = text
        self.name_clean = text_clean
        self.data_type = get_data_type_display
        self.description = description
        self.has_multiple_top_candidates = has_multiple_top_candidates
        self.threshold = kwargs.get('threshold', 0.1)

        self.project: Project = project
        self._correct_annotations_indexed = None

        project.add_label(self)
        if label_sets:
            [x.add_label(self) for x in label_sets]

    def __repr__(self):
        """Return string representation."""
        return self.name

    @property
    def label_sets(self):
        """Get the label sets in which this label is used."""
        label_sets = [x for x in self.project.label_sets if self in x.labels]
        return label_sets

    @property
    def annotations(self):
        """
        Add annotation to label.

        :return: Annotations
        """
        annotations = []
        for document in self.project.documents:
            annotations += document.annotations(label=self)
        return annotations

    def add_label_set(self, label_set: 'LabelSet'):
        """
        Add label set to label, if it does not exist.

        :param label_set: Label set to add
        """
        if label_set not in self.label_sets:
            self.label_sets.append(label_set)
        return self

    @property
    def correct_annotations(self):
        """Return correct annotations."""
        return [annotation for annotation in self.annotations if annotation.is_correct]

    @property
    def documents(self) -> List['Document']:
        """Return all documents which contain annotations of this label."""
        relevant_id = list(set([anno.document.id for anno in self.annotations]))
        return [doc for doc in self.project.documents if (doc.id in relevant_id)]

    def save(self) -> bool:
        """
        Save Label online.

        If no label sets are specified, the label is associated with the first default label set of the project.

        :return: True if the new label was created.
        """
        new_label_added = False
        try:
            if len(self.label_sets) == 0:
                prj_label_sets = self.project.label_sets
                label_set = [t for t in prj_label_sets if t.is_default][0]
                label_set.add_label(self)

            response = create_label(
                project_id=self.project.id,
                label_name=self.name,
                description=self.description,
                has_multiple_top_candidates=self.has_multiple_top_candidates,
                data_type=self.data_type,
                label_sets=self.label_sets,
            )
            self.id = response
            new_label_added = True
        except Exception:
            logger.error(f'Not able to save label {self.name}.')

        return new_label_added


class Annotation(Data):
    """
    An annotation is ~a single piece~ of a set of characters and/or bounding boxes that a label has been assigned to.

    One annotation can have mul. chr., words, lines, areas.
    """

    def __init__(
        self,
        start_offset: int,
        end_offset: int,
        label=None,
        is_correct: bool = False,
        revised: bool = False,
        id: int = None,
        accuracy: float = None,
        document=None,
        annotation_set=None,
        label_set_text=None,
        translated_string=None,
        label_set_id=None,
        *initial_data,
        **kwargs,
    ):
        """
        Initialize the Annotation.

        :param start_offset: Start of the offset string (int)
        :param end_offset: Ending of the offset string (int)
        :param label: ID of the label or Label object
        :param is_correct: If the annotation is correct or not (bool)
        :param revised: If the annotation is revised or not (bool)
        :param id: ID of the annotation (int)
        :param accuracy: Accuracy of the annotation (float)
        :param document: Document to annotate
        :param annotation_set: Annotation set of the document where the label belongs
        :param label_set_text: Name of the label set where the label belongs
        :param translated_string: Translated string
        :param label_set_id: ID of the label set where the label belongs
        """
        self.bottom = None  # Information about BoundingBox of text sequence
        self.document = document
        if document:
            document.add_annotation(self)
        self.end_offset = end_offset
        self.id = id  # Annotations can have None id, if they are not saved online and are only available locally
        self.is_correct = is_correct
        self.accuracy = accuracy

        if isinstance(label, int):
            self.label: Label = self.document.project.get_label_by_id(label)
        else:
            self.label: Label = label

        self.label_set = None
        self.define_annotation_set = True
        # if no label_set_id we check if is passed by section_label_id
        if label_set_id is None:
            label_set_id = kwargs.get('section_label_id')

        # handles association to an annotation set if the annotation belongs to a category
        if isinstance(label_set_id, int):
            self.label_set: LabelSet = self.document.project.get_label_set_by_id(label_set_id)
            if self.label_set.is_default:
                self.define_annotation_set = False

        if annotation_set is None:
            annotation_set = kwargs.get('section')
        self.annotation_set = annotation_set

        self.revised = revised
        self.start_offset = start_offset
        self.normalized = None
        self.translated_string = translated_string
        self.top = None
        self.top = None
        self.x0 = None
        self.x1 = None
        self.y0 = None
        self.y1 = None
        bbox = kwargs.get('bbox')
        if bbox:
            self.top = bbox.get('top')
            self.bottom = bbox.get('bottom')
            self.x0 = bbox.get('x0')
            self.x1 = bbox.get('x1')
            self.y0 = bbox.get('y0')
            self.y1 = bbox.get('y1')

        self.bboxes = kwargs.get('bboxes', None)
        self.selection_bbox = kwargs.get('selection_bbox', None)
        self.page_number = kwargs.get('page_number', None)

        # if no label_set_text we check if is passed by section_label_text
        if label_set_text is None:
            label_set_text = kwargs.get('section_label_text')

        self.annotation_set_text = label_set_text

    def __repr__(self):
        """Return string representation."""
        if self.label:
            return f'{self.label.name} ({self.start_offset}, {self.end_offset}): {self.offset_string}'
        else:
            return f'No Label ({self.start_offset}, {self.end_offset})'

    @property
    def is_online(self) -> Optional[int]:
        """Define if the Annotation is saved to the server."""
        return self.id is not None

    @property
    def offset_string(self) -> str:
        """View the string representation of the Annotation."""
        return self.document.text[self.start_offset : self.end_offset]

    def get_link(self):
        """Get link to the annotation in the SmartView."""
        return KONFUZIO_HOST + '/a/' + str(self.id)

    def save(self, document_annotations: list = None) -> bool:
        """
        Save Annotation online.

        If there is already an annotation in the same place as the current one, we will not be able to save the current
        annotation.

        In that case, we get the id of the original one to be able to track it.
        The verification of the duplicates is done by checking if the offsets and label match with any annotations
        online.
        To be sure that we are comparing with the information online, we need to have the document updated.
        The update can be done after the request (per annotation) or the updated annotations can be passed as input
        of the function (advisable when dealing with big documents or documents with many annotations).

        :param document_annotations: Annotations in the document (list)
        :return: True if new Annotation was created
        """
        new_annotation_added = False
        if not self.is_online:
            response = post_document_annotation(
                document_id=self.document.id,
                start_offset=self.start_offset,
                end_offset=self.end_offset,
                label_id=self.label.id,
                label_set_id=self.label_set.id,
                accuracy=self.accuracy,
                is_correct=self.is_correct,
                revised=self.revised,
                annotation_set=self.annotation_set,
                define_annotation_set=self.define_annotation_set,
                bboxes=self.bboxes,
                selection_bbox=self.selection_bbox,
                page_number=self.page_number,
            )
            if response.status_code == 201:
                json_response = json.loads(response.text)
                self.id = json_response['id']
                new_annotation_added = True
            elif response.status_code == 403:
                logger.error(response.text)
                try:
                    if 'In one project you cannot label the same text twice.' in response.text:
                        if document_annotations is None:
                            # get the annotation
                            self.document.update()
                            document_annotations = self.document.annotations()
                        # get the id of the existing annotation
                        is_duplicated = False
                        for annotation in document_annotations:
                            if (
                                annotation.start_offset == self.start_offset
                                and annotation.end_offset == self.end_offset
                                and annotation.label == self.label
                            ):
                                logger.error(f'ID of annotation online: {annotation.id}')
                                self.id = annotation.id
                                is_duplicated = True
                                break

                        # if there isn't a perfect match, the current annotation is considered incorrect
                        if not is_duplicated:
                            self.is_correct = False

                        new_annotation_added = False
                    else:
                        logger.exception(f'Unknown issue to create Annotation {self} in {self.document}')
                except KeyError:
                    logger.error(f'Not able to save annotation online: {response}')
        return new_annotation_added

    def delete(self) -> None:
        """Delete Annotation online."""
        for index, annotation in enumerate(self.document._annotations):
            if annotation == self:
                del self.document._annotations[index]

        if self.is_online:
            response = delete_document_annotation(document_id=self.document.id, annotation_id=self.id)
            if response.status_code == 204:
                self.id = None
            else:
                logger.exception(response.text)


class Document(Data):
    """Access the information about one document, which is available online."""

    session = konfuzio_session()
    annotation_class = Annotation

    def __init__(
        self,
        id: Union[int, None] = None,
        file_path: str = None,
        file_url: str = None,
        status=None,
        data_file_name: str = None,
        project=None,
        is_dataset: bool = None,
        dataset_status: int = None,
        updated_at: tzinfo = None,
        bbox: Dict = None,
        number_of_pages: int = None,
        *initial_data,
        **kwargs,
    ):
        """
        Check if the document root is available, otherwise create it.

        :param id: ID of the Document
        :param file_path: Path to a local file from which generate the Document object
        :param file_url: URL of the document
        :param status: Status of the document
        :param data_file_name: File name of the document
        :param project: Project where the document belongs
        :param is_dataset: Is dataset or not. (bool)
        :param dataset_status: Dataset status of the document (e.g. training)
        :param updated_at: Updated information
        :param bbox: Bounding box information per character in the PDF (dict)
        :param number_of_pages: Number of pages in the document
        """
        self.file_path = file_path
        self.annotation_file_path = None  # path to json containing the Annotations of a Document
        self.annotation_set_file_path = None  # path to json containing the Annotation Sets of a Document
        self._annotations: List[Annotation] = []
        # Bounding box information per character in the PDF
        # Only access this via self.get_bbox
        self.bbox = bbox
        self.file_url = file_url
        self.is_dataset = is_dataset
        self.dataset_status = dataset_status
        self.number_of_pages = number_of_pages
        if project:
            self.category = project.get_category_by_id(kwargs.get('category_template', None))
        self.id = id
        self.updated_at = None
        if updated_at:
            self.updated_at = dateutil.parser.isoparse(updated_at)

        self.name = data_file_name
        self.ocr_file_path = None  # Path to the ocred pdf (sandwich pdf)
        self.image_paths = []  # Path to the images
        self.status = status  # status of document online
        if self.is_without_errors and project:
            self.project = project
            project.add_document(self)  # check for duplicates by ID before adding the document to the project
        else:
            self.project = None
        self.annotation_file_path = os.path.join(self.root, 'annotations.json5')
        self.txt_file_path = os.path.join(self.root, 'document.txt')
        self.hocr_file_path = os.path.join(self.root, 'document.hocr')
        self.pages_file_path = os.path.join(self.root, 'pages.json5')

        self.bbox_file_path = None
        if os.path.exists(os.path.join(self.root, 'bbox.json5')):
            self.bbox_file_path = os.path.join(self.root, 'bbox.json5')

        self.bio_scheme_file_path = None

        self.text = kwargs.get('text')
        self.hocr = kwargs.get('hocr')

        # prepare local setup for document
        pathlib.Path(self.root).mkdir(parents=True, exist_ok=True)

    def __repr__(self):
        """Return the name of the document incl. the ID."""
        return f'{self.id}: {self.name}'

    @property
    def is_online(self) -> Optional[int]:
        """Define if the Document is saved to the server."""
        return self.id is not None

    def offset(self, start_offset: int, end_offset: int) -> List[Annotation]:
        """
        Convert an offset to a list of annotations.

        :param start_offset: Starting of the offset string (int)
        :param end_offset: Ending of the offset string (int)
        :return: annotations
        """
        if start_offset >= end_offset and start_offset >= 0 and end_offset <= len(self.text):
            raise ValueError(f'End offset ({end_offset}) must start after start_offset ({start_offset}).')

        filtered_annotations = []
        next_annotation_id = 0
        correct_annotations = self.annotations(start_offset=start_offset, end_offset=end_offset)
        chr_idx = start_offset
        while start_offset <= chr_idx <= end_offset:
            try:
                next_annotation = correct_annotations[next_annotation_id]
            except IndexError:  # in the offset we are not able to find labeled annotations
                next_annotation = None

            if next_annotation and next_annotation.start_offset > chr_idx:
                # create an artificial Annotation without label
                artificial_anno = self.annotation_class(
                    start_offset=chr_idx,
                    end_offset=next_annotation.start_offset,
                    document=self,
                    label=None,
                    is_correct=False,
                )
                filtered_annotations.append(artificial_anno)
                filtered_annotations.append(next_annotation)
                chr_idx = next_annotation.end_offset
                next_annotation_id += 1
            elif next_annotation is None:  # reached the end of the full offset or did not find any correct annotations
                # create an artificial Annotation without label
                artificial_anno = self.annotation_class(
                    start_offset=chr_idx, end_offset=end_offset, document=self, label=None, is_correct=False
                )
                if end_offset > chr_idx:  # in rare cases the end and start offset are equal
                    filtered_annotations.append(artificial_anno)
                chr_idx = end_offset + 1
            else:
                filtered_annotations.append(next_annotation)
                chr_idx = next_annotation.end_offset
                next_annotation_id += 1

        return filtered_annotations

    def annotations(
        self, label: Label = None, use_correct: bool = True, start_offset: int = None, end_offset: int = None
    ) -> List:
        """
        Filter available annotations. Exclude custom_offset_string annotations.

        You can specific an offset of a document, to filter the annotations by

        :param label: Label for which to filter the annotations
        :param use_correct: If to filter by correct annotations
        :param start_offset: Starting of the offset string (int)
        :param end_offset: Ending of the offset string (int)
        :return: Annotations in the document.
        """
        annotations = []
        for annotation in self._annotations:
            if annotation.start_offset is None or annotation.end_offset is None:
                continue
            # filter by correct information
            if (use_correct and annotation.is_correct) or not use_correct:
                # filter by start and end offset, be aware that this approach to filter annotations will include the
                if start_offset and end_offset:  # if the start and end offset are specified
                    latest_start = max(annotation.start_offset, start_offset)
                    earliest_end = min(annotation.end_offset, end_offset)
                    is_overlapping = latest_start - earliest_end <= 0
                else:
                    is_overlapping = True

                # filter by label
                if label is not None:
                    if annotation.label == label and is_overlapping:
                        annotations.append(annotation)
                elif is_overlapping:
                    annotations.append(annotation)

        return annotations

    @property
    def is_without_errors(self) -> bool:
        """Check if the document can be used for training clf."""
        if self.status is None:
            # Assumption: any Document without status, might be ok
            return True
        else:
            return self.status[0] == 2

    @property
    def root(self):
        """Get the path to the folder where all the document information is cached locally."""
        if self.project:
            return os.path.join(self.project.data_root, 'pdf', str(self.id))
        else:
            return os.path.join(FILE_ROOT, str(self.id))

    def get_file(self, ocr_version: bool = True, update: bool = False):
        """
        Get OCR version of the original file.

        :param ocr_version: Bool to get the ocr version of the pdf
        :param update: Update the downloaded file even if it is already available
        :return: Path to OCR or original file.
        """
        if self.is_without_errors and (not self.ocr_file_path or not is_file(self.ocr_file_path) or update):
            # for page_index in range(0, self.number_of_pages):
            filename = os.path.splitext(self.name)[0]
            if ocr_version:
                filename += '_ocr'

            filename += '.pdf'
            self.ocr_file_path = os.path.join(self.root, filename)
            if not is_file(self.ocr_file_path, raise_exception=False) or update:
                pdf_content = download_file_konfuzio_api(self.id, ocr=ocr_version, session=self.session)
                with open(self.ocr_file_path, 'wb') as f:
                    f.write(pdf_content)

        return self.ocr_file_path

    def get_images(self, update: bool = False):
        """
        Get document pages as png images.

        :param update: Update the downloaded images even they are already available
        :return: Path to OCR file.
        """
        session = konfuzio_session()

        self.image_paths = []
        for page in self.pages:

            if is_file(page['image'], raise_exception=False):
                self.image_paths.append(page['image'])
            else:
                page_path = os.path.join(self.root, f'page_{page["number"]}.png')
                self.image_paths.append(page_path)

                if not is_file(page_path, raise_exception=False) or update:
                    url = f'{KONFUZIO_HOST}{page["image"]}'
                    res = retry_get(session, url)
                    with open(page_path, 'wb') as f:
                        f.write(res.content)

    def get_document_details(self, update):
        """
        Get data from a document.

        :param update: Update the downloaded information even it is already available
        """
        self.annotation_file_path = os.path.join(self.root, 'annotations.json5')
        self.annotation_set_file_path = os.path.join(self.root, 'annotation_sets.json5')
        self.txt_file_path = os.path.join(self.root, 'document.txt')
        self.hocr_file_path = os.path.join(self.root, 'document.hocr')

        if update or not (
            is_file(self.annotation_file_path, raise_exception=False)
            and is_file(self.annotation_set_file_path, raise_exception=False)
            and is_file(self.txt_file_path, raise_exception=False)
            and is_file(self.pages_file_path, raise_exception=False)
        ):
            data = get_document_details(document_id=self.id, session=self.session, extra_fields='hocr')

            if data['text'] is None:
                # try get data again
                time.sleep(15)
                data = get_document_details(document_id=self.id, session=self.session, extra_fields='hocr')
                if data['text'] is None:
                    message = f'Document {self.id} is not fully processed yet. Please try again in some minutes.'
                    logger.error(message)
                    raise ValueError(message)

            raw_annotations = data['annotations']
            self.number_of_pages = data['number_of_pages']

            self.text = data['text']
            self.hocr = data['hocr'] or ''
            self.pages = data['pages']
            self._annotation_sets = data['sections']

            # write a file, even there are no annotations to support offline work
            with open(self.annotation_file_path, 'w') as f:
                json.dump(raw_annotations, f, indent=2, sort_keys=True)

            with open(self.annotation_set_file_path, 'w') as f:
                json.dump(data['sections'], f, indent=2, sort_keys=True)

            with open(self.txt_file_path, 'w', encoding="utf-8") as f:
                f.write(data['text'])

            with open(self.pages_file_path, 'w') as f:
                json.dump(data['pages'], f, indent=2, sort_keys=True)

            if self.hocr != '':
                with open(self.hocr_file_path, 'w', encoding="utf-8") as f:
                    f.write(data['hocr'])

        else:
            with open(self.txt_file_path, 'r', encoding="utf-8") as f:
                self.text = f.read()

            with open(self.annotation_file_path, 'rb') as f:
                raw_annotations = json.loads(f.read())

            with open(self.annotation_set_file_path, 'rb') as f:
                self._annotation_sets = json.loads(f.read())

            with open(self.pages_file_path, 'rb') as f:
                self.pages = json.loads(f.read())

            if is_file(self.hocr_file_path, raise_exception=False):
                # hocr might not be available (depends on the project settings)
                with open(self.hocr_file_path, 'r', encoding="utf-8") as f:
                    self.hocr = f.read()

        # add Annotations to the document and project
        if hasattr(self, 'project') and self.project:
            for raw_annotation in raw_annotations:
                if not raw_annotation['custom_offset_string']:
                    _ = self.annotation_class(document=self, **raw_annotation)
                else:
                    real_string = self.text[raw_annotation['start_offset'] : raw_annotation['end_offset']]
                    if real_string.replace(' ', '') == raw_annotation['offset_string'].replace(' ', ''):
                        _ = self.annotation_class(document=self, **raw_annotation)
                    else:
                        logger.warning(
                            f'Annotation {raw_annotation["id"]} is a custom string and, therefore, it will not be added'
                            f' to the document annotations {KONFUZIO_HOST}/a/{raw_annotation["id"]}.'
                        )

        return self

    def add_annotation(self, annotation, check_duplicate=True):
        """Add an annotation to a document.

        If check_duplicate is True, we only add an annotation after checking it doesn't exist in the document already.
        If check_duplicate is False, we add an annotation without checking, but it is considerably faster when the
        number of annotations in the document is large.

        :param annotation: Annotation to add in the document
        :param check_duplicate: If to check if the annotation already exists in the document
        :return: Input annotation.
        """
        if check_duplicate:
            if annotation not in self._annotations:
                self._annotations.append(annotation)
        else:
            self._annotations.append(annotation)
        return annotation

    def get_text_in_bio_scheme(self, update=False) -> List[Tuple[str, str]]:
        """
        Get the text of the document in the BIO scheme.

        :param update: Update the bio annotations even they are already available
        :return: list of tuples with each word in the text an the respective label
        """
        if not self.bio_scheme_file_path or not is_file(self.bio_scheme_file_path, raise_exception=False) or update:
            annotations = self.annotations()
            converted_text = []

            if len(annotations) > 0:
                annotations_in_doc = [
                    (annotation.start_offset, annotation.end_offset, annotation.label.name)
                    for annotation in annotations
                ]
                converted_text = convert_to_bio_scheme(self.text, annotations_in_doc)

            self.bio_scheme_file_path = os.path.join(self.root, 'bio_scheme.txt')

            with open(self.bio_scheme_file_path, 'w', encoding="utf-8") as f:
                for word, tag in converted_text:
                    f.writelines(word + ' ' + tag + '\n')
                f.writelines('\n')

        bio_annotations = []

        with open(self.bio_scheme_file_path, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                if line == '\n':
                    continue
                word, tag = line.replace('\n', '').split(' ')
                bio_annotations.append((word, tag))

        return bio_annotations

    def get_bbox(self, update=False):
        """
        Get bbox information per character of file.

        There are two ways to access it:
        - If the bbox attribute is set when creating the Document, it is returned immediately.
        - Otherwise, we open the file at bbox_file_path and return its content.

        In the second case, we do not store bbox as an attribute on Document because with many big
        documents this quickly fills the available memory. So it is first written to a file by
        get_document_details and then retrieved from that file when accessing it.

        :param update: Update the bbox information even if it's are already available
        :return: Bounding box information per character in the document.
        """
        if self.bbox is not None:
            return self.bbox

        if not self.bbox_file_path or not is_file(self.bbox_file_path, raise_exception=False) or update:
            self.bbox_file_path = os.path.join(self.root, 'bbox.json5')

            with open(self.bbox_file_path, 'w', encoding="utf-8") as f:
                data = get_document_details(document_id=self.id, session=self.session, extra_fields='bbox')
                json.dump(data['bbox'], f, indent=2, sort_keys=True)

        with open(self.bbox_file_path, 'r', encoding="utf-8") as f:
            bbox = json.loads(f.read())

        return bbox

    def save(self) -> bool:
        """
        Save or edit Document online.

        :return: True if the new document was created or existing document was updated.
        """
        document_saved = False
        category_template_id = None

        if hasattr(self, 'category') and self.category is not None:
            category_template_id = self.category.id

        if not self.is_online:
            response = upload_file_konfuzio_api(
                filepath=self.file_path,
                project_id=self.project.id,
                dataset_status=self.dataset_status,
                category_template_id=category_template_id,
            )
            if response.status_code == 201:
                self.id = json.loads(response.text)['id']
                document_saved = True
            else:
                logger.error(f'Not able to save document {self.file_path} online: {response.text}')
        else:
            response = update_file_konfuzio_api(
                document_id=self.id,
                file_name=self.name,
                dataset_status=self.dataset_status,
                category_template_id=category_template_id,
            )
            if response.status_code == 200:
                self.project.update_document(document=self)
                document_saved = True
            else:
                logger.error(f'Not able to update document {self.id} online: {response.text}')

        return document_saved

    def update(self):
        """Update document information."""
        self.delete()
        pathlib.Path(self.root).mkdir(parents=True, exist_ok=True)
        self._annotations = []
        self.get_document_details(update=True)
        return self

    def delete(self):
        """Delete all local information for the document."""
        try:
            shutil.rmtree(self.root)
        except FileNotFoundError:
            pass


class Project(Data):
    """Access the information of a project."""

    session = konfuzio_session()
    # classes are defined here to be able to redefine them if needed
    label_class = Label
    annotation_set_class = AnnotationSet
    label_set_class = LabelSet
    category_class = Category
    document_class = Document
    annotation_class = Annotation

    def __init__(self, id: int = KONFUZIO_PROJECT_ID, offline=False, data_root=False, **kwargs):
        """
        Set up the data using the Konfuzio Host.

        :param id: ID of the project
        :param offline: If to get the data from Konfuzio Host
        :param data_root: Path to the folder with the data. If not specified uses the data_root defined in the project,
        by default.
        """
        self.id = id
        self.categories: List[Category] = []
        self.label_sets: List[LabelSet] = []
        self.labels: List[Label] = []

        self.documents: List[Document] = []
        self.test_documents: List[Document] = []
        self.no_status_documents: List[Document] = []
        self.preparation_documents: List[Document] = []
        self.low_ocr_documents: List[Document] = []

        self.label_sets_file_path = None
        self.labels_file_path = None
        self.meta_file_path = None
        self.meta_data = None
        self._textcorpus = None
        self._correct_annotations_indexed = {}
        self.data_root = data_root if data_root else DATA_ROOT
        if not offline:
            self.make_paths()
            self.get()  # keep update to False, so once you have downloaded the data, don't do it again.
            self.load_annotation_sets()

    def __repr__(self):
        """Return string representation."""
        return f'Project {self.id}'

    def load_annotation_sets(self):
        """Load document annotation sets for all training and test documents."""
        label_set_mapper_dict = dict((x.id, x) for x in self.label_sets)
        for document in self.documents + self.test_documents:
            annotations = [(x.annotation_set, x) for x in document.annotations()]
            annotations_dict = {}
            for annotation_set_id, annotation in annotations:
                annotations_dict.setdefault(annotation_set_id, []).append(annotation)

            document_annotation_sets = []
            for annotation_set in document._annotation_sets:
                annotations = annotations_dict[annotation_set['id']] if annotation_set['id'] in annotations_dict else []
                annotation_set['label_set'] = label_set_mapper_dict[annotation_set['section_label']]
                # we only add the annotation_sets that match the category of the document
                # (ignore ghost annotation_sets that may exist)
                if (
                    annotation_set['label_set'] == document.category
                    or document.category in annotation_set['label_set'].categories
                ):
                    annotation_set_instance = self.annotation_set_class(
                        **annotation_set, document=document, annotations=annotations
                    )
                    document_annotation_sets.append(annotation_set_instance)
            document.annotation_sets = document_annotation_sets

            # Put legacy naming in place.
            document.sections = document.annotation_sets
            for section in document.sections:
                section.section_label = section.label_set

    def load_categories(self):
        """Load categories for all label sets in the project."""
        for label_set in self.label_sets:
            updated_list = []
            for category in label_set.categories:
                if isinstance(category, int):
                    updated_list.append(self.get_category_by_id(category))
                else:
                    updated_list.append(category)
            label_set.categories = updated_list

    def make_paths(self):
        """Create paths needed to store the project."""
        # create folders if not available
        # Ensure self.data_root exists
        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)
        pathlib.Path(self.data_root + '/pdf').mkdir(parents=True, exist_ok=True)

    def get(self, update=False):
        """
        Access meta information of the project.

        :param update: Update the downloaded information even it is already available
        """
        # if not self.meta_file_path or update:
        # add the labels first, before creating documents and annotations
        self.get_meta(update=update)
        self.get_labels(update=update)
        self.get_label_sets(update=update)
        self.get_categories(update=update)
        self.load_categories()
        self.clean_documents(update=update)
        self.get_documents(update=update)
        self.get_test_documents(update=update)

        return self

    def add_label_set(self, label_set: LabelSet):
        """
        Add label set to project, if it does not exist.

        :param label_set: Label Set to add in the project
        """
        if label_set not in self.label_sets:
            self.label_sets.append(label_set)

    def add_category(self, category: Category):
        """
        Add category to project, if it does not exist.

        :param category: Category to add in the project
        """
        if category not in self.categories:
            self.categories.append(category)

    def add_label(self, label: Label):
        """
        Add label to project, if it does not exist.

        :param label: Label to add in the project
        """
        if label not in self.labels:
            self.labels.append(label)

    def add_document(self, document):
        """
        Add document to project, if it does not exist.

        :param document: Document to add in the project
        """
        if (
            document
            not in self.documents
            + self.test_documents
            + self.no_status_documents
            + self.preparation_documents
            + self.low_ocr_documents
        ):
            if document.dataset_status == 2:
                self.documents.append(document)
            elif document.dataset_status == 3:
                self.test_documents.append(document)
            elif document.dataset_status == 0:
                self.no_status_documents.append(document)
            elif document.dataset_status == 1:
                self.preparation_documents.append(document)
            elif document.dataset_status == 4:
                self.low_ocr_documents.append(document)

    def update_document(self, document):
        """
        Update document in the project.

        Update can be in the dataset_status, name or category.
        First, we need to find the document (different list accordingly with dataset_status).
        Then, if we are just updating the name or category, we can change the fields in place.
        If we are updating the document dataset status, we need to move the document from the project list.

        :param document: Document to update in the project
        """
        current_status = document.dataset_status

        prj_docs = {
            0: self.no_status_documents,
            1: self.preparation_documents,
            2: self.documents,
            3: self.test_documents,
            4: self.low_ocr_documents,
        }

        # by default the status is None (even if not in the no_status_documents)
        previous_status = 0
        project_documents = []

        # get project list that contains the document
        for previous_status, project_list in prj_docs.items():
            if document in project_list:
                project_documents = project_list
                break

        # update name and category and get dataset status
        for doc in project_documents:
            if doc.id == document.id:
                doc.name = document.name
                doc.category = document.category
                break

        # if the document is new to the project, just add it
        if len(project_documents) == 0:
            doc = document

        # update project list if dataset status is different
        if current_status != previous_status:
            if doc in project_documents:
                project_documents.remove(doc)
            doc.dataset_status = current_status
            self.add_document(doc)

    def get_meta(self, update=False):
        """
        Get the list of all documents in the project and their information.

        :param update: Update the downloaded information even it is already available
        :return: Information of the documents in the project.
        """
        if not self.meta_data or update:
            self.meta_file_path = os.path.join(self.data_root, 'documents_meta.json5')

            if not is_file(self.meta_file_path, raise_exception=False) or update:
                self.meta_data = get_meta_of_files(self.session)
                with open(self.meta_file_path, 'w') as f:
                    json.dump(self.meta_data, f, indent=2, sort_keys=True)
            else:
                with open(self.meta_file_path, 'r') as f:
                    self.meta_data = json.load(f)

        return self.meta_data

    def get_categories(self, update=False):
        """
        Get Categories in the project.

        :param update: Update the downloaded information even it is already available
        :return: Categories in the project.
        """
        if not self.categories or update:
            if not self.label_sets:
                error_message = 'You need to get the label sets before getting the categories of the project.'
                logger.error(error_message)
                raise ValueError(error_message)

            for label_set in self.label_sets:
                if label_set.is_default:
                    temp_label_set = deepcopy(label_set)
                    temp_label_set.__dict__.pop('project', None)
                    self.category_class(project=self, **temp_label_set.__dict__)

        return self.categories

    def get_label_sets(self, update=False):
        """
        Get Label Sets in the project.

        :param update: Update the downloaded information even it is already available
        :return: Label Sets in the project.
        """
        if not self.label_sets or update:
            self.label_sets_file_path = os.path.join(self.data_root, 'label_sets.json5')
            if not is_file(self.label_sets_file_path, raise_exception=False) or update:
                label_sets_data = get_project_label_sets(session=self.session)
                if label_sets_data:
                    # the text of a document can be None
                    with open(self.label_sets_file_path, 'w') as f:
                        json.dump(label_sets_data, f, indent=2, sort_keys=True)
            else:
                with open(self.label_sets_file_path, 'r') as f:
                    label_sets_data = json.load(f)

            for label_set_data in label_sets_data:
                self.label_set_class(project=self, **label_set_data)

        return self.label_sets

    def get_labels(self, update=False):
        """
        Get ID and name of any label in the project.

        :param update: Update the downloaded information even it is already available
        :return: Labels in the project.
        """
        if not self.labels or update:
            self.labels_file_path = os.path.join(self.data_root, 'labels.json5')
            if not is_file(self.labels_file_path, raise_exception=False) or update:
                labels_data = get_project_labels(session=self.session)
                with open(self.labels_file_path, 'w') as f:
                    json.dump(labels_data, f, indent=2, sort_keys=True)
            else:
                with open(self.labels_file_path, 'r') as f:
                    labels_data = json.load(f)
            for label_data in labels_data:
                # Remove the project from label_data as we use the already present project reference.
                label_data.pop('project', None)
                self.label_class(project=self, **label_data)

        return self.labels

    def _init_document(self, document_data, document_list_cache, update):
        """
        Initialize Document.

        :param document_data: Document data
        :param document_list_cache: Cache with documents in the project
        :param update: Update the downloaded information even it is already available
        """
        if document_data['status'][0] != 2:
            logger.info(f"Document {document_data['id']} skipped due to: {document_data['status']}")
            return None

        needs_update = False  # basic assumption, document has not changed since the latest pull
        new_in_dataset = False
        if document_data['id'] not in [doc.id for doc in document_list_cache]:
            # it is a new document
            new_in_dataset = True
        else:
            # it might have been changed since our latest pull
            latest_change_online = dateutil.parser.isoparse(document_data['updated_at'])
            doc = [document for document in document_list_cache if document.id == document_data['id']][0]
            if doc.updated_at is None or (latest_change_online > doc.updated_at):
                needs_update = True

        if (new_in_dataset or needs_update) and update:
            data_path = os.path.join(self.data_root, 'df_data.pkl')
            test_data_path = os.path.join(self.data_root, 'df_test.pkl')
            feature_list_path = os.path.join(self.data_root, 'label_feature_list.pkl')

            if os.path.exists(data_path):
                os.remove(data_path)
            if os.path.exists(test_data_path):
                os.remove(test_data_path)
            if os.path.exists(feature_list_path):
                os.remove(feature_list_path)

        if (new_in_dataset and update) or (needs_update and update):
            doc = self.document_class(project=self, **document_data)
            doc.get_document_details(update=update)
            self.update_document(doc)
        else:
            doc = self.document_class(project=self, **document_data)
            doc.get_document_details(update=False)

    def get_documents(self, update=False):
        """
        Get all documents in a project which have been marked as available in the training dataset.

        Dataset status: training = 2

        :param update: Bool to update the meta-information from the project
        :return: training documents
        """
        document_list_cache = self.documents
        self.documents: List[Document] = []
        self.get_documents_by_status(dataset_statuses=[2], document_list_cache=document_list_cache, update=update)

        return self.documents

    def get_test_documents(self, update=False):
        """
        Get all documents in a project which have been marked as available in the test dataset.

        Dataset status: test = 3

        :param update: Bool to update the meta-information from the project
        :return: test documents
        """
        document_list_cache = self.test_documents
        self.test_documents: List[Document] = []
        self.get_documents_by_status(dataset_statuses=[3], document_list_cache=document_list_cache, update=update)

        return self.test_documents

    def get_documents_by_status(
        self, dataset_statuses: List[int] = [0], document_list_cache: List[Document] = [], update: bool = False
    ) -> List[Document]:
        """
        Get a list of documents with the specified dataset status from the project.

        Besides returning a list, the documents are also initialized in the project.
        They become accessible from the attributes of the class: self.test_documents, self.none_documents,...

        :param dataset_statuses: List of status of the documents to get
        :param document_list_cache: Cache with documents in the project
        :param update: Bool to update the meta-information from the project
        :return: Documents with the specified dataset status
        """
        documents = []

        for document_data in self.meta_data:
            if document_data['dataset_status'] in dataset_statuses:
                self._init_document(document_data, document_list_cache, update)

        if 0 in dataset_statuses:
            documents.extend(self.no_status_documents)
        if 1 in dataset_statuses:
            documents.extend(self.preparation_documents)
        if 2 in dataset_statuses:
            documents.extend(self.documents)
        if 3 in dataset_statuses:
            documents.extend(self.test_documents)
        if 4 in dataset_statuses:
            documents.extend(self.low_ocr_documents)

        return documents

    def clean_documents(self, update):
        """
        Clean the documents by removing those that have been removed from the App.

        Only if to update the project locally.

        :param update: Bool to update locally the documents in the project
        """
        if update:
            meta_data_document_ids = set([str(document['id']) for document in self.meta_data])
            existing_document_ids = set(
                [
                    str(document['id'])
                    for document in self.existing_meta_data
                    if document['dataset_status'] == 2 or document['dataset_status'] == 3
                ]
            )
            remove_document_ids = existing_document_ids.difference(meta_data_document_ids)
            for document_id in remove_document_ids:
                document_path = os.path.join(self.data_root, 'pdf', document_id)
                try:
                    shutil.rmtree(document_path)
                except FileNotFoundError:
                    pass

    def get_label_by_id(self, id: int) -> Label:
        """
        Return a label by ID.

        :param id: ID of the label to get.
        """
        for label in self.labels:
            if label.id == id:
                return label

    def get_label_set_by_id(self, id: int) -> LabelSet:
        """
        Return a Label Set by ID.

        :param id: ID of the Label Set to get.
        """
        for label_set in self.label_sets:
            if label_set.id == id:
                return label_set

    def get_category_by_id(self, id: int) -> Category:
        """
        Return a Category by ID.

        :param id: ID of the Category to get.
        """
        for category in self.categories:
            if category.id == id:
                return category

    def clean_meta(self):
        """Clean the meta-information about the Project, Labels, and Label Sets."""
        if self.meta_file_path:
            os.remove(self.meta_file_path)
        assert not is_file(self.meta_file_path, raise_exception=False)
        self.meta_data = None
        self.meta_file_path = None

        if self.labels_file_path:
            os.remove(self.labels_file_path)
        assert not is_file(self.labels_file_path, raise_exception=False)
        self.labels_file_path = None
        self.labels: List[Label] = []

        if self.label_sets_file_path:
            os.remove(self.label_sets_file_path)
        assert not is_file(self.label_sets_file_path, raise_exception=False)
        self.label_sets_file_path = None
        self.label_sets: List[LabelSet] = []

    def update(self):
        """Update the project and all documents by downloading them from the host. Note : It will not work offline."""
        # make sure you always update any changes to Labels, ProjectMeta
        self.existing_meta_data = self.meta_data  # get meta data of what currently exists locally
        self.clean_meta()
        self.get(update=True)
        self.load_annotation_sets()

        return self
