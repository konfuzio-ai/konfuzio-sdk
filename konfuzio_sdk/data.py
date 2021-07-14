"""Handle data from the API."""

import json
import logging
import os
import pathlib
import shutil
from datetime import tzinfo
from typing import Dict, Optional, List, Union

import dateutil.parser
from konfuzio_sdk import KONFUZIO_HOST, DATA_ROOT, KONFUZIO_PROJECT_ID, FILE_ROOT
from konfuzio_sdk.api import (
    get_document_details,
    konfuzio_session,
    download_file_konfuzio_api,
    get_meta_of_files,
    get_project_labels,
    post_document_annotation,
    get_project_templates,
    retry_get,
    delete_document_annotation,
    upload_file_konfuzio_api,
    create_label,
    update_file_status_konfuzio_api,
)
from konfuzio_sdk.utils import is_file

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


class Section(Data):
    """Represent a Section - group of annotations."""

    def __init__(self, id, document, template, annotations, **kwargs):
        """
        Create a section.

        :param id: ID of the section
        :param document: Document where the section belongs
        :param template: Template
        :param annotations: Annotations of the section
        """
        self.id = id
        self.document = document
        self.template = template
        self.annotations = annotations
        _annotations = [x for x in annotations if x.start_offset and x.end_offset]
        if len(_annotations) > 0:
            self.start_offset = min(x.start_offset for x in _annotations)
            self.end_offset = max(x.end_offset for x in _annotations)
        else:
            self.start_offset = None
            self.end_offset = None


class Template(Data):
    """A template is a set of labels."""

    def __init__(
        self,
        project,
        id: int,
        name: str,
        name_clean: str,
        labels: List[int],
        is_default=False,
        default_templates: ["Template"] = [],
        has_multiple_sections=False,
        **kwargs,
    ):
        """
        Create a named template.

        :param project: Project where the template belongs
        :param id: ID of the template
        :param name: Name of the template
        :param name_clean: Normalized name of the template
        :param labels: Labels that belong to the template (IDs)
        :param is_default: Bool for template to be the default one in the project
        :param default_templates: Default templates to which belongs
        :param has_multiple_sections: Bool to allow the template to have different sections in a document
        """
        self.id = id
        self.name = name
        self.name_clean = name_clean
        self.is_default = is_default
        if 'default_template' in kwargs:
            self.default_templates = [kwargs['default_template']]
        else:
            self.default_templates = default_templates
        self.has_multiple_sections = has_multiple_sections
        self.project: Project = project
        project.add_template(self)
        self.labels: List[Label] = []
        for label_id in labels:
            label = self.project.get_label_by_id(id=label_id)
            self.add_label(label)

    def __repr__(self):
        """Return string representation of the template."""
        return f'{self.name} ({self.id})'

    def add_label(self, label):
        """
        Add label to template, if it does not exist.

        :param label: Label ID to be added
        """
        if label not in self.labels:
            self.labels.append(label)


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
        templates: List[Template] = [],
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
        :param templates: Templates that use this label
        """
        self.id = id
        self.name = text
        self.name_clean = text_clean
        self.data_type = get_data_type_display
        self.description = description
        self.has_multiple_top_candidates = has_multiple_top_candidates

        self.project: Project = project
        self._correct_annotations_indexed = None

        project.add_label(self)
        if templates:
            [x.add_label(self) for x in templates]

    def __repr__(self):
        """Return string representation."""
        return self.name

    @property
    def templates(self):
        """Get the templates in which this label is used."""
        templates = [x for x in self.project.templates if self in x.labels]
        return templates

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

    def add_template(self, template):
        """
        Add template to label, if it does not exist.

        :param template: Template to add
        """
        if template not in self.templates:
            self.templates.append(template)
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

        If no templates are specified, the label is associated with the first default template of the project.

        :return: True if the new label was created.
        """
        new_label_added = False
        try:
            if len(self.templates) == 0:
                prj_templates = self.project.templates
                default_template = [t for t in prj_templates if t.is_default][0]
                default_template.add_label(self)

            response = create_label(project_id=self.project.id,
                                    label_name=self.name,
                                    description=self.description,
                                    has_multiple_top_candidates=self.has_multiple_top_candidates,
                                    data_type=self.data_type,
                                    templates=self.templates,
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
        section=None,
        template_text=None,
        translated_string=None,
        template_id=None,
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
        :param section: Section of the document where the label belongs
        :param template_text: Name of the template where the label belongs
        :param translated_string: Translated string
        :param template_id: ID of the template where the label belongs
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

        self.template = None
        self.define_section = True
        # if no template_id we check if is passed by section_label_id
        if template_id is None:
            template_id = kwargs.get('section_label_id')

        if isinstance(template_id, int):
            self.template: Template = self.document.project.get_template_by_id(template_id)
            if self.template.is_default:
                self.define_section = False

        self.revised = revised
        self.section = section
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

        # if no template_text we check if is passed by section_label_text
        if template_text is None:
            template_text = kwargs.get('section_label_text')

        self.section_text = template_text

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
        return 'https://app.konfuzio.com/a/' + str(self.id)

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
                template_id=self.template.id,
                accuracy=self.accuracy,
                is_correct=self.is_correct,
                revised=self.revised,
                section=self.section,
                define_section=self.define_section
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
                            if annotation.start_offset == self.start_offset and \
                                    annotation.end_offset == self.end_offset and annotation.label == self.label:
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
        self.section_file_path = None  # path to json containing the Sections of a Document
        self._annotations: List[Annotation] = []
        # Bounding box information per character in the PDF
        # Only access this via self.get_bbox
        self.bbox = bbox
        self.file_url = file_url
        self.is_dataset = is_dataset
        self.dataset_status = dataset_status
        self.number_of_pages = number_of_pages
        if project:
            self.category_template = project.get_template_by_id(kwargs.get('category_template', None))
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

        self.text = kwargs.get('text')
        self.hocr = kwargs.get('hocr')
        self.txt_file_path = None  # path to local text file containing the document text

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

    def get_file(self, update: bool = False):
        """
        Get OCR version of the original file.

        :param update: Update the downloaded file even if it is already available
        :return: Path to OCR file.
        """
        if self.is_without_errors and (not self.ocr_file_path or update):
            for page_index in range(0, self.number_of_pages):
                filename = os.path.splitext(self.name)[0] + '_ocr.pdf'
                self.ocr_file_path = os.path.join(self.root, filename)
                if not is_file(self.ocr_file_path, raise_exception=False) or update:
                    pdf_content = download_file_konfuzio_api(self.id, session=self.session)
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
        self.section_file_path = os.path.join(self.root, 'sections.json5')
        self.txt_file_path = os.path.join(self.root, 'document.txt')
        self.hocr_file_path = os.path.join(self.root, 'document.hocr')

        if update or not (
            is_file(self.annotation_file_path, raise_exception=False)
            and is_file(self.section_file_path, raise_exception=False)
            and is_file(self.txt_file_path, raise_exception=False)
            and is_file(self.pages_file_path, raise_exception=False)
        ):

            data = get_document_details(document_id=self.id, session=self.session)
            raw_annotations = data['annotations']
            self.number_of_pages = data['number_of_pages']
            self.text = data['text']
            self.hocr = data['hocr'] or ''
            self.pages = data['pages']
            self._sections = data['sections']

            # write a file, even there are no annotations to support offline work
            with open(self.annotation_file_path, 'w') as f:
                json.dump(raw_annotations, f, indent=2, sort_keys=True)

            with open(self.section_file_path, 'w') as f:
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

            with open(self.section_file_path, 'rb') as f:
                self._sections = json.loads(f.read())

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
                            f'Annotation {raw_annotation["id"]} is a custom string and, therefore, it will not be used '
                            f'in training {KONFUZIO_HOST}/a/{raw_annotation["id"]}.'
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

    def get_bbox(self):
        """
        Get bbox information per character of file.

        There are two ways to access it:
        - If the bbox attribute is set when creating the Document, it is returned immediately.
        - Otherwise, we open the file at bbox_file_path and return its content.

        In the second case, we do not store bbox as an attribute on Document because with many big
        documents this quickly fills the available memory. So it is first written to a file by
        get_document_details and then retrieved from that file when accessing it.

        :return: Bounding box information per character in the document.
        """
        if self.bbox is not None:
            return self.bbox

        if not self.bbox_file_path:
            self.bbox_file_path = os.path.join(self.root, 'bbox.json5')

            with open(self.bbox_file_path, 'w') as f:
                data = get_document_details(document_id=self.id, session=self.session)
                json.dump(data['bbox'], f, indent=2, sort_keys=True)

        with open(self.bbox_file_path, 'rb') as f:
            bbox = json.loads(f.read())

        return bbox

    def save(self) -> bool:
        """
        Save or update Document online.

        :return: True if the new document was created or existing document was updated.
        """
        document_saved = False
        if not self.is_online:
            response = upload_file_konfuzio_api(self.file_path,
                                                project_id=self.project.id,
                                                dataset_status=self.dataset_status)
            if response.status_code == 201:
                self.id = json.loads(response.text)['id']
                document_saved = True
            else:
                logger.error(f'Not able to save document  {self.file_path} online: {response.text}')
        else:
            response = update_file_status_konfuzio_api(document_id=self.id,
                                                       dataset_status=self.dataset_status,
                                                       file_name=self.name,
                                                       category_template=self.category_template.id)
            if response.status_code == 200:
                document_saved = True
                self.project.update_document(document=self)
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
        shutil.rmtree(self.root)


class Project(Data):
    """Access the information of a project."""

    session = konfuzio_session()
    # classes are defined here to be able to redefine them if needed
    label_class = Label
    section_class = Section
    template_class = Template
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
        self.templates: List[Template] = []
        self.labels: List[Label] = []

        self.documents: List[Document] = []
        self.test_documents: List[Document] = []
        self.no_status_documents: List[Document] = []
        self.preparation_documents: List[Document] = []
        self.low_ocr_documents: List[Document] = []

        self.templates_file_path = None
        self.labels_file_path = None
        self.meta_file_path = None
        self.meta_data = None
        self._textcorpus = None
        self._correct_annotations_indexed = {}
        self.data_root = data_root if data_root else DATA_ROOT
        if not offline:
            self.make_paths()
            self.get()  # keep update to False, so once you have downloaded the data, don't do it again.
        self.load_sections()

    def __repr__(self):
        """Return string representation."""
        return f'Project {self.id}'

    def load_sections(self):
        """Load document sections for all training and test documents."""
        template_mapper_dict = dict((x.id, x) for x in self.templates)
        for document in self.documents + self.test_documents:
            annotations = [(x.section, x) for x in document.annotations()]
            annotations_dict = {}
            for section_id, annotation in annotations:
                annotations_dict.setdefault(section_id, []).append(annotation)

            document_sections = []
            for section in document._sections:
                annotations = annotations_dict[section['id']] if section['id'] in annotations_dict else []
                section['template'] = template_mapper_dict[section['section_label']]
                # we only add the sections that match the category of the document
                # (ignore ghost sections that may exist)
                if section['template'] == document.category_template or \
                        document.category_template in section['template'].default_templates:
                    section_instance = self.section_class(**section, document=document, annotations=annotations)
                    document_sections.append(section_instance)
            document.sections = document_sections

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
        self.get_templates(update=update)
        self.clean_documents(update=update)
        self.get_documents(update=update)
        self.get_test_documents(update=update)

        return self

    def add_template(self, template: Template):
        """
        Add template to project, if it does not exist.

        :param template: Template to add in the project
        """
        if template not in self.templates:
            self.templates.append(template)

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
        if document not in self.documents + self.test_documents + self.no_status_documents + \
                self.preparation_documents + self.low_ocr_documents:
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

        prj_docs = {0: self.no_status_documents,
                    1: self.preparation_documents,
                    2: self.documents,
                    3: self.test_documents,
                    4: self.low_ocr_documents}

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
                doc.category_template = document.category_template
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

    def get_templates(self, update=False):
        """
        Get ID and name of any Template in the project.

        :param update: Update the downloaded information even it is already available
        :return: Templates in the project.
        """
        if not self.templates or update:
            self.templates_file_path = os.path.join(self.data_root, 'templates.json5')
            if not is_file(self.templates_file_path, raise_exception=False) or update:
                templates_data = get_project_templates(session=self.session)
                if templates_data:
                    # the text of a document can be None
                    with open(self.templates_file_path, 'w') as f:
                        json.dump(templates_data, f, indent=2, sort_keys=True)
            else:
                with open(self.templates_file_path, 'r') as f:
                    templates_data = json.load(f)

            for template_data in templates_data:
                self.template_class(project=self, **template_data)

        # Make default_template an Template instance
        for template in self.templates:
            updated_list = []
            for default_template in template.default_templates:
                if isinstance(default_template, int):
                    updated_list.append(self.get_template_by_id(default_template))
                else:
                    updated_list.append(default_template)
            template.default_templates = updated_list
        return self.templates

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
        self.get_documents_from_project(dataset_statuses=[2], document_list_cache=document_list_cache, update=update)

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
        self.get_documents_from_project(dataset_statuses=[3], document_list_cache=document_list_cache, update=update)

        return self.test_documents

    def get_documents_from_project(self, dataset_statuses: List[int] = [0], document_list_cache: List[Document] = [],
                                   update: bool = False) -> List[Document]:
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
            existing_document_ids = set([str(document['id']) for document in self.existing_meta_data])
            remove_document_ids = existing_document_ids.difference(meta_data_document_ids)
            for document_id in remove_document_ids:
                document_path = os.path.join(self.data_root, 'pdf', document_id)
                shutil.rmtree(document_path)

            # to restart lists and allow changes in the dataset status
            self.documents = []
            self.test_documents = []
            self.no_status_documents = []
            self.preparation_documents = []
            self.low_ocr_documents = []


    def get_label_by_id(self, id: int) -> Label:
        """
        Return a label by ID.

        :param id: ID of the label to get.
        """
        for label in self.labels:
            if label.id == id:
                return label

    def get_template_by_id(self, id: int) -> Template:
        """
        Return a section label by ID.

        :param id: ID of the section label to get.
        """
        for template in self.templates:
            if template.id == id:
                return template

    def clean_meta(self):
        """Clean the meta-information about the Project, Labels, and Templates."""
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

        if self.templates_file_path:
            os.remove(self.templates_file_path)
        assert not is_file(self.templates_file_path, raise_exception=False)
        self.templates_file_path = None
        self.templates: List[Template] = []

    def update(self):
        """Update the project and all documents by downloading them from the host. Note : It will not work offline."""
        # make sure you always update any changes to Labels, ProjectMeta
        self.existing_meta_data = self.meta_data  # get meta data of what currently exists locally
        self.clean_meta()
        self.get(update=True)
        self.load_sections()

        return self
