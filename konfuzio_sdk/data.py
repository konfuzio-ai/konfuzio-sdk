"""Handle data from the API."""
import io
import itertools
import json
import logging
import os
import pathlib
import re
import shutil
import time
import zipfile
from typing import Optional, List, Union, Tuple, Dict
from warnings import warn

import dateutil.parser
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from konfuzio_sdk.api import (
    _konfuzio_session,
    download_file_konfuzio_api,
    get_meta_of_files,
    get_project_details,
    post_document_annotation,
    get_document_details,
    update_document_konfuzio_api,
    get_page_image,
    delete_document_annotation,
)
from konfuzio_sdk.normalize import normalize
from konfuzio_sdk.regex import get_best_regex, regex_matches, suggest_regex_for_string, merge_regex
from konfuzio_sdk.urls import get_annotation_view_url
from konfuzio_sdk.utils import get_missing_offsets
from konfuzio_sdk.utils import is_file, convert_to_bio_scheme, amend_file_name, sdk_isinstance

logger = logging.getLogger(__name__)


class Data:
    """Collect general functionality to work with data from API."""

    id_iter = itertools.count()
    id_ = None
    id_local = None
    session = _konfuzio_session()
    _update = False
    _force_offline = False

    def __eq__(self, other) -> bool:
        """Compare any point of data with their ID, overwrite if needed."""
        if self.id_ is None and other and other.id_ is None:
            # Compare to virtual instances
            return self.id_local == other.id_local
        else:
            return self.id_ is not None and other is not None and other.id_ is not None and self.id_ == other.id_

    def __hash__(self):
        """Make any online or local concept hashable. See https://stackoverflow.com/a/7152650."""
        return hash(str(self.id_local))

    def __copy__(self):
        """Not yet modelled."""
        raise NotImplementedError

    def __deepcopy__(self, memodict):
        """Not yet modelled."""
        raise NotImplementedError

    @property
    def is_online(self) -> Optional[int]:
        """Define if the Document is saved to the server."""
        return (self.id_ is not None) and (not self._force_offline)

    # todo require to overwrite lose_weight via @abstractmethod
    def lose_weight(self):
        """Delete data of the instance."""
        self.session = None
        return self

    def set_offline(self):
        """Force data into offline mode."""
        self._force_offline = True
        self._update = False


class Page(Data):
    """Access the information about one Page of a document."""

    def __init__(
        self,
        id_: Union[int, None],
        document: 'Document',
        start_offset: int,
        end_offset: int,
        number: int,
        original_size: Tuple[float, float],
    ):
        """Create a Page for a Document."""
        self.id_ = id_
        self.document = document
        self.number = number
        self.index = number - 1
        document.add_page(self)
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.image = None
        self._original_size = original_size
        self.width = self._original_size[0]
        self.height = self._original_size[1]
        self.image_path = os.path.join(self.document.document_folder, f'page_{self.number}.png')

        check_page = True
        if self.index is None:
            logger.error(f'Page index is None of {self} in {self.document}.')
            check_page = False
        if self.height is None:  # todo why do we allow pages with height<=0?
            logger.error(f'Page Height is None of {self} in {self.document}.')
            check_page = False
        if self.width is None:  # todo why do we allow pages with width<=0?
            logger.error(f'Page Width is None of {self} in {self.document}.')
            check_page = False
        assert check_page

    def __hash__(self):
        """Define that one Page per Document is unique."""
        return (self.document, self.index)

    def __eq__(self, other: 'Page') -> bool:
        """Define how one Page is identical."""
        return self.__hash__() == other.__hash__()

    def __repr__(self):
        """Return the name of the Document incl. the ID."""
        return f"Page {self.index} in {self.document}"

    def get_image(self, update: bool = False):
        """Get Document Page as PNG."""
        if self.document.status[0] == 2 and (not is_file(self.image_path, raise_exception=False) or update):
            png_content = get_page_image(self.id_)
            with open(self.image_path, "wb") as f:
                f.write(png_content)
                self.image = Image.open(io.BytesIO(png_content))
        elif is_file(self.image_path, raise_exception=False):
            self.image = Image.open(self.image_path)
        return self.image

    def get_annotations_image(self, image: Image = None):
        """Get Document Page as PNG with annotations shown."""
        if image is None and self.document.id_ is not None:
            image = self.get_image()
        elif image is None and self.document.copy_of_id is not None:
            original_doc = self.document.project.get_document_by_id(self.document.copy_of_id)
            image = original_doc.get_page_by_index(self.index).get_image()
        image = image.convert('RGB')
        # bbox information are based on a downscaled image
        scale_mult = image.size[1] / self.height

        anns = self.view_annotations()

        for ann in anns:
            pos = [
                scale_mult * ann.spans[0].bbox().x0,
                scale_mult * (self.height - ann.spans[0].bbox().y0),
                scale_mult * ann.spans[0].bbox().x1,
                scale_mult * (self.height - ann.spans[0].bbox().y1),
            ]
            draw = ImageDraw.Draw(image)
            draw.rectangle(pos, outline='blue', width=2)

            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf", 24, encoding="unic"
                )
            except OSError:
                logger.warning('Font not found. Loading default.')
                font = ImageFont.load_default()

            draw.text((pos[0], pos[3] - 24), ann.label.name, fill='blue', font=font)
        return image

    @property
    def text(self):
        """Get Document text corresponding to the Page."""
        doc_text = self.document.text
        page_text = self.document.text[self.start_offset : self.end_offset]
        if doc_text.split('\f')[self.index] != page_text:
            raise IndexError(f'{self} text offsets do not match Document text.')
        return page_text

    @property
    def number_of_lines(self) -> int:
        """Calculate the number of lines in Page."""
        return len(self.text.split('\n'))

    def spans(
        self,
        label: 'Label' = None,
        use_correct: bool = False,
        start_offset: int = 0,
        end_offset: int = None,
        fill: bool = False,
    ) -> List['Span']:
        """Return all Spans of the Page."""
        spans = []
        annotations = self.annotations(
            label=label, use_correct=use_correct, start_offset=start_offset, end_offset=end_offset, fill=fill
        )
        for annotation in annotations:
            for span in annotation.spans:
                if span not in spans:
                    spans.append(span)

        return sorted(spans)

    def get_bbox(self):
        """Get bbox information per character of Page."""
        doc_bbox = self.document.get_bbox()
        page_bbox = {k: doc_bbox[k] for k in doc_bbox.keys() if doc_bbox[k]["page_number"] == self.number}
        return page_bbox

    def annotations(
        self,
        label: 'Label' = None,
        use_correct: bool = True,
        ignore_below_threshold: bool = False,
        start_offset: int = 0,
        end_offset: int = None,
        fill: bool = False,
    ) -> List['Annotation']:
        """Get Page Annotations."""
        start_offset = max(start_offset, self.start_offset)
        if end_offset is None:
            end_offset = self.end_offset
        else:
            end_offset = min(end_offset, self.end_offset)
        page_annotations = self.document.annotations(
            label=label,
            use_correct=use_correct,
            ignore_below_threshold=ignore_below_threshold,
            start_offset=start_offset,
            end_offset=end_offset,
            fill=fill,
        )
        return page_annotations

    def view_annotations(self) -> List['Annotation']:
        """Get the best Annotations, where the Spans are not overlapping in Page."""
        page_view_anns = self.document.view_annotations(start_offset=self.start_offset, end_offset=self.end_offset)
        return page_view_anns


class Bbox:
    """
    A bounding box relates to an area of a Document Page.

    :param strict_validation: If False, it allows bounding boxes to have zero width or height. This option is available
    and defaults to False for compatibility reasons since some OCR engines can sometimes return character level bboxes
    with zero width or height.
    """

    def __init__(self, x0: int, x1: int, y0: int, y1: int, page: Page, strict_validation: bool = False):
        """Store information and validate."""
        self.x0: int = x0
        self.x1: int = x1
        self.y0: int = y0
        self.y1: int = y1
        self.angle: float = 0.0  # not yet used
        self.page: Page = page
        self._valid(strict_validation)

    def __repr__(self):
        """Represent the Box."""
        return f'{self.__class__.__name__}: {self.x0} {self.x1} {self.y0} {self.y1} on Page {self.page}'

    def __hash__(self):
        """Return identical value for a Bounding Box."""
        return (self.x0, self.x1, self.y0, self.y1, self.page)

    def __eq__(self, other: 'Bbox') -> bool:
        """Define that one Bounding Box on the same page is identical."""
        return self.__hash__() == other.__hash__()

    def _valid(self, strict: bool = False):
        """
        Validate contained data.

        :param strict: If False, it allows bounding boxes to have zero width or height. This option is available
        and defaults to False for compatibility reasons since some OCR engines can sometimes return character level
        bboxes with zero width or height.
        """
        if self.x0 == self.x1:
            if strict:
                raise ValueError(f'{self} has no width in {self.page}.')
            else:
                # todo add link to documentation page relating to feature calculation for the RandomForest ExtractionAI
                logger.error(
                    f'{self} has no width in {self.page}. Some of our AIs use the area of bboxes as a feature'
                    f'during training and prediction, which will be zero.'
                )

        if self.x0 > self.x1:
            raise ValueError(f'{self} has negative width in {self.page}.')

        if self.y0 == self.y1:
            if strict:
                raise ValueError(f'{self} has no height in {self.page}.')
            else:
                # todo add link to documentation page relating to feature calculation for the RandomForest ExtractionAI
                logger.error(
                    f'{self} has no height in {self.page}. Some of our AIs use the area of bboxes as a feature'
                    f'during training and prediction, which will be zero.'
                )

        if self.y0 > self.y1:
            raise ValueError(f'{self} has negative height in {self.page}.')

        if self.y1 > self.page.height:
            raise ValueError(f'{self} exceeds height of {self.page}.')

        if self.x1 > self.page.width:
            raise ValueError(f'{self} exceeds width of {self.page}.')

        if self.y0 < 0:
            raise ValueError(f'{self} has negative y coordinate in {self.page}.')

        if self.x0 < 0:
            raise ValueError(f'{self} has negative x coordinate in {self.page}.')

    @property
    def area(self):
        """Return area covered by the Bbox."""
        return round(abs(self.x0 - self.x1) * abs(self.y0 - self.y1), 3)


class AnnotationSet(Data):
    """An Annotation Set is a group of Annotations. The Labels of those Annotations refer to the same Label Set."""

    def __init__(self, document, label_set: 'LabelSet', id_: Union[int, None] = None, **kwargs):
        """
        Create an Annotation Set.

        :param id: ID of the Annotation Set
        :param document: Document where the Annotation Set belongs
        :param label_set: Label set where the Annotation Set belongs to
        :param annotations: Annotations of the Annotation Set
        """
        self.id_local = next(Data.id_iter)
        self.id_ = id_
        self.label_set: LabelSet = label_set
        self.document: Document = document  # we don't add it to the Document as it's added via get_annotations
        self._force_offline = document._force_offline
        self._annotations = []
        document.add_annotation_set(self)

    def __repr__(self):
        """Return string representation of the Annotation Set."""
        return f"{self.__class__.__name__}({self.id_}) of {self.label_set} in {self.document}."

    def lose_weight(self):
        """Delete data of the instance."""
        self.label_set = None
        self.document = None

    def annotations(self, use_correct: bool = True, ignore_below_threshold: bool = False):
        """All Annotations currently in this Annotation Set."""
        if not self._annotations:
            for annotation in self.document.annotations(use_correct=False, ignore_below_threshold=False):
                if annotation.annotation_set == self:
                    self._annotations.append(annotation)

        annotations: List[Annotation] = []
        if use_correct:
            annotations = [ann for ann in self._annotations if ann.is_correct]
        elif ignore_below_threshold:
            annotations = [ann for ann in self._annotations if ann.is_correct or ann.confidence > ann.label.threshold]
        else:
            annotations = self._annotations
        return annotations

    @property
    def start_offset(self):
        """Calculate the earliest start based on all Annotations currently in this Annotation Set."""
        return min((s.start_offset for a in self.annotations() for s in a.spans), default=None)

    @property
    def start_line_index(self):
        """Calculate starting line of this Annotation Set."""
        return self.document.text[0 : self.start_offset].count('\n')

    @property
    def end_offset(self):
        """Calculate the end based on all Annotations currently in this Annotation Set."""
        return max((a.end_offset for a in self.annotations()), default=None)


class LabelSet(Data):
    """A Label Set is a group of labels."""

    def __init__(
        self,
        project,
        labels=None,
        id_: int = None,
        name: str = None,
        name_clean: str = None,
        is_default=False,
        categories=None,
        has_multiple_annotation_sets=False,
        **kwargs,
    ):
        """
        Create a named Label Set.

        :param project: Project where the Label Set belongs
        :param id_: ID of the Label Set
        :param name: Name of Label Set
        :param name_clean: Normalized name of the Label Set
        :param labels: Labels that belong to the Label Set (IDs)
        :param is_default: Bool for the Label Set to be the default one in the Project
        :param categories: Categories to which the Label Set belongs
        :param has_multiple_annotation_sets: Bool to allow the Label Set to have different Annotation Sets in a Document
        """
        if categories is None:
            categories = []
        if labels is None:
            labels = []
        self.id_local = next(Data.id_iter)
        self.id_ = id_
        self.name = name
        self.name_clean = name_clean
        self.is_default = is_default

        if not categories and "default_label_sets" in kwargs:
            self._default_of_label_set_ids = kwargs["default_label_sets"]
            self.categories = []
        elif not categories and "default_section_labels" in kwargs:
            self._default_of_label_set_ids = kwargs["default_section_labels"]
            self.categories = []
        else:
            self._default_of_label_set_ids = []
            self.categories = categories

        self.has_multiple_annotation_sets = has_multiple_annotation_sets

        if "has_multiple_sections" in kwargs:
            self.has_multiple_annotation_sets = kwargs["has_multiple_sections"]

        self.project: Project = project
        self._force_offline = project._force_offline
        self.labels: List[Label] = []

        # todo allow to create Labels either on Project or Label Set level, so they are (not) shared among Label Sets.
        for label in labels:
            if isinstance(label, int):
                label = self.project.get_label_by_id(id_=label)
            self.add_label(label)

        project.add_label_set(self)
        for category in self.categories:
            category.add_label_set(self)

    def __lt__(self, other: 'LabelSet'):
        """Sort Label Sets by name."""
        try:
            return self.name < other.name
        except TypeError:
            logger.error(f'Cannot sort {self} and {other}.')
            return False

    def __repr__(self):
        """Return string representation of the Label Set."""
        return f"{self.name} ({self.id_})"

    def add_category(self, category: 'Category'):
        """
        Add Category to Project, if it does not exist.

        :param category: Category to add in the Project
        """
        if category not in self.categories:
            self.categories.append(category)
        else:
            raise ValueError(f'In {self} the {category} is a duplicate and will not be added.')

    def add_label(self, label):
        """
        Add Label to Label Set, if it does not exist.

        :param label: Label ID to be added
        """
        if label not in self.labels:
            self.labels.append(label)
            label.add_label_set(self)
        else:
            raise ValueError(f'In {self} the {label} is a duplicate and will not be added.')
        return self

    def get_target_names(self, use_separate_labels: bool):
        """Get target string name for Annotation Label classification."""
        targets = []
        for label in self.labels:
            if use_separate_labels:
                targets.append(self.name + '__' + label.name)
            else:
                targets.append(label.name)
        return targets


class Category(Data):
    """Group Documents in a Project."""

    def __init__(self, project, id_: int = None, name: str = None, name_clean: str = None, *args, **kwargs):
        """Associate Label Sets to relate to Annotations."""
        self.id_local = next(Data.id_iter)
        self.id_ = id_
        self.name = name
        self.name_clean = name_clean
        self.project: Project = project
        self._force_offline = project._force_offline
        self.label_sets: List[LabelSet] = []
        self.project.add_category(category=self)

    @property
    def labels(self):
        """Return the Labels that belong to the Category and it's Label Sets."""
        labels = []
        # for label in self.project.labels:
        #     if self in label.label_sets:
        #         labels.append(label)
        for label_set in self.label_sets:
            labels += label_set.labels

        return list(set(labels))

    @property
    def fallback_name(self) -> str:
        """Turn the category name to lowercase, remove parentheses along with their contents, and trim spaces."""
        parentheses_removed = re.sub(r'\([^)]*\)', '', self.name.lower()).strip()
        single_spaces = parentheses_removed.replace("  ", " ")
        return single_spaces

    def documents(self):
        """Filter for Documents of this Category."""
        return [x for x in self.project.documents if x.category == self]

    def test_documents(self):
        """Filter for test Documents of this Category."""
        return [x for x in self.project.test_documents if x.category == self]

    def add_label_set(self, label_set):
        """Add Label Set to Category."""
        if label_set not in self.label_sets:
            self.label_sets.append(label_set)
        else:
            raise ValueError(f'In {self} the {label_set} is a duplicate and will not be added.')

    def __lt__(self, other: 'Category'):
        """Sort Categories by name."""
        try:
            return self.name < other.name
        except TypeError:
            logger.error(f'Cannot sort {self} and {other}.')
            return False

    def __repr__(self):
        """Return string representation of the Category."""
        return f"{self.name} ({self.id_})"


class Label(Data):
    """Group Annotations across Label Sets."""

    def __init__(
        self,
        project,
        id_: Union[int, None] = None,
        text: str = None,
        get_data_type_display: str = 'Text',
        text_clean: str = None,
        description: str = None,
        label_sets=None,
        has_multiple_top_candidates: bool = False,
        threshold: float = 0.1,
        *initial_data,
        **kwargs,
    ):
        """
        Create a named Label.

        :param project: Project where the Label belongs
        :param id_: ID of the label
        :param text: Name of the label
        :param get_data_type_display: Data type of the label
        :param text_clean: Normalized name of the label
        :param description: Description of the label
        :param label_sets: Label sets that use this label
        """
        self.id_local = next(Data.id_iter)
        self.id_ = id_
        self.name = text
        self.name_clean = text_clean
        self.data_type = get_data_type_display
        self.description = description
        self.has_multiple_top_candidates = has_multiple_top_candidates
        self.threshold = threshold
        self.project: Project = project
        self._force_offline = project._force_offline
        project.add_label(self)

        self.label_sets = []
        for label_set in label_sets or []:
            label_set.add_label(self)

        # Regex features
        self._tokens = {}
        self.tokens_file_path = None
        self._regex = {}  # : List[str] = []
        # self._combined_tokens = None
        self.regex_file_path = os.path.join(self.project.regex_folder, f'{self.name_clean}.json5')
        self._correct_annotations = []
        self._evaluations = {}  # used to do the duplicate check on Annotation level

    def __repr__(self):
        """Return string representation."""
        return f'Label: {self.name}'

    def __lt__(self, other: 'Label'):
        """Sort Spans by start offset."""
        try:
            return self.name < other.name
        except TypeError:
            logger.error(f'Cannot sort {self} and {other}.')
            return False

    def annotations(
        self, categories: List[Category], use_correct=True, ignore_below_threshold=False
    ) -> List['Annotation']:
        """Return related Annotations. Consider that one Label can be used across Label Sets in multiple Categories."""
        annotations = []
        for category in categories:
            for document in category.documents():
                for annotation in document.annotations(
                    label=self, use_correct=use_correct, ignore_below_threshold=ignore_below_threshold
                ):
                    annotations.append(annotation)

        if not annotations:
            logger.warning(f'{self} has no correct annotations.')

        return annotations

    def has_multiline_annotations(self, categories: List[Category]) -> bool:
        """Return if any Label annotations are multi-line."""
        for category in categories:
            for document in category.documents():
                for annotation in document.annotations(label=self):
                    if len(annotation.spans) > 1:
                        return True
        return False

    def add_label_set(self, label_set: "LabelSet"):
        """
        Add Label Set to label, if it does not exist.

        :param label_set: Label set to add
        """
        if label_set not in self.label_sets:
            self.label_sets.append(label_set)
        else:
            raise ValueError(f'In {self} the {label_set} is a duplicate and will not be added.')

    def evaluate_regex(self, regex, category: Category, annotations: List['Annotation'] = None, regex_quality=0):
        """
        Evaluate a regex on Categories.

        Type of regex allows you to group regex by generality

        Example:
            Three Annotations about the birthdate in two Documents and one regex to be evaluated
            1.doc: "My was born at the 12th of December 1980, you could also say 12.12.1980." (2 Annotations)
            2.doc: "My was born at 12.06.1997." (1 Annotations)
            regex: dd.dd.dddd (without escaped characters for easier reading)
            stats:
                  total_correct_findings: 2
                  correct_label_annotations: 3
                  total_findings: 2 --> precision 100 %
                  num_docs_matched: 2
                  Project.documents: 2  --> Document recall 100%

        """
        evaluations = []
        documents = category.documents()

        for document in documents:
            # todo: potential time saver: make sure we did a duplicate check for the regex before we run the evaluation
            evaluation = document.evaluate_regex(regex=regex, label=self, annotations=annotations)
            evaluations.append(evaluation)

        total_findings = sum(evaluation['count_total_findings'] for evaluation in evaluations)
        num_docs_matched = sum(evaluation['doc_matched'] for evaluation in evaluations)
        correct_findings = [finding for evaluation in evaluations for finding in evaluation['correct_findings']]
        total_correct_findings = sum(evaluation['count_total_correct_findings'] for evaluation in evaluations)
        processing_times = [evaluation['runtime'] for evaluation in evaluations]

        try:
            annotation_precision = total_correct_findings / total_findings
        except ZeroDivisionError:
            annotation_precision = 0

        try:
            annotation_recall = total_correct_findings / len(self.annotations(categories=[category]))
        except ZeroDivisionError:
            annotation_recall = 0

        try:
            document_recall = num_docs_matched / len(documents)
        except ZeroDivisionError:
            document_recall = 0

        try:
            f_score = 2 * (annotation_precision * annotation_recall) / (annotation_precision + annotation_recall)
        except ZeroDivisionError:
            f_score = 0

        if documents:
            evaluation = {
                'regex': regex,
                'regex_len': len(regex),  # the longer the regex the more conservative it is to use
                'runtime': sum(processing_times) / len(processing_times),  # time to process the regex
                'annotation_recall': annotation_recall,
                'annotation_precision': annotation_precision,
                'f1_score': f_score,
                'document_recall': document_recall,
                'regex_quality': regex_quality,
                # other stats
                'correct_findings': correct_findings,
                'total_findings': total_findings,
                'total_annotations': len(self.annotations(categories=[category])),
                'num_docs_matched': num_docs_matched,
                'total_correct_findings': total_correct_findings,
            }
            correct_matches_per_document = {
                f'document_{evaluation["id"]}': evaluation['correct_findings'] for evaluation in evaluations
            }
            evaluation.update(correct_matches_per_document)  # add the matching info per document

            return evaluation
        else:
            return {}

    def base_regex(self, category: 'Category', annotations: List['Annotation'] = None) -> str:
        """Find the best combination of regex in the list of all regex proposed by Annotations."""
        if category.id_ in self._tokens:
            return self._tokens[category.id_]

        logger.info(f"Beginning base regex search for Label {self.name}.")

        if annotations is None:
            all_annotations = self.annotations(categories=[category])  # default is use_correct = True
        else:
            all_annotations = annotations

        evaluated_proposals = []
        for annotation in all_annotations:
            annotation_proposals = annotation.tokens()
            evaluated_proposals += annotation_proposals

        self._evaluations[category.id_] = evaluated_proposals

        try:
            best_proposals = get_best_regex(evaluated_proposals)
        except ValueError:
            logger.error(f'We cannot find regexes for {self} with a f_score > 0.')
            best_proposals = []

        label_regex_token = merge_regex(best_proposals)

        self._tokens[category.id_] = label_regex_token

        return label_regex_token

    def find_regex(self, category: 'Category', max_findings_per_page=100) -> List[str]:
        """Find the best combination of regex for Label with before and after context."""
        all_annotations = self.annotations(categories=[category])  # default is use_correct = True
        if all_annotations == []:
            logger.error(f"We cannot find annotations for Label {self} and Category {category}.")
            return []

        label_regex_token = self.base_regex(category=category, annotations=all_annotations)

        search = [1, 3, 5]
        regex_to_remove_groupnames = re.compile('<.*?>')

        naive_proposal = label_regex_token
        regex_made = []
        regex_found = set()

        for annotation in all_annotations:
            new_proposals = []
            annotation.document.spans(fill=True)
            for span in annotation.spans:
                before_reg_dict = {}
                after_reg_dict = {}
                for spacer in search:  # todo fix this search, so that we take regex token from other spans into account
                    before_regex = ''
                    bef_spacer = spacer * 3 if spacer > 1 else spacer
                    before_start_offset = span.start_offset - bef_spacer  # spacer**2
                    for before_span in annotation.document.spans(
                        fill=True, start_offset=before_start_offset, end_offset=span.start_offset
                    ):
                        if before_span.annotation.label is self.project.no_label:
                            to_rep_offset_string = before_span.annotation.document.text[
                                max(before_start_offset, before_span.start_offset) : before_span.end_offset
                            ]
                            before_regex += suggest_regex_for_string(to_rep_offset_string, replace_characters=True)
                        else:
                            before_regex += before_span.annotation.label.base_regex(category)
                    before_reg_dict[spacer] = before_regex

                    after_regex = ''
                    after_end_offset = span.end_offset + spacer
                    for after_span in annotation.document.spans(
                        fill=True, start_offset=span.end_offset, end_offset=after_end_offset
                    ):
                        if after_span.annotation.label is self.project.no_label:
                            to_rep_offset_string = after_span.annotation.document.text[
                                after_span.start_offset : min(after_end_offset, after_span.end_offset)
                            ]
                            after_regex += suggest_regex_for_string(to_rep_offset_string, replace_characters=True)
                        else:
                            after_regex += after_span.annotation.label.base_regex(category)
                    after_reg_dict[spacer] = after_regex

                    spacer_proposals = [
                        before_regex + label_regex_token + after_regex,
                        before_reg_dict[search[0]] + label_regex_token + after_regex,
                        before_regex + label_regex_token + after_reg_dict[search[0]],
                    ]

                    # check for duplicates
                    for proposal in spacer_proposals:
                        new_regex = re.sub(regex_to_remove_groupnames, '', proposal)
                        if new_regex not in regex_found:
                            if max_findings_per_page:
                                num_matches = len(regex_matches(regex=proposal, doctext=annotation.document.text))
                                if num_matches / (annotation.document.number_of_pages) < max_findings_per_page:
                                    new_proposals.append(proposal)
                                else:
                                    logger.info(
                                        f'Skip to evaluate regex {repr(proposal)} as it finds {num_matches} in\
                                            {annotation.document}.'
                                    )
                            else:
                                new_proposals.append(proposal)
            for proposal in new_proposals:
                new_regex = re.sub(regex_to_remove_groupnames, '', proposal)
                if new_regex not in regex_found:
                    regex_made.append(proposal)
                    regex_found.add(new_regex)

        logger.info(
            f'For Label {self.name} we found {len(regex_made)} regex proposals for {len(all_annotations)} annotations.'
        )

        # todo replace by compare
        evaluations = [
            self.evaluate_regex(_regex_made, category=category, annotations=all_annotations)
            for _regex_made in regex_made
        ]

        logger.info(
            f'We compare {len(evaluations)} regex for {len(all_annotations)} correct Annotations for Category '
            f'{category}.'
        )

        try:
            logger.info(f'Evaluate {self} for best regex.')
            best_regex = get_best_regex(evaluations)
        except ValueError:
            logger.exception(f'We cannot find regex for {self} with a f_score > 0.')
            best_regex = []

        if best_regex == []:
            best_regex = [naive_proposal]

        return best_regex

    def regex(self, categories: List[Category], update=False) -> List:
        """Calculate regex to be used in the Extraction AI."""
        # if not is_file(self.regex_file_path, raise_exception=False) or update:
        logger.info(f'Build regexes for Label {self.name}.')
        regex = {}
        for category in categories:
            if category.id_ not in self._regex or update:
                regex_category_file_path = os.path.join(
                    self.project.regex_folder, f'{category.name}_{self.name_clean}_tokens.json5'
                )
                if not is_file(regex_category_file_path, raise_exception=False) or update:
                    category_regex = self.find_regex(category=category)
                    if os.path.exists(self.project.regex_folder):
                        with open(regex_category_file_path, 'w') as f:
                            json.dump(category_regex, f, indent=2, sort_keys=True)
                else:
                    logger.info(f'Start loading existing regexes for Label {self.name}.')
                    with open(regex_category_file_path, 'r') as f:
                        category_regex = json.load(f)
                regex[category.id_] = category_regex
            else:
                regex[category.id_] = self._regex[category.id_]

        self._regex = regex

        logger.info(f'Regexes are ready for Label {self.name}.')

        categories_ids = [category.id_ for category in categories]
        return {k: v for k, v in self._regex.items() if k in categories_ids}

    def spans_not_found_by_tokenizer(self, tokenizer, categories: List[Category], use_correct=False) -> List['Span']:
        """Find Label Spans that are not found by a tokenizer."""
        spans_not_found = []
        label_annotations = self.annotations(categories=categories, use_correct=use_correct)
        for annotation in label_annotations:
            for span in annotation.spans:
                if not tokenizer.span_match(span):
                    spans_not_found.append(span)
        return spans_not_found

    # def save(self) -> bool:
    #     """
    #     Save Label online.
    #
    #     If no Label Sets are specified, the Label is associated with the first default Label Set of the Project.
    #
    #     :return: True if the new Label was created.
    #     """
    #     if len(self.label_sets) == 0:
    #         prj_label_sets = self.project.label_sets
    #         label_set = [t for t in prj_label_sets if t.is_default][0]
    #         label_set.add_label(self)
    #
    #     response = create_label(
    #         project_id=self.project.id_,
    #         label_name=self.name,
    #         description=self.description,
    #         has_multiple_top_candidates=self.has_multiple_top_candidates,
    #         data_type=self.data_type,
    #         label_sets=self.label_sets,
    #     )
    #
    #     return True


class Span(Data):
    """A Span is a sequence of characters or whitespaces without line break."""

    def __init__(self, start_offset: int, end_offset: int, annotation=None):
        """
        Initialize the Span without bbox, to save storage.

        If Bbox should be calculated the bbox file of the Document will be automatically downloaded.

        :param start_offset: Start of the offset string (int)
        :param end_offset: Ending of the offset string (int)
        :param annotation: The Annotation the Span belong to
        """
        self.id_local = next(Data.id_iter)
        self.annotation: Annotation = annotation
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.top = None
        self.bottom = None
        self._line_index = None
        self._page: Union[Page, None] = None
        self._bbox: Union[Bbox, None] = None
        self.regex_matching = []
        annotation and annotation.add_span(self)  # only add if Span has access to an Annotation
        self._valid()

    def _valid(
        self,
    ):
        """Validate containted data."""
        if self.end_offset == self.start_offset == 0:
            logger.error(f'{self} is intentionally left empty.')
        elif self.end_offset < 0:
            raise ValueError(f'{self} must span text.')
        elif self.start_offset == self.end_offset:
            raise ValueError(f"{self} must span text: Start {self.start_offset} equals end.")
        elif self.end_offset < self.start_offset:
            raise ValueError(f"{self} length must be positive.")
        elif self.offset_string and ('\n' in self.offset_string or '\f' in self.offset_string):
            raise ValueError(f'{self} must not span more than one visual line.')
        return True

    @property
    def page(self) -> Page:
        """Return Page of Span."""
        if self.annotation is None or self.annotation.document is None:
            raise NotImplementedError
        elif self.annotation.document.text is None:
            logger.error(f'{self.annotation.document} does not provide text.')
            pass
        elif self._page is None and self.annotation.document.pages():
            text = self.annotation.document.text[: self.start_offset]
            page_index = len(text.split('\f')) - 1
            self._page = self.annotation.document.get_page_by_index(page_index=page_index)
        return self._page

    @property
    def line_index(self) -> int:
        """Return index of the line of the Span."""
        self._valid()
        if self.annotation.document.text and self._line_index is None:
            line_number = len(self.annotation.document.text[: self.start_offset].replace('\f', '\n').split('\n'))
            self._line_index = line_number - 1

        return self._line_index

    def __eq__(self, other) -> bool:
        """Compare any point of data with their position is equal."""
        return (
            type(self) == type(other)
            and self.start_offset == other.start_offset
            and self.end_offset == other.end_offset
        )

    def __lt__(self, other: 'Span'):
        """If we sort spans we do so by start offset."""
        return self.start_offset < other.start_offset

    def __repr__(self):
        """Return string representation."""
        return f"{self.__class__.__name__} ({self.start_offset}, {self.end_offset})"

    def __hash__(self):
        """Make any online or local concept hashable. See https://stackoverflow.com/a/7152650."""
        if not self.annotation:
            raise NotImplementedError('Span without Annotation is not hashable.')
        else:
            return hash((self.annotation, self.start_offset, self.end_offset))

    def regex(self):
        """Suggest a Regex for the offset string."""
        if self.annotation:
            # todo make the options to replace characters and string more granular
            full_replace = suggest_regex_for_string(self.offset_string, replace_characters=True, replace_numbers=True)
            return merge_regex([full_replace])
        else:
            raise NotImplementedError('A Span needs a Annotation and Document relation to suggest a Regex.')

    def bbox(self) -> Bbox:
        """Calculate the bounding box of a text sequence."""
        if not self.annotation:
            raise NotImplementedError
        if not self.page:
            logger.warning(f'{self} does not have a Page.')
            return None
        if not self.annotation.document.bboxes_available:
            logger.warning(f'{self.annotation.document} of {self} does not provide Bboxes.')
            return None
        _ = self.line_index  # quick validate if start and end is in the same line of text

        if self._bbox is None:
            warn('WIP: Modifications before the next stable release expected.', FutureWarning, stacklevel=2)
            # todo: verify that one Span relates to Character in on line of text
            character_range = range(self.start_offset, self.end_offset)
            document = self.annotation.document
            characters = {key: document.bboxes.get(key) for key in character_range if document.text[key] != ' '}
            if not all(characters.values()):
                logger.error(
                    f'{self} in {self.annotation.document} contains Characters that don\'t provide a Bounding Box.'
                )
            self._bbox = Bbox(
                x0=min([ch.x0 for c, ch in characters.items() if ch is not None]),
                x1=max([ch.x1 for c, ch in characters.items() if ch is not None]),
                y0=min([ch.y0 for c, ch in characters.items() if ch is not None]),
                y1=max([ch.y1 for c, ch in characters.items() if ch is not None]),
                page=self.page,
            )
        return self._bbox

    @property
    def normalized(self):
        """Normalize the offset string."""
        return normalize(self.offset_string, self.annotation.label.data_type)

    @property
    def offset_string(self) -> Union[str, None]:
        """Calculate the offset string of a Span."""
        if self.annotation and self.annotation.document and self.annotation.document.text:
            return self.annotation.document.text[self.start_offset : self.end_offset]
        else:
            return None

    def eval_dict(self):
        """Return any information needed to evaluate the Span."""
        if self.start_offset == self.end_offset == 0:
            span_dict = {
                "id_local": None,
                "id_": None,
                "confidence": None,
                "offset_string": None,
                "normalized": None,
                "start_offset": 0,  # to support compare function to evaluate True and False
                "end_offset": 0,  # to support compare function to evaluate True and False
                "is_correct": None,
                "created_by": None,
                "revised_by": None,
                "custom_offset_string": None,
                "revised": None,
                "label_threshold": None,
                "label_id": None,
                "label_set_id": None,
                "annotation_id": None,
                "annotation_set_id": 0,  # to allow grouping to compare boolean
                "document_id": 0,
                "document_id_local": 0,
                "category_id": 0,
                "x0": 0,
                "x1": 0,
                "y0": 0,
                "y1": 0,
                "line_index": 0,
                "page_index": None,
                "page_width": 0,
                "page_height": 0,
                "x0_relative": None,
                "x1_relative": None,
                "y0_relative": None,
                "y1_relative": None,
                "page_index_relative": None,
                "area_quadrant_two": 0,
                "area": 0,
                "label_name": None,
                "label_set_name": None,
                "data_type": None,
            }
        else:
            span_dict = {
                "id_local": self.annotation.id_local,
                "id_": self.annotation.id_,
                "confidence": self.annotation.confidence,
                "offset_string": self.offset_string,
                "normalized": self.normalized,
                "start_offset": self.start_offset,  # to support multiline
                "end_offset": self.end_offset,  # to support multiline
                "is_correct": self.annotation.is_correct,
                "created_by": self.annotation.created_by,
                "revised_by": self.annotation.revised_by,
                "custom_offset_string": self.annotation.custom_offset_string,
                "revised": self.annotation.revised,
                "label_threshold": self.annotation.label.threshold,  # todo: allow to optimize threshold
                "label_id": self.annotation.label.id_,
                "label_set_id": self.annotation.label_set.id_,
                "annotation_id": self.annotation.id_,
                "annotation_set_id": self.annotation.annotation_set.id_,
                "document_id": self.annotation.document.id_,
                "document_id_local": self.annotation.document.id_local,
                "category_id": self.annotation.document.category.id_,
                "line_index": self.line_index,
                "data_type": self.annotation.label.data_type,
            }

            if self.bbox():
                span_dict["x0"] = self.bbox().x0
                span_dict["x1"] = self.bbox().x1
                span_dict["y0"] = self.bbox().y0
                span_dict["y1"] = self.bbox().y1

                # https://www.cuemath.com/geometry/quadrant/
                span_dict["area_quadrant_two"] = self.bbox().x0 * self.bbox().y0
                span_dict["area"] = self.bbox().area

            if self.page:  # todo separate as eval_dict on Page level
                span_dict["page_index"] = self.page.index
                span_dict["page_width"] = self.page.width
                span_dict["page_height"] = self.page.height
                span_dict["x0_relative"] = self.bbox().x0 / self.page.width
                span_dict["x1_relative"] = self.bbox().x1 / self.page.width
                span_dict["y0_relative"] = self.bbox().y0 / self.page.height
                span_dict["y1_relative"] = self.bbox().y1 / self.page.height
                span_dict["page_index_relative"] = self.page.index / self.annotation.document.number_of_pages

            document_id = (
                self.annotation.document.id_
                if self.annotation.document.id_ is not None
                else self.annotation.document.copy_of_id
            )
            span_dict["document_id"] = document_id
            span_dict["label_name"] = self.annotation.label.name if self.annotation.label else None
            span_dict["label_set_name"] = self.annotation.label_set.name if self.annotation.label_set else None

        return span_dict


class Annotation(Data):
    """Hold information that a Label, Label Set and Annotation Set has been assigned to and combines Spans."""

    def __init__(
        self,
        document: 'Document',
        annotation_set_id: Union[int, None] = None,  # support to init from API output
        annotation_set: Union[AnnotationSet, None] = None,  # support to init from API output
        label: Union[int, Label, None] = None,
        label_set_id: Union[None, int] = None,
        label_set: Union[None, LabelSet] = None,
        is_correct: bool = False,
        revised: bool = False,
        normalized=None,
        id_: int = None,
        spans=None,
        accuracy: float = None,
        confidence: float = None,
        created_by: int = None,
        revised_by: int = None,
        translated_string: str = None,
        custom_offset_string: bool = False,
        offset_string: str = False,
        *args,
        **kwargs,
    ):
        """
        Initialize the Annotation.

        :param label: ID of the Annotation
        :param is_correct: If the Annotation is correct or not (bool)
        :param revised: If the Annotation is revised or not (bool)
        :param id_: ID of the Annotation (int)
        :param accuracy: Accuracy of the Annotation (float) which is the Confidence
        :param document: Document to annotate
        :param annotation: Annotation Set of the Document where the Label belongs
        :param label_set_text: Name of the Label Set where the Label belongs
        :param translated_string: Translated string
        :param custom_offset_string: String as edited by a user
        :param label_set_id: ID of the Label Set where the Label belongs
        """
        self.id_local = next(Data.id_iter)
        self.is_correct = is_correct
        self.revised = revised
        self.normalized = normalized
        self.translated_string = translated_string
        self.document = document
        self._force_offline = self.document._force_offline
        self.created_by = created_by
        self.revised_by = revised_by
        if custom_offset_string:
            self.custom_offset_string = offset_string
        else:
            self.custom_offset_string = None
        self.id_ = id_  # Annotations can have None id_, if they are not saved online and are only available locally
        self._spans: List[Span] = []

        if accuracy is not None:  # its a confidence
            self.confidence = accuracy
        elif confidence is not None:
            self.confidence = confidence
        elif self.id_ is not None and accuracy is None:  # hotfix: it's an online annotation crated by a human
            self.confidence = 1
        elif accuracy is None and confidence is None:
            self.confidence = None
        else:
            raise ValueError('Annotation has an id_ but does not provide a confidence.')

        if isinstance(label, int):
            self.label: Label = self.document.project.get_label_by_id(label)
        elif sdk_isinstance(label, Label):
            self.label: Label = label
        else:
            raise ValueError(f'{self.__class__.__name__} {self.id_local} has no Label.')

        # if no label_set_id we check if is passed by section_label_id
        if label_set_id is None and kwargs.get("section_label_id") is not None:
            label_set_id = kwargs.get("section_label_id")

        # handles association to an Annotation Set if the Annotation belongs to a Category
        if isinstance(label_set_id, int):
            self.label_set: LabelSet = self.document.project.get_label_set_by_id(label_set_id)
        elif sdk_isinstance(label_set, LabelSet):
            self.label_set = label_set
        else:
            self.label_set = None
            logger.info(f'{self.__class__.__name__} {self.id_local} has no Label Set.')

        # make sure an Annotation Set is available
        if isinstance(annotation_set_id, int):
            self.annotation_set: AnnotationSet = self.document.get_annotation_set_by_id(annotation_set_id)
        elif sdk_isinstance(annotation_set, AnnotationSet):
            # it's a save way to look up the annotation set first. Otherwise users can add annotation sets which
            # do not relate to the document
            self.annotation_set: AnnotationSet = self.document.get_annotation_set_by_id(annotation_set.id_)
        else:
            self.annotation_set = None
            logger.warning(f'{self} in {self.document} created but without Annotation Set information.')

        for span in spans or []:
            self.add_span(span)

        self.selection_bbox = kwargs.get("selection_bbox", None)

        # TODO START LEGACY to support multiline Annotations
        bboxes = kwargs.get("bboxes", None)
        if bboxes and len(bboxes) > 0:
            for bbox in bboxes:
                if "start_offset" in bbox.keys() and "end_offset" in bbox.keys():
                    Span(start_offset=bbox["start_offset"], end_offset=bbox["end_offset"], annotation=self)
                else:
                    ValueError(f'SDK cannot read bbox of Annotation {self.id_} in {self.document}: {bbox}')
        elif (
            bboxes is None
            and kwargs.get("start_offset", None) is not None
            and kwargs.get("end_offset", None) is not None
        ):
            # Legacy support for creating Annotations with a single offset
            bbox = kwargs.get('bbox', {})
            _ = Span(start_offset=kwargs.get("start_offset"), end_offset=kwargs.get("end_offset"), annotation=self)
            # self.add_span(sa)

            logger.warning(f'{self} is empty')

        self.top = None
        self.top = None
        self.x0 = None
        self.x1 = None
        self.y0 = None
        self.y1 = None

        # todo: remove this Annotation single Bbox
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
        # END LEGACY -

        # regex features
        self._tokens = []
        self._regex = None

        # Call add_annotation to document at the end, so all attributes for duplicate checking are available.
        self.document.add_annotation(self)

        if not self.document:
            raise NotImplementedError(f'{self} has no Document and cannot be created.')
        if not self.label_set:
            raise NotImplementedError(f'{self} has no Label Set and cannot be created.')
        if not self.label:
            raise NotImplementedError(f'{self} has no Label and cannot be created.')
        if not self.spans:
            raise NotImplementedError(f'{self} has no Spans and cannot be created.')

    def __repr__(self):
        """Return string representation."""
        if self.label and self.document:
            span_str = ', '.join(f'{x.start_offset, x.end_offset}' for x in self._spans)
            return f"Annotation ({self.get_link()}) {self.label.name} {span_str}"
        elif self.label:
            return f"Annotation ({self.get_link()}) {self.label.name} ({self._spans})"
        else:
            return f"Annotation ({self.get_link()}) without Label ({self.start_offset}, {self.end_offset})"

    def __eq__(self, other):
        """We compare an Annotation based on it's Label, Label-Sets if it's online otherwise on the id_local."""
        result = False
        if self.document and other.document and self.document == other.document:  # same Document
            # if self.is_correct and other.is_correct:  # for correct Annotations check if they are identical
            if self.label and other.label and self.label == other.label:  # same Label
                if self.spans == other.spans:  # logic changed from "one Span is identical" to "all Spans identical"
                    return True

        return result

    def __lt__(self, other):
        """If we sort Annotations we do so by start offset."""
        return min([span.start_offset for span in self.spans]) < min([span.start_offset for span in other.spans])

    def __hash__(self):
        """Identity of Annotation that does not change over time."""
        return hash((self.start_offset, self.end_offset, self.label_set, self.document, self.label))

    @property
    def is_multiline(self) -> int:
        """Calculate if Annotation spans multiple lines of text."""
        logger.error('We cannot calculate this. The indicator is unreliable.')
        return self.offset_string.count('\n')

    @property
    def normalize(self) -> str:
        """Provide one normalized offset string due to legacy."""
        logger.warning('You use normalize on Annotation Level which is legacy.')
        return normalize(self.offset_string, self.label.data_type)

    @property
    def start_offset(self) -> int:
        """Legacy: One Annotation can have multiple start offsets."""
        logger.warning('You use start_offset on Annotation Level which is legacy.')
        return min([sa.start_offset for sa in self._spans], default=None)

    @property
    def end_offset(self) -> int:
        """Legacy: One Annotation can have multiple end offsets."""
        logger.warning('You use end_offset on Annotation Level which is legacy.')
        return max([sa.end_offset for sa in self._spans], default=None)

    @property
    def offset_string(self) -> List[str]:
        """View the string representation of the Annotation."""
        if len(self.spans) > 1:
            logger.warning(f'You use offset string on {self} level which is legacy.')
        if not self.custom_offset_string and self.document.text:
            result = [span.offset_string for span in self.spans]
        elif self.custom_offset_string:
            result = self.custom_offset_string
        else:
            result = []
        return result

    @property
    def eval_dict(self) -> List[dict]:
        """Calculate the Span information to evaluate the Annotation."""
        return [span.eval_dict() for span in self.spans]

    def add_span(self, span: Span):
        """Add a Span to an Annotation incl. a duplicate check per Annotation."""
        if span not in self._spans:
            # add the Span first to make sure to bea able to do a duplicate check
            self._spans.append(span)  # one Annotation can span multiple Spans
            if span.annotation is not None and self != span.annotation:
                raise ValueError(f'{span} should be added to {self} but relates to {span.annotation}.')
            else:
                span.annotation = self  # todo feature to link one Span to many Annotations
        else:
            raise ValueError(f'In {self} the {span} is a duplicate and will not be added.')
        return self

    def get_link(self):
        """Get link to the Annotation in the SmartView."""
        if self.is_online:
            return get_annotation_view_url(self.id_)
        else:
            return None

    def save(self, document_annotations: list = None) -> bool:
        """
        Save Annotation online.

        If there is already an Annotation in the same place as the current one, we will not be able to save the current
        annotation.

        In that case, we get the id_ of the original one to be able to track it.
        The verification of the duplicates is done by checking if the offsets and Label match with any Annotations
        online.
        To be sure that we are comparing with the information online, we need to have the Document updated.
        The update can be done after the request (per annotation) or the updated Annotations can be passed as input
        of the function (advisable when dealing with big Documents or Documents with many Annotations).

        :param document_annotations: Annotations in the Document (list)
        :return: True if new Annotation was created
        """
        new_annotation_added = False
        if not self.label_set:
            label_set_id = None
        else:
            label_set_id = self.label_set.id_
        if self.is_online:
            raise ValueError(f"You cannot update Annotations once saved online: {self.get_link()}")
            # update_annotation(id_=self.id_, document_id=self.document.id_, project_id=self.project.id_)

        if not self.is_online:
            response = post_document_annotation(
                project_id=self.document.project.id_,
                document_id=self.document.id_,
                start_offset=self.start_offset,
                end_offset=self.end_offset,
                label_id=self.label.id_,
                label_set_id=label_set_id,
                confidence=self.confidence,
                is_correct=self.is_correct,
                revised=self.revised,
                annotation_set=self.annotation_set,
                # bboxes=self.bboxes,
                # selection_bbox=self.selection_bbox,
                page_number=self.page_number,
            )
            if response.status_code == 201:
                json_response = json.loads(response.text)
                self.id_ = json_response["id"]
                new_annotation_added = True
            elif response.status_code == 403:
                logger.error(response.text)
                try:
                    if "In one project you cannot label the same text twice." in response.text:
                        if document_annotations is None:
                            # get the Annotation
                            self.document.update()
                            document_annotations = self.document.annotations()
                        # get the id_ of the existing annotation
                        is_duplicated = False
                        for annotation in document_annotations:
                            if (
                                annotation.start_offset == self.start_offset
                                and annotation.end_offset == self.end_offset
                                and annotation.label == self.label
                            ):
                                logger.error(f"ID of annotation online: {annotation.id_}")
                                self.id_ = annotation.id_
                                is_duplicated = True
                                break

                        # if there isn't a perfect match, the current Annotation is considered incorrect
                        if not is_duplicated:
                            self.is_correct = False

                        new_annotation_added = False
                    else:
                        logger.exception(f"Unknown issue to create Annotation {self} in {self.document}")
                except KeyError:
                    logger.error(f"Not able to save annotation online: {response}")
        return new_annotation_added

    def regex_annotation_generator(self, regex_list) -> List[Span]:
        """
        Build Spans without Labels by regexes.

        :return: Return sorted list of Spans by start_offset
        """
        spans: List[Span] = []
        for regex in regex_list:
            dict_spans = regex_matches(doctext=self.document.text, regex=regex)
            for offset in list(set((x['start_offset'], x['end_offset']) for x in dict_spans)):
                try:
                    span = Span(start_offset=offset[0], end_offset=offset[1], annotation=self)
                    spans.append(span)
                except ValueError as e:
                    logger.error(str(e))
        spans.sort()
        return spans

    def token_append(self, new_regex, regex_quality: int):
        """Append token if it is not a duplicate."""
        category = self.document.category
        regex_to_remove_group_names = re.compile('<.*?>')
        previous_matchers = [re.sub(regex_to_remove_group_names, '', t['regex']) for t in self._tokens]
        found_for_label = [
            re.sub(regex_to_remove_group_names, '', t['regex']) for t in (self.label._evaluations.get(category.id_, []))
        ]
        new_matcher = re.sub(regex_to_remove_group_names, '', new_regex)
        if new_matcher not in previous_matchers + found_for_label:  # only run evaluation if the token is truly new
            evaluation = self.label.evaluate_regex(new_regex, regex_quality=regex_quality, category=category)
            self._tokens.append(evaluation)
            logger.debug(f'Added new regex Token {new_matcher}.')
        else:
            logger.debug(f'Annotation Token {repr(new_matcher)} or regex {repr(new_regex)} does exist.')

    def tokens(self) -> List[str]:
        """Create a list of potential tokens based on Spans of this Annotation."""
        if not self._tokens:
            for span in self.spans:
                # the original string, with harmonized whitespaces
                harmonized_whitespace = suggest_regex_for_string(span.offset_string, replace_numbers=False)
                regex_w = f'(?P<Label_{self.label.id_}_W_{self.id_}_{span.start_offset}>{harmonized_whitespace})'
                self.token_append(new_regex=regex_w, regex_quality=0)
                # the original string, with numbers replaced
                numbers_replaced = suggest_regex_for_string(span.offset_string)
                regex_n = f'(?P<Label_{self.label.id_}_N_{self.id_}_{span.start_offset}>{numbers_replaced})'
                self.token_append(new_regex=regex_n, regex_quality=1)
                # the original string, with characters and numbers replaced
                full_replacement = suggest_regex_for_string(span.offset_string, replace_characters=True)
                regex_f = f'(?P<Label_{self.label.id_}_F_{self.id_}_{span.start_offset}>{full_replacement})'
                self.token_append(new_regex=regex_f, regex_quality=2)
        return self._tokens

    # todo can we circumvent the combined tokens
    def regex(self):
        """Return regex of this annotation."""
        return self.label.combined_tokens(categories=[self.document.category])

    def delete(self, delete_online: bool = True) -> None:
        """Delete Annotation.

        :param delete_online: Whether the Annotation is deleted online or only locally.
        """
        if self.document.is_online and delete_online:
            delete_document_annotation(self.document.id_, self.id_, self.document.project.id_)
            self.document.update()
        else:
            self.document._annotations.remove(self)

    @property
    def spans(self) -> List[Span]:
        """Return default entry to get all Spans of the Annotation."""
        return sorted(self._spans)


class Document(Data):
    """Access the information about one document, which is available online."""

    def __init__(
        self,
        project: 'Project',
        id_: Union[int, None] = None,
        file_url: str = None,
        status=None,
        data_file_name: str = None,
        is_dataset: bool = None,
        dataset_status: int = None,
        updated_at: str = None,
        assignee: int = None,
        category_template: int = None,  # fix for Konfuzio Server API, it's actually an ID of a Category
        category: Category = None,
        text: str = None,
        bbox: dict = None,
        strict_bbox_validation: bool = False,
        pages: list = None,
        update: bool = None,
        copy_of_id: Union[int, None] = None,
        *args,
        **kwargs,
    ):
        """
        Create a Document and link it to its Project.

        :param id_: ID of the Document
        :param project: Project where the Document belongs to
        :param file_url: URL of the document
        :param status: Status of the document
        :param data_file_name: File name of the document
        :param is_dataset: Is dataset or not. (bool)
        :param dataset_status: Dataset status of the Document (e.g. training)
        :param updated_at: Updated information
        :param assignee: Assignee of the Document
        :param bbox: Bounding box information per character in the PDF (dict)
        :param strict_bbox_validation: If False, it allows bounding boxes to have zero width or height. This option is
                        available and defaults to False for compatibility reasons since some OCR engines can sometimes
                        return character level bboxes with zero width or height.
        :param pages: List of page sizes.
        :param update: Annotations, Annotation Sets will not be loaded by default. True will load it from the API.
                        False from local files
        :param copy_of_id: ID of the Document that originated the current Document
        """
        self._no_label_annotation_set = None
        self.id_local = next(Data.id_iter)
        self.id_ = id_
        self.assignee = assignee
        self._annotations: List[Annotation] = None
        self._annotation_sets: List[AnnotationSet] = None
        self.file_url = file_url
        self.is_dataset = is_dataset
        self.dataset_status = dataset_status
        self.assignee = assignee
        self._update = update
        self.copy_of_id = copy_of_id

        if project and category_template:
            self.category = project.get_category_by_id(category_template)
        elif category:
            self.category = category
        else:
            self.category = None

        if updated_at:
            self.updated_at = dateutil.parser.isoparse(updated_at)
        else:
            self.updated_at = None

        self.name = data_file_name
        self.status = status  # status of document online
        self.project = project
        self._force_offline = project._force_offline
        project.add_document(self)  # check for duplicates by ID before adding the Document to the project

        # use hidden variables to store low volume information in instance
        self._text: str = text
        self._text_hash = None
        self._characters: Dict[int, Bbox] = None
        self._bbox_hash = None
        self._bbox_json = bbox
        self.bboxes_available: bool = True if (self.is_online or self._bbox_json) else False
        self._strict_bbox_validation = strict_bbox_validation
        self._hocr = None
        self._pages: List[Page] = []
        self._n_pages = None

        # prepare local setup for document
        if self.is_online:
            pathlib.Path(self.document_folder).mkdir(parents=True, exist_ok=True)
        self.annotation_file_path = os.path.join(self.document_folder, "annotations.json5")
        self.annotation_set_file_path = os.path.join(self.document_folder, "annotation_sets.json5")
        self.txt_file_path = os.path.join(self.document_folder, "document.txt")
        self.hocr_file_path = os.path.join(self.document_folder, "document.hocr")
        self.pages_file_path = os.path.join(self.document_folder, "pages.json5")
        self.bbox_file_path = os.path.join(self.document_folder, "bbox.zip")
        self.bio_scheme_file_path = os.path.join(self.document_folder, "bio_scheme.txt")

        if pages:
            self.pages()  # create page instances

    def __repr__(self):
        """Return the name of the Document incl. the ID."""
        if self.copy_of_id:
            return f"Document {self.name} ({self.copy_of_id})"
        else:
            return f"Document {self.name} ({self.id_})"

    @property
    def file_path(self):
        """Return path to file."""
        return os.path.join(self.document_folder, amend_file_name(self.name))

    @property
    def ocr_file_path(self):
        """Return path to OCR PDF file."""
        return os.path.join(self.document_folder, amend_file_name(self.name, append_text="ocr", new_extension=".pdf"))

    @property
    def number_of_pages(self) -> int:
        """Calculate the number of pages."""
        if self._n_pages is None:
            self._n_pages = len(self.text.split('\f'))
        return self._n_pages

    @property
    def number_of_lines(self) -> int:
        """Calculate the number of lines."""
        return len(self.text.replace('\f', '\n').split('\n'))

    @property
    def no_label_annotation_set(self) -> AnnotationSet:
        """
        Return the Annotation Set for project.no_label Annotations.

        We need to load the Annotation Sets from Server first (call self.annotation_sets()).
        If we create the no_label_annotation_set in the first place, the data from the Server is not be loaded
        anymore because _annotation_sets will no longer be None.
        """
        if self._no_label_annotation_set is None:
            self.annotation_sets()
            self._no_label_annotation_set = AnnotationSet(document=self, label_set=self.project.no_label_set)

        return self._no_label_annotation_set

    def spans(
        self,
        label: Label = None,
        use_correct: bool = False,
        start_offset: int = 0,
        end_offset: int = None,
        fill: bool = False,
    ) -> List[Span]:
        """Return all Spans of the Document."""
        spans = []

        annotations = self.annotations(
            label=label, use_correct=use_correct, start_offset=start_offset, end_offset=end_offset, fill=fill
        )

        for annotation in annotations:
            for span in annotation.spans:
                if span not in spans:
                    spans.append(span)

        # if self.spans() == list(set(self.spans())):
        #     # todo deduplicate Spans. One text offset in a document can ber referenced by many Spans of Annotations
        #     raise NotImplementedError

        return sorted(spans)

    def eval_dict(self, use_correct=False) -> List[dict]:
        """Use this dict to evaluate Documents. The speciality: For every Span of an Annotation create one entry."""
        result = []
        annotations = self.annotations(use_correct=use_correct)
        if not annotations:  # if there are no Annotations in this Documents
            result.append(Span(start_offset=0, end_offset=0).eval_dict())
        else:
            for annotation in annotations:
                result += annotation.eval_dict

        return result

    def check_bbox(self) -> None:
        """
        Run validation checks on the document text and bboxes.

        This is run when the document is initialized, and usually it's not needed to be run again because a document's
        text and bboxes are not expected to change within the Konfuzio Server.

        You can run this manually instead if your pipeline allows changing the text or the bbox during the lifetime of
        a document. Will raise ValueError if the bboxes don't match with the text of the document, or if bboxes have
        invalid coordinates (outside page borders) or invalid size (negative width or height).

        This check is usually slow, and it can be made faster by calling document.set_text_bbox_hashes() right after
        initializing the document, which will enable running a hash comparison during this check.
        """
        warn('WIP: Modifications before the next stable release expected.', FutureWarning, stacklevel=2)
        if self._check_text_or_bbox_modified():
            self._characters = None
            _ = self.bboxes

    def __deepcopy__(self, memo) -> 'Document':
        """Create a new Document of the instance."""
        copy_id = self.id_ if self.id_ else self.copy_of_id
        document = Document(
            id_=None,
            project=self.project,
            category=self.category,
            text=self.text,
            copy_of_id=copy_id,
            bbox=self.get_bbox(),
        )
        for page in self.pages():
            _ = Page(
                id_=None,
                document=document,
                start_offset=page.start_offset,
                end_offset=page.end_offset,
                number=page.number,
                original_size=(page.width, page.height),
            )
        return document

    def check_annotations(self, update_document: bool = False) -> bool:
        """Check if Annotations are valid - no duplicates and correct Category."""
        valid = True
        assignee = None

        try:
            self.get_annotations()

        except ValueError as error_message:
            valid = False

            if "is a duplicate of" in str(error_message):
                logger.error(f'{self} has duplicated Annotations.')
                assignee = 1101  # duplicated-annotation@konfuzio.com

            elif "related to" in str(error_message):
                logger.error(f'{self} has Annotations from an incorrect Category.')
                assignee = 1118  # category-issue@konfuzio.com

            else:
                raise ValueError('Error not expected.')

        if update_document and assignee is not None:
            # set the dataset status of the Document to Excluded
            update_document_konfuzio_api(document_id=self.id_, file_name=self.name, dataset_status=4, assignee=assignee)

        return valid

    def annotation_sets(self):
        """Return Annotation Sets of Documents."""
        if self._annotation_sets is not None:
            return self._annotation_sets
        if self.is_online and not is_file(self.annotation_set_file_path, raise_exception=False):
            self.download_document_details()
        if is_file(self.annotation_set_file_path, raise_exception=False):
            with open(self.annotation_set_file_path, "r") as f:
                raw_annotation_sets = json.load(f)
            # first load all Annotation Sets before we create Annotations
            for raw_annotation_set in raw_annotation_sets:
                _ = AnnotationSet(
                    id_=raw_annotation_set["id"],
                    document=self,
                    label_set=self.project.get_label_set_by_id(raw_annotation_set["section_label"]),
                )
        elif self._annotation_sets is None:
            self._annotation_sets = []  # Annotation sets cannot be loaded from Konfuzio Server
        return self._annotation_sets

    def annotations(
        self,
        label: Label = None,
        use_correct: bool = True,
        ignore_below_threshold: bool = False,
        start_offset: int = 0,
        end_offset: int = None,
        fill: bool = False,
    ) -> List[Annotation]:
        """
        Filter available annotations.

        :param label: Label for which to filter the Annotations.
        :param use_correct: If to filter by correct annotations.
        :param ignore_below_threshold: To filter out annotations with confidence below Label prediction threshold.
        :return: Annotations in the document.
        """
        self.get_annotations()
        start_offset = max(start_offset, 0)
        if end_offset:
            end_offset = min(end_offset, len(self.text))
        annotations: List[Annotation] = []
        add = False
        for annotation in self._annotations:
            # filter by correct information
            if not annotation.is_correct:
                if ignore_below_threshold and (
                    not annotation.confidence or annotation.confidence < annotation.label.threshold
                ):
                    continue
            for span in annotation.spans:
                if (use_correct and annotation.is_correct) or not use_correct:
                    # todo: add option to filter for overruled Annotations where mult.=F
                    # todo: add option to filter for overlapping Annotations, `add_annotation` just checks for identical
                    # filter by start and end offset, include annotations that extend into the offset
                    if start_offset is not None and end_offset is not None:  # if the start and end offset are specified
                        latest_start = max(span.start_offset, start_offset)
                        earliest_end = min(span.end_offset, end_offset)
                        is_overlapping = latest_start - earliest_end < 0
                    else:
                        is_overlapping = True

                    if label is not None:  # filter by label
                        if label == annotation.label and is_overlapping:
                            add = True
                    elif is_overlapping:
                        add = True
            # as multiline Annotations will be added twice
            if add:
                annotations.append(annotation)
                add = False

        if fill:
            # todo: we cannot assure that the Document has a Category, so Annotations must not require label_set
            spans = [range(span.start_offset, span.end_offset) for anno in annotations for span in anno.spans]
            if end_offset is None:
                end_offset = len(self.text)
            missings = get_missing_offsets(start_offset=start_offset, end_offset=end_offset, annotated_offsets=spans)

            for missing in missings:
                new_spans = []
                offset_text = self.text[missing.start : missing.stop]
                # we split Spans which span multiple lines, so that one Span comprises one line
                offset_of_offset = 0
                line_breaks = [
                    offset_line for offset_line in re.split(r'(\n|\f)', offset_text) if offset_line != ''
                ]  # , '\n', '\f'}]
                if not line_breaks:
                    continue
                for offset in line_breaks:
                    start = missing.start + offset_of_offset
                    offset_of_offset += len(offset)
                    end = missing.start + offset_of_offset
                    new_span = Span(start_offset=start, end_offset=end)
                    new_spans.append(new_span)

                new_annotation = Annotation(
                    document=self,
                    annotation_set=self.no_label_annotation_set,
                    label=self.project.no_label,
                    label_set=self.project.no_label_set,
                    spans=new_spans,
                )

                annotations.append(new_annotation)

        return sorted(annotations)

    def view_annotations(self, start_offset: int = 0, end_offset: int = None) -> List[Annotation]:
        """Get the best Annotations, where the Spans are not overlapping."""
        self.get_annotations()
        annotations: List[Annotation] = []

        filled = 0  # binary number keeping track of filled offsets
        priority_annotations = sorted(
            self.annotations(use_correct=False, start_offset=start_offset, end_offset=end_offset),
            key=lambda x: (
                not x.is_correct,  # x.is_correct == True first
                -x.confidence if x.confidence else 0,  # higher confidence first
                min([span.start_offset for span in x.spans]),
            ),
        )

        no_label_duplicates = set()  # for top annotation filter
        for annotation in priority_annotations:
            if annotation.confidence is not None and annotation.label.threshold > annotation.confidence:
                continue
            if not annotation.is_correct and annotation.revised:  # if marked as incorrect by user
                continue
            spans_num = 0
            for span in annotation.spans:
                for i in range(span.start_offset, span.end_offset):
                    spans_num |= 1 << i
            if spans_num & filled:
                # if there's overlap
                continue
            # if (
            #     annotation.is_correct is False
            #     and annotation.label.has_multiple_top_candidates is False
            #     and annotation.label.id_ in no_label_duplicates
            # ):
            #     continue
            annotations.append(annotation)
            filled |= spans_num
            if not annotation.label.has_multiple_top_candidates:
                no_label_duplicates.add(annotation.label.id_)

        return sorted(annotations)

    @property
    def document_folder(self):
        """Get the path to the folder where all the Document information is cached locally."""
        return os.path.join(self.project.documents_folder, str(self.id_))

    def get_file(self, ocr_version: bool = True, update: bool = False):
        """
        Get OCR version of the original file.

        :param ocr_version: Bool to get the ocr version of the original file
        :param update: Update the downloaded file even if it is already available
        :return: Path to the selected file.
        """
        if ocr_version:
            file_path = self.ocr_file_path
        else:
            file_path = self.file_path

        if self.status[0] == 2 and (not file_path or not is_file(file_path, raise_exception=False) or update):
            pdf_content = download_file_konfuzio_api(self.id_, ocr=ocr_version, session=self.session)
            with open(file_path, "wb") as f:
                f.write(pdf_content)

        return file_path

    def get_images(self, update: bool = False):
        """
        Get Document Pages as PNG images.

        :param update: Update the downloaded images even they are already available
        :return: Path to PNG files.
        """
        return [page.get_image(update=update) for page in self.pages()]

    def download_document_details(self):
        """Retrieve data from a Document online in case documented has finished processing."""
        if self.is_online and self.status and self.status[0] == 2:
            data = get_document_details(document_id=self.id_, project_id=self.project.id_, session=self.session)

            # write a file, even there are no annotations to support offline work
            with open(self.annotation_file_path, "w") as f:
                json.dump(data["annotations"], f, indent=2, sort_keys=True)

            with open(self.annotation_set_file_path, "w") as f:
                json.dump(data["sections"], f, indent=2, sort_keys=True)

            with open(self.txt_file_path, "w", encoding="utf-8") as f:
                f.write(data["text"])

            with open(self.pages_file_path, "w") as f:
                json.dump(data["pages"], f, indent=2, sort_keys=True)
        else:
            raise NotImplementedError

        return self

    def add_annotation(self, annotation: Annotation):
        """Add an annotation to a document.

        :param annotation: Annotation to add in the document
        :return: Input annotation.
        """
        if self._annotations is None:
            self.annotations()
        if annotation not in self._annotations:
            # Hotfix Text Annotation Server:
            #  Annotation belongs to a Label / Label Set that does not relate to the Category of the Document.
            # todo: add test that the Label and Label Set of an Annotation belong to the Category of the Document
            if self.category is not None:
                if annotation.label_set is not None:
                    if annotation.label_set.categories:
                        if self.category in annotation.label_set.categories:
                            self._annotations.append(annotation)
                        else:
                            raise ValueError(
                                f'We cannot add {annotation} related to {annotation.label_set.categories} to {self} '
                                f'as the document has {self.category}'
                            )
                    else:
                        raise ValueError(f'{annotation} uses Label Set without Category, cannot be added to {self}.')
                else:
                    raise ValueError(f'{annotation} has no Label Set, which cannot be added to {self}.')
            else:
                raise ValueError(f'We cannot add {annotation} to {self} where the category is {self.category}')
        else:
            duplicated = [x for x in self._annotations if x == annotation]
            raise ValueError(f'In {self} the {annotation} is a duplicate of {duplicated} and will not be added.')

        return self

    def get_annotation_by_id(self, annotation_id: int) -> Annotation:
        """
        Return an Annotation by ID, searching within the document.

        :param id_: ID of the Annotation to get.
        """
        result = None
        if self._annotations is None:
            self.annotations()
        for annotation in self._annotations:
            if annotation.id_ == annotation_id:
                result = annotation
                break
        if result:
            return result
        else:
            raise IndexError(f"Annotation {annotation_id} is not part of {self}.")

    def add_annotation_set(self, annotation_set: AnnotationSet):
        """Add the Annotation Sets to the document."""
        if annotation_set.document and annotation_set.document != self:
            raise ValueError('One Annotation Set must only belong to one document.')
        if self._annotation_sets is None:
            self._annotation_sets = []
        if annotation_set not in self._annotation_sets:
            self._annotation_sets.append(annotation_set)
        else:
            raise ValueError(f'In {self} the {annotation_set} is a duplicate and will not be added.')
        return self

    def get_annotation_set_by_id(self, id_: int) -> AnnotationSet:
        """
        Return an Annotation Set by ID.

        :param id_: ID of the Annotation Set to get.
        """
        result = None
        if self._annotation_sets is None:
            self.annotation_sets()
        for annotation_set in self._annotation_sets:
            if annotation_set.id_ == id_:
                result = annotation_set
        if result:
            return result
        else:
            logger.error(f"Annotation Set {id_} is not part of Document {self.id_}.")
            raise IndexError

    def get_text_in_bio_scheme(self, update=False) -> List[Tuple[str, str]]:
        """
        Get the text of the Document in the BIO scheme.

        :param update: Update the bio annotations even they are already available
        :return: list of tuples with each word in the text and the respective label
        """
        converted_text = []
        if not is_file(self.bio_scheme_file_path, raise_exception=False) or update:
            converted_text = convert_to_bio_scheme(self)
            with open(self.bio_scheme_file_path, "w", encoding="utf-8") as f:
                for word, tag in converted_text:
                    f.writelines(word + " " + tag + "\n")
                f.writelines("\n")
        else:
            with open(self.bio_scheme_file_path, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    if not line.strip():
                        continue
                    split_line = line.strip().split(' ')
                    word = split_line[0]
                    tag = ' '.join(split_line[1:])  # tag allowed to have multiple words
                    converted_text.append((word, tag))

        return converted_text

    def get_bbox(self) -> Dict:
        """
        Get bbox information per character of file. We don't store bbox as an attribute to save memory.

        :return: Bounding box information per character in the document.
        """
        if self._bbox_json:
            bbox = self._bbox_json
        elif is_file(self.bbox_file_path, raise_exception=False):
            with zipfile.ZipFile(self.bbox_file_path, "r") as archive:
                bbox = json.loads(archive.read('bbox.json5'))
        elif self.is_online and self.status and self.status[0] == 2:
            # todo check for self.project.id_ and self.id_ and ?
            logger.warning(f'Start downloading bbox files of {len(self.text)} characters for {self}.')
            bbox = get_document_details(document_id=self.id_, project_id=self.project.id_, extra_fields="bbox")['bbox']
            # Use the `zipfile` module: `compresslevel` was added in Python 3.7
            with zipfile.ZipFile(
                self.bbox_file_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
            ) as zip_file:
                # Dump JSON data
                dumped: str = json.dumps(bbox, indent=2, sort_keys=True)
                # Write the JSON data into `data.json` *inside* the ZIP file
                zip_file.writestr('bbox.json5', data=dumped)
                # Test integrity of compressed archive
                zip_file.testzip()
        else:
            self.bboxes_available = False
            bbox = {}

        return bbox

    @property
    def _hashable_characters(self) -> Optional[frozenset]:
        """Convert bbox dict into a hashable type."""
        return frozenset(self._characters) if self._characters is not None else None

    def set_text_bbox_hashes(self) -> None:
        """Update hashes of document text and bboxes. Can be used for checking later on if any changes happened."""
        self._text_hash = hash(self._text)
        self._bbox_hash = hash(self._hashable_characters)

    def _check_text_or_bbox_modified(self) -> bool:
        """Check if either the document text or its bboxes have been modified in memory."""
        text_modified = self._text_hash != hash(self._text)
        bbox_modified = self._bbox_hash != hash(self._hashable_characters)
        return text_modified or bbox_modified

    @property
    def bboxes(self) -> Dict[int, Bbox]:
        """Use the cached bbox version."""
        warn('WIP: Modifications before the next stable release expected.', FutureWarning, stacklevel=2)
        if self.bboxes_available and self._characters is None:
            bbox = self.get_bbox()
            boxes = {}
            for character_index, box in bbox.items():
                x0 = box.get('x0')
                x1 = box.get('x1')
                y0 = box.get('y0')
                y1 = box.get('y1')
                page_index = box.get('page_number') - 1
                page = self.get_page_by_index(page_index=page_index)
                box_character = box.get('text')
                document_character = self.text[int(character_index)]
                if box_character not in [' ', '\f', '\n'] and box_character != document_character:
                    raise ValueError(
                        f'{self} Bbox provides Character "{box_character}" document text refers to '
                        f'"{document_character}" with ID "{character_index}".'
                    )
                boxes[int(character_index)] = Bbox(
                    x0=x0, x1=x1, y0=y0, y1=y1, page=page, strict_validation=self._strict_bbox_validation
                )
            self._characters = boxes
        return self._characters

    def set_bboxes(self, characters: Dict[int, Bbox]):
        """Set character Bbox dictionary."""
        characters = {int(key): bbox for key, bbox in characters.items()}

        for key, bbox in characters.items():
            bbox._valid(self._strict_bbox_validation)

        self._characters = characters
        self.bboxes_available = True

    @property
    def text(self):
        """Get Document text. Once loaded stored in memory."""
        if self._text is not None:
            return self._text
        if self.is_online and not is_file(self.txt_file_path, raise_exception=False):
            self.download_document_details()
        if is_file(self.txt_file_path, raise_exception=False):
            with open(self.txt_file_path, "r", encoding="utf-8") as f:
                self._text = f.read()
        return self._text

    def add_page(self, page: Page):
        """Add a Page to a Document."""
        if page not in self._pages:
            self._pages.append(page)
        else:
            raise ValueError(f'In {self} the {page} is a duplicate and will not be added.')

    def get_page_by_index(self, page_index: int):
        """Return the Page by index."""
        for page in self.pages():
            if page.index == page_index:
                return page
        raise IndexError(f'Page with Index {page_index} not available in {self}')

    def pages(self) -> List[Page]:
        """Get Pages of Document."""
        if self._pages:
            return self._pages
        if self.is_online and not is_file(self.pages_file_path, raise_exception=False):
            self.download_document_details()
            is_file(self.pages_file_path)
        if is_file(self.pages_file_path, raise_exception=False):
            with open(self.pages_file_path, "r") as f:
                pages_data = json.loads(f.read())

            page_texts = self.text.split('\f')
            assert len(page_texts) == len(pages_data)
            start_offset = 0
            for page_index, page_data in enumerate(pages_data):
                page_text = page_texts[page_index]
                end_offset = start_offset + len(page_text)
                _ = Page(
                    id_=page_data['id'],
                    document=self,
                    number=page_data['number'],
                    original_size=page_data['original_size'],
                    start_offset=start_offset,
                    end_offset=end_offset,
                )
                start_offset = end_offset + 1

        return self._pages

    @property
    def hocr(self):
        """Get HOCR of Document. Once loaded stored in memory."""
        if self._hocr:
            pass
        elif is_file(self.hocr_file_path, raise_exception=False):
            # hocr might not be available (depends on the Project settings)
            with open(self.hocr_file_path, "r", encoding="utf-8") as f:
                self._hocr = f.read()
        else:
            if self.status[0] == 2:
                data = get_document_details(
                    document_id=self.id_, project_id=self.project.id_, session=self.session, extra_fields="hocr"
                )

                if 'hocr' in data.keys() and data['hocr']:
                    self._hocr = data['hocr']
                    with open(self.hocr_file_path, "w", encoding="utf-8") as f:
                        f.write(self._hocr)
                else:
                    logger.warning(f'Please enable HOCR in {self.project} and upload {self} again to create HOCR.')

        return self._hocr

    def update(self):
        """Update document information."""
        self.delete()
        self.download_document_details()
        return self

    def delete(self):
        """Delete all local information for the document."""
        try:
            shutil.rmtree(self.document_folder)
        except FileNotFoundError:
            pass
        pathlib.Path(self.document_folder).mkdir(parents=True, exist_ok=True)
        self._annotations = None
        self._annotation_sets = None

    def merge_vertical(self, only_multiline_labels=True):
        """
        Merge Annotations with the same Label.

        :param only_multiline_labels: Only merge if multiline Label Annotation in category training set
        """
        labels_dict = {}
        for label in self.project.labels:
            if not only_multiline_labels or label.has_multiline_annotations(categories=[self.category]):
                labels_dict[label.id_local] = []

        for annotation in self.annotations(use_correct=False, ignore_below_threshold=True):
            if annotation.label.id_local in labels_dict:
                labels_dict[annotation.label.id_local].append(annotation)

        for label_id in labels_dict:
            buffer = []
            for annotation in labels_dict[label_id]:
                for span in annotation.spans:
                    # remove all spans in buffer more than 1 line apart
                    while buffer and span.line_index > buffer[0].line_index + 1:
                        buffer.pop(0)

                    if buffer and buffer[-1].page != span.page:
                        buffer = [span]
                        continue

                    # Do not merge new Span if Annotation part of AnnotationSet with more than 1 Annotation
                    # (except default annotationSet)
                    if (
                        span.annotation.annotation_set
                        and not span.annotation.annotation_set.label_set.is_default
                        and len(
                            span.annotation.annotation_set.annotations(use_correct=False, ignore_below_threshold=True)
                        )
                        > 1
                    ):
                        buffer.append(span)
                        continue
                    if len(annotation.spans) > 1:
                        buffer.append(span)
                        continue

                    for candidate in buffer:
                        # only looking for elements in line above
                        if candidate.line_index == span.line_index:
                            break
                        # overlap in x
                        # or next line
                        if (
                            not (span.bbox().x0 > candidate.bbox().x1 or span.bbox().x1 < candidate.bbox().x0)
                        ) or self.text[candidate.end_offset : span.start_offset].replace(' ', '').replace(
                            '\n', ''
                        ) == '':
                            span.annotation.delete()
                            span.annotation = None
                            candidate.annotation.add_span(span)
                            buffer.remove(candidate)
                    buffer.append(span)

    def evaluate_regex(self, regex, label: Label, annotations: List['Annotation'] = None):
        """Evaluate a regex based on the Document."""
        start_time = time.time()
        findings_in_document = regex_matches(
            doctext=self.text,
            regex=regex,
            keep_full_match=False,
            filtered_group=f'Label_{label.id_}'
            # filter by name of label: one regex can match multiple labels
        )
        processing_time = time.time() - start_time
        correct_findings = []

        label_annotations = self.annotations(label=label)

        label_annotations_offsets = {
            (span.start_offset, span.end_offset): ann for ann in label_annotations for span in ann.spans
        }

        for finding in findings_in_document:
            key = (finding['start_offset'], finding['end_offset'])
            if key in label_annotations_offsets:
                correct_findings.append(label_annotations_offsets[key])

        try:
            annotation_precision = len(correct_findings) / len(findings_in_document)
        except ZeroDivisionError:
            annotation_precision = 0

        try:
            annotation_recall = len(correct_findings) / len(label_annotations)
        except ZeroDivisionError:
            annotation_recall = 0

        try:
            f1_score = 2 * (annotation_precision * annotation_recall) / (annotation_precision + annotation_recall)
        except ZeroDivisionError:
            f1_score = 0
        return {
            'id': self.id_local,
            'regex': regex,
            # processing_time time can vary slightly between runs, round to ms so that this variation does not affect
            # the choice of regexes when values are below ms and metrics are the same
            'runtime': round(processing_time, 3),
            'count_total_findings': len(findings_in_document),
            'count_total_correct_findings': len(correct_findings),
            'count_correct_annotations': len(label_annotations),
            'count_correct_annotations_not_found': len(correct_findings) - len(label_annotations),
            'doc_matched': len(correct_findings) > 0,
            'annotation_precision': annotation_precision,
            'document_recall': 0,  # keep this key to be able to use the function get_best_regex
            'annotation_recall': annotation_recall,
            'f1_score': f1_score,
            'correct_findings': correct_findings,
        }

    def get_annotations(self) -> List[Annotation]:
        """Get Annotations of the Document."""
        if self.category is None:
            raise ValueError(f'Document {self} without Category must not have Annotations')

        annotation_file_exists = is_file(self.annotation_file_path, raise_exception=False)
        annotation_set_file_exists = is_file(self.annotation_set_file_path, raise_exception=False)

        if self._update or (self.is_online and (self._annotations is None or self._annotation_sets is None)):

            if self.is_online and (not annotation_file_exists or not annotation_set_file_exists or self._update):
                self.update()  # delete the meta of the Document details and download them again
                self._update = None  # Make sure we don't repeat to load once updated.

            self._annotation_sets = None  # clean Annotation Sets to not create duplicates
            self.annotation_sets()

            self._annotations = []  # clean Annotations to not create duplicates
            # We read the annotation file that we just downloaded
            with open(self.annotation_file_path, 'r') as f:
                raw_annotations = json.load(f)

            for raw_annotation in raw_annotations:
                raw_annotation['annotation_set_id'] = raw_annotation.pop('section')
                raw_annotation['label_set_id'] = raw_annotation.pop('section_label_id')
                _ = Annotation(document=self, id_=raw_annotation['id'], **raw_annotation)
            self._update = None  # Make sure we don't repeat to load once loaded.

        if self._annotations is None:
            self.annotation_sets()
            self._annotations = []
            # We load the annotation file if it exists
            if annotation_file_exists:
                with open(self.annotation_file_path, 'r') as f:
                    raw_annotations = json.load(f)

                for raw_annotation in raw_annotations:
                    raw_annotation['annotation_set_id'] = raw_annotation.pop('section')
                    raw_annotation['label_set_id'] = raw_annotation.pop('section_label_id')
                    _ = Annotation(document=self, id_=raw_annotation['id'], **raw_annotation)

        return self._annotations


class Project(Data):
    """Access the information of a Project."""

    def __init__(self, id_: Union[int, None], project_folder=None, update=False, **kwargs):
        """
        Set up the Data using the Konfuzio Host.

        :param id_: ID of the Project
        :param project_folder: Set a Project root folder, if empty "data_<id_>" will be used.
        """
        self.id_local = next(Data.id_iter)
        self.id_ = id_  # A Project with None ID is not retrieved from the HOST
        self._project_folder = project_folder
        self.categories: List[Category] = []
        self._label_sets: List[LabelSet] = []
        self._labels: List[Label] = []
        self._documents: List[Document] = []
        self._meta_data = []

        # paths
        self.meta_file_path = os.path.join(self.project_folder, "documents_meta.json5")
        self.labels_file_path = os.path.join(self.project_folder, "labels.json5")
        self.label_sets_file_path = os.path.join(self.project_folder, "label_sets.json5")

        if self.id_ or self._project_folder:
            self.get(update=update)

        # todo: list of Categories related to NO LABEL SET can be outdated, i.e. if the number of Categories changes
        self.no_label_set = LabelSet(project=self, categories=self.categories)
        self.no_label_set.name_clean = 'NO_LABEL_SET'
        self.no_label_set.name = 'NO_LABEL_SET'
        self.no_label = Label(project=self, text='NO_LABEL', label_sets=[self.no_label_set])
        self.no_label.name_clean = 'NO_LABEL'

    def __repr__(self):
        """Return string representation."""
        return f"Project {self.id_}"

    @property
    def documents(self):
        """Return Documents with status training."""
        return [doc for doc in self._documents if doc.dataset_status == 2]

    @property
    def virtual_documents(self):
        """Return Documents created virtually."""
        return [doc for doc in self._documents if doc.dataset_status is None or doc.id_ is None]

    @property
    def test_documents(self):
        """Return Documents with status test."""
        return [doc for doc in self._documents if doc.dataset_status == 3]

    @property
    def excluded_documents(self):
        """Return Documents which have been excluded."""
        return [doc for doc in self._documents if doc.dataset_status == 4]

    @property
    def preparation_documents(self):
        """Return Documents with status test."""
        return [doc for doc in self._documents if doc.dataset_status == 1]

    @property
    def no_status_documents(self):
        """Return Documents with status test."""
        return [doc for doc in self._documents if doc.dataset_status == 0]

    @property
    def project_folder(self) -> str:
        """Calculate the data document_folder of the Project."""
        if self._project_folder is not None:
            return self._project_folder
        else:
            return f"data_{self.id_}"

    @property
    def regex_folder(self) -> str:
        """Calculate the regex folder of the Project."""
        return os.path.join(self.project_folder, "regex")

    @property
    def documents_folder(self) -> str:
        """Calculate the regex folder of the Project."""
        return os.path.join(self.project_folder, "documents")

    @property
    def model_folder(self) -> str:
        """Calculate the model folder of the Project."""
        return os.path.join(self.project_folder, "models")

    def write_project_files(self):
        """Overwrite files with Project, Label, Label Set information."""
        data = get_project_details(project_id=self.id_)
        with open(self.label_sets_file_path, "w") as f:
            json.dump(data['section_labels'], f, indent=2, sort_keys=True)
        with open(self.labels_file_path, "w") as f:
            json.dump(data['labels'], f, indent=2, sort_keys=True)

        meta_data = get_meta_of_files(project_id=self.id_, session=self.session)
        with open(self.meta_file_path, "w") as f:
            json.dump(meta_data, f, indent=2, sort_keys=True)
        return self

    def get(self, update=False):
        """
        Access meta information of the Project.

        :param update: Update the downloaded information even it is already available
        """
        if is_file(self.meta_file_path, raise_exception=False):
            logger.debug("Keep your local information about Documents to be able to do a partial update.")
            with open(self.meta_file_path, "r") as f:
                self.old_meta_data = json.load(f)
        else:
            self.old_meta_data = []

        pathlib.Path(self.project_folder).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.documents_folder).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.regex_folder).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.model_folder).mkdir(parents=True, exist_ok=True)

        if self.id_ and (not is_file(self.meta_file_path, raise_exception=False) or update):
            self.write_project_files()
        self.get_meta(reload=True)
        self.get_labels(reload=True)
        self.get_label_sets(reload=True)
        self.get_categories()
        self.init_or_update_document()
        return self

    def add_label_set(self, label_set: LabelSet):
        """
        Add Label Set to Project, if it does not exist.

        :param label_set: Label Set to add in the Project
        """
        if label_set not in self._label_sets:
            self._label_sets.append(label_set)
        else:
            raise ValueError(f'In {self} the {label_set} is a duplicate and will not be added.')

    def add_category(self, category: Category):
        """
        Add Category to Project, if it does not exist.

        :param category: Category to add in the Project
        """
        if category not in self.categories:
            self.categories.append(category)
        else:
            raise ValueError(f'In {self} the {category} is a duplicate and will not be added.')

    def add_label(self, label: Label):
        """
        Add Label to Project, if it does not exist.

        :param label: Label to add in the Project
        """
        if label not in self._labels:
            self._labels.append(label)
        else:
            raise ValueError(f'In {self} the {label} is a duplicate and will not be added.')

    def add_document(self, document: Document):
        """Add Document to Project, if it does not exist."""
        if document not in self._documents:
            self._documents.append(document)
        else:
            raise ValueError(f'In {self} the {document} is a duplicate and will not be added.')

    def get_meta(self, reload=False):
        """
        Get the list of all Documents in the Project and their information.

        :return: Information of the Documents in the Project.
        """
        if not self._meta_data or reload:
            with open(self.meta_file_path, "r") as f:
                self._meta_data = json.load(f)
        return self._meta_data

    @property
    def meta_data(self):
        """Return Project meta data."""
        if not self._meta_data:
            self.get_meta()
        return self._meta_data

    def get_categories(self):
        """Load Categories for all Label Sets in the Project."""
        for label_set in self.label_sets:
            if label_set.is_default:
                # the _default_of_label_set_ids are the label sets used by the category
                pass
            else:
                # the _default_of_label_set_ids are the categories the label set is used in
                for label_set_id in label_set._default_of_label_set_ids:
                    category = self.get_category_by_id(label_set_id)
                    label_set.add_category(category)  # The Label Set is linked to a Category it created
                    category.add_label_set(label_set)

    def get_label_sets(self, reload=False):
        """Get LabelSets in the Project."""
        if not self._label_sets or reload:
            with open(self.label_sets_file_path, "r") as f:
                label_sets_data = json.load(f)

            self._label_sets = []  # clean up Label Sets to not create duplicates
            self.categories = []  # clean up Labels to not create duplicates
            for label_set_data in label_sets_data:
                label_set = LabelSet(project=self, id_=label_set_data['id'], **label_set_data)
                if label_set.is_default:
                    category = Category(project=self, id_=label_set_data['id'], **label_set_data)
                    category.label_sets.append(label_set)
                    label_set.categories.append(category)  # Konfuzio Server mixes the concepts, we use two instances
                    # self.add_category(category)

        return self._label_sets

    @property
    def label_sets(self):
        """Return Project LabelSets."""
        if not self._label_sets:
            self.get_label_sets()
        return self._label_sets

    def get_labels(self, reload=False) -> Label:
        """Get ID and name of any Label in the Project."""
        if not self._labels or reload:
            with open(self.labels_file_path, "r") as f:
                labels_data = json.load(f)
            self._labels = []  # clean up Labels to not create duplicates
            for label_data in labels_data:
                # Remove the project from label_data
                label_data.pop("project", None)
                Label(project=self, id_=label_data['id'], **label_data)

        return self._labels

    @property
    def labels(self):
        """Return Project Labels."""
        if not self._labels:
            self.get_labels()
        return self._labels

    def init_or_update_document(self):
        """Initialize Document to then decide about full, incremental or no update."""
        self._documents = []  # clean up Documents to not create duplicates
        for document_data in self.meta_data:
            if document_data['status'][0] == 2:  # NOQA - hotfix for Text Annotation Server # todo add test
                new_date = document_data["updated_at"]
                new = True
                updated = None
                if self.old_meta_data:
                    new = document_data["id"] not in [doc["id"] for doc in self.old_meta_data]
                    if not new:
                        last_date = [d["updated_at"] for d in self.old_meta_data if d['id'] == document_data["id"]][0]
                        updated = dateutil.parser.isoparse(new_date) > dateutil.parser.isoparse(last_date)

                if updated:
                    doc = Document(project=self, update=True, id_=document_data['id'], **document_data)
                    logger.info(f'{doc} was updated, we will download it again as soon you use it.')
                elif new:
                    doc = Document(project=self, update=True, id_=document_data['id'], **document_data)
                    logger.info(f'{doc} is not available on your machine, we will download it as soon you use it.')
                else:
                    doc = Document(project=self, update=False, id_=document_data['id'], **document_data)
                    logger.debug(f'Load local version of {doc} from {new_date}.')

    def get_document_by_id(self, document_id: int) -> Document:
        """Return Document by its ID."""
        for document in self._documents:
            if document.id_ == document_id:
                return document
        raise IndexError(f'Document id {document_id} was not found in {self}.')

    def get_label_by_name(self, name: str) -> Label:
        """Return Label by its name."""
        for label in self.labels:
            if label.name == name:
                return label
        raise IndexError(f'Label name {name} was not found in {self}.')

    def get_label_by_id(self, id_: int) -> Label:
        """
        Return a Label by ID.

        :param id_: ID of the Label to get.
        """
        for label in self.labels:
            if label.id_ == id_:
                return label
        raise IndexError(f'Label id {id_} was not found in {self}.')

    def get_label_set_by_name(self, name: str) -> LabelSet:
        """Return a Label Set by ID."""
        for label_set in self.label_sets:
            if label_set.name == name:
                return label_set
        raise IndexError(f'LabelSet name {name} was not found in {self}.')

    def get_label_set_by_id(self, id_: int) -> LabelSet:
        """
        Return a Label Set by ID.

        :param id_: ID of the Label Set to get.
        """
        for label_set in self.label_sets:
            if label_set.id_ == id_:
                return label_set
        raise IndexError(f'LabelSet id {id_} was not found in {self}.')

    def get_category_by_id(self, id_: int) -> Category:
        """
        Return a Category by ID.

        :param id_: ID of the Category to get.
        """
        for category in self.categories:
            if category.id_ == id_:
                return category
        raise IndexError(f'Category id {id_} was not found in {self}.')

    def delete(self):
        """Delete the Project folder."""
        shutil.rmtree(self.project_folder)

    def lose_weight(self):
        """Delete data of the instance."""
        super().lose_weight()
        for category in self.categories:
            category.lose_weight()
        for label_set in self.label_sets:
            label_set.lose_weight()
        for label in self.labels:
            label.lose_weight()
        self._documents = []
        self._test_documents = []
        return self


def download_training_and_test_data(id_: int):
    """
    Migrate your project to another HOST.

    See https://help.konfuzio.com/integrations/migration-between-konfuzio-server-instances/index.html
        #migrate-projects-between-konfuzio-server-instances
    """
    prj = Project(id_=id_, update=True)

    if len(prj.documents + prj.test_documents) == 0:
        raise ValueError("No Documents in the training or test set. Please add them.")

    for document in tqdm(prj.documents + prj.test_documents):
        document.download_document_details()
        document.get_file()
        document.get_file(ocr_version=False)
        document.get_bbox()
        document.get_images()

    print("[SUCCESS] Data downloading finished successfully!")
