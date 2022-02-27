"""Handle data from the API."""
import itertools
import json
import logging
import os
import pathlib
import re
import shutil
import zipfile
import time
from datetime import tzinfo
from typing import Optional, List, Union, Tuple

import dateutil.parser
import pandas as pd
from tqdm import tqdm

from konfuzio_sdk import KONFUZIO_HOST
from konfuzio_sdk.api import (
    _konfuzio_session,
    download_file_konfuzio_api,
    get_meta_of_files,
    get_project_details,
    post_document_annotation,
    delete_document_annotation,
    create_label,
    get_document_details,
)
from konfuzio_sdk.evaluate import compare
from konfuzio_sdk.normalize import normalize
from konfuzio_sdk.regex import get_best_regex, regex_spans, suggest_regex_for_string, merge_regex
from konfuzio_sdk.utils import get_bbox, get_missing_offsets
from konfuzio_sdk.utils import is_file, convert_to_bio_scheme, amend_file_name

logger = logging.getLogger(__name__)


class Data:
    """Collect general functionality to work with data from API."""

    id_iter = itertools.count()
    id_ = None
    id_local = None
    session = _konfuzio_session()

    def __eq__(self, other) -> bool:
        """Compare any point of data with their ID, overwrite if needed."""
        if self.id_ is None and other and other.id_ is None:
            # Compare to virtual instances
            return self.id_local == other.id_local
        else:
            return self.id_ and other and other.id_ and self.id_ == other.id_

    def __hash__(self):
        """Make any online or local concept hashable. See https://stackoverflow.com/a/7152650."""
        return hash(str(self.id_local))

    # todo review function to be defined as @abstractmethod
    def lose_weight(self):
        """Delete data of the instance."""
        if self.project:
            self.project = None
        return self


class AnnotationSet(Data):
    """Represent an Annotation Set - group of Annotations."""

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
        if document:
            document.add_annotation_set(self)

    def __repr__(self):
        """Return string representation of the Annotation Set."""
        return f"{self.__class__.__name__}({self.id_}) of {self.label_set} in {self.document}."

    def lose_weight(self):
        """Delete data of the instance."""
        self.label_set = None
        self.document = None

    @property
    def annotations(self):
        """All Annotations currently in this Annotation Set."""
        related_annotation = []
        # todo discuss [x for x in self.document.annotations() if x.label]:
        for annotation in self.document.annotations():
            if annotation.annotation_set == self:
                related_annotation.append(annotation)
        return related_annotation

    @property
    def start_offset(self):
        """Calculate earliest start based on all Annotations currently in this Annotation Set."""
        return min((s.start_offset for a in self.annotations for s in a.spans), default=None)

    @property
    def end_offset(self):
        """Calculate the end based on all Annotations currently in this Annotation Set."""
        return max((a.end_offset for a in self.annotations), default=None)


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
        categories=[],
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
        self.labels: List[Label] = []

        # todo the following lines are optional and serve as "separate_labels"
        for label in labels:
            if isinstance(label, int):
                label = self.project.get_label_by_id(id_=label)
            self.add_label(label)

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
            logger.error(f'{self} already has category {category}.')

    def add_label(self, label):
        """
        Add Label to Label Set, if it does not exist.

        :param label: Label ID to be added
        """
        if label not in self.labels:
            self.labels.append(label)
        else:
            logger.error(f'{self} already has {label}, which cannot be added twice.')


class Category(Data):
    """A Category is used to group Documents."""

    def __init__(self, project, id_: int = None, name: str = None, name_clean: str = None, *args, **kwargs):
        """Define a Category that is also a Label Set but cannot have other Categories associated to it."""
        self.id_local = next(Data.id_iter)
        self.id_ = id_
        self.name = name
        self.name_clean = name_clean
        self.project: Project = project
        self.label_sets: List[LabelSet] = []

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
            logger.error(f'{self} already has {label_set}, which cannot be added twice.')

    def __repr__(self):
        """Return string representation of the Category."""
        return f"{self.name} ({self.id_})"


class Label(Data):
    """A Label is the name of a group of individual pieces of information annotated in a type of document."""

    def __init__(
        self,
        project,
        id_: Union[int, None] = None,
        text: str = None,
        get_data_type_display: str = None,
        text_clean: str = None,
        description: str = None,
        label_sets=None,
        has_multiple_top_candidates: bool = False,
        threshold: float = None,  # todo should we really add the default 0.1?
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
        if label_sets is None:
            label_sets = []
        self.id_local = next(Data.id_iter)
        self.id_ = id_
        self.name = text
        self.name_clean = text_clean
        self.data_type = get_data_type_display
        self.description = description
        self.has_multiple_top_candidates = has_multiple_top_candidates
        self.threshold = threshold
        self.project: Project = project
        project.add_label(self)  # todo add feature as described in TestSeparateLabels
        if label_sets:  # todo add feature as described in TestSeparateLabels
            [x.add_label(self) for x in label_sets]

        # Regex features
        self._tokens = None
        self.tokens_file_path = None
        self._regex: List[str] = []
        self.regex_file_path = None
        self._combined_tokens = None
        self.regex_file_path = os.path.join(self.project.regex_folder, f'{self.name_clean}.json5')
        self._correct_annotations = []

    def __repr__(self):
        """Return string representation."""
        return self.name

    def __lt__(self, other: 'Label'):
        """If we sort spans we do so by start offset."""
        try:
            return self.name < other.name
        except TypeError:
            logger.error(f'Cannot sort {self} and {other}.')
            return False

    @property
    def label_sets(self) -> List[LabelSet]:
        """Get the Label Sets in which this Label is used."""
        label_sets = [x for x in self.project.label_sets if self in x.labels]
        return label_sets

    @property
    def translations(self):
        """Create a translation dictionary between offset string and business relevant representation."""
        return {
            annotation.offset_string: annotation.translated_string
            for annotation in self.annotations
            if annotation.translated_string
        }

    @property
    def annotations(self):
        """
        Add Annotation to Label.

        :return: Annotations
        """
        annotations = []
        for document in self.project.documents:
            annotations += document.annotations(label=self)
        return annotations

    def add_label_set(self, label_set: "LabelSet"):
        """
        Add Label Set to label, if it does not exist.

        :param label_set: Label set to add
        """
        if label_set not in self.label_sets:
            self.label_sets.append(label_set)
        else:
            logger.warning(f'{self} already has Label Set {label_set}.')
        return self

    @property
    def correct_annotations(self):
        """Return correct Annotations."""
        if not self._correct_annotations:
            logger.info(f'Find all correct Annotations of {self}.')
            self._correct_annotations = [annotation for annotation in self.annotations if annotation.is_correct]
        return self._correct_annotations

    @property
    def documents(self) -> List["Document"]:
        """Return all Documents which contain Annotations of this Label."""
        relevant_id = list(set([anno.document.id_ for anno in self.annotations]))
        return [doc for doc in self.project.documents if (doc.id_ in relevant_id)]

    # todo move to regex.py so it runs on a list of Annotations, run on Annotations
    def find_tokens(self):
        """Calculate the regex token of a label, which matches all offset_strings of all correct Annotations."""
        self._evaluations = []  # used to do the duplicate check on Annotation level
        for annotation in self.correct_annotations:
            self._evaluations += annotation.tokens()
        try:
            tokens = get_best_regex(self._evaluations, log_stats=True)
        except ValueError:
            logger.error(f'We cannot find tokens for {self} with a f_score > 0.')
            tokens = []
        return tokens

    def tokens(self, update=False) -> List[str]:
        """Calculate tokens to be used in the regex of the Label."""
        if not self._tokens or update:
            self.tokens_file_path = os.path.join(self.project.regex_folder, f'{self.name_clean}_tokens.json5')
            if not is_file(self.tokens_file_path, raise_exception=False) or update:

                logger.info(f'Build tokens for Label {self.name}.')
                self._tokens = self.find_tokens()

                with open(self.tokens_file_path, 'w') as f:
                    json.dump(self._tokens, f, indent=2, sort_keys=True)
            else:
                logger.info(f'Load existing tokens for Label {self.name}.')
                with open(self.tokens_file_path, 'r') as f:
                    self._tokens = json.load(f)
        return self._tokens

    def check_tokens(self):
        """Check if a list of regex do find the annotations. Log Annotations that we cannot find."""
        not_found = []
        for annotation in self.correct_annotations:
            for span in annotation.spans:
                valid_offset = span.offset_string.replace('\n', '').replace('\t', '').replace('\f', '').replace(' ', '')
                if valid_offset and span not in annotation.regex_annotation_generator(self.tokens()):
                    logger.error(
                        f'Please check Annotation ({span.annotation.get_link()}) >>{repr(span.offset_string)}<<.'
                    )
                    not_found.append(span)
        return not_found

    @property
    def combined_tokens(self):
        """Create one OR Regex for all relevant Annotations tokens."""
        if not self._combined_tokens:
            self._combined_tokens = merge_regex(self.tokens())
        return self._combined_tokens

    def evaluate_regex(self, regex, filtered_group=None, regex_quality=0):
        """
        Evaluate a regex on overall Project data.

        Type of regex allows you to group regex by generality

        Example:
            Three Annotations about the birth date in two Documents and one regex to be evaluated
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
        evaluations = [
            document.evaluate_regex(regex=regex, filtered_group=filtered_group, label=self)
            for document in self.project.documents
        ]

        total_findings = sum(evaluation['count_total_findings'] for evaluation in evaluations)
        correct_findings = [finding for evaluation in evaluations for finding in evaluation['correct_findings']]
        total_correct_findings = sum(evaluation['count_total_correct_findings'] for evaluation in evaluations)
        processing_times = [evaluation['runtime'] for evaluation in evaluations]

        try:
            annotation_precision = total_correct_findings / total_findings
        except ZeroDivisionError:
            annotation_precision = 0

        try:
            annotation_recall = total_correct_findings / len(self.correct_annotations)
        except ZeroDivisionError:
            annotation_recall = 0

        try:
            f_score = 2 * (annotation_precision * annotation_recall) / (annotation_precision + annotation_recall)
        except ZeroDivisionError:
            f_score = 0

        if self.project.documents:
            evaluation = {
                'regex': regex,
                'regex_len': len(regex),  # the longer the regex the more conservative it is to use
                'runtime': sum(processing_times) / len(processing_times),  # time to process the regex
                'annotation_recall': annotation_recall,
                'annotation_precision': annotation_precision,
                'f1_score': f_score,
                'regex_quality': regex_quality,
                # other stats
                'correct_findings': correct_findings,
                'total_findings': total_findings,
                'total_correct_findings': total_correct_findings,
            }
            return evaluation
        else:
            return {}

    def find_regex(self) -> List[str]:
        """Find the best combination of regex in the list of all regex proposed by Annotations."""
        if not self.correct_annotations:
            logger.warning(f'{self} has no correct annotations.')
            return []

        # todo: start duplicate check
        regex_made = []
        for annotation in self.annotations:
            for span in annotation._spans:
                proposals = annotation.document.regex(start_offset=span.start_offset, end_offset=span.end_offset)
                for proposal in proposals:
                    regex_to_remove_groupnames = re.compile('<.*?>')
                    regex_found = [re.sub(regex_to_remove_groupnames, '', reg) for reg in regex_made]
                    new_regex = re.sub(regex_to_remove_groupnames, '', proposal)
                    if new_regex not in regex_found:
                        regex_made.append(proposal)

        logger.info(
            f'For Label {self.name} we found {len(regex_made)} regex proposals for {len(self.correct_annotations)}'
            f' annotations.'
        )

        evaluations = [self.evaluate_regex(_regex_made, f'{self.name_clean}_') for _regex_made in regex_made]
        logger.info(f'We compare {len(evaluations)} regex for {len(self.correct_annotations)} correct Annotations.')

        try:
            logger.info(f'Evaluate {self} for best regex.')
            best_regex = get_best_regex(evaluations)
        except ValueError:
            logger.exception(f'We cannot find regex for {self} with a f_score > 0.')
            best_regex = []
        # todo: end duplicate check

        # for annotation in self.annotations:
        #     for span in annotation._spans:
        #         for regex in best_regex:
        #             _, _, spans = generic_candidate_function(regex)(annotation.document.text)
        #             if (span.start_offset, span.end_offset) in spans:
        #                 break
        #         else:
        #             logger.error(f'Fallback regex added for >>{span}<<.')
        #             _suggest = annotation.document.regex(span.start_offset, span.end_offset)
        #             match = check_for_match(
        #                 annotation.document.text, _suggest[:1], span.start_offset, span.end_offset
        #             )
        #             if match:
        #                 best_regex += _suggest

        # Final check.
        # for annotation in self.annotations:
        #     for regex in best_regex:
        #         x = generic_candidate_function(regex)(annotation.document.text)
        #         if (annotation.start_offset, annotation.end_offset) in x[2]:
        #             break
        #     else:
        #         logger.error(f'{annotation} could not be found by any regex.')
        return best_regex

    def regex(self, update=False) -> List:
        """Calculate regex to be used in the LabelExtractionModel."""
        if not self._regex or update:
            if not is_file(self.regex_file_path, raise_exception=False) or update:
                logger.info(f'Build regexes for Label {self.name}.')
                self._regex = self.find_regex()
                with open(self.regex_file_path, 'w') as f:
                    json.dump(self._regex, f, indent=2, sort_keys=True)
            else:
                logger.info(f'Start loading existing regexes for Label {self.name}.')
                with open(self.regex_file_path, 'r') as f:
                    self._regex = json.load(f)
        logger.info(f'Regexes are ready for Label {self.name}.')
        return self._regex

    def save(self) -> bool:
        """
        Save Label online.

        If no Label Sets are specified, the Label is associated with the first default Label Set of the Project.

        :return: True if the new Label was created.
        """
        new_label_added = False
        try:
            if len(self.label_sets) == 0:
                prj_label_sets = self.project.label_sets
                label_set = [t for t in prj_label_sets if t.is_default][0]
                label_set.add_label(self)

            response = create_label(
                project_id=self.project.id_,
                label_name=self.name,
                description=self.description,
                has_multiple_top_candidates=self.has_multiple_top_candidates,
                data_type=self.data_type,
                label_sets=self.label_sets,
            )
            self.id_ = response
            new_label_added = True
        except Exception:
            logger.error(f"Not able to save Label {self.name}.")

        return new_label_added


class Span(Data):
    """An Span is a single sequence of characters."""

    def __init__(self, start_offset: int, end_offset: int, annotation=None):
        """
        Initialize the Span without bbox, to save storage.

        If Bbox should be calculated the bbox file of the Document will be automatically downloaded.

        :param start_offset: Start of the offset string (int)
        :param end_offset: Ending of the offset string (int)
        :param annotation: The Annotation the Span belong to
        """
        if start_offset == end_offset:
            logger.warning(f"You created a {self.__class__.__name__} with start {start_offset} and no Text.")
        self.id_local = next(Data.id_iter)
        self.annotation = annotation
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.page_index = None
        self.top = None
        self.bottom = None
        self.x0 = None
        self.x1 = None
        self.y0 = None
        self.y1 = None

    def __eq__(self, other) -> bool:
        """Compare any point of data with their position is equal."""
        return (
            type(self) == type(other)
            and self.start_offset == other.start_offset
            and self.end_offset == other.end_offset
        )

    def __lt__(self, other: 'Span'):
        """If we sort spans we do so by start offset."""
        # todo check for overlapping
        return self.start_offset < other.start_offset

    def __repr__(self):
        """Return string representation."""
        return f"{self.__class__.__name__} ({self.start_offset}, {self.end_offset})"

    def bbox(self) -> 'Span':
        """Calculate the bounding box of a text sequence."""
        if self.annotation:
            b = get_bbox(self.annotation.document.get_bbox(), self.start_offset, self.end_offset)
            self.page_index = b['page_index']
            self.top = b['top']
            self.bottom = b['bottom']
            self.x0 = b['x0']
            self.x1 = b['x1']
            self.y0 = b['y0']
            self.y1 = b['y1']
        return self

    @property
    def line_index(self) -> int:
        """Calculate the index of the page on which the span starts, first page has index 0."""
        return self.annotation.document.text[0: self.start_offset].count('\n')

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
        if self.start_offset == 0 and self.end_offset == 0:
            eval = {
                "id_local": None,
                "id_": None,
                "confidence": None,
                "start_offset": 0,  # to support compare function to evaluate True and False
                "end_offset": 0,  # to support compare function to evaluate True and False
                "is_correct": None,
                "revised": None,
                "label_threshold": None,
                "label_id": None,
                "label_set_id": None,
                "annotation_set_id": 0,  # to allow grouping to compare boolean
            }
        else:
            eval = {
                "id_local": self.annotation.id_local,
                "id_": self.annotation.id_,
                "confidence": self.annotation.confidence,
                "start_offset": self.start_offset,  # to support multiline
                "end_offset": self.end_offset,  # to support multiline
                "is_correct": self.annotation.is_correct,
                "revised": self.annotation.revised,
                "label_threshold": self.annotation.label.threshold,  # todo: allow to optimize threshold
                "label_id": self.annotation.label.id_,
                "label_set_id": self.annotation.label_set.id_,
                "annotation_set_id": self.annotation.annotation_set.id_,
            }
        return eval


class Annotation(Data):
    """An Annotation holds information that a Label and Annotation Set has been assigned by a list of Spans."""

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
        translated_string=None,
        *initial_data,
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
        :param label_set_id: ID of the Label Set where the Label belongs
        """
        self.id_local = next(Data.id_iter)
        self.is_correct = is_correct
        self.revised = revised
        self.normalized = normalized
        self.translated_string = translated_string
        self.document = document
        self.id_ = id_  # Annotations can have None id_, if they are not saved online and are only available locally
        self._spans: List[Span] = []

        if accuracy:  # its a confidence
            self.confidence = accuracy
        elif self.id_ is not None and accuracy is None:  # todo hotfix: it's an online annotation crated by a human
            self.confidence = 1
        else:
            self.confidence = None

        if isinstance(label, int):
            self.label: Label = self.document.project.get_label_by_id(label)
        elif isinstance(label, Label):
            self.label: Label = label
        else:
            self.label: Label = None
            logger.info(f'{self.__class__.__name__} {self.id_local} has no Label.')

        # if no label_set_id we check if is passed by section_label_id
        if label_set_id is None and kwargs.get("section_label_id") is not None:
            label_set_id = kwargs.get("section_label_id")

        # handles association to an Annotation Set if the Annotation belongs to a Category
        if isinstance(label_set_id, int):
            self.label_set: LabelSet = self.document.project.get_label_set_by_id(label_set_id)
        elif isinstance(label_set, LabelSet):
            self.label_set = label_set
        else:
            self.label_set = None
            logger.info(f'{self.__class__.__name__} {self.id_local} has no Label Set.')

        # make sure an Annotation Set is available
        if isinstance(annotation_set_id, int):
            self.annotation_set = self.document.get_annotation_set_by_id(annotation_set_id)
        elif isinstance(annotation_set, AnnotationSet):
            self.annotation_set = annotation_set
        else:
            self.annotation_set = None
            logger.debug(f'{self} in {self.document} created but without Annotation Set information.')

        for span in spans or []:
            self._add_span(span)

        self.selection_bbox = kwargs.get("selection_bbox", None)
        # self.page_number = kwargs.get("page_number", None)

        bboxes = kwargs.get("bboxes", None)
        if bboxes and len(bboxes) > 0:
            for bbox in bboxes:
                if "start_offset" in bbox.keys() and "end_offset" in bbox.keys():
                    sa = Span(start_offset=bbox["start_offset"], end_offset=bbox["end_offset"], annotation=self)
                    self._add_span(sa)
                else:
                    logger.error(f'SDK cannot read bbox of Annotation {self.id_} in {self.document}: {bbox}')
        elif (
            bboxes is None
            and kwargs.get("start_offset", None) is not None
            and kwargs.get("end_offset", None) is not None
        ):
            # Legacy support for creating Annotations with a single offset
            bbox = kwargs.get('bbox', {})
            sa = Span(start_offset=kwargs.get("start_offset"), end_offset=kwargs.get("end_offset"), annotation=self)
            self._add_span(sa)

            logger.warning(f'{self} is empty')
        else:
            logger.debug(f'{self} created but without bbox information.')
            # raise NotImplementedError
            # todo is it ok to have no bbox ? raise NotImplementedError

        # TODO START LEGACY -
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

        self.bboxes = kwargs.get('bboxes', None)  # todo: smoothen implementation of multiple bboxes
        self.selection_bbox = kwargs.get('selection_bbox', None)
        self.page_number = kwargs.get('page_number', None)
        # TODO END LEGACY -

        # regex features
        self._tokens = []
        self._regex = None

        # Call add_annotation to document at the end, so all attributes for duplicate checking are available.
        # todo: @FZ please add test so this stays at this point of th init.
        self.document.add_annotation(self)

        if not self.document or not self.label_set or not self.label:
            raise NotImplementedError

    def __repr__(self):
        """Return string representation."""
        if self.label and self.document:
            span_str = ', '.join(f'{x.start_offset, x.end_offset}' for x in self._spans)
            return f"{self.__class__.__name__} {self.label.name} {span_str}"
        elif self.label:
            return f"{self.__class__.__name__} {self.label.name} ({self._spans})"
        else:
            return f"{self.__class__.__name__} without Label ({self.start_offset}, {self.end_offset})"

    def __eq__(self, other):
        """We compare a Annotation based on it's Label, Label-Sets."""
        result = False
        # TODO Here are two option on how to define __equal__
        # (1) it could be a "real" duplicate based on label, label_set and spans
        # (2) it could be mark spans only once as correct (in this case it could be used in the tokenizer for duplicate
        # checking, if we go with (1) we need another method for duplicate checking

        # if self.document and other.document and self.document == other.document:
        #    if self.is_correct == other.is_correct:
        #        if self.spans == other.spans:
        #            result = True
        # return result

        # if self.document and other.document and self.document == other.document:
        #     if self.label and other.label and self.label == other.label:
        #         if self.label_set and other.label_set and self.label_set == other.label_set:
        #             if self.spans == other.spans:
        #                 result = True
        # return result

        if self.document and other.document and self.document == other.document:
            if self.label and other.label and self.label == other.label:
                if self.label_set and other.label_set and self.label_set == other.label_set:
                    if self.spans == other.spans:
                        result = True
        return result

    def __lt__(self, other):
        """If we sort Annotations we do so by start offset."""
        # logger.warning('Legacy: Annotations can not be sorted consistently by start offset.')
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
        """Legacy: One Annotations can have multiple start offsets."""
        logger.warning('You use start_offset on Annotation Level which is legacy.')
        return min([sa.start_offset for sa in self._spans], default=None)

    @property
    def end_offset(self) -> int:
        """Legacy: One Annotations can have multiple end offsets."""
        logger.warning('You use end_offset on Annotation Level which is legacy.')
        return max([sa.end_offset for sa in self._spans], default=None)

    @property
    def is_online(self) -> Optional[int]:
        """Define if the Annotation is saved to the server."""
        return self.id_ is not None

    @property
    def offset_string(self) -> List[str]:
        """View the string representation of the Annotation."""
        if self.document.text:
            result = [span.offset_string for span in self.spans]
        else:
            result = []
        return result

    @property
    def eval_dict(self) -> List[dict]:
        """Calculate the Span information to evaluate the Annotation."""
        result = []
        if not self._spans:
            result.append(Span(start_offset=0, end_offset=0).eval_dict())
        else:
            for sa in self._spans:
                result.append(sa.eval_dict())
        return result

    def _add_span(self, span: Span):
        """
        Add a Span to an Annotation.

        This is a private method and should only be called in __init__ as otherwise the duplicate check for annotations
        is bypassed.
        """
        if span not in self._spans:
            self._spans.append(span)
            if span.annotation is not None:
                logger.error(f'{span} is added to {self} however it was assigned to {span.annotation} before.')
            span.annotation = self
        else:
            logger.error(f'In {self} the Span {span} is a duplicate and will not be added.')
        return self

    def get_link(self):
        """Get link to the Annotation in the SmartView."""
        return KONFUZIO_HOST + "/a/" + str(self.id_)

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
            logger.error("You cannot update Annotations once saved online.")
            # update_annotation(id_=self.id_, document_id=self.document.id_, project_id=self.project.id_)

        if not self.is_online:
            response = post_document_annotation(
                document_id=self.document.id_,
                start_offset=self.start_offset,
                end_offset=self.end_offset,
                label_id=self.label.id_,
                label_set_id=label_set_id,
                accuracy=self.confidence,
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
            dict_spans = regex_spans(doctext=self.document.text, regex=regex)
            for offset in list(set((x['start_offset'], x['end_offset']) for x in dict_spans)):
                span = Span(start_offset=offset[0], end_offset=offset[1], annotation=self)
                spans.append(span)
        spans.sort()
        return spans

    # def toJSON(self):
    #     """Convert Annotation to dict."""
    #     res_dict = {
    #         'start_offset': self.start_offset,
    #         'end_offset': self.end_offset,
    #         'label': self.label.id_,
    #         'revised': self.revised,
    #         'annotation_set': self.annotation_set,
    #         'label_set_id': self.label_set.id_,
    #         'accuracy': self.confidence,
    #         'is_correct': self.is_correct,
    #     }
    #
    #     res = {k: v for k, v in res_dict.items() if v is not None}
    #     return res

    # @property
    # def page_index(self) -> int:
    #     """Calculate the index of the page on which the Annotation starts, first page has index 0."""
    #     return self.document.text[0: self.start_offset].count('\f')
    #
    # @property
    # def line_index(self) -> int:
    #     """Calculate the index of the page on which the Annotation starts, first page has index 0."""
    #     return self.document.text[0: self.start_offset].count('\n')

    def token_append(self, evaluation, new_regex):
        """Append token if it is not a duplicate."""
        regex_to_remove_groupnames = re.compile('<.*?>')
        matchers = [re.sub(regex_to_remove_groupnames, '', t['regex']) for t in self._tokens]
        new_matcher = re.sub(regex_to_remove_groupnames, '', new_regex)
        if new_matcher not in matchers:
            self._tokens.append(evaluation)
        else:
            logger.info(f'Annotation Token {repr(new_matcher)} or regex {repr(new_regex)} does exist.')

    def tokens(self) -> List:
        """Create a list of potential tokens based on this annotation."""
        if not self._tokens:
            for span in self._spans:
                harmonized_whitespace = suggest_regex_for_string(span.offset_string, replace_numbers=False)
                numbers_replaced = suggest_regex_for_string(span.offset_string)
                full_replacement = suggest_regex_for_string(span.offset_string, replace_characters=True)

                # the original string, with harmonized whitespaces
                regex_w = f'(?P<{self.label.name_clean}_W_{self.id_}_{span.start_offset}>{harmonized_whitespace})'
                evaluation_w = self.label.evaluate_regex(regex_w, regex_quality=0)
                if evaluation_w['total_correct_findings'] > 1:
                    self.token_append(evaluation=evaluation_w, new_regex=regex_w)
                # the original string, numbers replaced
                if harmonized_whitespace != numbers_replaced:
                    regex_n = f'(?P<{self.label.name_clean}_N_{self.id_}_{span.start_offset}>{numbers_replaced})'
                    self.token_append(evaluation=self.label.evaluate_regex(regex_n, regex_quality=1), new_regex=regex_n)
                # numbers and characters replaced
                if numbers_replaced != full_replacement:
                    regex_f = f'(?P<{self.label.name_clean}_F_{self.id_}_{span.start_offset}>{full_replacement})'
                    self.token_append(evaluation=self.label.evaluate_regex(regex_f, regex_quality=2), new_regex=regex_f)
                if not self._tokens:  # fallback if every proposed token is equal
                    regex_w = f'(?P<{self.label.name_clean}_W_{self.id_}_fallback>{harmonized_whitespace})'
                    self.token_append(evaluation=self.label.evaluate_regex(regex_w, regex_quality=0), new_regex=regex_w)

                    regex_n = f'(?P<{self.label.name_clean}_N_{self.id_}_fallback>{numbers_replaced})'
                    self.token_append(evaluation=self.label.evaluate_regex(regex_n, regex_quality=1), new_regex=regex_n)
                    regex_f = f'(?P<{self.label.name_clean}_F_{self.id_}_fallback>{full_replacement})'
                    self.token_append(evaluation=self.label.evaluate_regex(regex_f, regex_quality=2), new_regex=regex_f)
        return self._tokens

    def regex(self):
        """Return regex of this annotation."""
        return self.label.combined_tokens

    def delete(self) -> None:
        """Delete Annotation online."""
        for index, annotation in enumerate(self.document._annotations):
            if annotation == self:
                del self.document._annotations[index]

        if self.is_online:
            response = delete_document_annotation(document_id=self.document.id_, annotation_id=self.id_)
            if response.status_code == 204:
                self.id_ = None
            else:
                logger.exception(response.text)

    @property
    def spans(self) -> List[Span]:
        """Return default entry to get all Spans of the Annotation."""
        return self._spans


class Document(Data):
    """Access the information about one document, which is available online."""

    def __init__(
        self,
        project,
        id_: Union[int, None] = None,
        file_url: str = None,
        status=None,
        data_file_name: str = None,
        is_dataset: bool = None,
        dataset_status: int = None,
        updated_at: tzinfo = None,
        category_template: int = None,  # fix for Konfuzio Server API, it's actually a id of a Category
        category: Category = None,
        text: str = None,
        bbox: dict = None,
        update: bool = None,
        *args,
        **kwargs,
    ):
        """
        Check if the Document document_folder is available, otherwise create it.

        :param id_: ID of the Document
        :param project: Project where the Document belongs to
        :param file_url: URL of the document
        :param status: Status of the document
        :param data_file_name: File name of the document
        :param is_dataset: Is dataset or not. (bool)
        :param dataset_status: Dataset status of the Document (e.g. training)
        :param updated_at: Updated information
        :param bbox: Bounding box information per character in the PDF (dict)
        :param number_of_pages: Number of pages in the document
        """
        self.id_local = next(Data.id_iter)
        self.id_ = id_
        self._annotations: List[Annotation] = []
        self._annotation_sets: List[AnnotationSet] = []
        self.file_url = file_url
        self.is_dataset = is_dataset
        self.dataset_status = dataset_status
        self._update = update  # the default is None: True will load it from the API, False from local files

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
        project.add_document(self)  # check for duplicates by ID before adding the Document to the project

        # use hidden variables to store low volume information in instance
        self._text = text
        self._bbox = bbox
        self._hocr = None
        self._pages = None

        # prepare local setup for document
        if self.id_:
            pathlib.Path(self.document_folder).mkdir(parents=True, exist_ok=True)
        self.image_paths = []  # Path to the images  # todo implement pages
        self.annotation_file_path = os.path.join(self.document_folder, "annotations.json5")
        self.annotation_set_file_path = os.path.join(self.document_folder, "annotation_sets.json5")
        self.txt_file_path = os.path.join(self.document_folder, "document.txt")
        self.hocr_file_path = os.path.join(self.document_folder, "document.hocr")
        self.pages_file_path = os.path.join(self.document_folder, "pages.json5")
        self.bbox_file_path = os.path.join(self.document_folder, "bbox.zip")
        self.bio_scheme_file_path = os.path.join(self.document_folder, "bio_scheme.txt")

    def __repr__(self):
        """Return the name of the Document incl. the ID."""
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
    def number_of_pages(self):
        """Calculate the number of pages."""
        return len(self.text.split('\f'))

    # todo goes to Trainer extract Method of AI Models
    def add_extractions_as_annotations(
        self, label: Label, extractions, label_set: LabelSet, annotation_set: AnnotationSet
    ):
        """Add the extraction of a model to the document."""
        annotations = extractions[extractions['Accuracy'] > 0.1][
            ['Start', 'End', 'Accuracy', 'page_index', 'x0', 'x1', 'y0', 'y1', 'top', 'bottom']
        ].sort_values(by='Accuracy', ascending=False)
        annotations.rename(columns={'Start': 'start_offset', 'End': 'end_offset'}, inplace=True)
        for annotation in annotations.to_dict('records'):  # todo ask Ana: are Start and End always ints
            _ = Annotation(
                document=self,
                label=label,
                accuracy=annotation['Accuracy'],
                label_set=label_set,
                annotation_set=annotation_set,
                bboxes=[annotation],
            )
            # todo: ask Flo why commented out?
            # self.add_annotation(anno)
        return self

    # todo: Goes to Trainer extract AI method
    def evaluate_extraction_model(self, path_to_model: str):
        """Run and evaluate model on this document."""
        # todo: tbd local import to prevent circular import - Can only be used by konfuzio Trainer users
        from konfuzio.load_data import load_pickle

        model = load_pickle(path_to_model)

        # build the doc from model results
        virtual_doc_for_extraction = Document(
            project=self.project,
            text=self.text,
            bbox=self.get_bbox(),
            category=self.category
        )
        extraction_result = model.extract(document=virtual_doc_for_extraction)
        virtual_doc = self.extraction_result_to_document(extraction_result)

        return compare(self, virtual_doc)

    # todo: Goes to Trainer extract AI method
    def extraction_result_to_document(self, extraction_result):
        """Return a virtual Document annotated with AI Model output."""
        virtual_doc = Document(project=self.project, text=self.text, bbox=self.get_bbox())
        virtual_annotation_set_id = 0  # counter for accross mult. Annotation Set groups of a Label Set

        # define Annotation Set for the Category Label Set: todo: this is unclear from API side
        category_label_set = self.project.get_label_set_by_id(self.category.id_)
        virtual_default_annotation_set = AnnotationSet(
            document=virtual_doc, label_set=category_label_set, id_=virtual_annotation_set_id
        )

        for label_or_label_set_name, information in extraction_result.items():
            if isinstance(information, pd.DataFrame):
                # annotations belong to the default Annotation Set
                # add default Annotation Set if there is any prediction for it
                if virtual_default_annotation_set not in virtual_doc.annotation_sets:
                    virtual_doc.add_annotation_set(virtual_default_annotation_set)

                label = self.project.get_label_by_name(label_or_label_set_name)
                virtual_doc.add_extractions_as_annotations(
                    label=label,
                    extractions=information,
                    label_set=category_label_set,
                    annotation_set=virtual_default_annotation_set,
                )

            else:  # process multi Annotation Sets where multiline is True
                label_set = self.project.get_label_set_by_name(label_or_label_set_name)

                if not isinstance(information, list):
                    information = [information]

                for entry in information:  # represents one of pot. multiple annotation-sets belonging of one LabelSet
                    virtual_annotation_set_id += 1
                    virtual_annotation_set = AnnotationSet(
                        document=virtual_doc, label_set=label_set, id_=virtual_annotation_set_id
                    )
                    # todo: ask Flo why commented out
                    # virtual_doc.add_annotation_set(virtual_annotation_set)

                    for label_name, extractions in entry.items():
                        label = self.project.get_label_by_name(label_name)
                        virtual_doc.add_extractions_as_annotations(
                            label=label,
                            extractions=extractions,
                            label_set=label_set,
                            annotation_set=virtual_annotation_set,
                        )
        return virtual_doc

    def eval_dict(self, use_correct=False) -> dict:
        """Use this dict to evaluate Documents. The speciality: For ever Span of an Annotation create one entry."""
        result = []
        annotations = self.annotations(use_correct=use_correct)
        if not annotations:  # if there are no annotations in this Documents
            result.append(Span(start_offset=0, end_offset=0).eval_dict())
        else:
            for annotation in annotations:
                result += annotation.eval_dict

        return result

    def check_bbox(self) -> bool:
        """Check if bbox matches text."""
        return all([self.text[int(k)] == v['text'] for k, v in self.get_bbox().items()])

    @property
    def is_online(self) -> Optional[int]:
        """Define if the Document is saved to the server."""
        return self.id_ is not None

    @property
    def annotation_sets(self):
        """Return Annotation Sets of Documents."""
        return self._annotation_sets

    def annotations(
        self,
        label: Label = None,
        use_correct: bool = True,
        start_offset: int = 0,
        end_offset: int = None,
        fill: bool = False,
    ) -> List[Annotation]:
        """
        Filter available annotations.

        :param label: Label for which to filter the Annotations.
        :param use_correct: If to filter by correct annotations.
        :return: Annotations in the document.
        """
        # make sure the Document has all required information
        if self._update is None:
            pass
        elif self._update:
            self.update()
            self._update = None  # Make sure we don't repeat to load once loaded.
        else:
            self.get_annotations()  # get_annotations has a fallback, if you deleted the raw json files
            self._update = None  # Make sure we don't repeat to load once loaded.

        annotations = []
        add = False
        for annotation in self._annotations:
            for span in annotation.spans:
                # filter by correct information
                if (use_correct and annotation.is_correct) or not use_correct:
                    # todo: add option to filter for overruled Annotations where mult.=F
                    # todo: add option to filter for overlapping Annotations, `add_annotation` just checks for identical
                    # filter by start and end offset, include annotations that extend into the offset
                    if start_offset and end_offset:  # if the start and end offset are specified
                        latest_start = max(span.start_offset, start_offset)
                        earliest_end = min(span.end_offset, end_offset)
                        is_overlapping = latest_start - earliest_end <= 0
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
            # add a None Label to the default Label Set of the Document
            # todo: we cannot assure that the Document has a Category, so Annotations must not require label_set
            default_label_set = self.project.get_label_set_by_id(self.category.id_)
            no_label = Label(project=self.project, label_sets=[default_label_set])

            spans = [range(span.start_offset, span.end_offset) for anno in annotations for span in anno.spans]
            if end_offset is None:
                end_offset = len(self.text)
            missings = get_missing_offsets(start_offset=start_offset, end_offset=end_offset, annotated_offsets=spans)

            for missing in missings:
                new_spans = []
                offset_text = self.text[missing.start : missing.stop]
                new_annotation = Annotation(document=self, label=no_label, label_set=default_label_set)
                # we split Spans which apan multiple lines, so that one Span comprises one line
                offset_of_offset = 0
                line_breaks = [offset_line for offset_line in re.split(r'(\n)', offset_text) if offset_line != '']
                for offset in line_breaks:
                    start = missing.start + offset_of_offset
                    offset_of_offset += len(offset)
                    end = missing.start + offset_of_offset
                    new_span = Span(start_offset=start, end_offset=end, annotation=new_annotation)
                    new_spans.append(new_span)
                if new_spans:
                    new_annotation._spans = new_spans
                    annotations.append(new_annotation)

        return annotations

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
            if not is_file(file_path, raise_exception=False) or update:
                pdf_content = download_file_konfuzio_api(self.id_, ocr=ocr_version, session=self.session)
                with open(file_path, "wb") as f:
                    f.write(pdf_content)

        return file_path

    def get_images(self, update: bool = False):
        """
        Get Document pages as png images.

        :param update: Update the downloaded images even they are already available
        :return: Path to OCR file.
        """
        self.image_paths = []
        for page in self.pages:

            if is_file(page["image"], raise_exception=False):
                self.image_paths.append(page["image"])
            else:
                page_path = os.path.join(self.document_folder, f'page_{page["number"]}.png')
                self.image_paths.append(page_path)

                if not is_file(page_path, raise_exception=False) or update:
                    url = f'{KONFUZIO_HOST}{page["image"]}'
                    res = self.session.get(url)
                    with open(page_path, "wb") as f:
                        f.write(res.content)

    def download_document_details(self):
        """
        Retrieve data from a Document online in case documented has finished processing.

        :param update: Update the downloaded information even it is already available
        """
        if self.status[0] == 2:
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
            logger.error(f'{self} is not available for download.')

        return self

    def add_annotation(self, annotation: Annotation):
        """Add an annotation to a document.

        :param annotation: Annotation to add in the document
        :param check_duplicate: If to check if the Annotation already exists in the document
        :return: Input annotation.
        """
        if annotation not in self._annotations:
            # Hotfix Text Annotation Server:
            #  Annotation belongs to a Label / Label Set that does not relate to the Category of the Document.
            # logger.debug('You are using a hotfix for API results from Konfuzio Server.')
            if self.category is not None:
                if annotation.label_set and annotation.label_set.categories:
                    if self.category in annotation.label_set.categories:
                        self._annotations.append(annotation)
                    else:
                        logger.error(
                            f'We cannot add {annotation} related to {annotation.label_set.categories} to {self} '
                            f'as the document has {self.category}'
                        )
                else:
                    logger.error(f'{annotation} has Label Set None, which cannot be added to {self}.')
            else:
                logger.error(f'We cannot add {annotation} to {self} where the category is {self.category}')
        else:
            message = f'In {self} the {annotation} is a duplicate and will not be added.'
            # todo: add ValueError to all add_* methods
            raise ValueError(message)

        return self

    def add_annotation_set(self, annotation_set: AnnotationSet):
        """Add the Annotation Sets to the document."""
        if annotation_set.document and annotation_set.document != self:
            raise ValueError('One Annotation Set must only belong to one document.')
        if annotation_set not in self._annotation_sets:
            # todo: skip Annotation Sets that don't belong to the Category: not possible via current API
            # if annotation_set.label_set.category == self.category:
            self._annotation_sets.append(annotation_set)
        else:
            # todo raise NotImplementedError
            logger.error(f'In {self} the {annotation_set} is a duplicate and will not be added.')
        return self

    def get_annotation_set_by_id(self, id_: int) -> AnnotationSet:
        """
        Return a Label Set by ID.

        :param id_: ID of the Label Set to get.
        """
        result = None
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
        :return: list of tuples with each word in the text an the respective label
        """
        # if not is_file(self.bio_scheme_file_path, raise_exception=False) or update:
        annotations_in_doc = []
        for annotation in self.annotations():
            for span in annotation.spans:
                annotations_in_doc.append((span.start_offset, span.end_offset, annotation.label.name))
        converted_text = convert_to_bio_scheme(self.text, annotations_in_doc)
        with open(self.bio_scheme_file_path, "w", encoding="utf-8") as f:
            for word, tag in converted_text:
                f.writelines(word + " " + tag + "\n")
            f.writelines("\n")

        return converted_text

    def get_bbox(self):
        """
        Get bbox information per character of file. We don't store bbox as an attribute to save memory.

        :return: Bounding box information per character in the document.
        """
        if self._bbox:
            return self._bbox
        elif is_file(self.bbox_file_path, raise_exception=False):
            with zipfile.ZipFile(self.bbox_file_path, "r") as archive:
                bbox = json.loads(archive.read('bbox.json5'))
        elif self.status and self.status[0] == 2:
            logger.warning(f'Start downloading bbox files of all characters {self}.')
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
            logger.error(f'{self} does not have bboxes.')
            return {}

        return bbox

    @property
    def text(self):
        """Get Document text. Once loaded stored in memory."""
        if self._text:
            return self._text
        if not is_file(self.txt_file_path, raise_exception=False):
            self.download_document_details()
        if is_file(self.txt_file_path, raise_exception=False):
            with open(self.txt_file_path, "r", encoding="utf-8") as f:
                self._text = f.read()

        return self._text

    @property
    def pages(self):
        """Get Pages of document. Once loaded stored in memory."""
        if self._pages:
            pass
        elif is_file(self.pages_file_path, raise_exception=False):
            with open(self.pages_file_path, "r") as f:
                self._pages = json.loads(f.read())
        else:
            logger.error(f'{self} does not provide information about pages.')
        return self._pages

    @property
    def hocr(self):
        """Get HOCR of document. Once loaded stored in memory."""
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

    # todo: add real workflow to add a new document to a Project
    # def save(self) -> bool:
    #     """
    #     Save or edit Document online.
    #
    #     :return: True if the new document was created or existing document was updated.
    #     """
    #     document_saved = False
    #     category_id = None
    #
    #     if hasattr(self, "category") and self.category is not None:
    #         category_id = self.category.id_
    #
    #     if not self.is_online:
    #         response = upload_file_konfuzio_api(
    #             filepath=self.file_path,
    #             project_id=self.project.id_,
    #             dataset_status=self.dataset_status,
    #             category_id=category_id,
    #         )
    #         if response.status_code == 201:
    #             self.id_ = json.loads(response.text)["id"]
    #             document_saved = True
    #         else:
    #             logger.error(f"Not able to save document {self.file_path} online: {response.text}")
    #     else:
    #         response = update_file_konfuzio_api(
    #             document_id=self.id_, file_name=self.name, dataset_status=self.dataset_status, category_id=category_id
    #         )
    #         if response.status_code == 200:
    #             self.project.update_document(document=self)
    #             document_saved = True
    #         else:
    #             logger.error(f"Not able to update document {self.id_} online: {response.text}")
    #
    #     return document_saved

    def update(self):
        """Update document information."""
        self.delete()
        self.download_document_details()
        self.get_annotations()
        return self

    def delete(self):
        """Delete all local information for the document."""
        try:
            shutil.rmtree(self.document_folder)
        except FileNotFoundError:
            pass
        pathlib.Path(self.document_folder).mkdir(parents=True, exist_ok=True)
        self._annotations = []
        self._annotation_sets = []

    def regex(self, start_offset: int, end_offset: int, max_findings_per_page=15) -> List[str]:
        """Suggest a list of regex which can be used to get the Span of a document."""
        proposals = []
        regex_to_remove_groupnames = re.compile('<.*?>')
        annotations = self.annotations(start_offset=start_offset, end_offset=end_offset)
        for annotation in annotations:
            for spacer in [1, 3, 5, 15]:
                before_regex = suggest_regex_for_string(self.text[start_offset - spacer ** 2 : start_offset])
                after_regex = suggest_regex_for_string(self.text[end_offset : end_offset + spacer])
                proposal = before_regex + annotation.regex() + after_regex

                # check for duplicates
                regex_found = [re.sub(regex_to_remove_groupnames, '', reg) for reg in proposals]
                new_regex = re.sub(regex_to_remove_groupnames, '', proposal)
                if new_regex not in regex_found:
                    if max_findings_per_page:
                        num_matches = len(re.findall(proposal, self.text))
                        if num_matches / (self.text.count('\f') + 1) < max_findings_per_page:
                            proposals.append(proposal)
                        else:
                            logger.info(f'Skip to evaluate regex {repr(proposal)} as it finds {num_matches} in {self}.')
                    else:
                        proposals.append(proposal)

        return proposals

    def evaluate_regex(self, regex, label: Label, filtered_group=None):
        """Evaluate a regex based on the document."""
        start_time = time.time()
        findings_in_document = regex_spans(
            doctext=self.text,
            regex=regex,
            keep_full_match=False,
            # filter by name of label: one regex can match multiple labels
            filtered_group=filtered_group,
        )
        processing_time = time.time() - start_time
        correct_findings = []

        label_annotations = self.annotations(label=label)
        for finding in findings_in_document:
            for annotation in label_annotations:
                for span in annotation._spans:
                    if span.start_offset == finding['start_offset'] and span.end_offset == finding['end_offset']:
                        correct_findings.append(annotation)

        try:
            annotation_precision = len(correct_findings) / len(findings_in_document)
        except ZeroDivisionError:
            annotation_precision = 0

        try:
            annotation_recall = len(correct_findings) / len(self.annotations(label=label))
        except ZeroDivisionError:
            annotation_recall = 0

        try:
            f1_score = 2 * (annotation_precision * annotation_recall) / (annotation_precision + annotation_recall)
        except ZeroDivisionError:
            f1_score = 0
        return {
            'id_': self.id_,
            'regex': regex,
            'runtime': processing_time,
            'count_total_findings': len(findings_in_document),
            'count_total_correct_findings': len(correct_findings),
            'count_correct_annotations': len(self.annotations(label=label)),
            'count_correct_annotations_not_found': len(correct_findings) - len(self.annotations(label=label)),
            # 'doc_matched': len(correct_findings) > 0,
            'annotation_precision': annotation_precision,
            'document_recall': 0,  # keep this key to be able to use the function get_best_regex
            'annotation_recall': annotation_recall,
            'f1_score': f1_score,
            'correct_findings': correct_findings,
        }

    def get_annotations(self):
        """
        Get Annotations of the Document.

        :param update: Update the downloaded information even it is already available
        :return: Annotations
        """
        # Check JSON for Annotation Sets as a fallback if update is False or None but the files do not exist
        if not is_file(self.annotation_set_file_path, raise_exception=False) or not is_file(
            self.annotation_file_path, raise_exception=False
        ):
            self.update()

        with open(self.annotation_set_file_path, "r") as f:
            raw_annotation_sets = json.load(f)

        # first load all Annotation Sets before we create Annotations
        for raw_annotation_set in raw_annotation_sets:
            # todo add parent to define default Annotation Set
            _ = AnnotationSet(
                id_=raw_annotation_set["id"],
                document=self,
                label_set=self.project.get_label_set_by_id(raw_annotation_set["section_label"]),
            )

        with open(self.annotation_file_path, 'r') as f:
            raw_annotations = json.load(f)

        for raw_annotation in raw_annotations:
            raw_annotation['annotation_set_id'] = raw_annotation.pop('section')
            raw_annotation['label_set_id'] = raw_annotation.pop('section_label_id')
            _ = Annotation(document=self, id_=raw_annotation['id'], **raw_annotation)
            # if raw_annotation["custom_offset_string"]:
            #     real_string = self.text[raw_annotation['start_offset'] : raw_annotation['end_offset']]
            #     if real_string == raw_annotation['offset_string']:
            #
            #         # self.add_annotation(annotation)
            #     else:
            #         logger.warning(
            #             f'Annotation {raw_annotation["id"]} has custom string and is not used '
            #             f'in training {KONFUZIO_HOST}/a/{raw_annotation["id"]}.'
            #         )
            # else:
            #
            #
            #     self.add_annotation(Annotation(document=self, id_=raw_annotation['id'], **raw_annotation))

        return self._annotations

    # todo: please add tests before adding this functionality
    # def check_annotations(self):
    #     """Check for annotations width more values than allowed."""
    #     labels = self.project.labels
    #     # Labels that can only have 1 value.
    #     labels_to_check = [label.name_clean for label in labels if not Label.has_multiple_top_candidates]
    #
    #     # Check is done per annotation_set.
    #     for annotation_set in self.annotation_sets:
    #         values_annotations = {}
    #         for annotation in annotation_set.annotations:
    #             annotation_label = annotation.label.name_clean
    #
    #             if annotation_label in labels_to_check:
    #                 annotation_value = annotation.normalize
    #
    #                 if annotation.normalized is None:
    #                     annotation_value = annotation.offset_string
    #
    #                 if annotation_label in values_annotations.keys():
    #                     if annotation_value not in values_annotations[annotation_label]:
    #                         values_annotations[annotation_label].extend([annotation_value])
    #
    #                 else:
    #                     values_annotations[annotation_label] = [annotation_value]
    #
    #         for label, values in values_annotations.items():
    #             if len(values) > 1:
    #                 logger.info(
    #                     f'[Warning] Doc {self.id_} - '
    #                     f'AnnotationSet {annotation.label_set.name_clean} ({annotation.label_set.id_})- '
    #                     f'Label "{label}" shouldn\'t have more than 1 value. Values = {values}'
    #                 )


class Project(Data):
    """Access the information of a Project."""

    def __init__(self, id_: Union[int, None], project_folder=None, update=False, **kwargs):
        """
        Set up the data using the Konfuzio Host.

        :param id_: ID of the project
        :param project_folder: Set a project_older if empty "data_<id_>" will be used.
        :param init_objects: Initialize objects of ths Project.
        """
        self.id_local = next(Data.id_iter)
        self.id_ = id_  # A Project with None ID is not retrieved from the HOST
        self._project_folder = project_folder
        self.categories: List[Category] = []
        self.label_sets: List[LabelSet] = []
        self.labels: List[Label] = []
        self._documents: List[Document] = []
        self.meta_data = []

        # paths
        self.meta_file_path = os.path.join(self.project_folder, "documents_meta.json5")
        self.labels_file_path = os.path.join(self.project_folder, "labels.json5")
        self.label_sets_file_path = os.path.join(self.project_folder, "label_sets.json5")

        if self.id_ or self._project_folder:
            self.get(update=update)

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
        """Return Documents wich have been excluded."""
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

        if not is_file(self.meta_file_path, raise_exception=False) or update:
            self.write_project_files()
        self.get_meta()
        self.get_labels()
        self.get_label_sets()
        self.get_categories()
        self.init_or_update_document()
        return self

    def add_label_set(self, label_set: LabelSet):
        """
        Add Label Set to Project, if it does not exist.

        :param label_set: Label Set to add in the Project
        """
        if label_set not in self.label_sets:
            self.label_sets.append(label_set)
        else:
            logger.error(f'{self} already has Label Set {label_set}.')

    def add_category(self, category: Category):
        """
        Add Category to Project, if it does not exist.

        :param category: Category to add in the Project
        """
        if category not in self.categories:
            self.categories.append(category)
        else:
            logger.error(f'{self} already has category {category}.')

    def add_label(self, label: Label):
        """
        Add Label to Project, if it does not exist.

        :param label: Label to add in the Project
        """
        # todo raise NotImplementedError  # There is no reason to add a Label to a Project
        if label not in self.labels:
            self.labels.append(label)

    def add_document(self, document: Document):
        """Add document to Project, if it does not exist."""
        if document not in self._documents:
            self._documents.append(document)
        else:
            logger.error(f"{document} does exist in {self} and will not be added.")

    def get_meta(self):
        """
        Get the list of all Documents in the Project and their information.

        :return: Information of the Documents in the Project.
        """
        with open(self.meta_file_path, "r") as f:
            self.meta_data = json.load(f)
        return self.meta_data

    def get_categories(self):
        """Load Categories for all Label Sets in the Project."""
        for label_set in self.label_sets:
            if label_set.is_default:
                # the _default_of_label_set_ids are the label sets used by the category
                pass  # todo ?
            else:
                # the _default_of_label_set_ids are the categories the label set is used in
                for label_set_id in label_set._default_of_label_set_ids:
                    category = self.get_category_by_id(label_set_id)
                    label_set.add_category(category)  # The Label Set is linked to a Category it created
                    category.add_label_set(label_set)

    def get_label_sets(self):
        """
        Get Label Sets in the Project.

        :param update: Update the downloaded information even it is already available
        :return: Label Sets in the Project.
        """
        with open(self.label_sets_file_path, "r") as f:
            label_sets_data = json.load(f)

        for label_set_data in label_sets_data:
            label_set = LabelSet(project=self, id_=label_set_data['id'], **label_set_data)
            if label_set.is_default:
                category = Category(project=self, id_=label_set_data['id'], **label_set_data)
                category.label_sets.append(label_set)
                label_set.categories.append(category)  # todo: Konfuzio Server mixes the concepts, we use two instances
                self.add_category(category)
            self.add_label_set(label_set)

        return self.label_sets

    def get_labels(self):
        """
        Get ID and name of any Label in the Project.

        :param update: Update the downloaded information even it is already available
        :return: Labels in the Project.
        """
        with open(self.labels_file_path, "r") as f:
            labels_data = json.load(f)
        # todo clean Labels before reading from file?
        for label_data in labels_data:
            # Remove the project from label_data
            label_data.pop("project", None)
            Label(project=self, id_=label_data['id'], **label_data)

        return self

    def init_or_update_document(self):
        """
        Initialize Document to then decide about full, incremental or no update.

        :param document_data: Document data
        :param update: Update the downloaded information even it is already available
        """
        for document_data in self.meta_data:
            if document_data['status'][0] == 2:  # NOQA - hotfix for Text Annotation Server # todo add test
                new_date = document_data["updated_at"]
                if self.old_meta_data:
                    last_date = [d["updated_at"] for d in self.old_meta_data if d['id'] == document_data["id"]][0]
                    new = document_data["id"] not in [doc["id"] for doc in self.old_meta_data]
                    updated = dateutil.parser.isoparse(new_date) > dateutil.parser.isoparse(last_date)
                else:
                    new = True
                    updated = None

                if updated:
                    doc = Document(project=self, update=True, id_=document_data['id'], **document_data)
                    logger.info(f'{doc} was updated, we will download it again as soon you use it.')
                elif new:
                    doc = Document(project=self, update=True, id_=document_data['id'], **document_data)
                    logger.info(f'{doc} is not available on your machine, we will download it as soon you use it.')
                else:
                    doc = Document(project=self, update=False, id_=document_data['id'], **document_data)
                    logger.debug(f'Load local version of {doc} from {new_date}.')
                self.add_document(doc)

    def get_document_by_id(self, document_id: int) -> Document:
        """Return document by it's ID."""
        for document in self._documents:
            if document.id_ == document_id:
                return document
        raise IndexError

    def get_label_by_name(self, name: str) -> Label:
        """Return Label by its name."""
        for label in self.labels:
            if label.name == name:
                return label
        raise IndexError

    def get_label_by_id(self, id_: int) -> Label:
        """
        Return a Label by ID.

        :param id_: ID of the Label to get.
        """
        for label in self.labels:
            if label.id_ == id_:
                return label
        raise IndexError

    def get_label_set_by_name(self, name: str) -> LabelSet:
        """
        Return a Label Set by ID.

        :param id_: ID of the Label Set to get.
        """
        for label_set in self.label_sets:
            if label_set.name == name:
                return label_set
        raise IndexError

    def get_label_set_by_id(self, id_: int) -> LabelSet:
        """
        Return a Label Set by ID.

        :param id_: ID of the Label Set to get.
        """
        for label_set in self.label_sets:
            if label_set.id_ == id_:
                return label_set
        raise IndexError

    def get_category_by_id(self, id_: int) -> Category:
        """
        Return a Category by ID.

        :param id_: ID of the Category to get.
        """
        for category in self.categories:
            if category.id_ == id_:
                return category

        raise IndexError

    def check_normalization(self):
        """Check normalized offset_strings."""
        for document in self.documents + self.test_documents:
            for annotation in document.annotations():
                for span in annotation.spans:
                    span.normalize()

    def delete(self):
        """Delete the Project folder."""
        shutil.rmtree(self.project_folder)


def download_training_and_test_data(id_: int):
    """
    Migrate your project to another HOST.

    See https://help.konfuzio.com/integrations/migration-between-konfuzio-server-instances/index.html
        #migrate-projects-between-konfuzio-server-instances
    """
    prj = Project(id_=id_, update=True)

    if len(prj.documents + prj.test_documents) == 0:
        raise ValueError("No documents in the training or test set. Please add them.")

    for document in tqdm(prj.documents + prj.test_documents):
        document.download_document_details()
        document.get_file()
        document.get_file(ocr_version=False)
        document.get_bbox()
        document.get_images()

    print("[SUCCESS] Data downloading finished successfully!")
