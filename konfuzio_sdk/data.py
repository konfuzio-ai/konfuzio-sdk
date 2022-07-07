"""Handle data from the API."""
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
from tqdm import tqdm

from konfuzio_sdk import KONFUZIO_HOST
from konfuzio_sdk.api import (
    _konfuzio_session,
    download_file_konfuzio_api,
    get_meta_of_files,
    get_project_details,
    post_document_annotation,
    get_document_details,
    update_document_konfuzio_api,
)
from konfuzio_sdk.normalize import normalize
from konfuzio_sdk.regex import get_best_regex, regex_matches, suggest_regex_for_string, merge_regex
from konfuzio_sdk.utils import get_bbox, get_missing_offsets
from konfuzio_sdk.utils import is_file, convert_to_bio_scheme, amend_file_name, sdk_isinstance

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

    def __copy__(self):
        """Not yet modelled."""
        raise NotImplementedError

    def __deepcopy__(self, memodict):
        """Not yet modelled."""
        raise NotImplementedError

    @property
    def is_online(self) -> Optional[int]:
        """Define if the Document is saved to the server."""
        return self.id_ is not None

    # todo require to overwrite lose_weight via @abstractmethod
    def lose_weight(self):
        """Delete data of the instance."""
        self.session = None
        return self


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
        for annotation in self.document.annotations():
            if annotation.annotation_set == self:
                related_annotation.append(annotation)
        return related_annotation

    @property
    def start_offset(self):
        """Calculate the earliest start based on all Annotations currently in this Annotation Set."""
        return min((s.start_offset for a in self.annotations for s in a.spans), default=None)

    @property
    def start_line_index(self):
        """Calculate starting line of this Annotation Set."""
        return self.document.text[0 : self.start_offset].count('\n')

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


class Category(Data):
    """Group Documents in a Project."""

    def __init__(self, project, id_: int = None, name: str = None, name_clean: str = None, *args, **kwargs):
        """Associate Label Sets to relate to Annotations."""
        self.id_local = next(Data.id_iter)
        self.id_ = id_
        self.name = name
        self.name_clean = name_clean
        self.project: Project = project
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
        threshold: float = None,
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
        project.add_label(self)

        self.label_sets = []
        for label_set in label_sets or []:
            label_set.add_label(self)

        # Regex features
        self._tokens = {}
        self.tokens_file_path = None
        self._regex: List[str] = []
        self._combined_tokens = None
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

    def annotations(self, categories: List[Category], use_correct=True):
        """Return related Annotations. Consider that one Label can be used across Label Sets in multiple Categories."""
        annotations = []
        for category in categories:
            for document in category.documents():
                for annotation in document.annotations(label=self, use_correct=use_correct):
                    annotations.append(annotation)

        return annotations

    def add_label_set(self, label_set: "LabelSet"):
        """
        Add Label Set to label, if it does not exist.

        :param label_set: Label set to add
        """
        if label_set not in self.label_sets:
            self.label_sets.append(label_set)
        else:
            raise ValueError(f'In {self} the {label_set} is a duplicate and will not be added.')

    # todo move to regex.py so it runs on a list of Annotations, run on Annotations
    def find_tokens(self, category: Category) -> List:
        """Calculate the regex token of a label, which matches all offset_strings of all correct Annotations."""
        for annotation in self.annotations(categories=[category]):
            if category.id_ in self._evaluations.keys():
                self._evaluations[category.id_] += annotation.tokens()
            else:
                self._evaluations[category.id_] = annotation.tokens()
        try:
            tokens = get_best_regex(self._evaluations.get(category.id_, []), log_stats=True)
        except ValueError:
            logger.error(f'We cannot find tokens for {self} with a f_score > 0.')
            tokens = []
        return tokens

    def tokens(self, categories: List[Category], update=False) -> dict:
        """Calculate tokens to be used in the regex of the Label."""
        for category in categories:
            tokens_file_path = os.path.join(
                self.project.regex_folder, f'{category.name}_{self.name_clean}_tokens.json5'
            )

            if not is_file(tokens_file_path, raise_exception=False) or update:
                # self._evaluations = []
                category_tokens = self.find_tokens(category=category)

                if os.path.exists(self.project.regex_folder):
                    with open(tokens_file_path, 'w') as f:
                        json.dump(category_tokens, f, indent=2, sort_keys=True)

            else:
                logger.info(f'Load existing tokens for Label {self.name} in Category {category}.')
                with open(tokens_file_path, 'r') as f:
                    category_tokens = json.load(f)

            self._tokens[category.id_] = category_tokens

        categories_ids = [category.id_ for category in categories]

        return {k: v for k, v in self._tokens.items() if k in categories_ids}

    # def check_tokens(self, categories: List[Category]):
    #     """Check if a list of regex do find the Annotations. Log Annotations that we cannot find."""
    #     not_found = []
    #     for annotation in self.annotations(categories=categories):
    #         for span in annotation.spans:
    #             valid_offset = span.offset_string.replace('\n', '').replace('\t', '').\
    #             replace('\f', '').replace(' ', '')
    #             categories_tokens = self.tokens(categories=categories)
    #             for _, category_tokens in categories_tokens.items():
    #                 created_regex = annotation.regex_annotation_generator(category_tokens)
    #                 if valid_offset and span not in created_regex:
    #                     logger.error(
    #                         f'Please check Annotation ({span.annotation.get_link()}) >>{repr(span.offset_string)}<<.'
    #                     )
    #                     not_found.append(span)
    #     return not_found

    def combined_tokens(self, categories: List[Category]):
        """Create one OR Regex for all relevant Annotations tokens."""
        if not self._combined_tokens:
            categories_tokens = self.tokens(categories=categories)
            all_tokens = []
            for category_id, category_tokens in categories_tokens.items():
                all_tokens.extend(category_tokens)
            self._combined_tokens = merge_regex(all_tokens)
        return self._combined_tokens

    def evaluate_regex(
        self, regex, category: Category, annotations: List['Annotation'] = None, filtered_group=None, regex_quality=0
    ):
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
            evaluation = document.evaluate_regex(
                regex=regex, filtered_group=filtered_group, label=self, annotations=annotations
            )
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

    def find_regex(self, category: 'Category') -> List[str]:
        """Find the best combination of regex in the list of all regex proposed by Annotations."""
        regex_made = []
        all_annotations = self.annotations(categories=[category])  # default is use_correct = True

        if not all_annotations:
            logger.warning(f'{self} has no correct annotations.')
            return []

        for annotation in all_annotations:
            for span in annotation.spans:
                proposals = annotation.document.regex(start_offset=span.start_offset, end_offset=span.end_offset)
                for proposal in proposals:
                    regex_to_remove_groupnames = re.compile('<.*?>')
                    regex_found = [re.sub(regex_to_remove_groupnames, '', reg) for reg in regex_made]
                    new_regex = re.sub(regex_to_remove_groupnames, '', proposal)
                    if new_regex not in regex_found:
                        regex_made.append(proposal)

        logger.info(
            f'For Label {self.name} we found {len(regex_made)} regex proposals for {len(all_annotations)} annotations.'
        )

        # todo replace by compare
        evaluations = [
            self.evaluate_regex(
                _regex_made, category=category, annotations=all_annotations, filtered_group=f'{self.id_}_'
            )
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

        return best_regex

    def regex(self, categories: List[Category], update=False) -> List:
        """Calculate regex to be used in the LabelExtractionModel."""
        if not self._regex or update:
            if not is_file(self.regex_file_path, raise_exception=False) or update:
                logger.info(f'Build regexes for Label {self.name}.')
                regex = []
                for category in categories:
                    regex.extend(self.find_regex(category=category))
                self._regex = regex
                if os.path.exists(self.regex_file_path):
                    with open(self.regex_file_path, 'w') as f:
                        json.dump(self._regex, f, indent=2, sort_keys=True)
            else:
                logger.warning(
                    f'Regexes loaded from file for {self} which might have been calculated for other category.'
                )
                logger.info(f'Start loading existing regexes for Label {self.name}.')
                with open(self.regex_file_path, 'r') as f:
                    self._regex = json.load(f)
        logger.info(f'Regexes are ready for Label {self.name}.')
        return self._regex

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
        if start_offset == end_offset:
            raise ValueError(f"You cannot created a {self.__class__.__name__} with start {start_offset} and no Text.")
        elif end_offset < start_offset:
            raise ValueError(
                f"You cannot created a {self.__class__.__name__} with start {start_offset} which is later than"
                f" end_offset {end_offset}."
            )
        self.id_local = next(Data.id_iter)
        self.annotation = annotation
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.top = None
        self.bottom = None
        self.x0 = None
        self.x1 = None
        self.y0 = None
        self.y1 = None

        # use hidden variables calculated information in instance
        self._page_index = None
        self._line_index = None

        annotation and annotation.add_span(self)  # only add if Span has access to an Annotation

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

    def bbox(self):
        """Calculate the bounding box of a text sequence."""
        if self.annotation:
            b = get_bbox(self.annotation.document.get_bbox(), self.start_offset, self.end_offset)

            if not any(b.values()):
                # Whitespaces will not provide bounding boxes, e.g. for those characters like '\x0c'
                raise ValueError(f'{self} does not have available characters bounding boxes.')

            # self.page_index = b['page_index']
            # "page_index" depends on the document text and should be independent of bboxes.
            self.top = b['top']
            self.bottom = b['bottom']
            self.x0 = b['x0']
            self.x1 = b['x1']
            self.y0 = b['y0']
            self.y1 = b['y1']

            if self.x0 and self.x1 and not self.x0 < self.x1:
                raise ValueError(
                    f'{self} in {self.annotation.document}: coordinate x1 should be bigger than x0. '
                    f'x0: {self.x0}, x1: {self.x1}, document: {self.annotation.document}.'
                )

            if self.y0 and self.y1 and not self.y0 < self.y1:
                raise ValueError(
                    f'{self} in {self.annotation.document}: coordinate y1 should be bigger than y0.'
                    f' y0: {self.y0}, y1: {self.y1}, document: {self.annotation.document}.'
                )

            check_page = True
            if self.page_index is None:
                logger.error(f'Page index is None of {self} in {self.annotation.document}.')
                check_page = False
            if self.page_height is None:
                logger.error(f'Page Height is None of {self} in {self.annotation.document}.')
                check_page = False
            if self.page_width is None:
                logger.error(f'Page Width is None of {self} in {self.annotation.document}.')
                check_page = False
            if check_page and (not (self.x1 < self.page_width) or not (self.y1 < self.page_height)):
                raise ValueError(
                    f'{self} in {self.annotation.document}: bounding box of span is located outside of '
                    f'the page, document: {self.annotation.document}.'
                )
            return self
        else:
            raise NotImplementedError

    @property
    def line_index(self) -> int:  # TODO line_index might not be needed.
        """Calculate the index of the line on which the span starts, first line has index 0."""
        if self.annotation and self.annotation.document.pages:
            if self._line_index is not None:
                return self._line_index
            else:
                self._line_index = self.annotation.document.text[0 : self.start_offset].count('\n')
                return self._line_index

    @property
    def page_index(self) -> Optional[int]:
        """Calculate the index of the page on which the span starts, first page has index 0."""
        if self.annotation and self.annotation.document.pages:
            if self._page_index is not None:
                return self._page_index
            else:
                self._page_index = self.annotation.document.text[0 : self.start_offset].count('\f')
                return self._page_index

    @property
    def page_width(self) -> Optional[float]:
        """Get width of the page of the Span. Used to calculate relative position on page."""
        if self.page_index is not None:
            return self.annotation.document.pages[self.page_index].original_size[0]
        else:
            return None

    @property
    def page_height(self) -> Optional[float]:
        """Get width of the page of the Span. Used to calculate relative position on page."""
        if self.page_index is not None:
            return self.annotation.document.pages[self.page_index].original_size[1]
        else:
            return None

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
        if self.start_offset == -1 and self.end_offset == 0:
            eval = {
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
                "page_width": 0,
                "page_height": 0,
                "page_index": 0,
                "line_index": 0,
            }
        else:
            eval = {
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
                "x0": self.x0,
                "x1": self.x1,
                "y0": self.y0,
                "y1": self.y1,
                "page_width": self.page_width,
                "page_height": self.page_height,
                "page_index": self.page_index,
                "line_index": self.line_index,  # used by label_set clf
            }

        # Add calculated values: # TODO is this the right place for these calculations?
        if self.x0 is not None and self.x1 is not None and self.page_width is not None:
            eval["x0_relative"] = self.x0 / self.page_width  # TODO should we calculate fields here?
            eval["x1_relative"] = self.x1 / self.page_width
        else:
            eval["x0_relative"] = None
            eval["x1_relative"] = None

        if self.y0 is not None and self.y1 is not None and self.page_height is not None:
            eval["y0_relative"] = self.y0 / self.page_height
            eval["y1_relative"] = self.y1 / self.page_height
        else:
            eval["y0_relative"] = None
            eval["y1_relative"] = None

        if self.page_index is not None and self.annotation.document.number_of_pages:
            eval["page_index_relative"] = self.page_index / self.annotation.document.number_of_pages
        else:
            eval["page_index_relative"] = None

        return eval


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
        elif isinstance(label, Label):
            self.label: Label = label
        else:
            raise ValueError(f'{self.__class__.__name__} {self.id_local} has no Label.')

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
        if self.id_:
            return KONFUZIO_HOST + "/a/" + str(self.id_)
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
                project_id=self.project.id_,
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

    def delete(self) -> None:
        """Delete Annotation online."""
        for index, annotation in enumerate(self.document._annotations):
            if annotation == self:
                del self.document._annotations[index]

    @property
    def spans(self) -> List[Span]:
        """Return default entry to get all Spans of the Annotation."""
        return sorted(self._spans)


class Page(Data):
    """Access the information about one Page of a document."""

    def __init__(
        self,
        id_: Union[int, None] = None,
        document: 'Document' = None,
        start_offset: int = None,
        end_offset: int = None,
        number: int = None,
        image: str = None,
        original_size: Tuple[float, float] = None,
        size: Tuple[float, float] = None,
    ):
        """Create a Page for a Document."""
        self.id_ = id_
        self.document = document
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.original_size = original_size
        self.size = size
        self.image = image
        self.number = number


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
        project.add_document(self)  # check for duplicates by ID before adding the Document to the project

        # use hidden variables to store low volume information in instance
        self._text = text
        self._bbox = bbox
        self._hocr = None
        self._pages = None

        # Use Page to initialize Pages of this Document
        self._load_pages(pages_data=pages)

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
    def number_of_pages(self) -> int:
        """Calculate the number of pages."""
        return len(self.text.split('\f'))

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

    @property
    def spans(self):
        """Return all Spans of the Document."""
        spans = []
        if self._annotations is None:
            self.annotations()

        for annotation in self._annotations:
            for span in annotation.spans:
                if span not in spans:
                    spans.append(span)

        # if self.spans == list(set(self.spans)):
        #     # todo deduplicate Spans. One text offset in a document can ber referenced by many Spans of Annotations
        #     raise NotImplementedError

        return sorted(spans)

    def _load_pages(self, pages_data: List[Union[Dict, Page]]):
        """Load Pages of document."""
        if pages_data is not None:
            _pages = []
            page_texts = self.text.split('\f')
            assert len(page_texts) == len(pages_data)

            start_offset = 0

            for page_index, page_data in enumerate(pages_data):
                page_text = page_texts[page_index]
                end_offset = start_offset + len(page_text)
                if sdk_isinstance(page_data, Page):
                    page = Page(
                        id_=page_data.id_,
                        number=page_data.number,
                        size=page_data.size,
                        image=page_data.image,
                        original_size=page_data.original_size,
                        start_offset=page_data.start_offset,
                        end_offset=page_data.end_offset,
                        document=self,
                    )
                else:
                    page_data['id_'] = page_data.pop('id', None)
                    if 'size' in page_data.keys():
                        page_data['size'] = tuple(page_data['size'])
                    if 'original_size' in page_data.keys():
                        page_data['original_size'] = tuple(page_data['original_size'])
                    page = Page(**page_data, document=self, start_offset=start_offset, end_offset=end_offset)
                _pages.append(page)
                start_offset = end_offset + 1

            self._pages = _pages

    def eval_dict(self, use_correct=False) -> List[dict]:
        """Use this dict to evaluate Documents. The speciality: For every Span of an Annotation create one entry."""
        result = []
        annotations = self.annotations(use_correct=use_correct)
        if not annotations:  # if there are no annotations in this Documents
            result.append(Span(start_offset=-1, end_offset=0).eval_dict())
        else:
            for annotation in annotations:
                result += annotation.eval_dict

        return result

    def check_bbox(self, update_document: bool = False) -> bool:
        """
        Check match between the text in the Document and its bounding boxes.

        Check that every character in the text is part of the bbox dictionary and that each bbox has a match in the
        document text. Also, validate the coordinates of the bounding boxes.
        """
        warn('This method is deprecated.', DeprecationWarning, stacklevel=2)
        self._bbox = self.get_bbox()
        for index, char in enumerate(self.text):
            if char in [' ', '\f', '\n']:
                continue
            try:
                if self.get_bbox()[str(index)]['text'] != char:
                    return False
            except KeyError:
                return False

        all_x = all([v['x1'] > v['x0'] for k, v in self.get_bbox().items()])
        all_y = all([v['y1'] > v['y0'] for k, v in self.get_bbox().items()])
        all_width = all(
            [v['x1'] <= self.pages[v['page_number'] - 1].original_size[0] for k, v in self.get_bbox().items()]
        )

        all_height = all(
            [v['y1'] <= self.pages[v['page_number'] - 1].original_size[1] for k, v in self.get_bbox().items()]
        )

        valid = all([all_x, all_y, all_width, all_height])

        if not valid:
            logger.error(f'{self} has invalid character bounding boxes.')

            if update_document:
                # set the dataset status of the Document to Excluded
                update_document_konfuzio_api(
                    document_id=self.id_,
                    file_name=self.name,
                    dataset_status=4,
                    assignee=1102,  # bbox-issue@konfuzio.com
                )

        return valid

    def __deepcopy__(self, memo) -> 'Document':
        """Create a new Document of the instance."""
        return Document(
            id=None,
            project=self.project,
            category=self.category,
            text=self.text,
            bbox=self.get_bbox(),
            pages=self.pages,
            copy_of_id=self.id_,
        )

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
        if self.id_ and not is_file(self.annotation_set_file_path, raise_exception=False):
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
        if self.category is None:
            raise ValueError(f'Document {self} without Category must not have Annotations')
        self.get_annotations()
        annotations: List[Annotation] = []
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
                line_breaks = [offset_line for offset_line in re.split(r'(\n)', offset_text) if offset_line != '']
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
        Get Document pages as png images.

        :param update: Update the downloaded images even they are already available
        :return: Path to OCR file.
        """
        self.image_paths = []
        for page in self.pages:
            if is_file(page.image, raise_exception=False):
                self.image_paths.append(page.image)
            else:
                page_path = os.path.join(self.document_folder, f'page_{page.number}.png')
                self.image_paths.append(page_path)

                if not is_file(page_path, raise_exception=False) or update:
                    url = f'{KONFUZIO_HOST}{page.image}'
                    res = self.session.get(url)
                    with open(page_path, "wb") as f:
                        f.write(res.content)

    def download_document_details(self):
        """Retrieve data from a Document online in case documented has finished processing."""
        if self.status and self.status[0] == 2:
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
        elif self.id_ is None:
            pass  # Document is not online
        else:
            logger.error(f'{self} is not available for download.')

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
        Return a Label Set by ID.

        :param id_: ID of the Label Set to get.
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
        if self._bbox is not None:
            return self._bbox
        elif is_file(self.bbox_file_path, raise_exception=False):
            with zipfile.ZipFile(self.bbox_file_path, "r") as archive:
                bbox = json.loads(archive.read('bbox.json5'))
        elif self.status and self.status[0] == 2:  # todo check for self.project.id_ and self.id_ and ?
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
            logger.error(f'{self} does not have bboxes.')
            return {}

        return bbox

    @property
    def text(self):
        """Get Document text. Once loaded stored in memory."""
        if self._text is not None:
            return self._text
        if not is_file(self.txt_file_path, raise_exception=False):
            self.download_document_details()
        if is_file(self.txt_file_path, raise_exception=False):
            with open(self.txt_file_path, "r", encoding="utf-8") as f:
                self._text = f.read()

        return self._text

    @property
    def pages(self):
        """Get Pages of Document. Once loaded stored in memory."""
        # todo: add Page as a Class to get access to the Image
        if self._pages is not None:
            pass
        if not is_file(self.pages_file_path, raise_exception=False):
            self.download_document_details()
        if is_file(self.pages_file_path, raise_exception=False):
            with open(self.pages_file_path, "r") as f:
                pages_data = json.loads(f.read())
            self._load_pages(pages_data)
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

    def regex(self, start_offset: int, end_offset: int, search=None, max_findings_per_page=100) -> List[str]:
        """Suggest a list of regex which can be used to get the Span of a document."""
        if search is None:
            search = [2, 5, 10]
        if start_offset < 0:
            raise IndexError(f'The start offset must be a positive number but is {start_offset}')
        if end_offset > len(self.text):
            raise IndexError(f'The end offset must not exceed the text length of the Document but is {end_offset}')
        proposals = []
        regex_to_remove_groupnames = re.compile('<.*?>')
        annotations = self.annotations(start_offset=start_offset, end_offset=end_offset)
        for annotation in annotations:
            for token in annotation.tokens():
                for spacer in search:  # todo fix this search, so that we take regex token from other spans into account
                    before_regex = suggest_regex_for_string(
                        self.text[start_offset - spacer ** 2 : start_offset], replace_characters=True
                    )
                    after_regex = suggest_regex_for_string(
                        self.text[end_offset : end_offset + spacer], replace_characters=True
                    )
                    # proposal = before_regex + token['regex'] + after_regex
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
                                logger.info(
                                    f'Skip to evaluate regex {repr(proposal)} as it finds {num_matches} in {self}.'
                                )
                        else:
                            proposals.append(proposal)

        return proposals

    def evaluate_regex(self, regex, label: Label, annotations: List['Annotation'] = None, filtered_group=None):
        """Evaluate a regex based on the Document."""
        start_time = time.time()
        findings_in_document = regex_matches(
            doctext=self.text,
            regex=regex,
            keep_full_match=False,
            filtered_group=f'Label_{label.id_}'
            # filter by name of label: one regex can match multiple labels
            # filtered_group=filtered_group,
        )
        processing_time = time.time() - start_time
        correct_findings = []

        label_annotations = self.annotations(label=label)
        if annotations is not None:
            label_annotations = [x for x in label_annotations if x in annotations]

        for finding in findings_in_document:
            for annotation in label_annotations:
                for span in annotation.spans:
                    # todo: if the regex finds subparts of the Span, we don't count this as a valid finding,
                    #   even we could merge the subparts afterwards
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
            'id': self.id_local,
            'regex': regex,
            # processing_time time can vary slightly between runs, round to ms so that this variation does not affect
            # the choice of regexes when values are below ms and metrics are the same
            'runtime': round(processing_time, 3),
            'count_total_findings': len(findings_in_document),
            'count_total_correct_findings': len(correct_findings),
            'count_correct_annotations': len(self.annotations(label=label)),
            'count_correct_annotations_not_found': len(correct_findings) - len(self.annotations(label=label)),
            'doc_matched': len(correct_findings) > 0,
            'annotation_precision': annotation_precision,
            'document_recall': 0,  # keep this key to be able to use the function get_best_regex
            'annotation_recall': annotation_recall,
            'f1_score': f1_score,
            'correct_findings': correct_findings,
        }

    def get_annotations(self) -> List[Annotation]:
        """Get Annotations of the Document."""
        if self._update or (self.is_online and (self._annotations is None or self._annotation_sets is None)):
            annotation_file_exists = is_file(self.annotation_file_path, raise_exception=False)
            annotation_set_file_exists = is_file(self.annotation_set_file_path, raise_exception=False)

            if self.id_ and (not annotation_file_exists or not annotation_set_file_exists or self._update):
                self.update()  # delete the meta of the Document details and download them again
                self._update = None  # Make sure we don't repeat to load once updated.

            self._annotation_sets = None  # clean Annotation Sets to not create duplicates
            self.annotation_sets()

            self._annotations = []  # clean Annotations to not create duplicates
            with open(self.annotation_file_path, 'r') as f:
                raw_annotations = json.load(f)

            for raw_annotation in raw_annotations:
                if not raw_annotation['is_correct'] and raw_annotation['revised']:
                    continue
                raw_annotation['annotation_set_id'] = raw_annotation.pop('section')
                raw_annotation['label_set_id'] = raw_annotation.pop('section_label_id')
                _ = Annotation(document=self, id_=raw_annotation['id'], **raw_annotation)
            self._update = None  # Make sure we don't repeat to load once loaded.

        if self._annotations is None:
            self._annotations = []

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
        if label not in self.labels:
            self.labels.append(label)
        else:
            raise ValueError(f'In {self} the {label} is a duplicate and will not be added.')

    def add_document(self, document: Document):
        """Add Document to Project, if it does not exist."""
        if document not in self._documents:
            self._documents.append(document)
        else:
            raise ValueError(f'In {self} the {document} is a duplicate and will not be added.')

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
                pass
            else:
                # the _default_of_label_set_ids are the categories the label set is used in
                for label_set_id in label_set._default_of_label_set_ids:
                    category = self.get_category_by_id(label_set_id)
                    label_set.add_category(category)  # The Label Set is linked to a Category it created
                    category.add_label_set(label_set)

    def get_label_sets(self):
        """Get Label Sets in the Project."""
        with open(self.label_sets_file_path, "r") as f:
            label_sets_data = json.load(f)

        self.label_sets = []  # clean up Label Sets to not create duplicates
        self.categories = []  # clean up Labels to not create duplicates
        for label_set_data in label_sets_data:
            label_set = LabelSet(project=self, id_=label_set_data['id'], **label_set_data)
            if label_set.is_default:
                category = Category(project=self, id_=label_set_data['id'], **label_set_data)
                category.label_sets.append(label_set)
                label_set.categories.append(category)  # Konfuzio Server mixes the concepts, we use two instances
                # self.add_category(category)

        return self.label_sets

    def get_labels(self) -> Label:
        """Get ID and name of any Label in the Project."""
        with open(self.labels_file_path, "r") as f:
            labels_data = json.load(f)
        self.labels = []  # clean up Labels to not create duplicates
        for label_data in labels_data:
            # Remove the project from label_data
            label_data.pop("project", None)
            Label(project=self, id_=label_data['id'], **label_data)

        return self

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
        """Return a Label Set by ID."""
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
