"""Handle data from the API."""
import itertools
import json
import logging
import os
import pathlib
import re
import shutil
from copy import deepcopy
import time
from datetime import tzinfo
from typing import Dict, Optional, List, Union, Tuple

import dateutil.parser
import pandas

from konfuzio_sdk import KONFUZIO_HOST
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
    get_document_annotations,
)
from konfuzio_sdk.utils import get_bbox
from konfuzio_sdk.evaluate import compare
from konfuzio_sdk.normalize import normalize
from konfuzio_sdk.regex import get_best_regex, regex_spans, suggest_regex_for_string
from konfuzio_sdk.utils import is_file, convert_to_bio_scheme, amend_file_name

logger = logging.getLogger(__name__)


class Data(object):
    """Collect general functionality to work with data from API."""

    id_iter = itertools.count()
    id = None
    session = konfuzio_session()

    def __eq__(self, other) -> bool:
        """Compare any point of data with their ID, overwrite if needed."""
        return hasattr(other, "id") and self.id and other.id and self.id == other.id

    def __hash__(self):
        """Return hash(self)."""
        return hash(str(self.id))

    def lose_weight(self):
        """Delete data of the instance."""
        if self.project:
            self.project = None
        return self


class AnnotationSet(Data):
    """Represent an Annotation Set - group of annotations."""

    def __init__(self, id, document, label_set, **kwargs):
        """
        Create an Annotation Set.

        :param id: ID of the annotation set
        :param document: Document where the annotation set belongs
        :param label_set: Label set where the annotation set belongs to
        :param annotations: Annotations of the annotation set
        """
        self.id_local = next(Data.id_iter)
        self.id = id
        self.label_set = label_set
        self.document = document
        document.add_annotation_set(self)

    @property
    def annotations(self):
        """All Annotations currently in this Annotation Set."""
        related_annotation = []
        for annotation in [x for x in self.document.annotations() if x.label]:
            if annotation.annotation_set == self:
                related_annotation.append(annotation)
        return related_annotation

    @property
    def start_offset(self):
        """Calculate earliest start based on all Annotations currently in this Annotation Set."""
        return min((a.start_offset for a in self.annotations), default=None)

    @property
    def end_offset(self):
        """Calculate the end based on all Annotations currently in this Annotation Set."""
        return max((a.end_offset for a in self.annotations), default=None)


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
        self.id_local = next(Data.id_iter)
        self.id = id
        self.name = name
        self.name_clean = name_clean
        self.is_default = is_default
        if "default_label_sets" in kwargs:
            self.categories = kwargs["default_label_sets"]
        elif "default_section_labels" in kwargs:
            self.categories = kwargs["default_section_labels"]
        else:
            self.categories = categories
        self.has_multiple_annotation_sets = has_multiple_annotation_sets

        if "has_multiple_sections" in kwargs:
            self.has_multiple_annotation_sets = kwargs["has_multiple_sections"]

        self.project: Project = project

        self.labels: List[Label] = []
        project.add_label_set(self)

        for label in labels:
            if isinstance(label, int):
                label = self.project.get_label_by_id(id=label)
            self.add_label(label)
        self.has_multiple_sections = self.has_multiple_annotation_sets
        self.default_templates = self.categories

    def evaluate(self, doc_model):
        """Evaluate templates."""
        if self.is_default:
            return None
        if not self.has_multiple_sections:
            return None
        if not hasattr(self, 'pattern') or not self.pattern:
            return None

        evaluation_df = pandas.DataFrame()
        for test_doc in self.project.test_docs:
            res = doc_model.extract(test_doc.text)
            evaluation_df = self.evaluate_document_templates(test_doc, doc_model, evaluation_df, res)

    def evaluate_document_templates(
        self, test_doc, doc_model, evaluation_df: pandas.DataFrame, res: Dict
    ) -> pandas.DataFrame:
        """Evaluate templates for a document."""
        if self.is_default:
            return evaluation_df
        if not self.has_multiple_sections:
            return evaluation_df
        item_annotations = [x for x in test_doc.annotations() if x.section_label.id == self.id]
        item_section_ids = set(x.section for x in item_annotations)

        if self.name in res.keys() and len(item_section_ids) == len(res[self.name]):
            evaluation = True
            length = len(item_section_ids)
            length_predicted = len(res[self.name])
        else:
            evaluation = False
            length = len(item_section_ids)
            length_predicted = len(res[self.name]) if self.name in res.keys() else 0

        evaluation_df = evaluation_df.append(
            {
                'evaluation': evaluation,
                'section_label': self.name,
                'document': test_doc.id,
                'length': length,
                'length_predicted': length_predicted,
            },
            ignore_index=True,
        )
        evaluation_df['document'] = evaluation_df.document.astype(int)
        evaluation_df['length'] = evaluation_df.length.astype(int)
        evaluation_df['length_predicted'] = evaluation_df.length_predicted.astype(int)
        return evaluation_df

    def __repr__(self):
        """Return string representation of the Label Set."""
        return f"{self.name} ({self.id})"

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
        self.id_local = next(Data.id_iter)
        self.is_default = True
        self.has_multiple_annotation_sets = False
        self.categories = []
        self.project.add_category(self)

    def documents(self):
        """Filter for documents of this category."""
        return [x for x in self.project.documents if x.category == self]

    def test_documents(self):
        """Filter for test documents of this category."""
        return [x for x in self.project.test_documents if x.category == self]


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
        self.id_local = next(Data.id_iter)
        self.id = id
        self.name = text
        self.name_clean = text_clean
        self.data_type = get_data_type_display
        self.description = description
        self.has_multiple_top_candidates = has_multiple_top_candidates
        self.threshold = kwargs.get("threshold", 0.1)

        self.project: Project = project

        project.add_label(self)
        if label_sets:
            [x.add_label(self) for x in label_sets]

        # Regex features
        self._tokens = None
        self.tokens_file_path = None
        self._regex: List[str] = []
        self.regex_file_path = None
        self._combined_tokens = None
        self.regex_file_path = os.path.join(self.project.regex_folder, f'{self.name_clean}.json5')

    def __repr__(self):
        """Return string representation."""
        return self.name

    @property
    def label_sets(self):
        """Get the label sets in which this label is used."""
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
        Add annotation to label.

        :return: Annotations
        """
        annotations = []
        for document in self.project.documents:
            annotations += document.annotations(label=self)
        return annotations

    def add_label_set(self, label_set: "LabelSet"):
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
    def documents(self) -> List["Document"]:
        """Return all documents which contain annotations of this label."""
        relevant_id = list(set([anno.document.id for anno in self.annotations]))
        return [doc for doc in self.project.documents if (doc.id in relevant_id)]

    def find_tokens(self):
        """Calculate the regex token of a label, which matches all offset_strings of all correct annotations."""
        evaluations = []
        for annotation in self.correct_annotations:
            tokens = annotation.tokens()
            for token in tokens:
                # remove duplicates
                matcher = re.sub(re.compile('<.*?>'), '', token['regex'])
                matchers = [re.sub(re.compile('<.*?>'), '', t['regex']) for t in evaluations]
                if matcher not in matchers:
                    evaluations.append(token)

        try:
            tokens = get_best_regex(evaluations, log_stats=True)
        except ValueError:
            logger.error(f'We cannot find tokens for {self} with a f_score > 0.')
            tokens = []

        for annotation in self.correct_annotations:
            for span in annotation._spans:
                valid_offset = span.offset_string.replace('\n', '').replace('\t', '').replace('\f', '').replace(' ', '')
                if valid_offset and span not in annotation.regex_annotation_generator(tokens):
                    logger.error(
                        f'Please check Annotation ({KONFUZIO_HOST}/a/{annotation.id})'
                        f' >>{repr(span.offset_string)}<<.'
                    )
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

    @property
    def combined_tokens(self):
        """Create one OR Regex for all relevant Annotations tokens."""
        if not self._combined_tokens:
            tokens = r'|'.join(sorted(self.tokens(), key=len, reverse=True))
            self._combined_tokens = f'(?:{tokens})'  # store them for later use
        return self._combined_tokens

    def evaluate_regex(self, regex, filtered_group=None, regex_quality=0):
        """
        Evaluate a regex on overall project data.

        Type of regex allows you to group regex by generality

        Example:
            Three annotations about the birth date in two documents and one regex to be evaluated
            1.doc: "My was born at the 12th of December 1980, you could also say 12.12.1980." (2 Annotations)
            2.doc: "My was born at 12.06.1997." (1 Annotations)
            regex: dd.dd.dddd (without escaped characters for easier reading)
            stats:
                  total_correct_findings: 2
                  correct_label_annotations: 3
                  total_findings: 2 --> precision 100 %
                  num_docs_matched: 2
                  project.documents: 2  --> document recall 100%

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
        """Find the best combination of regex in the list of all regex proposed by annotations."""
        if not self.correct_annotations:
            logger.warning(f'{self} has no correct annotations.')
            return []

        regex_made = []
        for annotation in self.annotations:
            for span in annotation._spans:
                # TODO: review multiline case
                proposals = annotation.document.regex(start_offset=span.start_offset, end_offset=span.end_offset)
                for proposal in proposals:
                    regex_to_remove_groupnames = re.compile('<.*?>')
                    regex_found = [re.sub(regex_to_remove_groupnames, '', reg) for reg in regex_made]
                    new_regex = re.sub(regex_to_remove_groupnames, '', proposal)
                    if new_regex not in regex_found:
                        regex_made.append(proposal)

        logger.info(
            f'For label {self.name} we found {len(regex_made)} regex proposals for {len(self.correct_annotations)}'
            f' annotations.'
        )

        evaluations = [self.evaluate_regex(_regex_made, f'{self.name_clean}_') for _regex_made in regex_made]
        logger.info(f'We compare {len(evaluations)} regex for {len(self.correct_annotations)} correct annotations.')

        # correct_findings = set(x for evaluation in evaluations for x in evaluation['correct_findings'])
        # if missing_annotations := set(self.annotations) - correct_findings:
        #    logger.error(f'Missing correct annotation when building regex: {missing_annotations}')
        logger.info(f'Evaluate {self} for best regex.')
        try:
            best_regex = get_best_regex(evaluations)
        except ValueError:
            logger.exception(f'We cannot find regex for {self} with a f_score > 0.')
            best_regex = []

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
            logger.error(f"Not able to save label {self.name}.")

        return new_label_added


class Span(Data):
    """An Span is a single sequence of characters."""

    def __init__(self, start_offset: int, end_offset: int, annotation=None):
        """
        Initialize the Span.

        :param start_offset: Start of the offset string (int)
        :param end_offset: Ending of the offset string (int)
        :param page_index: 0-based index of the page
        """
        if start_offset == end_offset:
            raise IndexError("Spans must not be empty.")
        self.id_local = next(Data.id_iter)
        self.annotation = annotation
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.bbox()

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
    def normalized(self):
        """Normalize the offset string."""
        return normalize(self.offset_string, self.annotation.label.data_type)

    @property
    def offset_string(self) -> str:
        """Calculate the offset string of a Span."""
        return self.annotation.document.text[self.start_offset : self.end_offset]

    def eval_dict(self):
        """Return any information needed to evaluate the Span."""
        if self.start_offset == 0 and self.end_offset == 0:
            eval = {
                "id_local": None,
                "id": None,
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
                "id": self.annotation.id,
                "confidence": self.annotation.confidence,
                "start_offset": self.start_offset,  # to support multiline
                "end_offset": self.end_offset,  # to support multiline
                "is_correct": self.annotation.is_correct,
                "revised": self.annotation.revised,
                "label_threshold": self.annotation.label.threshold,  # todo: allow to optimize threshold
                "label_id": self.annotation.label.id,
                "label_set_id": self.annotation.label_set.id,
                "annotation_set_id": self.annotation.annotation_set.id,
            }
        return eval


class Annotation(Data):
    """
    An Annotation holds information that a Label and Annotation Set has been assigned to.

    Keep the information of one sequence of the document.

    Todo: the endpoint test_get_project_labels does no longer include the document annotation_sets, as the relation of
    a label and a annotation_set can be configured by a user while labeling. We might ne to model the relation of many
    Annotations to one AnnotationSet in a more explicit way.

    Example document: "I earn 15 Euro per hour."

    Assume the word "15" should be labeled. The project contains the labels "Amount" and "Tax".

    # CREATE

    Annotations can be created by:

    - Human: Who is using the web interface
    - Import: A human user imports extractions and uses "Copy extractions to annotations" admin action
    - Training: Using the konfuzio package you create an annotation online, via an Bot user
    - Text FB: Text Feedback - External API user, sends new extraction without ID, which contains only the offset string
    - Extraction: Internal Process after we receive a new document from an External API user
    - Extraction FB: External Feedback - External API user, sends feedback to existing extraction incl. ID

    ID column: relates to the Annotation instance created in the database
    is_revised: A human revisor had a look at this annotation
    correct: Human claims that this annotation should be extracted in future documents

    The KONFUZIO package will use annotations which are revised or (no XOR) correct.

    | ID | Creator       | is_revised  | correct       | User      | Label   | Action  |
    |:---|:--------------|:------------|:------------- |:----------|:--------|:--------|
    | 1  | Human         | False       | True          | Human     | Amount  | ALLOWED |
    | 2  | Import        | False       | False         | None      | Amount  | ALLOWED | Extraction.created_by_import
    | 3  | Training      | False       | False         | Bot       | Amount  | ALLOWED |
    | 4  | Extraction    | False       | False         | External  | Amount  | ALLOWED | one annotation per extraction
    | X  | Text FB       | -----       | -----         | ---       | Amount  | see 2   | only create extraction

    # REVISE

    Annotations, as they heave been created, can be revised by:

    - Human: Who is using the web interface
    - Revise Feedback: ?

    ## Positive Feedback will change

    | ID | Revisor       | is_revised  | correct       | User      | Label   | Action  |
    |:---|:--------------|:------------|:------------- |:----------|:--------|:--------|
    | 1  | Human         | NA          | NA            | NA        | Amount  | HIDDEN  |
    | 2  | Human         | True        | True          | Human     | Amount  | ALLOWED |
    | 3  | Human         | True        | True          | Bot       | Amount  | ALLOWED | -> ? does PUT update User
    | 4  | Human         | NA          | NA            | External  | Amount  | HIDDEN  |
    | 1  | Extraction FB | True        | True          | Human     | Amount  | ALLOWED |
    | 2  | Extraction FB | ----        | ----          | ----      | ----    | ----    | External user does not get ID
    | 3  | Extraction FB | ----        | ----          | ----      | ----    | ----    | External user does not get ID
    | 4  | Extraction FB | True        | True          | Bot       | Amount  | ALLOWED |

    As positive feedback displays the annotation in the interface but stores them as correct examples, the
    word "15" should NOT be labeled anew. This time the creator might choose between label "Amount" and "Tax".

    | ID | Creator       | is_revised  | correct       | User      | Label   | Action  |
    |:---|:--------------|:------------|:------------- |:----------|:--------|:--------|
    | 5  | Human         | False       | True          | Human     | Amount  | DENIED  |
    | 6  | Import        | True        | False         | None      | Amount  | DENIED  |
    | 7  | Training      | False       | False         | Bot       | Amount  | DENIED  |
    | 8  | Extraction FB | ?           | ?             | ?         | Amount  | DENIED  |
    | 9  | Human         | False       | True          | Human     | Tax     | DENIED  |
    | 10 | Import        | ----        | ----          | ----      | Tax     | DENIED  | External user does not get ID
    | 11 | Training      | ----        | ----          | ----      | Tax     | DENIED  | External user does not get ID
    | 12 | Extraction FB | ?           | ?             | ?         | Tax     | DENIED  |

    ## Negative Feedback will change

    - The user clicks on delete button next to the annotation in the web interface.
    - Incorrect or deleted annotations will no longer be displayed in the web interface.

    | ID | Revisor       | is_revised  | correct       | User      | Label   | Action  |
    |:---|:--------------|:------------|:------------- |:----------|:--------|:--------|
    | 1  | Human         | DELTED      | DELTED        | DELTED    | Amount  | ALLOWED | delete revised=F, correct=T
    | 2  | Human         | True        | False         | None      | Amount  | ALLOWED | Update three fields
    | 3  | Human         | True        | False         | Bot       | Amount  | ALLOWED | Does update is_revised field
    | 4  | Human         | ?           | ?             | ?         | Amount  | ALLOWED |
    | 1  | Extraction FB | True        | False         | ?         | Amount  | ALLOWED |
    | 2  | Extraction FB | ----        | ----          | ----      | Amount  | ALLOWED | External user does not get ID
    | 3  | Extraction FB | ----        | ----          | ----      | Amount  | ALLOWED | External user does not get ID
    | 4  | Extraction FB | True        | False         | External  | Amount  | ALLOWED |

    As negative feedback removed any annotation from the web interface but stores them as incorrect examples, the
    word "15" can be labeled anew. This time the creator might choose between label "Amount" and "Tax".

    | ID | Creator       | is_revised  | correct       | User      | Label   | Action  |
    |:---|:--------------|:------------|:------------- |:----------|:--------|:--------|
    | 5  | Human         | False       | True          | Human     | Amount  | ?DENIED | -> in contrast to annotation 1
    | 6  | Import        | ---         | ---           | ---       | ---     | DENIED  |
    | 7  | Training      | ---         | ---           | ---       | ---     | DENIED  |
    | 8  | Extraction FB | ?           | ?             | ?         | Amount  | NA      | Need to send new document
    | 9  | Human         | False       | True          | Human     | Tax     | ALLOWED | now we have 2 annotations
    | 10 | Import        | False       | False         | None      | Tax     | ALLOWED |
    | 11 | Training      | False       | False         | Bot       | Tax     | ALLOWED |
    | 12 | Extraction FB | ?           | ?             | ?         | Tax     | NA      | Need to send new document

    """

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
        id: int = None,
        accuracy: float = None,
        translated_string=None,
        *initial_data,
        **kwargs,
    ):
        """
        Initialize the Annotation.

        :param label: ID of the Annotation
        :param is_correct: If the annotation is correct or not (bool)
        :param revised: If the annotation is revised or not (bool)
        :param id: ID of the annotation (int)
        :param accuracy: Accuracy of the annotation (float) which is the confidence
        :param document: Document to annotate
        :param annotation: Annotation set of the document where the label belongs
        :param label_set_text: Name of the label set where the label belongs
        :param translated_string: Translated string
        :param label_set_id: ID of the label set where the label belongs
        """
        self.id_local = next(Data.id_iter)
        self.is_correct = is_correct
        self.revised = revised
        self.normalized = normalized
        self.translated_string = translated_string
        self.document = document
        document.add_annotation(self)
        self._spans: List[Span] = []
        self.id = id  # Annotations can have None id, if they are not saved online and are only available locally

        if accuracy:  # its a confidence
            self.confidence = accuracy
        elif self.id is not None and accuracy is None:  # todo hotfix: it's an online annotation crated by a human
            self.confidence = 1
        else:
            self.confidence = None

        if isinstance(label, int):
            self.label: Label = self.document.project.get_label_by_id(label)
        elif isinstance(label, Label):
            self.label: Label = label
        elif label is None and self.__class__.__name__ == 'NoLabelAnnotation':
            self.label = None
        else:
            raise AttributeError(f'{self.__class__.__name__} {self.id_local} has no label.')

        # if no label_set_id we check if is passed by section_label_id
        if label_set_id is None and kwargs.get("section_label_id") is not None:
            label_set_id = kwargs.get("section_label_id")

        # handles association to an annotation set if the annotation belongs to a category
        if isinstance(label_set_id, int):
            self.label_set: LabelSet = self.document.project.get_label_set_by_id(label_set_id)
        elif isinstance(label_set, LabelSet):
            self.label_set = label_set
        elif self.__class__.__name__ == 'NoLabelAnnotation':
            self.label_set = None
        else:
            raise AttributeError(f'{self.__class__.__name__} {self.id_local} has no Label Set.')

        # make sure an Annotation Set is available
        if isinstance(annotation_set_id, int):
            self.annotation_set = self.document.get_annotation_set_by_id(annotation_set_id)
        elif isinstance(annotation_set, AnnotationSet):
            self.annotation_set = annotation_set
        elif self.__class__.__name__ == 'NoLabelAnnotation':
            self.annotation_set = None
        else:
            raise AttributeError(f'{self.__class__.__name__} {self.id_local} has no Annotation Set.')

        self.selection_bbox = kwargs.get("selection_bbox", None)
        self.page_number = kwargs.get("page_number", None)

        bboxes = kwargs.get("bboxes", None)
        if bboxes and len(bboxes) > 0:
            for bbox in bboxes:
                sa = Span(start_offset=bbox["start_offset"], end_offset=bbox["end_offset"], annotation=self)
                self.add_span(sa)
        elif (
            bboxes is None
            and kwargs.get("start_offset", None) is not None
            and kwargs.get("end_offset", None) is not None
        ):
            # Legacy support for creating annotations with a single offset
            bbox = kwargs.get('bbox', {})
            sa = Span(start_offset=kwargs.get("start_offset"), end_offset=kwargs.get("end_offset"), annotation=self)
            self.add_span(sa)

            logger.warning(f'{self} is empty')
        else:
            raise NotImplementedError
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
        if not hasattr(self, 'label'):
            raise
        # TODO END LEGACY -

        # regex features
        self._tokens = []
        self._regex = None

    def __repr__(self):
        """Return string representation."""
        if self.label and self.document:
            span_str = ', '.join(f'{x.start_offset, x.end_offset}' for x in self._spans)
            return f"{self.label.name} {span_str}: {self.offset_string}"
        elif self.label and self.offset_string:
            return f"{self.label.name} ({self.start_offset}, {self.end_offset})"
        elif self.__class__.__name__ == 'NoLabelAnnotation':
            return f"NoLabelAnnotation ({self.start_offset}, {self.end_offset})"
        else:
            logger.error(f"{self.__class__.__name__} without Label ({self.start_offset}, {self.end_offset})")

    def __lt__(self, other):
        """If we sort Annotations we do so by start offset."""
        # todo check for overlapping
        return self.start_offset < other.start_offset

    @property
    def is_multiline(self) -> bool:
        """Calculate if Annotation spans multiple lines of text."""
        # TODO: clean code after issue 6230 is solved (TODO: move to sdk)
        if (
            self.bboxes is not None
            and self.bboxes
            and (
                ('line_number' in self.bboxes[0].keys() and len(set([bbox['line_number'] for bbox in self.bboxes])) > 1)
                or (  # NOQA
                    'line_index' in self.bboxes[0].keys() and len(set([bbox['line_index'] for bbox in self.bboxes])) > 1
                )
            )
        ):  # NOQA
            is_multiline = True
        else:
            is_multiline = False
        return is_multiline

    @property
    def normalize(self) -> str:
        """Provide one normalized offset string due to legacy."""
        logger.warning('You use normalize on Annotation Level which is legacy.')
        return normalize(self.offset_string, self.label.data_type)

    @property
    def start_offset(self) -> int:
        """Legacy: One Annotations can have multiple start offsets."""
        return min([sa.start_offset for sa in self._spans], default=None)

    @property
    def end_offset(self) -> int:
        """Legacy: One Annotations can have multiple end offsets."""
        return max([sa.end_offset for sa in self._spans], default=None)

    @property
    def is_online(self) -> Optional[int]:
        """Define if the Annotation is saved to the server."""
        return self.id is not None

    @property
    def offset_string(self) -> List[str]:
        """View the string representation of the Annotation."""
        if self.document.text:
            result = [self.document.text[span.start_offset : span.end_offset] for span in self._spans]
        else:
            result = []
        return result

    @property
    def eval_dict(self) -> List[dict]:
        """Calculate the span information to evaluate the Annotation."""
        result = []
        if not self._spans:
            result.append(Span(start_offset=0, end_offset=0).eval_dict())
        else:
            for sa in self._spans:
                result.append(sa.eval_dict())
        return result

    def add_span(self, span: Span, check_duplicate=True):
        """Add an span to a document.

        If check_duplicate is True, we only add an span after checking it doesn't exist in the
        document already. If check_duplicate is False, we add an span without checking, but it is
        considerably faster when the number of span in the document is large.

        :param span: Annotation to add in the document
        :param check_duplicate: If to check if the annotation already exists in the document
        :return: Input annotation.
        """
        if check_duplicate:
            if span not in self._spans:
                self._spans.append(span)
        else:
            self._spans.append(span)
        return span

    def get_link(self):
        """Get link to the annotation in the SmartView."""
        return KONFUZIO_HOST + "/a/" + str(self.id)

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
        if not self.label_set:
            label_set_id = None
        else:
            label_set_id = self.label_set.id
        if not self.is_online:
            response = post_document_annotation(
                document_id=self.document.id,
                start_offset=self.start_offset,
                end_offset=self.end_offset,
                label_id=self.label.id,
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
                self.id = json_response["id"]
                new_annotation_added = True
            elif response.status_code == 403:
                logger.error(response.text)
                try:
                    if "In one project you cannot label the same text twice." in response.text:
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
                                logger.error(f"ID of annotation online: {annotation.id}")
                                self.id = annotation.id
                                is_duplicated = True
                                break

                        # if there isn't a perfect match, the current annotation is considered incorrect
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
        Build candidates for regexes.

        # todo add more info from regex span function to Span.

        :return: Return sorted list of Spans by start_offset
        """
        spans: List[Span] = []
        for regex in regex_list:
            dict_spans = regex_spans(doctext=self.document.text, regex=regex)
            for offset in list(set((x['start_offset'], x['end_offset']) for x in dict_spans)):
                span = Span(start_offset=offset[0], end_offset=offset[1])
                spans.append(span)
        return sorted(spans, key=lambda x: x.start_offset)

    def toJSON(self):
        """Convert Annotation to dict."""
        res_dict = {
            'start_offset': self.start_offset,
            'end_offset': self.end_offset,
            'label': self.label.id,
            'revised': self.revised,
            'annotation_set': self.annotation_set,
            'label_set_id': self.label_set.id,
            'accuracy': self.confidence,
            'is_correct': self.is_correct,
        }

        res = {k: v for k, v in res_dict.items() if v is not None}
        return res

    @property
    def page_index(self) -> int:
        """Calculate the index of the page on which the Annotation starts, first page has index 0."""
        return self.document.text[0 : self.start_offset].count('\f')

    @property
    def line_index(self) -> int:
        """Calculate the index of the page on which the Annotation starts, first page has index 0."""
        return self.document.text[0 : self.start_offset].count('\n')

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
            if len(self._spans) > 1:
                print(1)
            for span in self._spans:
                harmonized_whitespace = suggest_regex_for_string(span.offset_string, replace_numbers=False)
                numbers_replaced = suggest_regex_for_string(span.offset_string)
                full_replacement = suggest_regex_for_string(span.offset_string, replace_characters=True)

                # the original string, with harmonized whitespaces
                regex_w = f'(?P<{self.label.name_clean}_W_{self.id}_{span.start_offset}>{harmonized_whitespace})'
                evaluation_w = self.label.evaluate_regex(regex_w, regex_quality=0)
                if evaluation_w['total_correct_findings'] > 1:
                    self.token_append(evaluation=evaluation_w, new_regex=regex_w)
                # the original string, numbers replaced
                if harmonized_whitespace != numbers_replaced:
                    regex_n = f'(?P<{self.label.name_clean}_N_{self.id}_{span.start_offset}>{numbers_replaced})'
                    self.token_append(evaluation=self.label.evaluate_regex(regex_n, regex_quality=1), new_regex=regex_n)
                # numbers and characters replaced
                if numbers_replaced != full_replacement:
                    regex_f = f'(?P<{self.label.name_clean}_F_{self.id}_{span.start_offset}>{full_replacement})'
                    self.token_append(evaluation=self.label.evaluate_regex(regex_f, regex_quality=2), new_regex=regex_f)
                if not self._tokens:  # fallback if every proposed token is equal
                    regex_w = f'(?P<{self.label.name_clean}_W_{self.id}_fallback>{harmonized_whitespace})'
                    self.token_append(evaluation=self.label.evaluate_regex(regex_w, regex_quality=0), new_regex=regex_w)

                    regex_n = f'(?P<{self.label.name_clean}_N_{self.id}_fallback>{numbers_replaced})'
                    self.token_append(evaluation=self.label.evaluate_regex(regex_n, regex_quality=1), new_regex=regex_n)
                    regex_f = f'(?P<{self.label.name_clean}_F_{self.id}_fallback>{full_replacement})'
                    self.token_append(evaluation=self.label.evaluate_regex(regex_f, regex_quality=2), new_regex=regex_f)
        return self._tokens

    def regex(self):
        """Return regex of this annotation."""
        if not self._regex:
            if len(self.label.tokens()) == 0:
                raise NotImplementedError
                # We have tried to find tokens without success, use the tokens of the Annotation
                # tokens = r'|'.join(sorted([x['regex'] for x in self.tokens()], key=len, reverse=True))
            else:
                # Use the token of the Label
                self._regex = self.label.combined_tokens
        return self._regex

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

    @property
    def spans(self):
        """Return default entry to get all Spans of the Annotation."""
        return self._spans


class NoLabelAnnotation(Annotation):
    """
    This is an Annotation created by a tokenizer which is not labeled.

    Because of this is does not have 'id', 'label', 'annotation_set'

    """
    pass


class Document(Data):
    """Access the information about one document, which is available online."""

    def __init__(
        self,
        project,
        id: Union[int, None] = None,
        file_path: str = None,
        file_url: str = None,
        status=None,
        data_file_name: str = None,
        is_dataset: bool = None,
        dataset_status: int = None,
        updated_at: tzinfo = None,
        number_of_pages: int = None,
        *initial_data,
        **kwargs,
    ):
        """
        Check if the document document_folder is available, otherwise create it.

        :param id: ID of the Document
        :param project: Project where the document belongs to
        :param file_path: Path to a local file from which generate the Document object
        :param file_url: URL of the document
        :param status: Status of the document
        :param data_file_name: File name of the document
        :param is_dataset: Is dataset or not. (bool)
        :param dataset_status: Dataset status of the document (e.g. training)
        :param updated_at: Updated information
        :param bbox: Bounding box information per character in the PDF (dict)
        :param number_of_pages: Number of pages in the document
        """
        self.id_local = next(Data.id_iter)
        self.file_path = file_path
        self.annotation_file_path = None  # path to json containing the Annotations of a Document
        self.annotation_set_file_path = None  # path to json containing the Annotation Sets of a Document
        self._annotations: List[Annotation] = []
        self._annotation_sets: List[AnnotationSet] = []
        self.file_url = file_url
        self.is_dataset = is_dataset
        self.dataset_status = dataset_status
        self.number_of_pages = number_of_pages
        if project:
            self.category = project.get_category_by_id(kwargs.get("category_template", None))
        self.id = id
        if updated_at:
            self.updated_at = dateutil.parser.isoparse(updated_at)
        else:
            self.updated_at = None

        self.name = data_file_name
        self.ocr_file_path = None  # Path to the ocred pdf (sandwich pdf)
        self.image_paths = []  # Path to the images
        self.status = status  # status of document online

        self.project = project
        project.add_document(self)  # check for duplicates by ID before adding the document to the project

        self.text = kwargs.get("text")
        self.hocr = kwargs.get("hocr")
        self._bbox = None

        # prepare local setup for document
        pathlib.Path(self.document_folder).mkdir(parents=True, exist_ok=True)
        self.annotation_file_path = os.path.join(self.document_folder, "annotations.json5")
        self.annotation_set_file_path = os.path.join(self.document_folder, "annotation_sets.json5")
        self.txt_file_path = os.path.join(self.document_folder, "document.txt")
        self.hocr_file_path = os.path.join(self.document_folder, "document.hocr")
        self.pages_file_path = os.path.join(self.document_folder, "pages.json5")
        self.bbox_file_path = os.path.join(self.document_folder, "bbox.json5")
        self.bio_scheme_file_path = os.path.join(self.document_folder, "bio_scheme.txt")

    def __repr__(self):
        """Return the name of the document incl. the ID."""
        return f"{self.name}: {self.id}"

    def add_extractions_as_annotations(self, label: Label, extractions, label_set: LabelSet, annotation_set: int):
        """Add the extraction of a model on the document."""
        annotations = extractions[extractions['Accuracy'] > 0.1][
            ['Start', 'End', 'Accuracy', 'page_index', 'x0', 'x1', 'y0', 'y1', 'top', 'bottom']
        ].sort_values(by='Accuracy', ascending=False)
        annotations.rename(columns={'Start': 'start_offset', 'End': 'end_offset'}, inplace=True)
        for annotation in annotations.to_dict('records'):  # todo ask Ana: are Start and End always ints
            anno = Annotation(
                label=label,
                accuracy=annotation['Accuracy'],
                label_set=label_set,
                annotation_set=annotation_set,
                bboxes=[annotation],
            )
            self.add_annotation(anno)
        return self

    def evaluate_extraction_model(self, path_to_model: str):
        """Run and evaluate model on this document."""
        # todo: tbd local import to prevent circular import - Can only be used by konfuzio Trainer users
        from konfuzio.load_data import load_pickle

        model = load_pickle(path_to_model)
        extraction_result = model.extract(
            text=self.text, bbox=self.get_bbox()  # todo: ask ana: why does it not save those as attribute of the doc
        )
        # build the doc from model results
        virtual_doc = Document(project=self.project)
        virtual_annotation_set = 0  # counter for accross mult. annotation set groups of a label set

        for label_or_label_set_name, information in extraction_result.items():
            if not isinstance(information, list):
                label = self.project.get_label_by_name(label_or_label_set_name)
                virtual_doc.add_extractions_as_annotations(
                    label=label, extractions=information, label_set=self.category, annotation_set=self.category.id
                )
            else:  # process multi annotation sets where multiline is True
                label_set = self.project.get_label_set_by_name(label_or_label_set_name)
                for entry in information:  # represents one of pot. multiple annotation-sets belonging of one label set
                    virtual_annotation_set += 1
                    for label_name, extractions in entry.items():
                        label = self.project.get_label_by_name(label_name)
                        virtual_doc.add_extractions_as_annotations(
                            label=label,
                            extractions=extractions,
                            label_set=label_set,
                            annotation_set=virtual_annotation_set,
                        )
        return compare(self, virtual_doc)

    def eval_dict(self, use_correct=False) -> dict:
        """Use this dict to evaluate Documents. The speciality: For ever Span of an Annotation create one entry."""
        result = []
        annotations = self.annotations(use_correct=use_correct)
        if not annotations:  # if there are no annotations in this documents
            result.append(Span(start_offset=0, end_offset=0).eval_dict())
        else:
            for annotation in annotations:
                result += annotation.eval_dict

        return result

    @property
    def is_online(self) -> Optional[int]:
        """Define if the Document is saved to the server."""
        return self.id is not None

    def spans(self, start_offset: int, end_offset: int) -> List[Span]:
        """
        Translate an offset into where offsets between the Spans of Annotations are filled with no Label Annotations.

        todo: If a regex can be used as a combination of Span regex Tokens.

        :param start_offset: Start of the offset to analyze in the document.
        :param end_offset: End of the offset to analyze in the document.
        :return: Returns a sorted lists of Spans.
        """
        raise NotImplementedError

    def annotations(
        self, label: Label = None, use_correct: bool = True, start_offset: int = None, end_offset: int = None
    ) -> List[Annotation]:
        """
        Filter available annotations.

        :param label: Label for which to filter the annotations.
        :param use_correct: If to filter by correct annotations.
        :return: Annotations in the document.
        """
        annotations = []
        artificial_annotations = []
        self._annotations.sort()  # use .sort() not sorted([list]): sort the instances not copy them
        for annotation in self._annotations:
            if annotation.start_offset is None or annotation.end_offset is None:
                raise NotImplementedError
            # filter by correct information
            if (use_correct and annotation.is_correct) or not use_correct:
                # todo: add option to filter for overruled Annotations where mult.=F
                # todo: add option to filter for overlapping Annotations, `add_annotation` just checks for identical
                # filter by start and end offset, include annotations that extend into the offset
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

            # if (use_correct and annotation.is_correct) or not use_correct:
            #     # filter by label
            #     if label is not None and annotation.label == label:
            #         annotations.append(annotation)
            #
            #     elif label is None:
            #         annotations.append(annotation)

            # if fill:
            #    raise NotImplementedError
            #     start_offset = 0
            #     end_offset = len(self.text)
            #     if not annotations:
            #         artificial_anno = Annotation(
            #             start_offset=start_offset, end_offset=end_offset, document=self, label=None, is_correct=False
            #         )
            #         artificial_annotations.append(artificial_anno)
            #
            #     else:
            #         for annotation in annotations:
            #             # create artificial annotations for offsets before the correct annotations spans
            #             for annotation_span in annotation._spans:
            #                 if annotation_span.start_offset > start_offset:
            #                     artificial_anno = Annotation(
            #                         start_offset=start_offset,
            #                         end_offset=annotation_span.start_offset,
            #                         document=self,
            #                         label=None,
            #                         is_correct=False,
            #                     )
            #                     artificial_annotations.append(artificial_anno)
            #
            #                 start_offset = annotation_span.end_offset
            #
            #         if max(span.end_offset for span in annotation._spans) < end_offset:
            #             # create annotation for offsets between last correct annotation and end offset
            #             artificial_anno = Annotation(
            #                 start_offset=max(span.end_offset for span in annotation._spans),
            #                 end_offset=end_offset,
            #                 document=self,
            #                 label=None,
            #                 is_correct=False,
            #             )
            #
            #             artificial_annotations.append(artificial_anno)

        return annotations + artificial_annotations

    @property
    def is_without_errors(self) -> bool:
        """Check if the document can be used for training clf."""
        if self.status is None:
            # Assumption: any Document without status, might be ok
            return True
        else:
            return self.status[0] == 2

    @property
    def document_folder(self):
        """Get the path to the folder where all the document information is cached locally."""
        return os.path.join(self.project.project_folder, "pdf", str(self.id))

    def get_file(self, ocr_version: bool = True, update: bool = False):
        """
        Get OCR version of the original file.

        :param ocr_version: Bool to get the ocr version of the original file
        :param update: Update the downloaded file even if it is already available
        :return: Path to the selected file.
        """
        filename = self.name.replace(":", "-")
        if ocr_version:
            filename = amend_file_name(filename, append_text="ocr", new_extension=".pdf")
            self.ocr_file_path = os.path.join(self.document_folder, filename)

        file_path = os.path.join(self.document_folder, filename)
        if self.is_without_errors and (not file_path or not is_file(file_path, raise_exception=False) or update):
            if not is_file(file_path, raise_exception=False) or update:
                pdf_content = download_file_konfuzio_api(self.id, ocr=ocr_version, session=self.session)
                with open(file_path, "wb") as f:
                    f.write(pdf_content)

        return self.ocr_file_path

    def get_images(self, update: bool = False):
        """
        Get document pages as png images.

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
                    res = retry_get(self.session, url)
                    with open(page_path, "wb") as f:
                        f.write(res.content)

    def get_document_details(self, update):
        """
        Get data from a document.

        :param update: Update the downloaded information even it is already available
        """
        if update or not (
            is_file(self.annotation_file_path, raise_exception=False)
            and is_file(self.annotation_set_file_path, raise_exception=False)
            and is_file(self.txt_file_path, raise_exception=False)
            and is_file(self.pages_file_path, raise_exception=False)
        ):
            data = get_document_details(
                document_id=self.id, project_id=self.project.id, session=self.session, extra_fields="hocr"
            )

            raw_annotations = data["annotations"]
            raw_annotation_sets = data["sections"]
            self.number_of_pages = data["number_of_pages"]

            self.text = data["text"]
            self.hocr = data["hocr"] or None
            self.pages = data["pages"]

            # write a file, even there are no annotations to support offline work
            with open(self.annotation_file_path, "w") as f:
                json.dump(raw_annotations, f, indent=2, sort_keys=True)

            with open(self.annotation_set_file_path, "w") as f:
                json.dump(raw_annotation_sets, f, indent=2, sort_keys=True)

            with open(self.txt_file_path, "w", encoding="utf-8") as f:
                f.write(data["text"])

            with open(self.pages_file_path, "w") as f:
                json.dump(data["pages"], f, indent=2, sort_keys=True)

            if self.hocr is not None:
                with open(self.hocr_file_path, "w", encoding="utf-8") as f:
                    f.write(data["hocr"])

        else:
            with open(self.txt_file_path, "r", encoding="utf-8") as f:
                self.text = f.read()

            with open(self.annotation_file_path, "rb") as f:
                raw_annotations = json.loads(f.read())

            with open(self.annotation_set_file_path, "rb") as f:
                raw_annotation_sets = json.loads(f.read())

            with open(self.pages_file_path, "rb") as f:
                self.pages = json.loads(f.read())

            if is_file(self.hocr_file_path, raise_exception=False):
                # hocr might not be available (depends on the project settings)
                with open(self.hocr_file_path, "r", encoding="utf-8") as f:
                    self.hocr = f.read()

        # first load all Annotation Sets before we create Annotations
        for raw_annotation_set in raw_annotation_sets:
            annotation_set = AnnotationSet(
                id=raw_annotation_set["id"], document=self, label_set=raw_annotation_set["section_label"]
            )  # todo rename
            # todo add parent to define default annotation set
            self.add_annotation_set(annotation_set)

        for raw_annotation in raw_annotations:
            if raw_annotation["custom_offset_string"]:
                logger.warning(
                    f'Annotation {raw_annotation["id"]} is a custom string and, therefore, it will not be added'
                    f' to the document annotations {KONFUZIO_HOST}/a/{raw_annotation["id"]}.'
                )
            else:
                raw_annotation['annotation_set_id'] = raw_annotation.pop('section')
                raw_annotation['label_set_id'] = raw_annotation.pop('section_label_id')
                _ = Annotation(document=self, **raw_annotation)

        return self

    def add_annotation(self, annotation: Annotation, check_duplicate=True):
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

    def add_annotation_set(self, annotation_set: AnnotationSet, check_duplicate=True):
        """Add the annotation sets to the document."""
        if check_duplicate:
            if annotation_set not in self._annotation_sets:
                # todo: skip annotation sets that don't belong to the category: not possilbe via current API
                # if annotation_set.label_set.category == self.category:
                self._annotation_sets.append(annotation_set)
        else:
            self._annotation_sets.append(annotation_set)
        return annotation_set

    def get_annotation_set_by_id(self, id: int) -> AnnotationSet:
        """
        Return a Label Set by ID.

        :param id: ID of the Label Set to get.
        """
        result = None
        for annotation_set in self._annotation_sets:
            if annotation_set.id == id:
                result = annotation_set
        if result:
            return result
        else:
            raise IndexError

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

            with open(self.bio_scheme_file_path, "w", encoding="utf-8") as f:
                for word, tag in converted_text:
                    f.writelines(word + " " + tag + "\n")
                f.writelines("\n")

        bio_annotations = []

        with open(self.bio_scheme_file_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                if line == "\n":
                    continue
                word, tag = line.replace("\n", "").split(" ")
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
        if self._bbox and not update:
            pass
        elif is_file(self.bbox_file_path, raise_exception=False) and not update:
            with open(self.bbox_file_path, "r", encoding="utf-8") as f:
                self._bbox = json.loads(f.read())
        else:
            # todo: reduce number of api calls to get_document_details
            self._bbox = get_document_details(
                document_id=self.id, project_id=self.project.id, session=self.session, extra_fields="bbox"
            )['bbox']
            with open(self.bbox_file_path, "w", encoding="utf-8") as f:
                json.dump(self._bbox, f, indent=2, sort_keys=True)

        return self._bbox

    def save(self) -> bool:
        """
        Save or edit Document online.

        :return: True if the new document was created or existing document was updated.
        """
        document_saved = False
        category_id = None

        if hasattr(self, "category") and self.category is not None:
            category_id = self.category.id

        if not self.is_online:
            response = upload_file_konfuzio_api(
                filepath=self.file_path,
                project_id=self.project.id,
                dataset_status=self.dataset_status,
                category_id=category_id,
            )
            if response.status_code == 201:
                self.id = json.loads(response.text)["id"]
                document_saved = True
            else:
                logger.error(f"Not able to save document {self.file_path} online: {response.text}")
        else:
            response = update_file_konfuzio_api(
                document_id=self.id, file_name=self.name, dataset_status=self.dataset_status, category_id=category_id
            )
            if response.status_code == 200:
                self.project.update_document(document=self)
                document_saved = True
            else:
                logger.error(f"Not able to update document {self.id} online: {response.text}")

        return document_saved

    def update(self):
        """Update document information."""
        self.delete()
        pathlib.Path(self.document_folder).mkdir(parents=True, exist_ok=True)
        self._annotations = []
        self._annotation_sets = []
        self.get_document_details(update=True)
        return self

    def delete(self):
        """Delete all local information for the document."""
        try:
            shutil.rmtree(self.document_folder)
        except FileNotFoundError:
            pass

    # def regex_new(self, label, start_offset: int, end_offset: int, max_findings_per_page=15) -> List[str]:
    #     """Suggest a list of regex which can be used to get the specified offset of a document."""
    #     proposals = []
    #     for spacer in [0, 1, 3, 5, 8, 10]:
    #         proposal = ''
    #         for annotation in self.annotations():
    #             for span in annotation._spans:
    #                 proposal += f'(?:(?P<{label.name_clean}_SUGGEST>{suggest_regex_for_string(span.offset_string)}))'
    #
    #         # do not add duplicates
    #         if re.sub(r'_\d+\>', r'\>', proposal) not in [re.sub(r'_\d+\>', r'\>', cor) for cor in proposals]:
    #             try:
    #                 num_matches = len(re.findall(proposal, self.text))
    #                 if num_matches / (self.text.count('\f') + 1) < max_findings_per_page:
    #                     proposals.append(proposal)
    #                 else:
    #                    logger.info(f'Skip to evaluate regex {repr(proposal)} as it finds {num_matches} in {self}.')
    #             except re.error:
    #                 logger.error('Not able to run regex. Probably the same token is used twice in proposal.')
    #
    #     return proposals

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
            'id': self.id,
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

    def get_annotations(self, update: bool = False):
        """
        Get annotations of the Document.

        :param update: Update the downloaded information even it is already available
        :return: Annotations
        """
        if self.is_without_errors and (not self._annotations or update):
            self.annotation_file_path = os.path.join(self.document_folder, 'annotations.json5')
            if not is_file(self.annotation_file_path, raise_exception=False) or update:
                annotations = get_document_annotations(document_id=self.id, session=self.session)
                # write a file, even there are no annotations to support offline work
                with open(self.annotation_file_path, 'w') as f:
                    json.dump(annotations, f, indent=2, sort_keys=True)
            else:
                with open(self.annotation_file_path, 'r') as f:
                    annotations = json.load(f)

            # add Annotations to the document
            for annotation_data in annotations:
                if not annotation_data['custom_offset_string']:
                    annotation = Annotation(document=self, **annotation_data)
                    self.add_annotation(annotation)
                else:
                    real_string = self.text[annotation_data['start_offset'] : annotation_data['end_offset']]
                    if real_string == annotation_data['offset_string']:
                        annotation = Annotation(document=self, **annotation_data)
                        self.add_annotation(annotation)
                    else:
                        logger.warning(
                            f'Annotation {annotation_data["id"]} is not used '
                            f'in training {KONFUZIO_HOST}/a/{annotation_data["id"]}.'
                        )
        return self._annotations

    # todo: please add tests before adding this functionality
    # def check_annotations(self):
    #     """Check for annotations width more values than allowed."""
    #     labels = self.project.labels
    #     # Labels that can only have 1 value.
    #     labels_to_check = [label.name_clean for label in labels if not label.has_multiple_top_candidates]
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
    #                     f'[Warning] Doc {self.id} - '
    #                     f'AnnotationSet {annotation.label_set.name_clean} ({annotation.label_set.id})- '
    #                     f'Label "{label}" shouldn\'t have more than 1 value. Values = {values}'
    #                 )


class Project(Data):
    """Access the information of a project."""

    def __init__(self, id: int, offline=False, **kwargs):
        """
        Set up the data using the Konfuzio Host.

        :param id: ID of the project
        :param offline: If to get the data from Konfuzio Host
        by default.
        """
        self.id_local = next(Data.id_iter)
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
        if not offline:
            pathlib.Path(self.project_folder).mkdir(parents=True, exist_ok=True)
            pathlib.Path(self.regex_folder).mkdir(parents=True, exist_ok=True)
            pathlib.Path(self.model_folder).mkdir(parents=True, exist_ok=True)
            self.get()  # keep update to False, so once you have downloaded the data, don't do it again.

    def __repr__(self):
        """Return string representation."""
        return f"Project {self.id}"

    @property
    def project_folder(self) -> str:
        """Calculate the data document_folder of the project."""
        return f"data_{self.id}"

    @property
    def regex_folder(self) -> str:
        """Calculate the regex folder of the project."""
        return os.path.join(self.project_folder, "regex")

    @property
    def model_folder(self) -> str:
        """Calculate the model folder of the project."""
        return os.path.join(self.project_folder, "models")

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

    def add_document(self, document: Document):
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
            self.meta_file_path = os.path.join(self.project_folder, "documents_meta.json5")

            if not is_file(self.meta_file_path, raise_exception=False) or update:
                self.meta_data = get_meta_of_files(project_id=self.id, session=self.session)
                with open(self.meta_file_path, "w") as f:
                    json.dump(self.meta_data, f, indent=2, sort_keys=True)
            else:
                with open(self.meta_file_path, "r") as f:
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
                error_message = "You need to get the label sets before getting the categories of the project."
                logger.error(error_message)
                raise ValueError(error_message)

            for label_set in self.label_sets:
                if label_set.is_default:
                    temp_label_set = deepcopy(label_set)
                    temp_label_set.__dict__.pop("project", None)
                    Category(project=self, **temp_label_set.__dict__)

        return self.categories

    def get_label_sets(self, update=False):
        """
        Get Label Sets in the project.

        :param update: Update the downloaded information even it is already available
        :return: Label Sets in the project.
        """
        if not self.label_sets or update:
            self.label_sets_file_path = os.path.join(self.project_folder, "label_sets.json5")
            if not is_file(self.label_sets_file_path, raise_exception=False) or update:
                label_sets_data = get_project_label_sets(project_id=self.id, session=self.session)
                if label_sets_data:
                    # the text of a document can be None
                    with open(self.label_sets_file_path, "w") as f:
                        json.dump(label_sets_data, f, indent=2, sort_keys=True)
            else:
                with open(self.label_sets_file_path, "r") as f:
                    label_sets_data = json.load(f)

            for label_set_data in label_sets_data:
                LabelSet(project=self, **label_set_data)

        return self.label_sets

    def get_labels(self, update=False):
        """
        Get ID and name of any label in the project.

        :param update: Update the downloaded information even it is already available
        :return: Labels in the project.
        """
        if not self.labels or update:
            self.labels_file_path = os.path.join(self.project_folder, "labels.json5")
            if not is_file(self.labels_file_path, raise_exception=False) or update:
                labels_data = get_project_labels(project_id=self.id, session=self.session)
                with open(self.labels_file_path, "w") as f:
                    json.dump(labels_data, f, indent=2, sort_keys=True)
            else:
                with open(self.labels_file_path, "r") as f:
                    labels_data = json.load(f)
            for label_data in labels_data:
                # Remove the project from label_data as we use the already present project reference.
                label_data.pop("project", None)
                Label(project=self, **label_data)

        return self.labels

    def _init_document(self, document_data, document_list_cache, update):
        """
        Initialize Document.

        :param document_data: Document data
        :param document_list_cache: Cache with documents in the project
        :param update: Update the downloaded information even it is already available
        """
        if document_data["status"][0] != 2:
            logger.info(f"Document {document_data['id']} skipped due to: {document_data['status']}")
            return None

        needs_update = False  # basic assumption, document has not changed since the latest pull
        new_in_dataset = False
        if document_data["id"] not in [doc.id for doc in document_list_cache]:
            # it is a new document
            new_in_dataset = True
        else:
            # it might have been changed since our latest pull
            latest_change_online = dateutil.parser.isoparse(document_data["updated_at"])
            doc = [document for document in document_list_cache if document.id == document_data["id"]][0]
            if doc.updated_at is None or (latest_change_online > doc.updated_at):
                needs_update = True

        if (new_in_dataset or needs_update) and update:
            data_path = os.path.join(self.project_folder, "df_data.pkl")
            test_data_path = os.path.join(self.project_folder, "df_test.pkl")
            feature_list_path = os.path.join(self.project_folder, "label_feature_list.pkl")

            if os.path.exists(data_path):
                os.remove(data_path)
            if os.path.exists(test_data_path):
                os.remove(test_data_path)
            if os.path.exists(feature_list_path):
                os.remove(feature_list_path)

        if (new_in_dataset and update) or (needs_update and update):
            doc = Document(project=self, **document_data)
            doc.get_document_details(update=update)
            self.update_document(doc)
        else:
            doc = Document(project=self, **document_data)
            doc.get_document_details(update=False)

    def get_document_by_id(self, document_id: int) -> Document:
        """Return document by it's ID."""
        for document in self.documents:
            if document.id == document_id:
                return document
        raise IndexError

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
            if document_data["dataset_status"] in dataset_statuses:
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
            meta_data_document_ids = set([str(document["id"]) for document in self.meta_data])
            existing_document_ids = set(
                [
                    str(document["id"])
                    for document in self.existing_meta_data
                    if document["dataset_status"] == 2 or document["dataset_status"] == 3
                ]
            )
            remove_document_ids = existing_document_ids.difference(meta_data_document_ids)
            for document_id in remove_document_ids:
                document_path = os.path.join(self.project_folder, "pdf", document_id)
                try:
                    shutil.rmtree(document_path)
                except FileNotFoundError:
                    pass

    def get_label_by_name(self, name: str) -> Label:
        """Return label by its name."""
        for label in self.labels:
            if label.name == name:
                return label

    def get_label_by_id(self, id: int) -> Label:
        """
        Return a label by ID.

        :param id: ID of the label to get.
        """
        for label in self.labels:
            if label.id == id:
                return label

    def get_label_set_by_name(self, name: str) -> LabelSet:
        """
        Return a Label Set by ID.

        :param id: ID of the Label Set to get.
        """
        for label_set in self.label_sets:
            if label_set.name == name:
                return label_set

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

    def get_label_sets_for_category(self, category_id):
        """Get all LabelSets that belong to a category (or represent the category)."""
        project_label_sets = [
            label_set
            for label_set in self.label_sets
            if label_set.id == category_id or category_id in [x.id for x in label_set.categories if x is not None]
        ]
        return project_label_sets

    def check_normalization(self):
        """Check normalized offset_strings."""
        for document in self.documents + self.test_documents:
            for annotation in document.annotations():
                annotation.normalize()

    def update(self):
        """Update the project and all documents by downloading them from the host. Note : It will not work offline."""
        # make sure you always update any changes to Labels, ProjectMeta
        self.existing_meta_data = self.meta_data  # get meta data of what currently exists locally
        self.clean_meta()
        self.get(update=True)

        return self

    def delete(self):
        """Delete all project document data."""
        for document in self.documents:
            document.delete()
        self.clean_meta()
