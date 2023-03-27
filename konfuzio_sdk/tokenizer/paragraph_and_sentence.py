"""Sentence and Paragraph tokenizers."""
import logging
from typing import List, Dict, Tuple, Union
import collections
import time

from konfuzio_sdk.data import Annotation, Document, Span, Bbox, Label, AnnotationSet
from konfuzio_sdk.tokenizer.base import AbstractTokenizer, ProcessingStep

from konfuzio_sdk.utils import sdk_isinstance, detectron_get_paragraph_bbox_and_label_name, get_spans_from_bbox, \
    detectron_get_paragraph_bboxes
from konfuzio_sdk.api import get_results_from_segmentation

logger = logging.getLogger(__name__)


class ParagraphTokenizer(AbstractTokenizer):
    """Tokenizer splitting Document into paragraphs."""

    def __init__(
        self,
        mode: str = 'detectron',
        line_height_ratio: float = 0.8,
        height: Union[int, float, None] = None,
        create_detectron_labels: bool = False,
    ):
        """
        Initialize Paragraph Tokenizer.

        :param mode: line_distance or detectron
        :param line_height_ratio: ratio of the median line height to use as threshold to create new paragraph.
        :param height: Height line threshold to use instead of the automatically calculated one.
        :param create_detectron_labels: Apply the labels given by the detectron model. If they don't exist, they are
        created.
        """
        super().__init__()
        self.mode = mode
        self.line_height_ratio = line_height_ratio
        if height is not None:
            if not (isinstance(height, int) or isinstance(height, float)):
                raise TypeError(f'Parameter must be of type int or float. It is {type(height)}.')
        self.height = height
        self.create_detectron_labels = create_detectron_labels

    def _new_detectron_tokenize(self, document: Document) -> Document:
        """Create one multiline Annotation per paragraph detected by detectron2."""
        document_id = document.id_ if document.id_ else document.copy_of_id

        # todo cache results
        detectron_document_results = get_results_from_segmentation(document_id, document.project.id_)
        all_paragraph_bboxes: List[List['Bbox']] = detectron_get_paragraph_bboxes(detectron_document_results, document)

        # Check if detrectron paragraphs overlap. TODO seems to be the case.
        for x in all_paragraph_bboxes:
            for y1 in x:
                for y2 in x:
                    assert not y1.check_overlap(y2)

        if self.create_detectron_labels:
            label_set = document.category.project.get_label_set_by_name(document.category.name)
            annotation_set = AnnotationSet(document=document, label_set=label_set, id_=1)
        else:
            label_set = document.project.no_label_set
            annotation_set = document.no_label_annotation_set

        for document_paragraph_bboxes in all_paragraph_bboxes:
            for paragraph_bbox in document_paragraph_bboxes:
                spans = get_spans_from_bbox(paragraph_bbox)

                if self.create_detectron_labels:
                    try:
                        label = document.project.get_label_by_name(paragraph_bbox._label_name)
                    except IndexError:
                        label = Label(project=document.project, text=paragraph_bbox._label_name, label_sets=[label_set])
                else:
                    label = document.project.no_label

                annotation = Annotation(
                    document=document,
                    annotation_set=annotation_set,
                    label=label,
                    label_set=document.project.no_label_set,
                    category=document.category,
                    spans=spans,
                )
        return document

    def __repr__(self):
        """Return string representation of the class."""
        return f"{self.__class__.__name__}: {self.mode=}"

    def __hash__(self):
        """Get unique hash for RegexTokenizer."""
        return hash(repr(self))

    def tokenize(self, document: Document) -> Document:
        """Create one multiline Annotation per paragraph detected."""
        assert sdk_isinstance(document, Document)

        before_none = len(document.annotations(use_correct=False, label=document.project.no_label))

        t0 = time.monotonic()

        if not document.bboxes_available:
            raise ValueError(
                f"Cannot tokenize Document {document} with tokenizer {self}: Missing Character Bbox information."
            )

        if self.mode == 'detectron':
            self._detectron_tokenize(document=document)
        elif self.mode == 'line_distance':
            self._line_distance_tokenize(document=document)

        after_none = len(document.annotations(use_correct=False, label=document.project.no_label))
        logger.info(f'{after_none - before_none} new Annotations in {document} by {repr(self)}.')

        self.processing_steps.append(ProcessingStep(self, document, time.monotonic() - t0))

        return document

    def _detectron_tokenize(self, document: Document) -> Document:
        """Create one multiline Annotation per paragraph detected by detectron2."""
        document_id = document.id_ if document.id_ else document.copy_of_id
        detectron_document_results = get_results_from_segmentation(document_id, document.project.id_)
        paragraph_bboxes_and_labels = detectron_get_paragraph_bbox_and_label_name(detectron_document_results, document)

        if self.create_detectron_labels:
            category_label_set = document.category.project.get_label_set_by_name(document.category.name)
            default_annotation_set = AnnotationSet(document=document, label_set=category_label_set, id_=1)

        for page in document.pages():
            page_paragraph_bboxes_and_labels = paragraph_bboxes_and_labels[page.index]
            paragraph_span_bboxes = collections.defaultdict(list)
            current_paragraph_and_label: Tuple[Bbox, str] = None
            paragraph_annotations: Dict[Bbox, Annotation] = {}
            for bbox in sorted(page.get_bbox().values(), key=lambda x: x['char_index']):
                if bbox['text'] == ' ':
                    continue
                for paragraph_bbox_and_label in page_paragraph_bboxes_and_labels:
                    paragraph_bbox_and_label: Tuple[Bbox, str]
                    if paragraph_bbox_and_label[0].check_overlap(bbox):
                        if not current_paragraph_and_label:
                            current_paragraph_and_label = paragraph_bbox_and_label
                        if paragraph_span_bboxes[current_paragraph_and_label] and (
                            (
                                paragraph_span_bboxes[current_paragraph_and_label][-1]['line_number']
                                != bbox['line_number']
                            )
                            or (paragraph_bbox_and_label is not current_paragraph_and_label)
                        ):
                            span = Span(
                                start_offset=paragraph_span_bboxes[current_paragraph_and_label][0]['char_index'],
                                end_offset=paragraph_span_bboxes[current_paragraph_and_label][-1]['char_index'] + 1,
                            )
                            if current_paragraph_and_label not in paragraph_annotations:
                                if self.create_detectron_labels:
                                    label_set = category_label_set
                                    annotation_set = default_annotation_set
                                    label_name = current_paragraph_and_label[1]
                                    try:
                                        label = document.project.get_label_by_name(label_name)
                                    except IndexError:
                                        label = Label(project=document.project, text=label_name, label_sets=[label_set])
                                else:
                                    label = document.project.no_label
                                    annotation_set = document.no_label_annotation_set
                                    label_set = document.project.no_label_set

                                annotation = Annotation(
                                    document=document,
                                    annotation_set=annotation_set,
                                    label=label,
                                    label_set=document.project.no_label_set,
                                    category=document.category,
                                    spans=[span],
                                )
                                paragraph_annotations[current_paragraph_and_label] = annotation
                            else:
                                paragraph_annotations[current_paragraph_and_label].add_span(span)
                            paragraph_span_bboxes[current_paragraph_and_label] = []

                            current_paragraph_and_label = paragraph_bbox_and_label
                            paragraph_span_bboxes[current_paragraph_and_label] = [bbox]
                        else:
                            paragraph_span_bboxes[current_paragraph_and_label].append(bbox)
                        break
            for paragraph_bbox_and_label, span_bboxes in paragraph_span_bboxes.items():
                if span_bboxes:
                    span = Span(start_offset=span_bboxes[0]['char_index'], end_offset=span_bboxes[-1]['char_index'] + 1)
                    if paragraph_bbox_and_label not in paragraph_annotations:
                        if self.create_detectron_labels:
                            label_set = category_label_set
                            annotation_set = default_annotation_set
                            label_name = current_paragraph_and_label[1]
                            try:
                                label = document.project.get_label_by_name(label_name)
                            except IndexError:
                                label = Label(project=document.project, text=label_name, label_sets=[label_set])
                        else:
                            label = document.project.no_label
                            annotation_set = document.no_label_annotation_set
                            label_set = document.project.no_label_set

                        annotation = Annotation(
                            document=document,
                            annotation_set=annotation_set,
                            label=label,
                            label_set=document.project.no_label_set,
                            category=document.category,
                            spans=[span],
                        )
                        paragraph_annotations[current_paragraph_and_label] = annotation
                    else:
                        paragraph_annotations[current_paragraph_and_label].add_span(span)

        return document

    def _line_distance_tokenize(self, document: Document) -> Document:
        """Create one multiline Annotation per paragraph detected by line distance based rule based algorithm."""
        from statistics import median

        for page in document.pages():
            page_char_bboxes = list(page.get_bbox().values())

            # set line_threshold
            if self.height is None:
                # calculate median vertical character size for Page
                line_threshold = round(
                    self.line_height_ratio * median(bbox['y1'] - bbox['y0'] for bbox in page_char_bboxes),
                    6,
                )
            else:
                line_threshold = self.height

            # go through lines to find paragraphs
            previous_y0 = None
            paragraph_spans = []
            for span in page.lines():

                if not paragraph_spans or previous_y0 - span.bbox().y1 < line_threshold:
                    paragraph_spans.append(span)
                else:
                    _ = Annotation(
                        document=document,
                        annotation_set=document.no_label_annotation_set,
                        label=document.project.no_label,
                        label_set=document.project.no_label_set,
                        category=document.category,
                        spans=paragraph_spans,
                    )
                    paragraph_spans = [span]

                # Update botton edge of previous Paragraph to the Paragraph we just assigned
                previous_y0 = span.bbox().y0

            _ = Annotation(
                document=document,
                annotation_set=document.no_label_annotation_set,
                label=document.project.no_label,
                label_set=document.project.no_label_set,
                category=document.category,
                spans=paragraph_spans,
            )
        return document

    def found_spans(self, document: Document):
        """Sentence found spans."""
        pass


class SentenceTokenizer(AbstractTokenizer):
    """Tokenizer splitting Document into Sentences."""

    def __init__(
        self,
        mode: str = 'detectron',
        line_height_ratio: float = 0.8,
        height: Union[int, float, None] = None,
        create_detectron_labels: bool = False,
    ):
        """
        Initialize Sentence Tokenizer.

        :param mode: line_distance or detectron
        :param line_height_ratio: ratio of the median line height to use as threshold to create new paragraph.
        :param height: Height line threshold to use instead of the automatically calulated one.
        """
        super().__init__()
        self.mode = mode
        self.line_height_ratio = line_height_ratio
        if height is not None:
            if not (isinstance(height, int) or isinstance(height, float)):
                raise TypeError(f'Parameter must be of type int or float. It is {type(height)}.')
        self.height = height
        self.punctuation = {'.', '!', '?'}
        self.create_detectron_labels = create_detectron_labels

    def __repr__(self):
        """Return string representation of the class."""
        return f"{self.__class__.__name__}: {self.mode=}"

    def __hash__(self):
        """Get unique hash for RegexTokenizer."""
        return hash(repr(self))

    def tokenize(self, document: Document) -> Document:
        """Create one multiline Annotation per paragraph detected."""
        assert sdk_isinstance(document, Document)

        before_none = len(document.annotations(use_correct=False, label=document.project.no_label))

        t0 = time.monotonic()

        if not document.bboxes_available:
            raise ValueError(
                f"Cannot tokenize Document {document} with tokenizer {self}: Missing Character Bbox information."
            )

        if self.mode == 'detectron':
            self._detectron_tokenize(document=document)
        elif self.mode == 'line_distance':
            self._line_distance_tokenize(document=document)

        after_none = len(document.annotations(use_correct=False, label=document.project.no_label))
        logger.info(f'{after_none - before_none} new Annotations in {document} by {repr(self)}.')

        self.processing_steps.append(ProcessingStep(self, document, time.monotonic() - t0))

        return document

    def _detectron_tokenize(self, document: Document) -> Document:
        """Create one multiline Annotation per sentence detected in paragraph detected by detectron."""
        document_id = document.id_ if document.id_ else document.copy_of_id
        detectron_document_results = get_results_from_segmentation(document_id, document.project.id_)
        paragraph_bboxes_and_labels = detectron_get_paragraph_bbox_and_label_name(detectron_document_results, document)

        if self.create_detectron_labels:
            category_label_set = document.category.project.get_label_set_by_name(document.category.name)
            default_annotation_set = AnnotationSet(document=document, label_set=category_label_set, id_=1)

        for _, (page, page_paragraph_bboxes_and_labels) in enumerate(
            zip(document.pages(), paragraph_bboxes_and_labels)
        ):
            paragraph_span_bboxes = collections.defaultdict(list)
            current_paragraph_and_label: Tuple[Bbox, str] = None
            paragraph_sentence_anns: Dict[Bbox, List[Annotation]] = {}
            for bbox in sorted(page.get_bbox().values(), key=lambda x: x['char_index']):
                if bbox['text'] == ' ':
                    continue
                for paragraph_bbox_and_label in page_paragraph_bboxes_and_labels:
                    paragraph_bbox_and_label: Tuple[Bbox, str]
                    if paragraph_bbox_and_label[0].check_overlap(bbox):
                        if not current_paragraph_and_label:
                            current_paragraph_and_label = paragraph_bbox_and_label
                        if paragraph_span_bboxes[current_paragraph_and_label] and (
                            (
                                paragraph_span_bboxes[current_paragraph_and_label][-1]['line_number']
                                != bbox['line_number']
                            )
                            or (paragraph_bbox_and_label is not current_paragraph_and_label)
                            or (paragraph_span_bboxes[current_paragraph_and_label][-1]['text'] in self.punctuation)
                        ):
                            span = Span(
                                start_offset=paragraph_span_bboxes[current_paragraph_and_label][0]['char_index'],
                                end_offset=paragraph_span_bboxes[current_paragraph_and_label][-1]['char_index'] + 1,
                            )
                            if current_paragraph_and_label not in paragraph_sentence_anns:
                                paragraph_sentence_anns[current_paragraph_and_label] = []

                            if (
                                not paragraph_sentence_anns[current_paragraph_and_label]
                                or paragraph_sentence_anns[current_paragraph_and_label][-1].spans[-1].offset_string[-1]
                                in self.punctuation
                            ):
                                if self.create_detectron_labels:
                                    label_set = category_label_set
                                    annotation_set = default_annotation_set
                                    label_name = current_paragraph_and_label[1]
                                    try:
                                        label = document.project.get_label_by_name(label_name)
                                    except IndexError:
                                        label = Label(project=document.project, text=label_name, label_sets=[label_set])
                                else:
                                    label = document.project.no_label
                                    annotation_set = document.no_label_annotation_set
                                    label_set = document.project.no_label_set

                                annotation = Annotation(
                                    document=document,
                                    annotation_set=annotation_set,
                                    label=label,
                                    label_set=document.project.no_label_set,
                                    category=document.category,
                                    spans=[span],
                                )
                                paragraph_sentence_anns[current_paragraph_and_label].append(annotation)
                            else:
                                paragraph_sentence_anns[current_paragraph_and_label][-1].add_span(span)

                            paragraph_span_bboxes[current_paragraph_and_label] = []

                            current_paragraph_and_label = paragraph_bbox_and_label
                            paragraph_span_bboxes[current_paragraph_and_label] = [bbox]
                        else:
                            paragraph_span_bboxes[current_paragraph_and_label].append(bbox)
                        break

            for paragraph_bbox_and_label, span_bboxes in paragraph_span_bboxes.items():
                if span_bboxes:
                    span = Span(start_offset=span_bboxes[0]['char_index'], end_offset=span_bboxes[-1]['char_index'] + 1)
                    if current_paragraph_and_label not in paragraph_sentence_anns:
                        paragraph_sentence_anns[current_paragraph_and_label] = []

                    if (
                        not paragraph_sentence_anns[current_paragraph_and_label]
                        or paragraph_sentence_anns[current_paragraph_and_label][-1].spans[-1].offset_string[-1]
                        in self.punctuation
                    ):
                        if self.create_detectron_labels:
                            label_set = category_label_set
                            annotation_set = default_annotation_set
                            label_name = current_paragraph_and_label[1]
                            try:
                                label = document.project.get_label_by_name(label_name)
                            except IndexError:
                                label = Label(project=document.project, text=label_name, label_sets=[label_set])
                        else:
                            label = document.project.no_label
                            annotation_set = document.no_label_annotation_set
                            label_set = document.project.no_label_set

                        annotation = Annotation(
                            document=document,
                            annotation_set=annotation_set,
                            label=label,
                            label_set=document.project.no_label_set,
                            category=document.category,
                            spans=[span],
                        )
                        paragraph_sentence_anns[current_paragraph_and_label].append(annotation)
                    else:
                        paragraph_sentence_anns[current_paragraph_and_label][-1].add_span(span)

        return document

    def _line_distance_tokenize(self, document: Document) -> Document:
        """Create one multiline Annotation per sentence in paragraph detected by line distance based rule based algo."""
        from statistics import median

        for page in document.pages():
            page_char_bboxes = page.get_bbox().values()
            # set line_threshold
            if self.height is None:
                # calculate median vertical character size for Page
                line_threshold = round(
                    self.line_height_ratio * median(bbox['y1'] - bbox['y0'] for bbox in page_char_bboxes),
                    6,
                )
            else:
                line_threshold = self.height

            # assemble bboxes by line and Page
            sentence_spans = []
            previous_y0 = None
            page_char_bboxes = sorted(page_char_bboxes, key=lambda x: x['char_index'])

            for line_span in page.lines():
                new_span_start_offset = line_span.start_offset
                max_y1 = line_span.bbox().y1

                if sentence_spans and previous_y0 - max_y1 >= line_threshold:
                    _ = Annotation(
                        document=document,
                        annotation_set=document.no_label_annotation_set,
                        label=document.project.no_label,
                        label_set=document.project.no_label_set,
                        category=document.category,
                        spans=sentence_spans,
                    )
                    sentence_spans = []
                for i, line_character in enumerate(line_span.offset_string):
                    if line_character and line_character in self.punctuation:
                        start_offset = new_span_start_offset
                        end_offset = line_span.start_offset + i + 1
                        new_span_start_offset = end_offset
                        sentence_spans.append(Span(start_offset=start_offset, end_offset=end_offset))
                        _ = Annotation(
                            document=document,
                            annotation_set=document.no_label_annotation_set,
                            label=document.project.no_label,
                            label_set=document.project.no_label_set,
                            category=document.category,
                            spans=sentence_spans,
                        )
                        sentence_spans = []
                if new_span_start_offset < line_span.end_offset:
                    sentence_spans.append(Span(start_offset=new_span_start_offset, end_offset=line_span.end_offset))
                previous_y0 = line_span.bbox().y0

        if sentence_spans:
            _ = Annotation(
                document=document,
                annotation_set=document.no_label_annotation_set,
                label=document.project.no_label,
                label_set=document.project.no_label_set,
                category=document.category,
                spans=sentence_spans,
            )

        return document

    def found_spans(self, document: Document):
        """Sentence found spans."""
        pass
