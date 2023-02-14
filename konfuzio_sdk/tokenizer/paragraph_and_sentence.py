"""Sentence and Paragraph tokenizers."""
import logging
from typing import List, Dict, Union
import collections
import time

from konfuzio_sdk.data import Annotation, Document, Span, Bbox
from konfuzio_sdk.tokenizer.base import AbstractTokenizer, ProcessingStep

from konfuzio_sdk.utils import sdk_isinstance, detectron_get_paragraph_bbox
from konfuzio_sdk.api import get_results_from_segmentation

logger = logging.getLogger(__name__)


class ParagraphTokenizer(AbstractTokenizer):
    """Tokenizer splitting Document into paragraphs."""

    def __init__(self, mode: str = 'detectron', line_height_ratio: float = 0.8, height: Union[int, float, None] = None):
        """
        Initialize Paragraph Tokenizer.

        :param mode: line_distance or detectron
        :param line_height_ratio: ratio of the median line height to use as threshold to create new paragraph.
        :param height: Height line threshold to use instead of the automatically calculated one.
        """
        super().__init__()
        self.mode = mode
        self.line_height_ratio = line_height_ratio
        if height is not None:
            if not (isinstance(height, int) or isinstance(height, float)):
                raise TypeError(f'Parameter must be of type int or float. It is {type(height)}.')
        self.height = height

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
        detectron_results = get_results_from_segmentation(document_id, document.project.id_)
        paragraph_bboxes = detectron_get_paragraph_bbox(detectron_results, document)

        for page in document.pages():
            page_paragraph_bboxes = paragraph_bboxes[page.index]
            paragraph_span_bboxes = collections.defaultdict(list)
            current_paragraph: Bbox = None
            paragraph_annotations: Dict[Bbox, Annotation] = {}
            for bbox in sorted(page.get_bbox().values(), key=lambda x: x['char_index']):
                for paragraph_bbox in page_paragraph_bboxes:
                    if paragraph_bbox.check_overlap(bbox):
                        if not current_paragraph:
                            current_paragraph = paragraph_bbox
                        if paragraph_span_bboxes[current_paragraph] and (
                            (paragraph_span_bboxes[current_paragraph][-1]['line_number'] != bbox['line_number'])
                            or (paragraph_bbox is not current_paragraph)
                        ):
                            span = Span(
                                start_offset=paragraph_span_bboxes[current_paragraph][0]['char_index'],
                                end_offset=paragraph_span_bboxes[current_paragraph][-1]['char_index'] + 1,
                            )
                            if current_paragraph not in paragraph_annotations:
                                annotation = Annotation(
                                    document=document,
                                    annotation_set=document.no_label_annotation_set,
                                    label=document.project.no_label,
                                    label_set=document.project.no_label_set,
                                    category=document.category,
                                    spans=[span],
                                )
                                paragraph_annotations[current_paragraph] = annotation
                            else:
                                paragraph_annotations[current_paragraph].add_span(span)
                            paragraph_span_bboxes[current_paragraph] = []

                            current_paragraph = paragraph_bbox
                            paragraph_span_bboxes[current_paragraph] = [bbox]
                        else:
                            paragraph_span_bboxes[current_paragraph].append(bbox)
                        break
            for paragraph_bbox, span_bboxes in paragraph_span_bboxes.items():
                if span_bboxes:
                    span = Span(start_offset=span_bboxes[0]['char_index'], end_offset=span_bboxes[-1]['char_index'] + 1)
                    if paragraph_bbox not in paragraph_annotations:
                        annotation = Annotation(
                            document=document,
                            annotation_set=document.no_label_annotation_set,
                            label=document.project.no_label,
                            label_set=document.project.no_label_set,
                            category=document.category,
                            spans=[span],
                        )
                        paragraph_annotations[current_paragraph] = annotation
                    else:
                        paragraph_annotations[current_paragraph].add_span(span)

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

    def __init__(self, mode: str = 'detectron', line_height_ratio: float = 0.8, height: Union[int, float, None] = None):
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
        detectron_results = get_results_from_segmentation(document_id, document.project.id_)
        paragraph_bboxes = detectron_get_paragraph_bbox(detectron_results, document)

        for _, (page, page_paragraph_bboxes) in enumerate(zip(document.pages(), paragraph_bboxes)):
            paragraph_span_bboxes = collections.defaultdict(list)
            current_paragraph = None
            paragraph_sentence_anns: Dict[Bbox, List[Annotation]] = {}
            for bbox in sorted(page.get_bbox().values(), key=lambda x: x['char_index']):

                for paragraph_bbox in page_paragraph_bboxes:
                    if paragraph_bbox.check_overlap(bbox):
                        if not current_paragraph:
                            current_paragraph = paragraph_bbox
                        if paragraph_span_bboxes[current_paragraph] and (
                            (paragraph_span_bboxes[current_paragraph][-1]['line_number'] != bbox['line_number'])
                            or (paragraph_bbox is not current_paragraph)
                            or (paragraph_span_bboxes[current_paragraph][-1]['text'] in self.punctuation)
                        ):
                            span = Span(
                                start_offset=paragraph_span_bboxes[current_paragraph][0]['char_index'],
                                end_offset=paragraph_span_bboxes[current_paragraph][-1]['char_index'] + 1,
                            )
                            if current_paragraph not in paragraph_sentence_anns:
                                paragraph_sentence_anns[current_paragraph] = []

                            if (
                                not paragraph_sentence_anns[current_paragraph]
                                or paragraph_sentence_anns[current_paragraph][-1].spans[-1].offset_string[-1]
                                in self.punctuation
                            ):
                                annotation = Annotation(
                                    document=document,
                                    annotation_set=document.no_label_annotation_set,
                                    label=document.project.no_label,
                                    label_set=document.project.no_label_set,
                                    category=document.category,
                                    spans=[span],
                                )
                                paragraph_sentence_anns[current_paragraph].append(annotation)
                            else:
                                paragraph_sentence_anns[current_paragraph][-1].add_span(span)

                            paragraph_span_bboxes[current_paragraph] = []

                            current_paragraph = paragraph_bbox
                            paragraph_span_bboxes[current_paragraph] = [bbox]
                        else:
                            paragraph_span_bboxes[current_paragraph].append(bbox)
                        break

            for paragraph_bbox, span_bboxes in paragraph_span_bboxes.items():
                if span_bboxes:
                    span = Span(start_offset=span_bboxes[0]['char_index'], end_offset=span_bboxes[-1]['char_index'] + 1)
                    if current_paragraph not in paragraph_sentence_anns:
                        paragraph_sentence_anns[current_paragraph] = []

                    if (
                        not paragraph_sentence_anns[current_paragraph]
                        or paragraph_sentence_anns[current_paragraph][-1].spans[-1].offset_string[-1]
                        in self.punctuation
                    ):
                        annotation = Annotation(
                            document=document,
                            annotation_set=document.no_label_annotation_set,
                            label=document.project.no_label,
                            label_set=document.project.no_label_set,
                            category=document.category,
                            spans=[span],
                        )
                        paragraph_sentence_anns[current_paragraph].append(annotation)
                    else:
                        paragraph_sentence_anns[current_paragraph][-1].add_span(span)

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
