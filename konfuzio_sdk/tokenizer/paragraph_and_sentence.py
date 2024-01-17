"""Sentence and Paragraph tokenizers."""
import logging
import time
from typing import List, Union

from konfuzio_sdk.data import Annotation, Bbox, Document, Label, Span
from konfuzio_sdk.tokenizer.base import AbstractTokenizer, ProcessingStep
from konfuzio_sdk.utils import (
    detectron_get_paragraph_bboxes,
    get_spans_from_bbox,
    sdk_isinstance,
)

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
        self.mode = mode
        self.line_height_ratio = line_height_ratio
        if height is not None:
            if not (isinstance(height, int) or isinstance(height, float)):
                raise TypeError(f'Parameter must be of type int or float. It is {type(height)}.')
        self.height = height
        self.create_detectron_labels = create_detectron_labels

    def __repr__(self):
        """Return string representation of the class."""
        return f'{self.__class__.__name__}: {self.mode=}'

    def __hash__(self):
        """Get unique hash for RegexTokenizer."""
        return hash(repr(self))

    def tokenize(self, document: Document) -> Document:
        """Create one multiline Annotation per paragraph detected."""
        assert sdk_isinstance(document, Document)

        before_none = len(document.annotations(use_correct=False, label=document.project.no_label))

        t0 = time.monotonic()

        if not document.bboxes_available:
            logger.warning(
                f'Cannot tokenize Document {document} with tokenizer {self}: Missing Character Bbox ' f'information.'
            )
            return document

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
        detectron_document_results = document.get_segmentation()
        all_paragraph_bboxes: List[List['Bbox']] = detectron_get_paragraph_bboxes(detectron_document_results, document)

        if self.create_detectron_labels:
            label_set = document.category.default_label_set
            annotation_set = document.default_annotation_set
        else:
            label_set = document.category.project.no_label_set
            annotation_set = document.no_label_annotation_set

        for document_paragraph_bboxes in all_paragraph_bboxes:
            for paragraph_bbox in document_paragraph_bboxes:
                spans = get_spans_from_bbox(paragraph_bbox)
                if not spans:
                    continue

                if self.create_detectron_labels:
                    try:
                        label = document.category.project.get_label_by_name(paragraph_bbox._label_name)
                    except IndexError:
                        label = Label(
                            project=document.category.project, text=paragraph_bbox._label_name, label_sets=[label_set]
                        )
                else:
                    label = document.category.project.no_label

                try:
                    confidence = None
                    if self.create_detectron_labels:
                        confidence = 1.0
                    annotation = Annotation(
                        document=document,
                        annotation_set=annotation_set,
                        label=label,
                        spans=spans,
                        confidence=confidence,
                    )
                    logger.debug(f'Created new Annotation {annotation}.')
                except ValueError as e:
                    if 'is a duplicate of' in str(e):
                        logger.warning(
                            f'New Annotation with {spans} cannot be tokenized because it is an exact '
                            'duplicated of an existing tokenized Annotation.'
                        )
                    else:
                        raise e
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
        :param create_detectron_labels: Apply the labels given by the detectron model. If they don't exist, they are
        created.
        """
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
        return f'{self.__class__.__name__}: {self.mode=}'

    def __hash__(self):
        """Get unique hash for RegexTokenizer."""
        return hash(repr(self))

    def tokenize(self, document: Document) -> Document:
        """Create one multiline Annotation per sentence detected."""
        assert sdk_isinstance(document, Document)

        before_none = len(document.annotations(use_correct=False, label=document.project.no_label))

        t0 = time.monotonic()

        if not document.bboxes_available:
            logger.warning(
                f'Cannot tokenize Document {document} with tokenizer {self}: Missing Character Bbox ' f'information.'
            )
            return document

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
        detectron_document_results = document.get_segmentation()
        all_paragraph_bboxes: List[List['Bbox']] = detectron_get_paragraph_bboxes(detectron_document_results, document)

        if self.create_detectron_labels:
            label_set = document.category.default_label_set
            annotation_set = document.default_annotation_set
        else:
            label_set = document.project.no_label_set
            annotation_set = document.no_label_annotation_set

        for document_paragraph_bboxes in all_paragraph_bboxes:
            for paragraph_bbox in document_paragraph_bboxes:
                spans = get_spans_from_bbox(paragraph_bbox)
                if not spans:
                    continue

                sentence_spans = Span.get_sentence_from_spans(spans=spans, punctuation=self.punctuation)
                for spans in sentence_spans:
                    if self.create_detectron_labels:
                        try:
                            label = document.category.project.get_label_by_name(paragraph_bbox._label_name)
                        except IndexError:
                            label = Label(
                                project=document.category.project,
                                text=paragraph_bbox._label_name,
                                label_sets=[label_set],
                            )
                    else:
                        label = document.project.no_label

                    try:
                        annotation = Annotation(
                            document=document,
                            annotation_set=annotation_set,
                            label=label,
                            category=document.category,
                            spans=spans,
                        )
                        logger.debug(f'Created new Annotation {annotation}.')
                    except ValueError as e:
                        if 'is a duplicate of' in str(e):
                            logger.warning(
                                f'New Annotation with {spans} cannot be tokenized because it is an exact '
                                'duplicated of an existing tokenized Annotation.'
                            )
                        else:
                            raise e
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
                spans=sentence_spans,
            )

        return document

    def found_spans(self, document: Document):
        """Sentence found spans."""
        pass
