"""Sentence and Paragraph tokenizers."""
import logging
import abc
from typing import List, Dict
import collections
import time

from konfuzio_sdk.data import Annotation, Document, Span, Bbox
from konfuzio_sdk.tokenizer.base import AbstractTokenizer, ProcessingStep

from konfuzio_sdk.utils import sdk_isinstance
from konfuzio_sdk.api import get_results_from_segmentation

logger = logging.getLogger(__name__)


class BlockTokenizer(AbstractTokenizer):
    """Block tokenizer for methods shared between Paragraph and Sentence tokenizers."""

    def __init__(self, mode: str, line_height_ratio: float):
        """Initialize Block Tokenizer."""
        self.mode = mode
        self.line_height_ratio = line_height_ratio

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

    def _detectron_get_paragraph_bbox(self, document: Document) -> Dict[Bbox, List[Dict]]:
        """Call detectron Bbox corresponding to each paragraph."""
        assert isinstance(document.project.id_, int)
        doc_id = document.id_ if document.id_ else document.copy_of_id
        detectron_paragraph_bboxes = get_results_from_segmentation(doc_id, document.project.id_)

        assert len(detectron_paragraph_bboxes) == document.number_of_pages

        paragraph_bboxes: List[List[Bbox]] = []
        for i, page in enumerate(document.pages()):
            paragraph_bboxes.append([])
            page = document.get_page_by_index(i)
            scale_factor = page.height / page.full_height
            for bbox in detectron_paragraph_bboxes[i]:
                paragraph_bboxes[-1].append(
                    Bbox(
                        x0=bbox['x0'] * scale_factor,
                        x1=bbox['x1'] * scale_factor,
                        y1=page.height - bbox['y0'] * scale_factor,
                        y0=page.height - bbox['y1'] * scale_factor,
                        page=page,
                    )
                )
        assert len(document.pages()) == len(paragraph_bboxes)

        return paragraph_bboxes

    @abc.abstractmethod
    def _line_distance_tokenize(self, document: Document, height=None) -> Document:
        """Use line distance rule based algorithm to perform tokenization."""

    @abc.abstractmethod
    def _detectron_tokenize(self, document: Document) -> Document:
        """Use Detectron endpoint to perform tokenization."""

    def found_spans(self, document: Document):
        """Sentence found spans."""
        pass


class ParagraphTokenizer(BlockTokenizer):
    """Tokenizer splitting Document into paragraphs."""

    def __init__(self, mode: str = 'detectron', line_height_ratio: float = 0.8):
        """
        Initialize Paragraph Tokenizer.

        :param mode: line_distance or detectron
        """
        super().__init__(mode=mode, line_height_ratio=line_height_ratio)

    def _detectron_tokenize(self, document: Document) -> Document:
        """Create one multiline Annotation per paragraph detected by detectron2."""
        paragraph_bboxes = self._detectron_get_paragraph_bbox(document)

        for i, (page, page_paragraph_bboxes) in enumerate(zip(document.pages(), paragraph_bboxes)):
            page = document.get_page_by_index(i)
            paragraph_span_bboxes = collections.defaultdict(list)
            curr_paragraph = None
            paragraph_anns: Dict[Bbox, Annotation] = {}
            for bbox in sorted(page.get_bbox(), key=lambda x: x['char_index']):

                for paragraph_bbox in page_paragraph_bboxes:
                    if paragraph_bbox.check_overlap(bbox):
                        if not curr_paragraph:
                            curr_paragraph = paragraph_bbox
                        if paragraph_span_bboxes[curr_paragraph] and (
                            (paragraph_span_bboxes[curr_paragraph][-1]['line_number'] != bbox['line_number'])
                            or (paragraph_bbox is not curr_paragraph)
                        ):
                            span = Span(
                                start_offset=paragraph_span_bboxes[curr_paragraph][0]['char_index'],
                                end_offset=paragraph_span_bboxes[curr_paragraph][-1]['char_index'] + 1,
                            )
                            if curr_paragraph not in paragraph_anns:
                                annotation = Annotation(
                                    document=document,
                                    annotation_set=document.no_label_annotation_set,
                                    label=document.project.no_label,
                                    label_set=document.project.no_label_set,
                                    category=document.category,
                                    spans=[span],
                                )
                                paragraph_anns[curr_paragraph] = annotation
                            else:
                                paragraph_anns[curr_paragraph].add_span(span)
                            paragraph_span_bboxes[curr_paragraph] = []

                            curr_paragraph = paragraph_bbox
                            paragraph_span_bboxes[curr_paragraph] = [bbox]
                        else:
                            paragraph_span_bboxes[curr_paragraph].append(bbox)
                        break

            for paragraph_bbox, span_bboxes in paragraph_span_bboxes.items():
                if span_bboxes:
                    span = Span(start_offset=span_bboxes[0]['char_index'], end_offset=span_bboxes[-1]['char_index'] + 1)
                    if paragraph_bbox not in paragraph_anns:
                        annotation = Annotation(
                            document=document,
                            annotation_set=document.no_label_annotation_set,
                            label=document.project.no_label,
                            label_set=document.project.no_label_set,
                            category=document.category,
                            spans=[span],
                        )
                        paragraph_anns[curr_paragraph] = annotation
                    else:
                        paragraph_anns[curr_paragraph].add_span(span)

        return document

    def _line_distance_tokenize(self, document: Document) -> Document:
        """Create one multiline Annotation per paragraph detected by line distance based rule based algorithm."""
        height = None
        if height is not None:
            if not (isinstance(height, int) or isinstance(height, float)):
                raise TypeError(f'Parameter must be of type int or float. It is {type(height)}.')

        from statistics import median

        # assemble bboxes by their page
        pages_char_bboxes = [[] for _ in document.pages()]
        for char_index, bbox in document.get_bbox().items():
            bbox['char_index'] = int(char_index)
            pages_char_bboxes[bbox['page_number'] - 1].append(bbox)

        for page_char_bboxes in pages_char_bboxes:
            # assemble bboxes by line and Page
            page_lines = []
            line_bboxes = []
            for bbox in sorted(page_char_bboxes, key=lambda x: x['char_index']):
                if not line_bboxes or line_bboxes[-1]['line_number'] == bbox['line_number']:
                    line_bboxes.append(bbox)
                else:
                    page_lines.append(line_bboxes)
                    line_bboxes = [bbox]
            page_lines.append(line_bboxes)

            # set line_threshold
            if height is None:
                # calculate median vertical character size for Page
                line_threshold = round(
                    self.line_height_ratio * median(bbox['y1'] - bbox['y0'] for bbox in page_char_bboxes),
                    6,
                )
            else:
                line_threshold = height

            # go through lines to find paragraphs
            previous_y0 = None
            paragraph_spans = []
            for line in page_lines:
                assert line[0]['line_number'] == line[-1]['line_number']
                max_y1 = max([bbox['y1'] for bbox in line])
                min_y0 = min([bbox['y0'] for bbox in line])
                span = Span(start_offset=line[0]['char_index'], end_offset=line[-1]['char_index'] + 1)
                if not paragraph_spans or previous_y0 - max_y1 < line_threshold:
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

                previous_y0 = min_y0
            _ = Annotation(
                document=document,
                annotation_set=document.no_label_annotation_set,
                label=document.project.no_label,
                label_set=document.project.no_label_set,
                category=document.category,
                spans=paragraph_spans,
            )
        return document


class SentenceTokenizer(BlockTokenizer):
    """Tokenizer splitting Document into Sentences."""

    def __init__(self, mode: str = 'detectron', line_height_ratio: float = 0.8):
        """
        Initialize Sentence Tokenizer.

        :param mode: line_distance or detectron
        """
        super().__init__(mode=mode, line_height_ratio=line_height_ratio)
        self.punctuation = {'.', '!', '?'}

    def _detectron_tokenize(self, document: Document) -> Document:
        """Create one multiline Annotation per sentence detected in paragraph detected by detectron."""
        paragraph_bboxes = self._detectron_get_paragraph_bbox(document)

        for i, (page, page_paragraph_bboxes) in enumerate(zip(document.pages(), paragraph_bboxes)):
            page = document.get_page_by_index(i)
            paragraph_span_bboxes = collections.defaultdict(list)
            curr_paragraph = None
            paragraph_sentence_anns: Dict[Bbox, List[Annotation]] = {}
            for bbox in sorted(page.get_bbox(), key=lambda x: x['char_index']):

                for paragraph_bbox in page_paragraph_bboxes:
                    if paragraph_bbox.check_overlap(bbox):
                        if not curr_paragraph:
                            curr_paragraph = paragraph_bbox
                        if paragraph_span_bboxes[curr_paragraph] and (
                            (paragraph_span_bboxes[curr_paragraph][-1]['line_number'] != bbox['line_number'])
                            or (paragraph_bbox is not curr_paragraph)
                            or (paragraph_span_bboxes[curr_paragraph][-1]['text'] in self.punctuation)
                        ):
                            span = Span(
                                start_offset=paragraph_span_bboxes[curr_paragraph][0]['char_index'],
                                end_offset=paragraph_span_bboxes[curr_paragraph][-1]['char_index'] + 1,
                            )
                            if curr_paragraph not in paragraph_sentence_anns:
                                paragraph_sentence_anns[curr_paragraph] = []

                            if (
                                not paragraph_sentence_anns[curr_paragraph]
                                or paragraph_sentence_anns[curr_paragraph][-1].spans[-1].offset_string[-1]
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
                                paragraph_sentence_anns[curr_paragraph].append(annotation)
                            else:
                                paragraph_sentence_anns[curr_paragraph][-1].add_span(span)

                            paragraph_span_bboxes[curr_paragraph] = []

                            curr_paragraph = paragraph_bbox
                            paragraph_span_bboxes[curr_paragraph] = [bbox]
                        else:
                            paragraph_span_bboxes[curr_paragraph].append(bbox)
                        break

            for paragraph_bbox, span_bboxes in paragraph_span_bboxes.items():
                if span_bboxes:
                    span = Span(start_offset=span_bboxes[0]['char_index'], end_offset=span_bboxes[-1]['char_index'] + 1)
                    if curr_paragraph not in paragraph_sentence_anns:
                        paragraph_sentence_anns[curr_paragraph] = []

                    if (
                        not paragraph_sentence_anns[curr_paragraph]
                        or paragraph_sentence_anns[curr_paragraph][-1].spans[-1].offset_string[-1] in self.punctuation
                    ):
                        annotation = Annotation(
                            document=document,
                            annotation_set=document.no_label_annotation_set,
                            label=document.project.no_label,
                            label_set=document.project.no_label_set,
                            category=document.category,
                            spans=[span],
                        )
                        paragraph_sentence_anns[curr_paragraph].append(annotation)
                    else:
                        paragraph_sentence_anns[curr_paragraph][-1].add_span(span)

        return document

    def _line_distance_tokenize(self, document: Document, height=None) -> Document:
        """Create one multiline Annotation per sentence in paragraph detected by line distance based rule based algo."""
        if height is not None:
            if not (isinstance(height, int) or isinstance(height, float)):
                raise TypeError(f'Parameter must be of type int or float. It is {type(height)}.')

        from statistics import median

        # assemble bboxes by their page
        pages_char_bboxes = [[] for _ in document.pages()]
        for char_index, bbox in document.get_bbox().items():
            bbox['char_index'] = int(char_index)
            pages_char_bboxes[bbox['page_number'] - 1].append(bbox)

        for page_char_bboxes in pages_char_bboxes:
            # set line_threshold
            if height is None:
                # calculate median vertical character size for Page
                line_threshold = round(
                    self.line_height_ratio * median(bbox['y1'] - bbox['y0'] for bbox in page_char_bboxes),
                    6,
                )
            else:
                line_threshold = height

            page_lines = []
            line_bboxes = []
            for bbox in sorted(page_char_bboxes, key=lambda x: x['char_index']):
                if not line_bboxes or line_bboxes[-1]['line_number'] == bbox['line_number']:
                    line_bboxes.append(bbox)
                else:
                    page_lines.append(line_bboxes)
                    line_bboxes = [bbox]
            page_lines.append(line_bboxes)

            # assemble bboxes by line and Page
            span_bboxes = []
            sentence_spans = []
            previous_y0 = None
            for line in page_lines:
                max_y1 = max([bbox['y1'] for bbox in line])
                min_y0 = min([bbox['y0'] for bbox in line])
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
                for bbox in line:
                    if span_bboxes and span_bboxes[-1]['text'] in self.punctuation:
                        sentence_spans.append(
                            Span(
                                start_offset=span_bboxes[0]['char_index'], end_offset=span_bboxes[-1]['char_index'] + 1
                            )
                        )
                        _ = Annotation(
                            document=document,
                            annotation_set=document.no_label_annotation_set,
                            label=document.project.no_label,
                            label_set=document.project.no_label_set,
                            category=document.category,
                            spans=sentence_spans,
                        )
                        sentence_spans = []
                        span_bboxes = [bbox]
                    else:
                        span_bboxes.append(bbox)
                if span_bboxes:
                    sentence_spans.append(
                        Span(start_offset=span_bboxes[0]['char_index'], end_offset=span_bboxes[-1]['char_index'] + 1)
                    )
                    span_bboxes = []
                previous_y0 = min_y0

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
