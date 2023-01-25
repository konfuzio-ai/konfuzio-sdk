"""Paragraph tokenizer."""
import logging
import collections
import time

from konfuzio_sdk.data import Annotation, Document, Span
from konfuzio_sdk.tokenizer.base import AbstractTokenizer, ProcessingStep

from konfuzio_sdk.utils import sdk_isinstance, get_paragraphs_by_line_space
from konfuzio_sdk.api import get_results_from_segmentation

logger = logging.getLogger(__name__)


class ParagraphTokenizer(AbstractTokenizer):
    """Tokenizer based on a single regex."""

    def __init__(self, mode: str = 'detectron'):
        """
        Initialize Paragraph Tokenizer.

        :param mode: line_distance or detectron
        """
        self.mode = mode

    def __repr__(self):
        """Return string representation of the class."""
        return f"{self.__class__.__name__}: {self.mode=}"

    def __hash__(self):
        """Get unique hash for RegexTokenizer."""
        return hash(repr(self))

    def __eq__(self, other) -> bool:
        """Compare Tokenizer with another Tokenizer."""
        return hash(self) == hash(other)

    def tokenize(self, document: Document) -> Document:
        """Create one multiline Annotation per paragraph detected."""
        assert sdk_isinstance(document, Document)
        assert isinstance(document.project.id_, int)  # isinstance(document.id_, int) and

        before_none = len(document.annotations(use_correct=False, label=document.project.no_label))

        t0 = time.monotonic()

        if not document.bboxes_available:
            raise ValueError(
                f"Cannot tokenize Document {document} with tokenizer {self}: Missing Character Bbox information."
            )

        if self.mode == 'detectron':
            return self._detectron_tokenize(document=document)
        elif self.mode == 'line_distance':
            return self._line_distance_tokenize(document=document)

        after_none = len(document.annotations(use_correct=False, label=document.project.no_label))
        logger.info(f'{after_none - before_none} new Annotations in {document} by {repr(self)}.')

        self.processing_steps.append(ProcessingStep(self, document, time.monotonic() - t0))

    def _detectron_tokenize(self, document: Document) -> Document:
        """Create one multiline Annotation per paragraph detected by detectron2."""
        doc_id = document.id_ if document.id_ else document.copy_of_id
        paragraph_bboxes = get_results_from_segmentation(doc_id, document.project.id_)

        assert len(paragraph_bboxes) == document.number_of_pages

        # normalize
        for i, page in enumerate(document.pages()):
            scale_mult = page.height / page.full_height
            for bbox in paragraph_bboxes[i]:
                bbox['x0'] *= scale_mult
                bbox['x1'] *= scale_mult
                bbox['y0'] *= scale_mult
                bbox['y1'] *= scale_mult

        # for page_bboxes in paragraph_bboxes:
        pages_char_bboxes = [[] for _ in document.pages()]
        for char_index, bbox in document.get_bbox().items():
            pages_char_bboxes[bbox['page_number'] - 1].append((int(char_index), bbox))

        assert len(pages_char_bboxes) == len(paragraph_bboxes)

        paragraph_char_bboxes = collections.defaultdict(list)
        for i, (page_char_booxes, page_paragraph_bboxes) in enumerate(zip(pages_char_bboxes, paragraph_bboxes)):
            page = document.get_page_by_index(i)
            # page_paragraph_bboxes = [Bbox(**bbox, page=page) for bbox in page_paragraph_bboxes]
            for bbox in sorted(page_char_booxes, key=lambda x: x[0]):
                for paragraph_bbox in page_paragraph_bboxes:
                    if (
                        bbox[1]['x0'] <= paragraph_bbox['x1']
                        and bbox[1]['x1'] >= paragraph_bbox['x0']
                        and page.height - bbox[1]['y0'] <= paragraph_bbox['y1']
                        and page.height - bbox[1]['y1'] >= paragraph_bbox['y0']
                    ):
                        paragraph_char_bboxes[(int(paragraph_bbox['x0']), int(paragraph_bbox['y0']))].append(bbox)
                        break

        for k, bboxes in paragraph_char_bboxes.items():
            spans = []
            line_bboxes = []
            for bbox in bboxes:
                if line_bboxes == [] or bbox[1]['line_number'] == line_bboxes[0][1]['line_number']:
                    line_bboxes.append(bbox)
                    continue
                else:
                    spans.append(Span(start_offset=line_bboxes[0][0], end_offset=line_bboxes[-1][0] + 1))
                    line_bboxes = [bbox]
            spans.append(Span(start_offset=line_bboxes[0][0], end_offset=line_bboxes[-1][0] + 1))

            _ = Annotation(
                document=document,
                annotation_set=document.no_label_annotation_set,
                label=document.project.no_label,
                label_set=document.project.no_label_set,
                category=document.category,
                spans=spans,
            )

        return document

    def found_spans(self, document: Document):
        pass

    # def __eq__, __hash__, fit, found_spans

    def _line_distance_tokenize(self, document: Document) -> Document:
        """Create one multiline Annotation per paragraph detected by line distance based rule based algorithm."""
        paragraph_bboxes = get_paragraphs_by_line_space(document.bboxes, document.text)
