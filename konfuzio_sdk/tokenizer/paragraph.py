"""Paragraph tokenizer."""
import logging

from konfuzio_sdk.data import Annotation, Document, Category, Span
from konfuzio_sdk.tokenizer.base import AbstractTokenizer, ProcessingStep

from konfuzio_sdk.utils import sdk_isinstance, get_paragraphs_by_line_space
from konfuzio_sdk.api import get_results_from_segmentation

logger = logging.getLogger(__name__)


class ParagraphTokenizer(AbstractTokenizer):
    """Tokenizer based on a single regex."""

    def __init__(self, mode: str):
        """
        Initialize Paragraph Tokenizer.

        :param mode: line_distance or detectron
        """
        self.mode = mode

    def tokenize(self, document: Document) -> Document:
        """Create one multiline Annotation per paragraph detected."""
        assert sdk_isinstance(document, Document)

    def _detectron_tokenize(self, document: Document) -> Document:
        """Create one multiline Annotation per paragraph detected by detectron2."""
        assert isinstance(document.id_, int) and isinstance(document.project.id_, int)
        bboxes = get_results_from_segmentation(document.id_, document.project.id_)

    def _line_distance_tokenize(self, document: Document) -> Document:
        """Create one multiline Annotation per paragraph detected by line distance based rule based algorithm."""
        pass
