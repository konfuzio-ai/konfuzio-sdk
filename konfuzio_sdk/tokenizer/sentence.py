"""Sentence tokenizer."""
import logging
import collections
import time

from konfuzio_sdk.data import Annotation, Document, Span
from konfuzio_sdk.tokenizer.base import AbstractTokenizer, ProcessingStep

from konfuzio_sdk.utils import sdk_isinstance
from konfuzio_sdk.api import get_results_from_segmentation

logger = logging.getLogger(__name__)


class SentenceTokenizer(AbstractTokenizer):
    """Tokenizer splitting Document into Sentences."""

    def __init__(self, mode: str = 'detectron'):
        """
        Initialize Sentence Tokenizer.

        :param mode: line_distance or detectron
        """
        self.mode = mode
