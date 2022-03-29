"""Generic tokenizer."""

import abc
import logging
from typing import List

import pandas as pd

from konfuzio_sdk.data import Document, Category
from konfuzio_sdk.evaluate import compare

logger = logging.getLogger(__name__)


class AbstractTokenizer(metaclass=abc.ABCMeta):
    """Abstract definition of a tokenizer."""

    @abc.abstractmethod
    def fit(self, category: Category):
        """Fit the tokenizer accordingly with the Documents of the Category."""

    @abc.abstractmethod
    def tokenize(self, document: Document):
        """Create Annotations with 1 Span based on the result of the Tokenizer."""

    def evaluate(self, document: Document) -> pd.DataFrame:
        """
        Compare a Document with its tokenized version.

        :param document: Document to evaluate
        :return: Evaluation DataFrame.
        """
        assert isinstance(document, Document)

        virtual_doc = Document(
            project=document.category.project, text=document.text, bbox=document.get_bbox(), category=document.category
        )

        self.tokenize(virtual_doc)
        return compare(document, virtual_doc)


class ListTokenizer(AbstractTokenizer):
    """Use multiple tokenizers."""

    def __init__(self, tokenizers: List['AbstractTokenizer']):
        """Initialize the list of tokenizers."""
        self.tokenizers = tokenizers

    def fit(self):
        """Call fit on all tokenizers."""
        for tokenizer in self.tokenizers:
            tokenizer.fit()

    def tokenize(self, document: Document) -> Document:
        """Run tokenize in the given order on a Document."""
        for tokenizer in self.tokenizers:
            tokenizer.tokenize()

        return document
