"""Generic tokenizer."""
import logging

import pandas as pd

from konfuzio_sdk.data import Document, Category
from konfuzio_sdk.evaluate import compare

logger = logging.getLogger(__name__)


class AbstractTokenizer:
    """Abstract definition of a tokenizer."""

    def fit(self, category: Category):
        """Fit the tokenizer accordingly with the Documents of the Category."""
        assert isinstance(category, Category)
        return self

    def tokenize(self, document: Document) -> Document:
        """
        Create Annotations with 1 Span based on the result of the Tokenizer.

        :param document: Document to tokenize
        :return: Document with Spans created by the Tokenizer.
        """
        assert isinstance(document, Document)
        return document

    def evaluate(self, document: Document) -> pd.DataFrame:
        """Compare a Document with its tokenized version."""
        assert isinstance(document, Document)

        virtual_doc = Document(
            project=document.category.project, text=document.text, bbox=document.get_bbox(), category=document.category
        )

        virtual_doc = self.tokenize(virtual_doc)
        return compare(document, virtual_doc)
