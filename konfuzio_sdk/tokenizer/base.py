"""Generic tokenizer."""

import abc
import logging
from typing import List

import pandas as pd

from konfuzio_sdk.data import Document, Category
from konfuzio_sdk.evaluate import compare
from konfuzio_sdk.utils import sdk_isinstance

logger = logging.getLogger(__name__)


class ProcessingStep:
    """Track runtime of Tokenizer functions."""

    def __init__(self, tokenizer: str, document: Document, runtime: float):
        """Initialize the processing step."""
        self.tokenizer = tokenizer
        self.document = document
        self.runtime = runtime

    def eval_dict(self):
        """Return any information needed to evaluate the ProcessingStep."""
        step_eval = {
            'tokenizer_name': str(self.tokenizer),
            'document_id': self.document.id_ or self.document.copy_of_id,
            'number_of_pages': self.document.number_of_pages,
            'runtime': self.runtime,
        }
        return step_eval


class AbstractTokenizer(metaclass=abc.ABCMeta):
    """Abstract definition of a tokenizer."""

    processing_steps = []

    def __repr__(self):
        """Return string representation of the class."""
        return f"{self.__class__.__name__}"

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
        :return: Evaluation DataFrame and Processing time DataFrame.
        """
        assert sdk_isinstance(document, Document)

        virtual_doc = Document(
            project=document.category.project,
            text=document.text,
            bbox=document.get_bbox(),
            category=document.category,
            copy_of_id=document.id_,
        )

        self.tokenize(virtual_doc)
        return compare(document, virtual_doc)

    def evaluate_category(self, category: Category) -> pd.DataFrame:
        """Compare test Documents of a Category with their tokenized version.

        :param category: Category to evaluate
        :return: Evaluation DataFrame containing the evaluation of all Documents in the Category.
        """
        assert isinstance(category, Category)

        if not category.test_documents():
            raise ValueError(f"Category {category.__repr__()} has no test documents.")

        evaluation = []
        for document in category.test_documents():
            doc_evaluation = self.evaluate(document)
            evaluation.append(doc_evaluation)

        return pd.concat(evaluation, ignore_index=True)

    def get_runtime_info(self) -> pd.DataFrame:
        """
        Get the processing runtime information as DataFrame.

        :return: processing time Dataframe containing the processing duration of all steps of the tokenization.
        """
        data = [x.eval_dict() for x in self.processing_steps]
        return pd.DataFrame(data)

    def lose_weight(self):
        """Delete processing steps."""
        self.processing_steps = []


class ListTokenizer(AbstractTokenizer):
    """Use multiple tokenizers."""

    def __init__(self, tokenizers: List['AbstractTokenizer']):
        """Initialize the list of tokenizers."""
        self.tokenizers = tokenizers
        self.processing_steps = []

    def fit(self, category: Category):
        """Call fit on all tokenizers."""
        assert isinstance(category, Category)

        for tokenizer in self.tokenizers:
            tokenizer.fit(category)

    def tokenize(self, document: Document) -> Document:
        """Run tokenize in the given order on a Document."""
        assert sdk_isinstance(document, Document)

        for tokenizer in self.tokenizers:
            tokenizer.tokenize(document)
            if tokenizer.processing_steps:
                self.processing_steps.append(tokenizer.processing_steps[-1])

        return document

    def lose_weight(self):
        """Delete processing steps."""
        self.processing_steps = []
        for tokenizer in self.tokenizers:
            tokenizer.lose_weight()
