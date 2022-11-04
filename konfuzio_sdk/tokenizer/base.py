"""Generic tokenizer."""

import abc
import logging
from typing import List
from copy import deepcopy

import pandas as pd

from konfuzio_sdk.data import Document, Category, Span
from konfuzio_sdk.evaluate import compare, Evaluation
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
    """Abstract definition of a Tokenizer."""

    processing_steps = []

    def __repr__(self):
        """Return string representation of the class."""
        return f"{self.__class__.__name__}"

    @abc.abstractmethod
    def __eq__(self, other) -> bool:
        """Check if two Tokenizers are the same."""

    @abc.abstractmethod
    def __hash__(self):
        """Get unique hash for Tokenizer."""

    @abc.abstractmethod
    def fit(self, category: Category):
        """Fit the tokenizer accordingly with the Documents of the Category."""

    @abc.abstractmethod
    def tokenize(self, document: Document):
        """Create Annotations with 1 Span based on the result of the Tokenizer."""

    @abc.abstractmethod
    def found_spans(self, document: Document) -> List[Span]:
        """Find all Spans in a Document that can be found by a Tokenizer."""

    def evaluate(self, document: Document) -> pd.DataFrame:
        """
        Compare a Document with its tokenized version.

        :param document: Document to evaluate
        :return: Evaluation DataFrame
        """
        assert sdk_isinstance(document, Document)
        document.annotations()  # Load Annotations before doing tokenization

        virtual_doc = deepcopy(document)
        self.tokenize(virtual_doc)
        evaluation = compare(document, virtual_doc)
        logger.warning(
            f'{evaluation["tokenizer_true_positive"].sum()} of {evaluation["is_correct"].sum()} corrects'
            f' Spans are found by Tokenizer'
        )
        return evaluation

    def evaluate_dataset(self, dataset_documents: List[Document]) -> Evaluation:
        """
        Evaluate the tokenizer on a dataset of documents.

        :param dataset_documents: Documents to evaluate
        :return: Evaluation instance
        """
        eval_list = []
        for document in dataset_documents:
            assert sdk_isinstance(document, Document), f"Invalid document type: {type(document)}. Should be Document."
            document.annotations()  # Load Annotations before doing tokenization
            virtual_doc = deepcopy(document)
            self.tokenize(virtual_doc)
            eval_list.append((document, virtual_doc))
        return Evaluation(eval_list)

    def missing_spans(self, document: Document) -> List[Span]:
        """
        Apply a Tokenizer on a Document and find all Spans that cannot be found.

        Use this approach to sequentially work on remaining Spans after a Tokenizer ran on a List of Documents.

        :param document: A Document

        :return: A list containing all missing Spans.

        """
        self.found_spans(document)
        missing_spans_list = [span for span in document.spans(use_correct=True) if span.regex_matching == []]

        return missing_spans_list

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
        self.tokenizers = list(dict.fromkeys(tokenizers))
        self.processing_steps = []

    def __eq__(self, other) -> bool:
        """Compare ListTokenizer with another Tokenizer."""
        if type(other) is ListTokenizer:
            return self.tokenizers == other.tokenizers
        else:
            return False

    def __hash__(self):
        """Get unique hash for ListTokenizer."""
        return hash(tuple(self.tokenizers))

    def fit(self, category: Category):
        """Call fit on all tokenizers."""
        assert sdk_isinstance(category, Category)

        for tokenizer in self.tokenizers:
            tokenizer.fit(category)

    def tokenize(self, document: Document) -> Document:
        """Run tokenize in the given order on a Document."""
        assert sdk_isinstance(document, Document)

        for tokenizer in self.tokenizers:
            # todo: running multiple tokenizers on one document
            #  should support that multiple Tokenizers can create identical Spans
            tokenizer.tokenize(document)
            if tokenizer.processing_steps:
                self.processing_steps.append(tokenizer.processing_steps[-1])

        return document

    def found_spans(self, document: Document) -> List[Span]:
        """Run found_spans in the given order on a Document."""
        found_spans_list = []
        for tokenizer in self.tokenizers:
            found_spans_list += tokenizer.found_spans(document)
        return found_spans_list

    def span_match(self, span: 'Span') -> bool:
        """Run span_match in the given order."""
        for tokenizer in self.tokenizers:
            if tokenizer.span_match(span):
                return True
        return False

    def lose_weight(self):
        """Delete processing steps."""
        self.processing_steps = []
        for tokenizer in self.tokenizers:
            tokenizer.lose_weight()
