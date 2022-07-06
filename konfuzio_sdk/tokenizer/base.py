"""Generic tokenizer."""

import abc
import logging
from typing import List
from warnings import warn

import pandas as pd

from konfuzio_sdk.data import Document, Category, Project, AnnotationSet, Span, Annotation
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
    """Abstract definition of a Tokenizer."""

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
        document.annotations()  # Load Annotations before doing tokenization

        virtual_doc = Document(
            project=document.project,
            text=document.text,
            bbox=document.get_bbox(),
            category=document.category,
            copy_of_id=document.id_,
            pages=document.pages,
        )

        self.tokenize(virtual_doc)
        evaluation = compare(document, virtual_doc)
        logger.warning(
            f'{evaluation["is_found_by_tokenizer"].sum()} of {evaluation["is_correct"].sum()} corrects'
            f' Spans are found by Tokenizer'
        )
        return evaluation

    def missing_spans(self, document: Document) -> Document:
        """
        Apply a Tokenizer on a list of Document and remove all Spans that can be found.

        Use this approach to sequentially work on remaining Spans after a Tokenizer ran on a List of Documents.

        :param tokenizer: A Tokenizer that runs on a list of Documents
        :param documents: Any list of Documents

        :return: A new Document containing all missing Spans contained in a copied version of all Documents.

        """
        warn(
            'This method is WIP, as we return a new instance, however the document should be kept and missing Span '
            'instances of this document should be returned.',
            FutureWarning,
            stacklevel=2,
        )
        virtual_project = Project(None)
        virtual_category = Category(project=virtual_project)
        virtual_label_set = virtual_project.no_label_set
        virtual_label = virtual_project.no_label

        compared = self.evaluate(document=document)  # todo summarize evaluation, as we are calculating it
        # return all Spans that were not found
        missing_spans = compared[(compared['is_correct']) & (compared['is_found_by_tokenizer'] == 0)]
        remaining_span_doc = Document(
            bbox=document.get_bbox(),
            pages=document.pages,
            text=document.text,
            project=virtual_project,
            category=virtual_category,
            dataset_status=document.dataset_status,
            copy_of_id=document.id_,
        )
        annotation_set_1 = AnnotationSet(id_=None, document=remaining_span_doc, label_set=virtual_label_set)
        # add Spans to the virtual Document in case the Tokenizer was not able to find them
        for index, span_info in missing_spans.iterrows():
            # todo: Schema for bbox format https://gitlab.com/konfuzio/objectives/-/issues/8661
            new_span = Span(start_offset=span_info['start_offset'], end_offset=span_info['end_offset'])
            # todo add Tokenizer used to create Span
            _ = Annotation(
                id_=int(span_info['id_']),
                document=remaining_span_doc,
                is_correct=True,
                annotation_set=annotation_set_1,
                label=virtual_label,
                label_set=virtual_label_set,
                spans=[new_span],
            )
        logger.warning(
            f'{len(remaining_span_doc.spans)} of {len(document.spans)} '
            f'correct Spans in {document} the abstract Tokenizer did not find.'
        )
        return remaining_span_doc

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
            # todo: running multiple tokenizers on one document
            #  should support that multiple Tokenizers can create identical Spans
            tokenizer.tokenize(document)
            if tokenizer.processing_steps:
                self.processing_steps.append(tokenizer.processing_steps[-1])

        return document

    def lose_weight(self):
        """Delete processing steps."""
        self.processing_steps = []
        for tokenizer in self.tokenizers:
            tokenizer.lose_weight()
