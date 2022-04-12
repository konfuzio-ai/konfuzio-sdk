"""Generic tokenizer."""

import abc
import logging
from typing import List, Tuple

import pandas as pd

from konfuzio_sdk.data import Document, Category, Project
from konfuzio_sdk.evaluate import compare

logger = logging.getLogger(__name__)


class ProcessingStep:
    """Track runtime of Tokenizer functions."""

    def __init__(self, tokenizer_name: str, document: Document, runtime: float):
        """Initialize the processing step."""
        self.tokenizer_name = tokenizer_name
        if document.id_ is None:
            document_id = document.copy_of_id
        else:
            document_id = document.id_
        self.document_id = document_id
        self.number_of_pages = document.number_of_pages
        self.runtime = runtime


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

    def evaluate(self, document: Document) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compare a Document with its tokenized version.

        :param document: Document to evaluate
        :return: Evaluation DataFrame and Processing time DataFrame.
        """
        assert isinstance(document, Document)

        virtual_doc = Document(
            project=document.category.project,
            text=document.text,
            bbox=document.get_bbox(),
            category=document.category,
            copy_of_id=document.id_,
        )

        self.tokenize(virtual_doc)
        data = {
            'tokenizer_name': [x.tokenizer_name for x in self.processing_steps],
            'document_id': [x.document_id for x in self.processing_steps],
            'number_of_pages': [x.number_of_pages for x in self.processing_steps],
            'runtime': [x.runtime for x in self.processing_steps],
        }
        return compare(document, virtual_doc), pd.DataFrame(data)

    def evaluate_category(self, category: Category) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compare test Documents of a Category with their tokenized version.

        :param category: Category to evaluate
        :return: Evaluation DataFrame containing the evaluation of all Documents in the Category and processing time
        Dataframe containing the processing duration of all steps of the tokenization.
        """
        assert isinstance(category, Category)

        if not category.test_documents():
            raise ValueError(f"Category {category.__repr__()} has no test documents.")

        evaluation = pd.DataFrame()
        for document in category.test_documents():
            doc_evaluation, _ = self.evaluate(document)
            evaluation = evaluation.append(doc_evaluation)

        data = {
            'tokenizer_name': [x.tokenizer_name for x in self.processing_steps],
            'document_id': [x.document_id for x in self.processing_steps],
            'number_of_pages': [x.number_of_pages for x in self.processing_steps],
            'runtime': [x.runtime for x in self.processing_steps],
        }
        return evaluation, pd.DataFrame(data)

    def evaluate_project(self, project: Project) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compare test Documents of the Categories in a Project with their tokenized version.

        :param project: Project to evaluate
        :return: Evaluation DataFrame containing the evaluation of all Documents in all Categories and processing time
        Dataframe containing the processing duration of all steps of the tokenization.
        """
        assert isinstance(project, Project)

        if not project.categories:
            raise ValueError(f"Project {project.__repr__()} has no Categories.")

        if not project.test_documents:
            raise ValueError(f"Project {project.__repr__()} has no test Documents.")

        evaluation = pd.DataFrame()
        for category in project.categories:
            try:
                docs_evaluation, _ = self.evaluate_category(category)
                evaluation = evaluation.append(docs_evaluation)
            except ValueError as e:
                # Category may not have test Documents
                logger.info(f'Evaluation of the Tokenizer for {category} not possible, because of {e}.')
                continue

        data = {
            'tokenizer_name': [x.tokenizer_name for x in self.processing_steps],
            'document_id': [x.document_id for x in self.processing_steps],
            'number_of_pages': [x.number_of_pages for x in self.processing_steps],
            'runtime': [x.runtime for x in self.processing_steps],
        }
        return evaluation, pd.DataFrame(data)

    def reset_processing_steps(self):
        """Reset tracking runtime of Tokenizer functions."""
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
        assert isinstance(document, Document)

        for tokenizer in self.tokenizers:
            tokenizer.tokenize(document)
            self.processing_steps.append(tokenizer.processing_steps[-1])

        return document

    def reset_processing_steps(self) -> None:
        """Reset tracking runtime of Tokenizer functions."""
        for tokenizer in self.tokenizers:
            tokenizer.reset_processing_steps()
        self.processing_steps = []
