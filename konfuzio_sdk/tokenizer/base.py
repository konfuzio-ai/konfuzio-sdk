"""Generic tokenizer."""

import abc
import logging
from typing import List

import pandas as pd

from konfuzio_sdk.data import Document, Category, Project
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

    def evaluate_category(self, category: Category) -> pd.DataFrame:
        """Compare test Documents of a Category with their tokenized version.

        :param category: Category to evaluate
        :return: Evaluation DataFrame containing the evaluation of all Documents in the Category.
        """
        assert isinstance(category, Category)

        if not category.test_documents():
            raise ValueError(f"Category {category.__repr__()} has no test documents.")

        evaluation = pd.DataFrame()
        for document in category.test_documents():
            evaluation = evaluation.append(self.evaluate(document))

        return evaluation

    def evaluate_project(self, project: Project) -> pd.DataFrame:
        """Compare test Documents of the Categories in a Project with their tokenized version.

        :param project: Project to evaluate
        :return: Evaluation DataFrame containing the evaluation of all Documents in all Categories.
        """
        assert isinstance(project, Project)

        if not project.categories:
            raise ValueError(f"Project {project.__repr__()} has no Categories.")

        if not project.test_documents:
            raise ValueError(f"Project {project.__repr__()} has no test Documents.")

        evaluation = pd.DataFrame()
        for category in project.categories:
            try:
                evaluation = evaluation.append(self.evaluate_category(category))
            except ValueError:
                # Category may not have test Documents
                continue

        return evaluation


class ListTokenizer(AbstractTokenizer):
    """Use multiple tokenizers."""

    def __init__(self, tokenizers: List['AbstractTokenizer']):
        """Initialize the list of tokenizers."""
        self.tokenizers = tokenizers

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

        return document
