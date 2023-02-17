"""Implements a Categorization Model."""

import abc
import inspect
import logging

from copy import deepcopy
from inspect import signature
from typing import List
from warnings import warn

from konfuzio_sdk.data import Document, Category, Page, CategoryAnnotation
from konfuzio_sdk.evaluate import CategorizationEvaluation

logger = logging.getLogger(__name__)

warn('This module is WIP: https://gitlab.com/konfuzio/objectives/-/issues/9481', FutureWarning, stacklevel=2)


class AbstractCategorizationAI(metaclass=abc.ABCMeta):
    """Abstract definition of a CategorizationAI."""

    def __init__(self, categories: List[Category], *args, **kwargs):
        """Initialize AbstractCategorizationAI."""
        self.documents = None
        self.test_documents = None
        self.categories = categories
        self.name = self.__class__.__name__
        self.evaluation = None

    @abc.abstractmethod
    def fit(self) -> None:
        """Train the Categorization AI."""

    @abc.abstractmethod
    def save(self, output_dir: str, include_konfuzio=True):
        """Save the model to disk."""

    @abc.abstractmethod
    def _categorize_page(self, page: Page) -> Page:
        """Run categorization on a Page.

        :param page: Input Page
        :returns: The input Page with added CategoryAnnotation information
        """

    def categorize(self, document: Document, recategorize: bool = False, inplace: bool = False) -> Document:
        """Run categorization on a Document.

        :param document: Input Document
        :param recategorize: If the input Document is already categorized, the already present Category is used unless
        this flag is True

        :param inplace: Option to categorize the provided Document in place, which would assign the Category attribute
        :returns: Copy of the input Document with added CategoryAnnotation information
        """
        if inplace:
            virtual_doc = document
        else:
            virtual_doc = deepcopy(document)
        if (document.category not in [None, document.project.no_category]) and (not recategorize):
            logger.info(
                f'In {document}, the Category was already specified as {document.category}, so it wasn\'t categorized '
                f'again. Please use recategorize=True to force running the Categorization AI again on this Document.'
            )
            return virtual_doc

        # Categorize each Page of the Document.
        for page in virtual_doc.pages():
            self._categorize_page(page)

        return virtual_doc

    def evaluate(self, use_training_docs: bool = False) -> CategorizationEvaluation:
        """
        Evaluate the full Categorization pipeline on the pipeline's Test Documents.

        :param use_training_docs: Bool for whether to evaluate on the Training Documents instead of Test Documents.
        :return: Evaluation object.
        """
        eval_list = []
        if not use_training_docs:
            eval_docs = self.test_documents
        else:
            eval_docs = self.documents

        for document in eval_docs:
            predicted_doc = self.categorize(document=document, recategorize=True)
            eval_list.append((document, predicted_doc))

        self.evaluation = CategorizationEvaluation(self.categories, eval_list)

        return self.evaluation

    @staticmethod
    @abc.abstractmethod
    def has_compatible_interface(external):
        """
        Validate that an instance of an external model is similar to that of the class.

        :param external: An instance of an external model to compare with.
        """


class FallbackCategorizationModel(AbstractCategorizationAI):
    """A simple, non-trainable model that predicts a Category for a given Document based on a predefined rule.

    It checks for whether the name of the Category is present in the input Document (case insensitive; also see
    Category.fallback_name). This can be an effective fallback logic to categorize Documents when no Categorization AI
    is available.
    """

    def fit(self) -> None:
        """Use as placeholder Function."""
        raise NotImplementedError(
            f'{self} uses a fallback logic for categorizing Documents, and does not train a classifier.'
        )

    def save(self, output_dir: str, include_konfuzio=True):
        """Use as placeholder Function."""
        raise NotImplementedError(
            f'{self} uses a fallback logic for categorizing Documents, this will not save model to disk.'
        )

    def _categorize_page(self, page: Page) -> Page:
        """Run categorization on a Page.

        :param page: Input Page
        :returns: The input Page with added Category information
        """
        for training_category in self.categories:
            if training_category.fallback_name in page.text.lower():
                _ = CategoryAnnotation(category=training_category, confidence=1.0, page=page)
                break
        if page.category is None:
            logger.info(
                f'{self} could not find the Category of {page} by using the fallback categorization logic.'
                f'We will now apply the same Category of the first Page to this Page (if any).'
            )
            first_page = page.document.pages()[0]
            _ = CategoryAnnotation(category=first_page.category, confidence=1.0, page=page)
        return page

    @staticmethod
    def has_compatible_interface(external):
        """
        Validate that an instance of an external model is similar to that of the class.

        :param external: An instance of an external model to compare with.
        """
        try:
            if (
                signature(external.__init__).parameters['categories'].annotation is List[Category]
                and signature(external.save).parameters['output_dir'].annotation is str
                and signature(external.save).parameters['include_konfuzio'].annotation
                and signature(external.fit).return_annotation is inspect._empty
                and signature(external._categorize_page).parameters['page'].annotation is Page
                and signature(external._categorize_page).return_annotation is Page
                and signature(external.has_compatible_interface).parameters['external']
                and signature(external.has_compatible_interface).return_annotation is bool
            ):
                return True
            else:
                return False
        except KeyError:
            return False
        except AttributeError:
            return False
