"""Implements a Categorization Model."""

import logging
from copy import deepcopy
from typing import List
from warnings import warn

from konfuzio_sdk.data import Document, Category, Page
from konfuzio_sdk.evaluate import CategorizationEvaluation

logger = logging.getLogger(__name__)

warn('This module is WIP: https://gitlab.com/konfuzio/objectives/-/issues/9481', FutureWarning, stacklevel=2)


class FallbackCategorizationModel:
    """A simple, non-trainable model that predicts a Category for a given Document based on a predefined rule.

    It checks for whether the name of the Category is present in the input Document (case insensitive; also see
    Category.fallback_name). This can be an effective fallback logic to categorize Documents when no Categorization AI
    is available.
    """

    def __init__(self, categories: List[Category], *args, **kwargs):
        """Initialize FallbackCategorizationModel."""
        # Go through keyword arguments, and either save their values to our
        # instance, or raise an error.
        self.categories = categories
        self.name = self.__class__.__name__

        self.evaluation = None

    def fit(self) -> None:
        """Use as placeholder Function."""
        raise NotImplementedError(
            f'{self} uses a fallback logic for categorizing documents, and does not train a classifier.'
        )

    def save(self, output_dir: str, include_konfuzio=True):
        """Use as placeholder Function."""
        raise NotImplementedError(
            f'{self} uses a fallback logic for categorizing documents, this will not save model to disk.'
        )

    def evaluate(self, use_training_docs: bool = False) -> CategorizationEvaluation:
        """
        Evaluate the full Categorization pipeline on the pipeline's Test Documents.

        :param use_training_docs: Bool for whether to evaluate on the training Documents instead of test Documents.
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

    def _categorize_page(self, page: Page) -> Page:
        """Run categorization on a Page.

        :param page: Input Page
        :returns: The input Page with added Category information
        """
        for training_category in self.categories:
            if training_category.fallback_name in page.text.lower():
                page.category = training_category
                break
        if page.category is None:
            logger.warning(f'{self} could not find the category of {page} by using the fallback categorization logic.')
        return page

    def categorize(self, document: Document, recategorize: bool = False, inplace: bool = False) -> Document:
        """Run categorization on a Document.

        :param document: Input document
        :param recategorize: If the input Document is already categorized, the already present Category is used unless
        this flag is True

        :param inplace: Option to categorize the provided Document in place, which would assign the Category attribute
        :returns: Copy of the input Document with added categorization information
        """
        if inplace:
            virtual_doc = document
        else:
            virtual_doc = deepcopy(document)
        if (document.category is not None) and (not recategorize):
            logger.info(
                f'In {document}, the category was already specified as {document.category}, so it wasn\'t categorized '
                f'again. Please use recategorize=True to force running the Categorization AI again on this document.'
            )
            return virtual_doc
        elif recategorize:
            virtual_doc._category = None
            for page in virtual_doc.pages():
                page.category = None

        # Categorize each Page of the Document.
        for page in virtual_doc.pages():
            self._categorize_page(page)

        return virtual_doc
