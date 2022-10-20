"""Implements a Categorization Model."""

import logging
from copy import deepcopy
from typing import Union
from warnings import warn

from konfuzio_sdk.data import Project, Document

logger = logging.getLogger(__name__)

warn('This module is WIP: https://gitlab.com/konfuzio/objectives/-/issues/9481', FutureWarning, stacklevel=2)


class FallbackCategorizationModel:
    """A non-trainable model that predicts a category for a given document based on predefined rules.

    This can be an effective fallback logic to categorize documents when no categorization AI is available.
    """

    def __init__(self, project: Union[int, Project], *args, **kwargs):
        """Initialize FallbackCategorizationModel."""
        # Go through keyword arguments, and either save their values to our
        # instance, or raise an error.
        if isinstance(project, int):
            self.project = Project(id_=project)
        elif isinstance(project, Project):
            self.project = project
        else:
            raise NotImplementedError

        self.categories = None
        self.name = self.__class__.__name__

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

    def evaluate(self):
        """Use as placeholder Function."""
        raise NotImplementedError(
            f'{self} uses a fallback logic for categorizing documents, without using Training or Test documents for '
            f'evaluation.'
        )

    def categorize(self, document: Document, recategorize: bool = False, inplace: bool = False) -> Document:
        """Run categorization.

        :param document: Input document
        :param recategorize: If the input document is already categorized, the already present category is used unless
        this flag is True

        :param inplace: Option to categorize the provided document in place, which would assign the category attribute
        :returns: Copy of the input document with added categorization information
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
            virtual_doc.category = None

        relevant_categories = [training_category.fallback_name for training_category in self.categories]
        found_category_name = None
        doc_text = virtual_doc.text.lower()
        for candidate_category_name in relevant_categories:
            if candidate_category_name in doc_text:
                found_category_name = candidate_category_name
                break

        if found_category_name is None:
            logger.warning(
                f'{self} could not find the category of {document} by using the fallback logic '
                f'with pre-defined common categories.'
            )
            return virtual_doc
        found_category = [category for category in self.categories if category.fallback_name in found_category_name][0]
        virtual_doc.category = found_category
        return virtual_doc
