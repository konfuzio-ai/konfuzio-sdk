"""Implements a DocumentModel."""

# import os
import re

# import sys
import logging
from copy import deepcopy
from typing import Union, List
from warnings import warn

# import pathlib

# import cloudpickle

from konfuzio_sdk.data import Project, Document, Category

logger = logging.getLogger(__name__)

warn('This module is WIP: https://gitlab.com/konfuzio/objectives/-/issues/9481', FutureWarning, stacklevel=2)


def get_category_name_for_fallback_prediction(category: Union[Category, str]) -> str:
    """Turn a category name to lowercase, remove parentheses along with their contents, and trim spaces."""
    if isinstance(category, Category):
        category_name = category.name.lower()
    elif isinstance(category, str):
        category_name = category.lower()
    else:
        raise NotImplementedError
    parentheses_removed = re.sub(r'\([^)]*\)', '', category_name).strip()
    single_spaces = parentheses_removed.replace("  ", " ")
    return single_spaces


def build_list_of_relevant_categories(training_categories: List[Category]) -> List[List[str]]:
    """Filter for category name variations which correspond to the given categories, starting from a predefined list."""
    relevant_categories = []
    for training_category in training_categories:
        category_name = get_category_name_for_fallback_prediction(training_category)
        relevant_categories.append(category_name)
    return relevant_categories


class FallbackCategorizationModel:
    """A model that predicts a category for a given document."""

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
        """Run categorization."""
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

        relevant_categories = build_list_of_relevant_categories(self.categories)
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
        found_category = [
            category
            for category in self.categories
            if get_category_name_for_fallback_prediction(category) in found_category_name
        ][0]
        virtual_doc.category = found_category
        return virtual_doc
