"""Implements a DocumentModel."""

import os
import re

# import sys
import logging
from copy import deepcopy
from typing import Union, List
from warnings import warn

# import pathlib

# import cloudpickle

from konfuzio_sdk.utils import get_timestamp
from konfuzio_sdk.data import Project, Document, Category
from konfuzio_sdk.evaluate import CategoryEvaluation

logger = logging.getLogger(__name__)

warn('This module is WIP: https://gitlab.com/konfuzio/objectives/-/issues/9481', FutureWarning, stacklevel=2)

# Common category names translated in English and German in multiple variations.
SUPPORTED_CATEGORY_NAMES = [
    ['invoice', 'rechnung'],
    ['certificate', 'zertifikat'],
    ['receipt', 'quittung', 'kundenbeleg'],
    ['energy certificate', 'energieausweis'],
    ['delivery note', 'lieferschein'],
    ['identity card', 'personalausweis'],
    ['payslip', 'lohnabrechnung', 'bezüge'],
    ['passport', 'reisepass'],
]


def get_category_name_for_fallback_prediction(category: Category) -> str:
    """Turn a category name to lowercase, remove parentheses and brackets with their contents, and trim spaces."""
    return re.sub(r'\([^)]*\)', '', category.name.lower()).strip()


def build_list_of_relevant_categories(training_categories: List[Category]) -> List[List[str]]:
    """Filter for category name variations which correspond to the given categories, starting from a predefined list."""
    relevant_categories = [
        category
        for category in SUPPORTED_CATEGORY_NAMES
        if any([get_category_name_for_fallback_prediction(c) in category for c in training_categories])
    ]
    for training_category in training_categories:
        relevant_categories.append([training_category.name.lower()])
    return relevant_categories


class BaseCategorizationModel:
    """A model that predicts a category for a given document."""

    def __init__(self, project: Union[int, Project], *args, **kwargs):
        """Initialize BaseCategorizationModel."""
        # Go through keyword arguments, and either save their values to our
        # instance, or raise an error.
        if isinstance(project, int):
            self.project = Project(id_=project)
        elif isinstance(project, Project):
            self.project = project
        else:
            raise NotImplementedError

        self.clf = None
        self.categories = None
        self.documents = None
        self.test_documents = None

        self.name = self.__class__.__name__

        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
        self.X_test = None
        self.y_test = None

        self.pipeline_path = None
        self.evaluation = None

    def fit(self) -> None:
        """Use as placeholder Function."""
        logger.warning(f'{self} uses a fallback logic for categorizing documents, and does not train a classifier.')
        pass

    def save(self, output_dir: str, include_konfuzio=True) -> str:
        """Use as placeholder Function."""
        # todo implementation
        # todo how to unify with Trainer.save() of information_extraction.py ?
        logger.warning(f'{self} uses a fallback logic for categorizing documents, this will not save model to disk.')
        pkl_file_path = os.path.join(output_dir, f'{get_timestamp()}_categorization_prj{self.project.id_}.pkl')
        return pkl_file_path

    def evaluate(self) -> CategoryEvaluation:
        """Evaluate the categorization pipeline on the pipeline's Test Documents."""
        # todo implementation
        logger.error(f'{self} not implemented.')
        self.evaluation = CategoryEvaluation(documents=())
        return self.evaluation

    def categorize(self, document: Document, recategorize: bool = True) -> Document:
        """Run categorization."""
        virtual_doc = deepcopy(document)
        if (document.category is not None) and (not recategorize):
            logger.info(
                f'In {document}, the category was already specified as {document.category}, so it wasn\'t categorized '
                f'again. Please use recategorize=True to force running the Categorization AI again on this document.'
            )
            return virtual_doc

        relevant_categories = build_list_of_relevant_categories(self.categories)
        found_compatible_category_names = None
        doc_text = virtual_doc.text.lower()
        for alternative_names_for_category in relevant_categories:
            if found_compatible_category_names is not None:
                break
            for candidate_category_name in alternative_names_for_category:
                if candidate_category_name in doc_text:
                    found_compatible_category_names = alternative_names_for_category
                    break

        if found_compatible_category_names is None:
            logger.warning(
                f'{self} could not find the category of {document} by using the fallback logic '
                f'with pre-defined common categories.'
            )
            return virtual_doc
        found_category = [
            category
            for category in self.categories
            if get_category_name_for_fallback_prediction(category) in found_compatible_category_names
        ][0]
        virtual_doc.category = found_category
        return virtual_doc
