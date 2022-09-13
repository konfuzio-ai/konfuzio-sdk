"""Implements a DocumentModel."""

import os

# import sys
import logging
from copy import deepcopy
from typing import Union
from warnings import warn

# import pathlib

# import cloudpickle

from konfuzio_sdk.utils import get_timestamp
from konfuzio_sdk.data import Project, Document
from konfuzio_sdk.evaluate import CategoryEvaluation

logger = logging.getLogger(__name__)

warn('This module is WIP: https://gitlab.com/konfuzio/objectives/-/issues/9481', FutureWarning, stacklevel=2)

# Common category names translated in common languages and normalized (no accents).
# Language order: English, German, Italian, Spanish.
SUPPORTED_CATEGORY_NAMES = [
    ['invoice', 'rechnung', 'fattura', 'factura'],
    ['certificate', 'zertifikat', 'certificato', 'certificado'],
    ['receipt', 'quittung', 'scontrino', 'recibo'],
    ['energy certificate', 'energieausweis', 'certificato energetico', 'certificado energetico'],
    ['delivery note', 'lieferschein', 'bolla accompagnamento', 'nota entrega'],
    ['identity card', 'personalausweis', 'carta identita', 'tarjeta identificacion'],
    ['payslip', 'lohnabrechnung', 'busta paga', 'nomina'],
    ['passport', 'reisepass', 'passaporto', 'pasaporte'],
]


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
        logger.warning(f'{self} uses a fallback logic for categorizing documents, no need to save model to disk.')
        pkl_file_path = os.path.join(output_dir, f'{get_timestamp()}_categorization_prj{self.project.id_}.pkl')
        open(pkl_file_path, 'w')
        return pkl_file_path

    def evaluate(self) -> CategoryEvaluation:
        """Evaluate the categorization pipeline on the pipeline's Test Documents."""
        # todo implementation
        logger.error(f'{self} not implemented.')
        self.evaluation = CategoryEvaluation(documents=())
        return self.evaluation

    def categorize(self, document: Document) -> Document:
        """Run categorization."""
        virtual_doc = deepcopy(document)
        category_classes = [cat for cat in SUPPORTED_CATEGORY_NAMES if any([c.name in cat for c in self.categories])]
        found_cat_class = None
        for cat in category_classes:
            if found_cat_class is not None:
                break
            for search_cat_name in cat:
                if search_cat_name in virtual_doc.text:
                    found_cat_class = cat
                    break
        if found_cat_class is None:
            logger.warning(
                f'{self} could not find the category of {document} by using the fallback logic '
                f'with pre-defined common categories.'
            )
            return virtual_doc
        found_cat = [c for c in self.categories if c.name in found_cat_class][0]
        virtual_doc.category = found_cat
        return virtual_doc
