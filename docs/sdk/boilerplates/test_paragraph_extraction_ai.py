"""Test creation of custom ParagraphExtractionAI model."""
import logging
from copy import deepcopy

from konfuzio_sdk.trainer.information_extraction import AbstractExtractionAI
from konfuzio_sdk.tokenizer.paragraph_and_sentence import ParagraphTokenizer
from konfuzio_sdk.data import Category, Document, Project

from tests.variables import TEST_PROJECT_ID, TEST_DOCUMENT_ID


logger = logging.getLogger(__name__)


class ParagraphExtractionAI(AbstractExtractionAI):
    """Extract and label text regions using Detectron2."""

    requires_images = True
    requires_text = True
    requires_segmentation = True

    def __init__(
        self,
        category: Category = None,
        *args,
        **kwargs,
    ):
        """Init ParagraphExtractionAI."""
        logger.info("Initializing ParagraphExtractionAI.")
        super().__init__(category=category, *args, **kwargs)
        self.tokenizer = ParagraphTokenizer(mode='detectron', create_detectron_labels=True)

    @property
    def project(self):
        """Get Project."""
        if not self.category:
            raise AttributeError(f'{self} has no Category.')
        return self.category.project

    def extract(self, document: Document) -> Document:
        """
        Infer information from a given Document.

        :param document: Document object
        :return: Document with predicted Labels

        :raises:
        AttributeError: When missing a Tokenizer
        """
        logger.info(f"Starting extraction of {document}.")

        self.check_is_ready()

        inference_document = deepcopy(document)

        inference_document = self.tokenizer.tokenize(inference_document)

        return inference_document

    def check_is_ready(self):
        """
        Check if the ExtractionAI is ready for the inference.

        It is assumed that the model is ready if a Tokenizer and a Category were set.

        :raises AttributeError: When no Category is specified.
        """
        logger.info(f"Checking if {self} is ready for extraction.")
        if not self.category:
            raise AttributeError(f'{self} requires a Category.')


def test_paragraph_extraction_ai():
    """Test custom ParagraphExtractionAI model."""
    project = Project(id_=TEST_PROJECT_ID)

    document = project.get_document_by_id(TEST_DOCUMENT_ID)

    paragraph_extraction_ai = ParagraphExtractionAI()

    paragraph_extraction_ai.check_is_ready()

    paragraph_extraction_ai.extract(document)
