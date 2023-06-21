"""Test creation of custom ParagraphExtractionAI model."""
import logging
import os

# start imports
from copy import deepcopy

from konfuzio_sdk.trainer.information_extraction import AbstractExtractionAI
from konfuzio_sdk.tokenizer.paragraph_and_sentence import ParagraphTokenizer
from konfuzio_sdk.data import Category, Document, Project, Label

# start imports

from tests.variables import TEST_PROJECT_ID, TEST_DOCUMENT_ID, TEST_PAYSLIPS_CATEGORY_ID


logger = logging.getLogger(__name__)


class ParagraphExtractionAI(AbstractExtractionAI):
    """Extract and label text regions using Detectron2."""

    # start model requirements
    requires_images = True
    requires_text = True
    requires_segmentation = True
    # end model requirements

    def __init__(
        self,
        category: Category = None,
        *args,
        **kwargs,
    ):
        """Initialize ParagraphExtractionAI."""
        logger.info("Initializing ParagraphExtractionAI.")
        super().__init__(category=category, *args, **kwargs)
        self.tokenizer = ParagraphTokenizer(mode='detectron', create_detectron_labels=True)

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
        :raises IndexError: When the Category does not contain the required Labels.
        """
        super().check_is_ready()

        self.project.get_label_by_name('figure')
        self.project.get_label_by_name('table')
        self.project.get_label_by_name('list')
        self.project.get_label_by_name('text')
        self.project.get_label_by_name('title')


def test_paragraph_extraction_ai():
    """Test custom ParagraphExtractionAI model."""
    # start use model
    labels = ['figure', 'table', 'list', 'text', 'title']

    project = Project(id_=TEST_PROJECT_ID)
    category = project.get_category_by_id(TEST_PAYSLIPS_CATEGORY_ID)

    # creating Labels in case they do not exist
    label_set = project.get_label_set_by_name(category.name)  # default Category label set

    for label_name in labels:
        try:
            project.get_label_by_name(label_name)
        except IndexError:
            Label(project=project, text=label_name, label_sets=[label_set])

    document = project.get_document_by_id(TEST_DOCUMENT_ID)

    paragraph_extraction_ai = ParagraphExtractionAI(category=category)

    assert paragraph_extraction_ai.check_is_ready() is True

    paragraph_extraction_ai.extract(document)

    model_path = paragraph_extraction_ai.save()
    # end use model

    # delete model
    os.remove(model_path)
