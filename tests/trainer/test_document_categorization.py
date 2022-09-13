# -*- coding: utf-8 -*-
"""Test to train a Categorization AI."""
import os
import logging
import unittest

# from requests import HTTPError

from konfuzio_sdk.data import Project, Document

# from konfuzio_sdk.api import upload_ai_model
from tests.variables import OFFLINE_PROJECT, TEST_DOCUMENT_ID
from konfuzio_sdk.trainer.document_categorization import BaseCategorizationModel

logger = logging.getLogger(__name__)


class TestBaseCategorizationModel(unittest.TestCase):
    """Test New SDK fallback logic for Categorization."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the Data and Categorization Pipeline."""
        cls.project = Project(id_=None, project_folder=OFFLINE_PROJECT)
        cls.categorization_pipeline = BaseCategorizationModel(cls.project)

    def test_1_configure_pipeline(self) -> None:
        """Make sure the Data and Pipeline is configured."""
        self.categorization_pipeline.categories = self.project.categories
        self.categorization_pipeline.documents = self.categorization_pipeline.project.documents
        self.categorization_pipeline.test_documents = self.categorization_pipeline.project.test_documents

    def test_2_fit(self) -> None:
        """Start to train the Model."""
        # since we are using the fallback logic, this should do nothing and print a logger warning
        self.categorization_pipeline.fit()

    def test_3_save_model(self):
        """Save the model."""
        # since we are using the fallback logic, this should do nothing and print a logger warning
        self.categorization_pipeline.pipeline_path = self.categorization_pipeline.save(
            output_dir=self.project.model_folder
        )
        assert os.path.isfile(self.categorization_pipeline.pipeline_path)
        # os.remove(self.pipeline.pipeline_path)  # cleanup

    @unittest.skip(reason="To be defined how to upload a categorization model.")
    def test_4_upload_ai_model(self):
        """Upload the model."""
        assert os.path.isfile(self.categorization_pipeline.pipeline_path)

        # try:
        #    upload_ai_model(ai_model_path=self.categorization_pipeline.pipeline_path,
        #    category_ids=[self.categorization_pipeline.category.id_])
        # except HTTPError as e:
        #    assert '403' in str(e)

    def test_5_evaluate(self):
        """Evaluate BaseCategorizationModel."""
        evaluation = self.categorization_pipeline.evaluate()

        assert evaluation.f1() is None

    def test_6_categorize_test_document(self):
        """Test extracted category by categorizing a randomly selected Test Document."""
        test_document = self.project.get_document_by_id(TEST_DOCUMENT_ID)
        self.result = self.categorization_pipeline.categorize(document=test_document)

        assert isinstance(self.result, Document)
        assert self.result.category is not None
        assert self.result.category.id_ == 63
