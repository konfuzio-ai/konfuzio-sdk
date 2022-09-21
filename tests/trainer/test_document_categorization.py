# -*- coding: utf-8 -*-
"""Test to train a Categorization AI."""
import os
import logging
import unittest

# from requests import HTTPError

from konfuzio_sdk.data import Project, Document

# from konfuzio_sdk.api import upload_ai_model
from tests.variables import (
    OFFLINE_PROJECT,
    TEST_DOCUMENT_ID,
    TEST_PAYSLIPS_CATEGORY_ID,
    TEST_CATEGORIZATION_DOCUMENT_ID,
    TEST_RECEIPTS_CATEGORY_ID,
)
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
        payslips_training_documents = self.project.get_category_by_id(TEST_PAYSLIPS_CATEGORY_ID).documents()
        receipts_training_documents = self.project.get_category_by_id(TEST_PAYSLIPS_CATEGORY_ID).documents()
        self.categorization_pipeline.documents = payslips_training_documents + receipts_training_documents
        payslips_test_documents = self.project.get_category_by_id(TEST_PAYSLIPS_CATEGORY_ID).test_documents()
        receipts_test_documents = self.project.get_category_by_id(TEST_PAYSLIPS_CATEGORY_ID).test_documents()
        self.categorization_pipeline.test_documents = payslips_test_documents + receipts_test_documents

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
        assert not os.path.isfile(self.categorization_pipeline.pipeline_path)
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

    @unittest.skip(reason="Categorization Evaluation not implemented.")
    def test_5_evaluate(self):
        """Evaluate BaseCategorizationModel."""
        evaluation = self.categorization_pipeline.evaluate()

        assert evaluation.f1() == 1.0

    def test_6_categorize_test_documents(self):
        """Test extracted category by categorizing two randomly selected Test Documents."""
        test_payslip_document = self.project.get_document_by_id(TEST_DOCUMENT_ID)
        test_payslip_document.category = None
        result = self.categorization_pipeline.categorize(document=test_payslip_document)

        assert isinstance(result, Document)
        assert result.category is not None
        assert result.category.id_ == TEST_PAYSLIPS_CATEGORY_ID

        test_receipt_document = self.project.get_document_by_id(TEST_CATEGORIZATION_DOCUMENT_ID)
        test_receipt_document.category = None
        result = self.categorization_pipeline.categorize(document=test_receipt_document)

        assert isinstance(result, Document)
        assert result.category is not None
        assert result.category.id_ == TEST_RECEIPTS_CATEGORY_ID

    def test_7_cannot_categorize_test_document(self):
        """Test cannot extract category for a specifically selected Test Document."""
        test_receipt_document = self.project.get_category_by_id(TEST_RECEIPTS_CATEGORY_ID).test_documents()[0]
        test_receipt_document.category = None
        result = self.categorization_pipeline.categorize(document=test_receipt_document)

        assert isinstance(result, Document)
        assert result.category is None
