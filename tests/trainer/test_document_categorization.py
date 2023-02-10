# -*- coding: utf-8 -*-
"""Test to train a Categorization AI."""
import logging
import unittest
from copy import deepcopy

from konfuzio_sdk.data import Project, Document

from tests.variables import (
    OFFLINE_PROJECT,
    TEST_DOCUMENT_ID,
    TEST_CATEGORIZATION_DOCUMENT_ID,
    TEST_RECEIPTS_CATEGORY_ID,
    TEST_PAYSLIPS_CATEGORY_ID,
)
from konfuzio_sdk.trainer.document_categorization import (
    FallbackCategorizationModel,
)

logger = logging.getLogger(__name__)


class TestFallbackCategorizationModel(unittest.TestCase):
    """Test the fallback logic for predicting the Category of a Document."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the Data and Categorization Pipeline."""
        cls.project = Project(id_=None, project_folder=OFFLINE_PROJECT)
        cls.categorization_pipeline = FallbackCategorizationModel(cls.project.categories)
        cls.payslips_category = cls.project.get_category_by_id(TEST_PAYSLIPS_CATEGORY_ID)
        cls.receipts_category = cls.project.get_category_by_id(TEST_RECEIPTS_CATEGORY_ID)

    def test_1_configure_pipeline(self) -> None:
        """
        No pipeline to configure for the fallback logic.

        Documents can be specified to calculate Evaluation metrics.
        """
        assert self.categorization_pipeline.categories is not None

        payslips_training_documents = self.project.get_category_by_id(TEST_PAYSLIPS_CATEGORY_ID).documents()
        receipts_training_documents = self.project.get_category_by_id(TEST_RECEIPTS_CATEGORY_ID).documents()
        self.categorization_pipeline.documents = payslips_training_documents + receipts_training_documents
        assert all(doc.category is not None for doc in self.categorization_pipeline.documents)

        payslips_test_documents = self.project.get_category_by_id(TEST_PAYSLIPS_CATEGORY_ID).test_documents()
        receipts_test_documents = self.project.get_category_by_id(TEST_RECEIPTS_CATEGORY_ID).test_documents()
        self.categorization_pipeline.test_documents = payslips_test_documents + receipts_test_documents
        assert all(doc.category is not None for doc in self.categorization_pipeline.test_documents)

    def test_2_fit(self) -> None:
        """Start to train the Model."""
        # since we are using the fallback logic, this should not require training anything
        with self.assertRaises(NotImplementedError):
            self.categorization_pipeline.fit()

    def test_3_save_model(self):
        """Save the model."""
        # since we are using the fallback logic, this should not save any model to disk
        with self.assertRaises(NotImplementedError):
            self.categorization_pipeline.pipeline_path = self.categorization_pipeline.save(
                output_dir=self.project.model_folder
            )

    def test_4_evaluate(self):
        """Evaluate FallbackCategorizationModel."""
        categorization_evaluation = self.categorization_pipeline.evaluate()
        # can't categorize any of the 3 payslips docs since they don't contain the word "lohnabrechnung"
        assert categorization_evaluation.f1(self.categorization_pipeline.categories[0]) == 1.0
        # can categorize 1 out of 2 receipts docs since one contains the word "quittung"
        # therefore recall == 1/2 and precision == 1.0, implying f1 == 2/3
        assert categorization_evaluation.f1(self.categorization_pipeline.categories[1]) == 1.0
        # global f1 score
        assert categorization_evaluation.f1(None) == 1.0

    def test_5_categorize_test_document(self):
        """Test extract Category for a selected Test Document with the Category name contained within its text."""
        test_receipt_document = deepcopy(self.project.get_document_by_id(TEST_CATEGORIZATION_DOCUMENT_ID))
        # reset each Page.category attribute to test that it can be categorized successfully
        test_receipt_document.set_category(None)
        result = self.categorization_pipeline.categorize(document=test_receipt_document)
        assert isinstance(result, Document)
        assert result.category == self.receipts_category
        for page in result.pages():
            assert page.category == self.receipts_category

    def test_6a_categorize_document_with_no_pages(self):
        """Test extract Category for a Document without a Page."""
        document_with_no_pages = Document(project=self.project, text="hello")
        result = self.categorization_pipeline.categorize(document=document_with_no_pages)
        assert isinstance(result, Document)
        assert result.category is None
        assert result.pages() == []

    def test_7_already_existing_categorization(self):
        """Test that the existing Category attribute for a Test Document will be reused as the fallback result."""
        test_payslip_document = self.project.get_document_by_id(TEST_DOCUMENT_ID)
        result = self.categorization_pipeline.categorize(document=test_payslip_document)
        assert isinstance(result, Document)
        assert result.category == test_payslip_document.category
        for page in result.pages():
            assert page.category == test_payslip_document.category

    def test_8_cannot_categorize_test_documents_with_category_name_not_contained_in_text(self):
        """Test cannot extract Category for two Test Document if their texts don't contain the Category name."""
        test_receipt_document = self.project.get_category_by_id(TEST_RECEIPTS_CATEGORY_ID).test_documents()[0]
        # reset each Page.category attribute to test that it can't be categorized successfully
        test_receipt_document.category = self.project.no_category
        result = self.categorization_pipeline.categorize(document=test_receipt_document)
        assert isinstance(result, Document)
        assert result.category == self.project.no_category
        for page in result.pages():
            assert page.category is None

        test_payslip_document = self.project.get_document_by_id(TEST_DOCUMENT_ID)
        # reset each Page.category attribute to test that it can't be categorized successfully
        test_payslip_document.category = self.project.no_category
        result = self.categorization_pipeline.categorize(document=test_payslip_document)
        assert isinstance(result, Document)
        assert result.category == self.project.no_category
        for page in result.pages():
            assert page.category is None

    def test_9_force_categorization(self):
        """Test re-extract Category for two selected Test Documents that already contain a Category attribute."""
        # This Document can be recategorized successfully because its text contains the word "quittung" (receipt) in it.
        # Recall that the check is case insensitive.
        test_receipt_document = self.project.get_document_by_id(TEST_CATEGORIZATION_DOCUMENT_ID)
        result = self.categorization_pipeline.categorize(document=test_receipt_document, recategorize=True)
        assert isinstance(result, Document)
        assert result.category == test_receipt_document.category
        for page in result.pages():
            assert page.category == test_receipt_document.category

        # This Document is originally categorized as "Lohnabrechnung".
        # It cannot be recategorized successfully because its text does not contain
        # the word "lohnabrechnung" (payslip) in it.
        # Recall that the check is case insensitive.
        test_payslip_document = self.project.get_document_by_id(TEST_DOCUMENT_ID)
        result = self.categorization_pipeline.categorize(document=test_payslip_document, recategorize=True)
        assert isinstance(result, Document)
        assert result.category is None
        for page in result.pages():
            assert page.category is None

    def test_9a_categorize_in_place(self):
        """Test in-place re-extract Category for a selected Test Document that already contains a Category attribute."""
        # This Document can be recategorized successfully because its text contains the word "quittung" (receipt) in it.
        # Recall that the check is case insensitive.
        test_receipt_document = self.project.get_document_by_id(TEST_CATEGORIZATION_DOCUMENT_ID)
        test_receipt_document.set_category(None)
        self.categorization_pipeline.categorize(document=test_receipt_document, inplace=True)
        assert test_receipt_document.category == self.receipts_category
        for page in test_receipt_document.pages():
            assert page.category == self.receipts_category

    def test_9b_categorize_in_place_document_with_no_pages(self):
        """Test extract Category in place for a Document without a Page."""
        document_with_no_pages = Document(project=self.project, text="hello")
        result = self.categorization_pipeline.categorize(document=document_with_no_pages, inplace=True)
        assert isinstance(result, Document)
        assert result.category is None
        assert result.pages() == []

    def test_9c_categorize_defaults_not_in_place(self):
        """Test cannot re-extract Category for a selected Test Document that already contain a Category attribute."""
        # This Document can be recategorized successfully because its text contains the word "quittung" (receipt) in it.
        # Recall that the check is case insensitive.
        test_receipt_document = self.project.get_document_by_id(TEST_CATEGORIZATION_DOCUMENT_ID)
        test_receipt_document.set_category(None)
        self.categorization_pipeline.categorize(document=test_receipt_document)
        assert test_receipt_document.category is None
        for page in test_receipt_document.pages():
            assert page.category is None
