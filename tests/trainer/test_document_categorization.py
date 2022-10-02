# -*- coding: utf-8 -*-
"""Test to train a Categorization AI."""
import os
import logging
import unittest
import pytest
from copy import deepcopy

from konfuzio_sdk.data import Project, Document
from konfuzio_sdk.trainer.tokenization import get_tokenizer

# from konfuzio_sdk.api import upload_ai_model
from tests.variables import (
    OFFLINE_PROJECT,
    TEST_DOCUMENT_ID,
    TEST_PAYSLIPS_CATEGORY_ID,
    TEST_CATEGORIZATION_DOCUMENT_ID,
    TEST_RECEIPTS_CATEGORY_ID,
)
from konfuzio_sdk.trainer.document_categorization import (
    FallbackCategorizationModel,
    get_category_name_for_fallback_prediction,
    build_list_of_relevant_categories,
    CustomDocumentModel,
    create_transformations_dict,
    build_template_category_vocab,
    get_timestamp,
)

logger = logging.getLogger(__name__)


class TestFallbackCategorizationModel(unittest.TestCase):
    """Test New SDK fallback logic for Categorization."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the Data and Categorization Pipeline."""
        cls.project = Project(id_=None, project_folder=OFFLINE_PROJECT)
        cls.categorization_pipeline = FallbackCategorizationModel(cls.project)
        cls.categorization_pipeline.categories = cls.project.categories

    def test_1_configure_pipeline(self) -> None:
        """No pipeline to configure for the fallback logic."""
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
        """Evaluate FallbackCategorizationModel."""
        categorization_evaluation = self.categorization_pipeline.evaluate()
        # can't categorize any of the 3 payslips docs since they don't contain the word "lohnabrechnung"
        assert categorization_evaluation.f1(self.categorization_pipeline.categories[0]) == 0.0
        # can categorize 1 out of 2 receipts docs since one contains the word "quittung"
        # therefore recall == 1/2 and precision == 1.0, implying f1 == 2/3
        assert categorization_evaluation.f1(self.categorization_pipeline.categories[1]) == 2 / 3
        # global f1 score
        assert categorization_evaluation.f1(None) == 0.26666666666666666

    def test_6_categorize_test_document(self):
        """Test extract category for a selected Test Document with the category name contained within its text."""
        test_receipt_document = deepcopy(self.project.get_document_by_id(TEST_CATEGORIZATION_DOCUMENT_ID))
        # reset the category attribute to test that it can be categorized successfully
        test_receipt_document.category = None
        result = self.categorization_pipeline.categorize(document=test_receipt_document)
        assert isinstance(result, Document)
        assert result.category is not None
        assert result.category.id_ == TEST_RECEIPTS_CATEGORY_ID

    def test_7_already_existing_categorization(self):
        """Test that the existing category attribute for a Test Document will be reused as the fallback result."""
        test_payslip_document = self.project.get_document_by_id(TEST_DOCUMENT_ID)
        result = self.categorization_pipeline.categorize(document=test_payslip_document)
        assert isinstance(result, Document)
        assert result.category is not None
        assert result.category.id_ == test_payslip_document.category.id_

    def test_8_cannot_categorize_test_documents_with_category_name_not_contained_in_text(self):
        """Test cannot extract category for two Test Document if their texts don't contain the category name."""
        test_receipt_document = self.project.get_category_by_id(TEST_RECEIPTS_CATEGORY_ID).test_documents()[0]
        # reset the category attribute to test that it can't be categorized successfully
        test_receipt_document.category = None
        result = self.categorization_pipeline.categorize(document=test_receipt_document)
        assert isinstance(result, Document)
        assert result.category is None

        test_payslip_document = self.project.get_document_by_id(TEST_DOCUMENT_ID)
        # reset the category attribute to test that it can't be categorized successfully
        test_payslip_document.category = None
        result = self.categorization_pipeline.categorize(document=test_payslip_document)
        assert isinstance(result, Document)
        assert result.category is None

    def test_9_force_categorization(self):
        """Test extract category for two selected Test Documents that already contain a category attribute."""
        # this document can be recategorized successfully because its text contains the word "quittung" (receipt) in it
        test_receipt_document = self.project.get_document_by_id(TEST_CATEGORIZATION_DOCUMENT_ID)
        result = self.categorization_pipeline.categorize(document=test_receipt_document, recategorize=True)
        assert isinstance(result, Document)
        assert result.category is not None
        assert result.category.id_ == test_receipt_document.category.id_

        # this document cannot be recategorized successfully because its text does not contain
        # the word "lohnabrechnung" (payslip) in it
        test_payslip_document = self.project.get_document_by_id(TEST_DOCUMENT_ID)
        result = self.categorization_pipeline.categorize(document=test_payslip_document, recategorize=True)
        assert isinstance(result, Document)
        assert result.category is None

    def test_9a_categorize_in_place(self):
        """Test extract category for two selected Test Documents that already contain a category attribute."""
        # this document can be recategorized successfully because its text contains the word "quittung" (receipt) in it
        test_receipt_document = self.project.get_document_by_id(TEST_CATEGORIZATION_DOCUMENT_ID)
        test_receipt_document.category = None
        self.categorization_pipeline.categorize(document=test_receipt_document, inplace=True)
        assert test_receipt_document.category is not None
        assert test_receipt_document.category.id_ == TEST_RECEIPTS_CATEGORY_ID

    def test_9b_categorize_defaults_not_in_place(self):
        """Test extract category for two selected Test Documents that already contain a category attribute."""
        # this document can be recategorized successfully because its text contains the word "quittung" (receipt) in it
        test_receipt_document = self.project.get_document_by_id(TEST_CATEGORIZATION_DOCUMENT_ID)
        test_receipt_document.category = None
        self.categorization_pipeline.categorize(document=test_receipt_document)
        assert test_receipt_document.category is None


def test_get_category_name_for_fallback_prediction():
    """Test turn a category name to lowercase, remove parentheses along with their contents, and trim spaces."""
    project = Project(id_=None, project_folder=OFFLINE_PROJECT)
    payslips_category = project.get_category_by_id(TEST_PAYSLIPS_CATEGORY_ID)
    receipts_category = project.get_category_by_id(TEST_RECEIPTS_CATEGORY_ID)
    assert get_category_name_for_fallback_prediction(payslips_category) == "lohnabrechnung"
    assert get_category_name_for_fallback_prediction(payslips_category.name) == "lohnabrechnung"
    assert get_category_name_for_fallback_prediction(receipts_category) == "quittung"
    assert get_category_name_for_fallback_prediction(receipts_category.name) == "quittung"
    assert get_category_name_for_fallback_prediction("Test Category Name") == "test category name"
    assert get_category_name_for_fallback_prediction("Test Category Name (content)") == "test category name"
    assert get_category_name_for_fallback_prediction("Te(s)t Category Name (content content)") == "tet category name"


def test_build_list_of_relevant_categories():
    """Filter for category name variations which correspond to the given categories, starting from a predefined list."""
    project = Project(id_=None, project_folder=OFFLINE_PROJECT)
    assert set(build_list_of_relevant_categories(project.categories)) == {"lohnabrechnung", "quittung"}


class TestDocumentModel(unittest.TestCase):
    """Test trainable DocumentModel."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the Data and Categorization Pipeline."""
        cls.training_prj = Project(id_=None, project_folder=OFFLINE_PROJECT)
        for document in cls.training_prj.documents + cls.training_prj.test_documents:
            document.get_images()

        cls.categorization_pipeline = CustomDocumentModel(
            cls.training_prj,
            tokenizer=get_tokenizer('phrasematcher', project=cls.training_prj),
            image_preprocessing=create_transformations_dict(
                possible_transforms=['invert', 'target_size', 'grayscale'],
                args=None,
            ),
            image_augmentation=create_transformations_dict(
                possible_transforms=['rotate'],
                args=None,
            ),
            document_classifier_config={
                'image_module': {'name': 'efficientnet_b0'},
                'text_module': {'name': 'nbowselfattention'},
                'multimodal_module': {'name': 'concatenate'},
            },
            category_vocab=build_template_category_vocab([cls.training_prj]),
            use_cuda=False,
        )
        cls.categorization_pipeline.categories = cls.training_prj.categories

    def test_1_configure_pipeline(self) -> None:
        """Test configure categories, with training and test docs for the Document Model."""
        assert self.categorization_pipeline.categories is not None

        payslips_training_documents = self.training_prj.get_category_by_id(TEST_PAYSLIPS_CATEGORY_ID).documents()[:4]
        receipts_training_documents = self.training_prj.get_category_by_id(TEST_RECEIPTS_CATEGORY_ID).documents()[:4]
        self.categorization_pipeline.documents = payslips_training_documents + receipts_training_documents
        assert all(doc.category is not None for doc in self.categorization_pipeline.documents)

        payslips_test_documents = self.training_prj.get_category_by_id(TEST_PAYSLIPS_CATEGORY_ID).test_documents()[:1]
        receipts_test_documents = self.training_prj.get_category_by_id(TEST_RECEIPTS_CATEGORY_ID).test_documents()[:1]
        self.categorization_pipeline.test_documents = payslips_test_documents + receipts_test_documents
        assert all(doc.category is not None for doc in self.categorization_pipeline.test_documents)

    def test_2_fit(self) -> None:
        """Start to train the Model."""
        self.categorization_pipeline.fit(
            document_training_config={
                'valid_ratio': 0.2,
                'batch_size': 2,
                'max_len': None,
                'n_epochs': 5,
                'patience': 1,
                'optimizer': {'name': 'Adam'},
            }
        )

    def test_3_save_model(self) -> None:
        """Test save .pt file to disk."""
        model_type = 'TestDocumentModel'
        path = os.path.join(self.training_prj.project_folder, 'models', f'{get_timestamp()}_{model_type}.pt')
        self.categorization_pipeline.save(path=path)
        assert os.path.isfile(path)

    @pytest.mark.skip(reason="To be defined how to upload a categorization model.")
    def test_4_upload_ai_model(self) -> None:
        """Upload the model."""
        raise NotImplementedError

    @pytest.mark.xfail(reason="100% score on test categorization project to be achieved.")
    def test_5_evaluate(self) -> None:
        """Evaluate DocumentModel."""
        categorization_evaluation = self.categorization_pipeline.evaluate()
        assert categorization_evaluation.f1(self.categorization_pipeline.categories[0]) == 1.0
        assert categorization_evaluation.f1(self.categorization_pipeline.categories[1]) == 1.0
        # global f1 score
        assert categorization_evaluation.f1(None) == 1.0

    def test_6_categorize_test_document(self) -> None:
        """Test categorize a test document."""
        test_receipt_document = self.training_prj.get_document_by_id(TEST_CATEGORIZATION_DOCUMENT_ID)
        # reset the category attribute to test that it can be categorized successfully
        test_receipt_document.category = None
        result = self.categorization_pipeline.categorize(document=test_receipt_document)
        assert isinstance(result, Document)
        assert result.category is not None
        assert result.category.id_ == TEST_RECEIPTS_CATEGORY_ID
