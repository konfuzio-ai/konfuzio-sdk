# -*- coding: utf-8 -*-
"""Test to train a Categorization AI."""
import os
import logging
import unittest

# from requests import HTTPError
from copy import deepcopy
from io import BytesIO

import torch
from PIL import Image as pil_image

from konfuzio_sdk.data import Project, Document
from konfuzio_sdk.trainer.document_categorization import build_category_document_model

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


# document_classifier_config = {'image_module': {'name': 'efficientnet_b0',
#                                                'pretrained': True,
#                                                'freeze': True
#                                                },
#                               'text_module': {'name': 'nbowselfattention',
#                                               'emb_dim': 104
#                                               },
#                               'multimodal_module': {'name': 'concatenate',
#                                                     'hid_dim': 250
#                                                     },
#                               }
#
# document_training_config = {'valid_ratio': 0.2,
#                             'batch_size': 6,
#                             'max_len': None,
#                             'n_epochs': 100,
#                             'patience': 3,
#                             'optimizer': {'name': 'Adam'},
#                             }
#
# img_args = {'invert': False,
#             'target_size': (1000, 1000),
#             'grayscale': True,
#             'rotate': 5
#             }

# None should by default be equal to the above (see data.py build_category_document_model)
category_ai_model_parameters = {
    "document_training_config": {
        'valid_ratio': 0.2,
        'batch_size': 2,
        'max_len': None,
        'n_epochs': 5,
        'patience': 1,
        'optimizer': {'name': 'Adam'},
    },
    "document_classifier_config": None,
    "img_args": None,
}


def test_categorization_model_build():
    """Test trainable DocumentModel."""
    project_id = 1680
    training_prj = Project(id_=project_id)
    for document in training_prj.documents:
        document.get_images()

    document_classifier_config = category_ai_model_parameters['document_classifier_config']
    document_training_config = category_ai_model_parameters['document_training_config']
    img_args = category_ai_model_parameters['img_args']

    torch.cuda.empty_cache()
    model_path, doc_model = build_category_document_model(
        project=training_prj,
        document_classifier_config=document_classifier_config,
        document_training_config=document_training_config,
        img_args=img_args,
        output_dir=os.path.join(training_prj.project_folder, "models"),
        return_model=True,
    )

    test_doc = training_prj.test_documents[-1]
    page_path = test_doc.pages()[0].image_path

    img_data = pil_image.open(page_path)
    buf = BytesIO()
    img_data.save(buf, format='PNG')
    docs_data_images = [buf]

    docs_text = test_doc.text

    # extract from pt file
    (predicted_category, predicted_confidence), _ = doc_model.extract(page_images=docs_data_images, text=docs_text)

    # assert category
    assert predicted_category == 5350

    # assert confidence
    assert predicted_confidence >= 0.5
