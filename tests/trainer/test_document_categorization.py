# -*- coding: utf-8 -*-
"""Test to train a Categorization AI."""
import os
import logging
import unittest

# from requests import HTTPError
from copy import deepcopy

from konfuzio_sdk.data import Project, Document

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


class TestBaseCategorizationModel(unittest.TestCase):
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
        # payslips_training_documents = self.project.get_category_by_id(TEST_PAYSLIPS_CATEGORY_ID).documents()
        # receipts_training_documents = self.project.get_category_by_id(TEST_PAYSLIPS_CATEGORY_ID).documents()
        # self.categorization_pipeline.documents = payslips_training_documents + receipts_training_documents
        # payslips_test_documents = self.project.get_category_by_id(TEST_PAYSLIPS_CATEGORY_ID).test_documents()
        # receipts_test_documents = self.project.get_category_by_id(TEST_PAYSLIPS_CATEGORY_ID).test_documents()
        # self.categorization_pipeline.test_documents = payslips_test_documents + receipts_test_documents

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
        with self.assertRaises(NotImplementedError):
            self.categorization_pipeline.evaluate()

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


# class TestDocumentCategorizationModel(unittest.TestCase):
#
#     def test_documents_categorization_documentmodel(self):
#         """Test categorization of specific documents with a DocumentModel.pt model."""
#         # confidence for each document with model 2020-11-30-16-17-57_DocumentModel.pt
#         # Current model is failing for doc 2!!
#         # doc_folders = {'1': ('Versicherungspolicen Erweitert (Multipolicen, KFZ, Haftpflicht, ..)', 0.9830012),
#         # '2': ('KONFUZIO_PAYSLIP_TESTS', 0.3158103),
#         # '3': ('KONFUZIO_PAYSLIP_TESTS', 0.9176251),
#         # '4': ('Kontoauszug', 0.74001426)}
#
#         doc_folders = {'1': '23',
#                        '2': '39',
#                        '3': '46',
#                        '4': '34',
#                        '5': '39',
#                        }
#
#         model = load_pickle(get_latest_document_model('*DocumentModel.pt'))
#         confidence_threshold = 0.375
#
#         for doc_folder in doc_folders.keys():
#             # # Code to create files (when new)
#             # import konfuzio
#             # from konfuzio.image import get_png_list
#             # from konfuzio.ocr import FileScanner
#             #
#             # path_file = glob.glob(os.path.join(data_root, doc_folder + '/*.pdf'))[0]
#             #
#             # scanner_ocr = FileScanner(file_path=path_file)
#             # scanner_ocr.ocr()
#             #
#             # with open(os.path.join(data_root, doc_folder + '/document.txt'), 'w') as f:
#             #     f.write(scanner_ocr.text)
#             #
#             # with open(path_file, 'rb') as f:
#             #     pngs_files = get_png_list(file=f)
#             #
#             # for i, png_file in enumerate(pngs_files):
#             #     png_file.save(filename=os.path.join(data_root, doc_folder + '/page_' + str(i) + '.png'))
#
#             # load data
#             paths_doc_images = glob.glob(os.path.join(data_root, doc_folder + '/*.png'))
#
#             # get document text
#             with open(data_root + '/' + doc_folder + '/document.txt', 'r') as f:
#                 content = f.read()
#
#             docs_text = str(content)
#
#             # get images of the document in BytesIO format
#             docs_data_images = []
#
#             for img in paths_doc_images:
#                 img_data = pil_image.open(img)
#                 buf = BytesIO()
#                 img_data.save(buf, format='PNG')
#                 docs_data_images.append(buf)
#
#             # extract from pt file
#             (predicted_category, predicted_confidence), _ = model.extract(
#             page_images=docs_data_images, text=docs_text)
#
#             # assert category
#             assert predicted_category == doc_folders[doc_folder]
#
#             # assert confidence
#             assert predicted_confidence >= confidence_threshold
#
#     def test_documents_categorization_metaclf(self):
#         """Test categorization of specific documents with a metaclf.pt model."""
#         # confidence for each document with model 2020-11-21-14-16-59_metaclf.pt
#         # doc_folders = {'1': ('23', 0.948), '2': ('39', 0.999), '3': ('46', 1.0), '4': ('34', 0.63)}
#
#         # doc_folders = {'1': '23', '2': '39', '3': '46', '4': '34'}
#         doc_folders = {'1': '23', '2': '39', '3': '46'}
#
#         model = load_pickle(get_latest_document_model('*metaclf.pt'))
#         model.batch_size = 1
#         confidence_threshold = 0.1
#
#         for doc_folder in doc_folders.keys():
#             # # Code to create files (when new)
#             # import konfuzio
#             # from konfuzio.image import get_png_list
#             # from konfuzio.ocr import FileScanner
#             #
#             # path_file = glob.glob(os.path.join(data_root, doc_folder + '/*.pdf'))[0]
#             #
#             # scanner_ocr = FileScanner(file_path=path_file)
#             # scanner_ocr.ocr()
#             #
#             # with open(os.path.join(data_root, doc_folder + '/document.txt'), 'w') as f:
#             #     f.write(scanner_ocr.text)
#             #
#             # with open(path_file, 'rb') as f:
#             #     pngs_files = get_png_list(file=f)
#             #
#             # for i, png_file in enumerate(pngs_files):
#             #     png_file.save(filename=os.path.join(data_root, doc_folder + '/page_' + str(i) + '.png'))
#
#             # load data
#             paths_doc_images = glob.glob(os.path.join(data_root, doc_folder + '/*.png'))
#
#             # get document text
#             with open(data_root + '/' + doc_folder + '/document.txt', 'r') as f:
#                 content = f.read()
#
#             docs_text = str(content)
#
#             # get images of the document in BytesIO format
#             docs_data_images = []
#
#             for img in paths_doc_images:
#                 img_data = pil_image.open(img)
#                 buf = BytesIO()
#                 img_data.save(buf, format='PNG')
#                 docs_data_images.append(buf)
#
#             # extract from pt file
#             (predicted_category, predicted_confidence), _ = model.extract(
#             page_images=docs_data_images, text=docs_text)
#
#             # assert category
#             assert predicted_category == doc_folders[doc_folder]
#
#             # assert confidence
#             assert predicted_confidence >= confidence_threshold
