# -*- coding: utf-8 -*-
"""Test to train an Extraction AI."""

import logging
import unittest

from konfuzio_sdk.trainer.information_extraction import DocumentAnnotationMultiClassModel
from konfuzio_sdk.api import upload_ai_model
from konfuzio_sdk.data import Project
from konfuzio_sdk.tokenizer.regex import WhitespaceTokenizer

logger = logging.getLogger(__name__)


class TestInformationExtraction(unittest.TestCase):
    """Test to train an extraction Model for Documents."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the Data and Pipeline."""
        cls.project = Project(id_=46)  # todo use offline project
        cls.pipeline = DocumentAnnotationMultiClassModel()

    def test_1_configure_pipeline(self):
        """Make sure the Data and Pipeline is configured."""
        self.pipeline.tokenizer = WhitespaceTokenizer()
        self.pipeline.category = self.project.get_category_by_id(id_=63)
        self.pipeline.documents = self.pipeline.category.documents()[:1]
        self.pipeline.test_documents = self.pipeline.category.test_documents()[:1]

    def test_2_make_features(self):
        """Make sure the Data and Pipeline is configured."""
        self.pipeline.df_train, self.pipeline.label_feature_list = self.pipeline.feature_function(
            documents=self.pipeline.documents
        )
        self.pipeline.df_test, self.pipeline.test_label_feature_list = self.pipeline.feature_function(
            documents=self.pipeline.test_documents
        )

    def test_3_fit(self) -> None:
        """Start to train the Model."""
        self.pipeline.fit()

    def test_4_save_model(self):
        """Evaluate the model."""
        self.pipeline_path = self.pipeline.save(output_dir=self.project.model_folder)

    def test_5_evaluate_model(self):
        """Evaluate the model."""
        self.pipeline.evaluate()

    def test_6_extract_test_document(self):
        """Extract a randomly selected Test Document."""
        test_document = self.project.get_document_by_id(44823)
        self.pipeline.extract(document=test_document)

    @unittest.skip(reason='Test run offline.')
    def test_7_upload_ai_model(self):
        """Upload the model."""
        upload_ai_model(ai_model_path=self.pipeline_path, category_ids=[self.pipeline.category.id_])
