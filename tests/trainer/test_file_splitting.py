"""Test file splitting model and its training pipeline."""
import unittest

import numpy as np

from konfuzio_sdk.data import Project
from konfuzio_sdk.trainer import file_splitting
from tests.variables import TEST_PROJECT_ID

from pathlib import Path


class TestFileSplittingModel(unittest.TestCase):
    """Test filesplitting model."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the tested class."""
        cls.fusion_model = file_splitting.FusionModel(project_id=TEST_PROJECT_ID, split_point=0.5)
        cls.project = Project(id_=TEST_PROJECT_ID)
        cls.train_data = cls.project.documents
        cls.test_data = cls.project.test_documents
        cls.bert_model, cls.bert_tokenizer = cls.fusion_model.init_bert()
        cls.file_splitter = file_splitting.PageSplitting('fusion.h5', project_id=TEST_PROJECT_ID)
        cls.image_data_generator = cls.fusion_model._prepare_image_data_generator

    def test_model_training(self):
        """Check that the trainer runs and saves trained model."""
        self.fusion_model.train()
        assert Path(self.project.model_folder + '/fusion.h5').exists()

    def test_get_logits_bert(self):
        """Test that BERT inputs are transformed into logits."""
        texts = [page.text for doc in self.train_data for page in doc.pages()]
        logits = self.fusion_model.get_logits_bert(texts, self.bert_tokenizer, self.bert_model)
        assert len(texts) == len(logits)
        for logit in logits:
            assert type(logit) is int

    def test_get_logits_vgg16(self):
        """Test that VGG16 inputs are transformed into logits."""
        pages = [page.image_path for doc in self.train_data for page in doc.pages()]
        model = self.fusion_model.train_vgg(self.image_data_generator)
        logits = self.fusion_model.get_logits_vgg16(pages, model)
        assert len(pages) == len(logits)
        for logit in logits:
            assert type(logit) is np.ndarray

    def test_squash_logits(self):
        """Test that logits are merged for further input to MLP."""
        texts = [page.text for doc in self.train_data for page in doc.pages()]
        text_logits = self.fusion_model.get_logits_bert(texts, self.bert_tokenizer, self.bert_model)
        pages = [page.image_path for doc in self.train_data for page in doc.pages()]
        model = self.fusion_model.train_vgg()
        img_logits = self.fusion_model.get_logits_vgg16(pages, model)
        logits = self.fusion_model.squash_logits(img_logits, text_logits)
        assert len(logits) == len(text_logits)
        assert len(logits) == len(img_logits)
        for logit in logits:
            assert len(logit) == 4

    def test_predict_first_page(self):
        """Check if first Pages are predicted correctly."""
        for doc in self.train_data:
            pred = self.file_splitter.propose_mappings(doc)
            assert len(pred) == 1
