"""Test file splitting model and its training pipeline."""
import unittest

import numpy as np

from konfuzio_sdk.data import Project
from konfuzio_sdk.trainer import file_splitting
from tests.variables import TEST_PROJECT_ID

from keras.models import load_model
from pathlib import Path
from zipfile import ZipFile


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
        cls.file_splitter = file_splitting.SplittingAI(
            cls.project.model_folder + '/splitting_ai_models.zip', project_id=TEST_PROJECT_ID
        )
        cls.image_data_generator = cls.fusion_model._prepare_image_data_generator

    def test_model_training(self):
        """Check that the trainer runs and saves trained model."""
        self.fusion_model.train()
        assert Path(self.project.model_folder + '/splitting_ai_models.zip').exists()

    def test_transform_logits_bert(self):
        """Test that BERT inputs are transformed into logits."""
        texts = [page.text for doc in self.train_data for page in doc.pages()]
        logits = self.fusion_model.get_logits_bert(texts, self.bert_tokenizer, self.bert_model)
        assert len(texts) == len(logits)
        for logit in logits:
            assert type(logit) is int

    def test_transform_logits_vgg16(self):
        """Test that VGG16 inputs are transformed into logits."""
        pages = [page.image_path for doc in self.train_data for page in doc.pages()]
        with ZipFile(self.project.model_folder + '/splitting_ai_models.zip') as zip_file:
            len(zip_file)
            model = load_model('vgg16.h5')
        # model = self.fusion_model.train_vgg(self.image_data_generator)
        logits = self.fusion_model.get_logits_vgg16(pages, model)
        assert len(pages) == len(logits)
        for logit in logits:
            assert type(logit) is np.ndarray

    def test_squash_logits(self):
        """Test that logits are merged for further input to MLP."""
        texts = [page.text for doc in self.train_data for page in doc.pages()]
        text_logits = self.fusion_model.get_logits_bert(texts, self.bert_tokenizer, self.bert_model)
        pages = [page.image_path for doc in self.train_data for page in doc.pages()]
        with ZipFile(self.project.model_folder + '/splitting_ai_models.zip') as zip_file:
            len(zip_file)
            model_vgg16 = load_model('vgg16.h5')
        img_logits = self.fusion_model.get_logits_vgg16(pages, model_vgg16)
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
