"""Test file splitting model and its training pipeline."""
import tarfile
import unittest

from konfuzio_sdk.data import Project
from konfuzio_sdk.trainer import file_splitting
from tests.variables import TEST_PROJECT_ID

from keras.models import load_model
from pathlib import Path


class TestFileSplittingModel(unittest.TestCase):
    """Test filesplitting model."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the tested class."""
        cls.fusion_model = file_splitting.FileSplittingModel(project_id=TEST_PROJECT_ID, split_point=0.5)
        cls.project = Project(id_=TEST_PROJECT_ID)
        cls.train_data = cls.project.documents
        cls.test_data = cls.project.test_documents
        cls.bert_model, cls.bert_tokenizer = cls.fusion_model.init_bert()
        cls.image_data_generator = cls.fusion_model._prepare_image_data_generator

    def test_model_training(self):
        """Check that the trainer runs and saves trained model."""
        self.fusion_model.train()
        assert Path(self.project.model_folder + '/splitting_ai_models.tar.gz').exists()
        tar = tarfile.open(self.project.model_folder + '/splitting_ai_models.tar.gz', "r:gz")
        tar.extractall()

    def test_squash_logits(self):
        """Test that logits are merged for further input to MLP."""
        texts = [page.text for doc in self.train_data for page in doc.pages()]
        text_logits = self.fusion_model.get_logits_bert(texts, self.bert_tokenizer, self.bert_model)
        pages = [page.image_path for doc in self.train_data for page in doc.pages()]
        model_vgg16 = load_model(self.project.model_folder + '/vgg16.h5')
        img_logits = self.fusion_model.get_logits_vgg16(pages, model_vgg16)
        logits = self.fusion_model.squash_logits(img_logits, text_logits)
        assert len(logits) == len(text_logits)
        assert len(logits) == len(img_logits)
        for logit in logits:
            assert len(logit) == 4
        Path(self.project.model_folder + '/vgg16.h5').unlink()
        Path(self.project.model_folder + '/fusion.h5').unlink()
        Path(self.project.model_folder + '/splitting_ai_models.tar.gz').unlink()

    def test_predict_first_page(self):
        """Check if first Pages are predicted correctly."""
        file_splitter = file_splitting.SplittingAI(
            self.project.model_folder + '/splitting_ai_models.tar.gz', project_id=TEST_PROJECT_ID
        )
        for doc in self.train_data:
            pred = file_splitter.propose_split_documents(doc)
            assert len(pred) == 1

    def test_split_document(self):
        """Propose splittings for a document."""
        doc = self.train_data[0]
        splitting_ai = file_splitting.SplittingAI(
            self.project.model_folder + '/splitting_ai_models.tar.gz', project_id=TEST_PROJECT_ID
        )
        proposed = doc.propose_splitting(splitting_ai)
        assert len(proposed) == 1
