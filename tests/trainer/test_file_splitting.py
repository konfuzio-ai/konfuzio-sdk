"""Test file splitting model and its training pipeline."""
import unittest

from konfuzio_sdk.data import Project
from konfuzio_sdk.trainer import file_splitting
from tests.variables import TEST_PROJECT_ID

from pathlib import Path


class TestFileSplittingModel(unittest.TestCase):
    """Test filesplitting model."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the tested class."""
        cls.fusion_model = file_splitting.FileSplittingModel(project_id=TEST_PROJECT_ID)
        cls.project = Project(id_=TEST_PROJECT_ID)
        cls.train_data = cls.project.documents
        cls.test_data = cls.project.test_documents
        cls.bert_model, cls.bert_tokenizer = cls.fusion_model.init_bert()

    def test_model_training(self):
        """Check that the trainer runs and saves trained model."""
        self.fusion_model.train()
        assert Path(self.project.model_folder + '/fusion.pickle').exists()

    def test_predict_first_page(self):
        """Check if first Pages are predicted correctly."""
        file_splitter = file_splitting.SplittingAI(
            self.project.model_folder + '/fusion.pickle', project_id=TEST_PROJECT_ID
        )
        for doc in self.train_data:
            pred = file_splitter.propose_split_documents(doc)
            assert len(pred) == 1

    def test_split_document(self):
        """Propose splittings for a document."""
        doc = self.train_data[0]
        splitting_ai = file_splitting.SplittingAI(
            self.project.model_folder + '/fusion.pickle', project_id=TEST_PROJECT_ID
        )
        proposed = doc.propose_splitting(splitting_ai)
        assert len(proposed) == 1
        Path(self.project.model_folder + '/fusion.pickle').unlink()
