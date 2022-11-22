"""Test file splitting model and its training pipeline."""
import unittest

from konfuzio_sdk.data import Project
from konfuzio_sdk.trainer import file_splitting
from tests.variables import TEST_PROJECT_ID


class TestFileSplittingModel(unittest.TestCase):
    """Test filesplitting model."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the tested class."""
        cls.project = Project(id_=TEST_PROJECT_ID)
        cls.train_data = cls.project.documents
        cls.test_data = cls.project.test_documents

    def test_split_document_model(self):
        """Propose splittings for a document using the model."""
        doc = self.train_data[0]
        splitting_ai = file_splitting.SplittingAI(project_id=46, category_id=63)
        first_page_spans = splitting_ai.train()
        proposed = doc.propose_splitting(splitting_ai, first_page_spans)
        assert len(proposed) == 1

    def test_split_document_fallback_logic(self):
        """Propose splittings for a document using the fallback logic."""
        splitting_ai = file_splitting.SplittingAI(project_id=46, category_id=63)
        first_page_spans = splitting_ai.train()
        for doc in self.test_data:
            pred = splitting_ai.propose_split_documents(doc, first_page_spans)
            assert len(pred) == 1
