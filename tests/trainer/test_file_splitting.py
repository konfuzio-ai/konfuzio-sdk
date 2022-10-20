"""Test file splitting model and its training pipeline."""
import unittest

from konfuzio_sdk.data import Project
from konfuzio_sdk.trainer import file_splitting
from tests.variables import TEST_PROJECT_ID

# draft


class TestFileSplittingModel(unittest.TestCase):
    """Test filesplitting model."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the class for filesplitting model testing."""
        cls.project = Project(id_=TEST_PROJECT_ID)

    def test_model_training(self):
        """Check that the trainer runs and saves trained model."""
        file_splitter = file_splitting.FusionModel(project_id=TEST_PROJECT_ID, split_point=0.5)
        file_splitter.train()
