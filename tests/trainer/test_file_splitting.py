"""Test file splitting model and its training pipeline."""
import unittest

from konfuzio_sdk.data import Project
from tests.variables import TEST_PROJECT_ID


class TestFileSplittingModel(unittest.TestCase):
    """Test filesplitting model."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the class for filesplitting model testing."""
        cls.project = Project(id_=TEST_PROJECT_ID)
