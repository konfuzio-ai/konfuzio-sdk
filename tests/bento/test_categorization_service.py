"""Test interfaces created for containerization of Categorization AIs."""
import unittest

import pytest

from konfuzio_sdk.settings_importer import is_dependency_installed


@pytest.mark.skipif(
    not is_dependency_installed('torch'),
    reason='Required dependencies not installed.',
)
class TestCategorizationAIBento(unittest.TestCase):
    """Test that Bento-based AI works."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create a model and its Bento instance of Categorization AI."""
