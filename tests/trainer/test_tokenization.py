"""Test tokenizers created for AI models."""
import parameterized
import pytest
import unittest

from konfuzio_sdk.data import Project
from konfuzio_sdk.settings_importer import is_dependency_installed
from konfuzio_sdk.trainer.tokenization import TransformersTokenizer


@pytest.mark.skipif(
    not is_dependency_installed('timm')
    and not is_dependency_installed('torch')
    and not is_dependency_installed('transformers')
    and not is_dependency_installed('torchvision'),
    reason='Required dependencies not installed.',
)
@parameterized.parameterized_class(
    ('tokenizer_name',),
    [
        ('prajjwal1/bert-tiny',),
        # ('bert-base-german-dbmdz-cased',),  # commented out for passing on push in github actions (RAM limitations).
        # feel free to uncomment when testing locally
        # ('distilbert-base-german-cased',),
        # ('albert-base-v2',),
        # ('facebook/bart-large-cnn',),
        # ('t5-small',),
        # ('camembert-base',),
        # ('bert-base-chinese',),
        # ('bert-base-german-cased',),
    ],
)
class TestTransformersTokenizer(unittest.TestCase):
    """Test different types of TransformersTokenizer."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the TransformersTokenizer."""
        cls.tokenizer = TransformersTokenizer(tokenizer_name=cls.tokenizer_name)
        cls.project = Project(id_=14392)

    def test_proper_tokenizer_type(self):
        """Test that a correct type of tokenizer is fetched for a current tokenizer name."""
        pass

    def test_no_supported_tokenizer_error(self):
        """Test that a non-supported tokenizer is not possible to use with the class."""
        pass

    def test_padding(self):
        """Test that padding happens to a text shorter than the selected max_length, if enabled."""
        pass

    def test_truncation(self):
        """Test that the text is truncated to the selected max_length, if enabled."""
        pass
