"""Test tokenizers created for AI models."""
import parameterized
import pytest
import unittest

from konfuzio_sdk.data import Project
from konfuzio_sdk.extras import torch
from konfuzio_sdk.settings_importer import is_dependency_installed
from konfuzio_sdk.trainer.tokenization import TransformersTokenizer, HF_ALLOWED_TOKENIZERS


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
        ('bert-base-german-dbmdz-cased',),
        ('distilbert-base-german-cased',),
        ('albert-base-v2',),
        ('bert-base-chinese',),
        ('bert-base-german-cased',),
        ('google/mobilebert-uncased',),
        ('vinai/phobert-base',),
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
        assert isinstance(self.tokenizer.tokenizer, tuple(HF_ALLOWED_TOKENIZERS))

    def test_padding_max_length(self):
        """Test that padding happens to a text shorter than the selected max_length, if enabled."""
        test_text = "This is the text to be padded."
        padded = self.tokenizer(test_text, padding='max_length')
        assert len(padded['input_ids']) == 512
        if self.tokenizer_name != 'vinai/phobert-base':
            assert padded['input_ids'][-1] == 0
        else:
            assert padded['input_ids'][-1] == 1

    def test_no_padding(self):
        """Test that padding does not happen if it is turned off."""
        test_text = "This is the text to not be padded."
        not_padded = self.tokenizer(test_text, padding='do_not_pad')
        assert len(not_padded['input_ids']) <= 15

    def test_truncation(self):
        """Test that the text is truncated to the selected max_length, if enabled."""
        test_text = ' '.join(['word' for w in range(520)])
        truncated = self.tokenizer(test_text, truncation=True)
        assert len(truncated['input_ids']) == 512

    def test_no_truncation(self):
        """Test that the text is not truncated if disabled."""
        test_text = ' '.join(['word' for w in range(520)])
        truncated = self.tokenizer(test_text, truncation=False)
        assert len(truncated['input_ids']) > 512

    def test_return_tensors(self):
        """Test that the tokenizer returns tensors if return_tensors is set to pt."""
        test_text = "This is the text to be returned as tensors after tokenization."
        tensors = self.tokenizer(test_text, return_tensors='pt')
        assert isinstance(tensors['input_ids'], torch.Tensor)

    def test_return_list(self):
        """Test that the tokenizer returns list if return_tensors is set to None."""
        test_text = "This is the text to be returned as list of encoded tokens after tokenization."
        tokens = self.tokenizer(test_text, return_tensors=None)
        assert isinstance(tokens['input_ids'], list)


@pytest.mark.skipif(
    not is_dependency_installed('timm')
    and not is_dependency_installed('torch')
    and not is_dependency_installed('transformers')
    and not is_dependency_installed('torchvision'),
    reason='Required dependencies not installed.',
)
def test_no_supported_tokenizer_error():
    """Test that a non-supported tokenizer is not possible to use with the class."""
    with pytest.raises(ValueError, match="is not supported"):
        TransformersTokenizer('t5-small')
