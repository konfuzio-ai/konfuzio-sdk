"""Test tokenizer."""
import logging
import unittest

import pandas as pd

from konfuzio_sdk.data import Project, Annotation, Document, Label, AnnotationSet, LabelSet, Span, Category
from konfuzio_sdk.tokenizer import DummyTokenizer, AbstractTokenizer

logger = logging.getLogger(__name__)


class TestAbstractTokenizer(unittest.TestCase):
    """Test create an instance of the AbstractTokenizer."""

    def test_create_instance(self):
        """Test create instance of the AbstractTokenizer."""
        with self.assertRaises(TypeError) as context:
            _ = AbstractTokenizer()
            assert "Can't instantiate abstract class AbstractTokenizer with abstract methods" in context


class TestDummyTokenizer(unittest.TestCase):
    """Test definition of the tokenizer."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the tokenizer and test setup."""
        cls.tokenizer = DummyTokenizer()

        cls.project = Project(id_=None)
        cls.category = Category(project=cls.project, id_=1)
        cls.document = Document(project=cls.project, category=cls.category, text="Good morning.")

        label_set = LabelSet(id_=2, project=cls.project, categories=[cls.category])
        label = Label(id_=3, text='LabelName', project=cls.project, label_sets=[label_set])
        annotation_set = AnnotationSet(id_=4, document=cls.document, label_set=label_set)
        cls.span = Span(start_offset=0, end_offset=4)

        _ = Annotation(
            document=cls.document,
            is_correct=True,
            annotation_set=annotation_set,
            label=label,
            label_set=label_set,
            spans=[cls.span],
        )

    def test_fit_input(self):
        """Test input for the fit of the tokenizer."""
        with self.assertRaises(AssertionError):
            self.tokenizer.fit(self.document)

    def test_fit_output(self):
        """Test output for the fit of the tokenizer."""
        assert self.tokenizer.fit(self.category) == self.tokenizer

    def test_tokenize_input(self):
        """Test input for the tokenize method."""
        with self.assertRaises(AssertionError):
            self.tokenizer.tokenize(self.project)

    def test_tokenize_output(self):
        """Test output for the tokenize method."""
        assert self.tokenizer.tokenize(self.document) == self.document

    def test_evaluate_input(self):
        """Test input for the evaluate method."""
        with self.assertRaises(AssertionError):
            self.tokenizer.evaluate(self.category)

    def test_evaluate_output_format(self):
        """Test output format for the evaluate method."""
        assert isinstance(self.tokenizer.evaluate(self.document), pd.DataFrame)

    def test_evaluate_output_with_empty_document(self):
        """Test output for the evaluate method with an empty Document."""
        document = Document(project=self.project, category=self.category)
        result = self.tokenizer.evaluate(document)
        assert result.shape == (1, 30)
        assert result.is_correct[0] is None
        assert result.is_found_by_tokenizer.sum() == 0

    def test_evaluate_output_with_document(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        result = self.tokenizer.evaluate(self.document)
        assert result.shape == (2, 30)
        assert result.is_correct.sum() == 1
        assert result.is_found_by_tokenizer.sum() == 0

    def test_evaluate_output_offsets_with_document(self):
        """Test offsets in output for the evaluate method with a Document with 1 Span."""
        result = self.tokenizer.evaluate(self.document)
        assert result.start_offset[0] == self.span.start_offset
        assert result.end_offset[0] == self.span.end_offset
