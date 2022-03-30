"""Test regex tokenizers."""
import logging
import unittest

from konfuzio_sdk.data import Project, Annotation, Document, Label, AnnotationSet, LabelSet, Span, Category
from konfuzio_sdk.tokenizer.regex import RegexTokenizer

logger = logging.getLogger(__name__)


class TestRegexTokenizer(unittest.TestCase):
    """Test the RegexTokenizer."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the tokenizer and test setup."""
        cls.project = Project(id_=None)
        cls.category = Category(project=cls.project, id_=1)
        cls.project.add_category(cls.category)
        cls.document = Document(project=cls.project, category=cls.category, text="Good morning.")

        cls.label_set = LabelSet(id_=2, project=cls.project, categories=[cls.category])
        cls.label = Label(id_=3, text='LabelName', project=cls.project, label_sets=[cls.label_set])
        annotation_set = AnnotationSet(id_=4, document=cls.document, label_set=cls.label_set)
        cls.span = Span(start_offset=0, end_offset=4)

        _ = Annotation(
            document=cls.document,
            is_correct=True,
            annotation_set=annotation_set,
            label=cls.label,
            label_set=cls.label_set,
            spans=[cls.span],
        )

        cls.regex = 'Good'
        cls.tokenizer = RegexTokenizer(regex=cls.regex)

    def test_initialization_missing_argument(self):
        """Test initialization of the class instance without necessary arguments."""
        with self.assertRaises(TypeError):
            _ = RegexTokenizer()

    def test_initialization_input_regex(self):
        """Test default regex."""
        assert self.tokenizer.regex == self.regex

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
            self.tokenizer.tokenize(self.category)

    def test_tokenize_document_with_matching_span(self):
        """
        Test tokenize a Document with Annotation that can be found by the tokenizer.

        This will result in 2 Spans with the same start and end offset but with an Annotation with a different Label.
        """
        document = Document(project=self.project, category=self.category, text="Good morning.")
        annotation_set = AnnotationSet(id_=1, document=document, label_set=self.label_set)
        span = Span(start_offset=0, end_offset=4)
        _ = Annotation(
            document=document,
            is_correct=True,
            annotation_set=annotation_set,
            label=self.label,
            label_set=self.label_set,
            spans=[span],
        )

        self.tokenizer.tokenize(document)
        no_label_annotations = document.annotations(use_correct=False, label=self.project.no_label)
        # tokenizer can create Annotations with Spans that overlap correct Spans
        assert no_label_annotations.__len__() == document.annotations().__len__() == 1
        assert no_label_annotations[0].spans[0] == span
        # assert annotations[0].spans[0].created_by == "human"

    def test_tokenize_document_no_matching_span(self):
        """Test tokenize a Document with Annotation that cannot be found by the tokenizer."""
        document = Document(project=self.project, category=self.category, text="Good morning.")
        annotation_set = AnnotationSet(id_=1, document=document, label_set=self.label_set)
        span = Span(start_offset=0, end_offset=3)
        _ = Annotation(
            document=document,
            is_correct=True,
            annotation_set=annotation_set,
            label=self.label,
            label_set=self.label_set,
            spans=[span],
        )

        self.tokenizer.tokenize(document)
        no_label_annotations = document.annotations(use_correct=False, label=self.project.no_label)
        assert no_label_annotations.__len__() == document.annotations().__len__() == 1
        assert no_label_annotations[0].spans[0] != span
        # assert annotations[0].spans[0].created_by == self.__repr__()

    def test_tokenize_with_empty_document(self):
        """Test tokenize a Document without text."""
        document = Document(project=self.project, category=self.category)
        self.tokenizer.tokenize(document)
        assert document.annotations().__len__() == 0

    def test_evaluate_output_with_document(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        result = self.tokenizer.evaluate(self.document)
        assert result.shape[0] == 1
        assert result.is_correct.sum() == 1
        assert result.is_found_by_tokenizer.sum() == 1

    def test_evaluate_output_offsets_with_document(self):
        """Test offsets in output for the evaluate method with a Document with 1 Span."""
        result = self.tokenizer.evaluate(self.document)
        assert result.start_offset[0] == self.span.start_offset
        assert result.end_offset[0] == self.span.end_offset
