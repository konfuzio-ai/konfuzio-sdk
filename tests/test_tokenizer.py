"""Test base tokenizers."""
import logging
import unittest

import pandas as pd
import pytest

from konfuzio_sdk.data import Project, Annotation, Document, Label, AnnotationSet, LabelSet, Span, Category
from konfuzio_sdk.tokenizer.base import AbstractTokenizer, ListTokenizer
from konfuzio_sdk.tokenizer.regex import RegexTokenizer

logger = logging.getLogger(__name__)


class TestAbstractTokenizer(unittest.TestCase):
    """Test create an instance of the AbstractTokenizer."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the tokenizer and test setup."""
        # DummyTokenizer definition

        class DummyTokenizer(AbstractTokenizer):
            def fit(self, category: Category):
                assert isinstance(category, Category)
                pass

            def tokenize(self, document: Document):
                assert isinstance(document, Document)
                pass

        cls.tokenizer = DummyTokenizer()

        cls.project = Project(id_=None)
        cls.category_1 = Category(project=cls.project, id_=1)
        cls.category_2 = Category(project=cls.project, id_=2)
        cls.project.add_category(cls.category_1)
        cls.project.add_category(cls.category_2)
        label_set = LabelSet(id_=2, project=cls.project, categories=[cls.category_1, cls.category_2])
        label = Label(id_=3, text='LabelName', project=cls.project, label_sets=[label_set])

        cls.document = Document(project=cls.project, category=cls.category_1, text="Good morning.", dataset_status=3)
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

        cls.document_2 = Document(project=cls.project, category=cls.category_2, text="Good day.", dataset_status=3)
        annotation_set_2 = AnnotationSet(id_=5, document=cls.document_2, label_set=label_set)
        cls.span_2 = Span(start_offset=0, end_offset=4)
        _ = Annotation(
            document=cls.document_2,
            is_correct=True,
            annotation_set=annotation_set_2,
            label=label,
            label_set=label_set,
            spans=[cls.span_2],
        )

    def test_create_instance(self):
        """Test create instance of the AbstractTokenizer."""
        with self.assertRaises(TypeError) as context:
            _ = AbstractTokenizer()
            assert "Can't instantiate abstract class AbstractTokenizer with abstract methods" in context

    def test_fit_input(self):
        """Test input for the fit of the tokenizer."""
        with self.assertRaises(AssertionError):
            self.tokenizer.fit(self.document)

    def test_fit_output(self):
        """Test output for the fit of the tokenizer."""
        self.assertIsNone(self.tokenizer.fit(self.category_1))

    def test_tokenize_input(self):
        """Test input for the tokenize method."""
        with self.assertRaises(AssertionError):
            self.tokenizer.tokenize(self.project)

    def test_tokenize_output(self):
        """Test output for the tokenize method - no Spans added with ."""
        self.tokenizer.tokenize(self.document)
        spans_after_tokenize = [annotation.spans for annotation in self.document.annotations()]
        spans_after_tokenize = [item for sublist in spans_after_tokenize for item in sublist]
        assert spans_after_tokenize == [self.span]

    def test_evaluate_input(self):
        """Test input for the evaluate method."""
        with self.assertRaises(AssertionError):
            self.tokenizer.evaluate(self.category_1)

    def test_evaluate_output_format(self):
        """Test output format for the evaluate method."""
        assert isinstance(self.tokenizer.evaluate(self.document), pd.DataFrame)

    def test_evaluate_output_with_empty_document(self):
        """Test output for the evaluate method with an empty Document."""
        document = Document(project=self.project, category=self.category_1)
        result = self.tokenizer.evaluate(document)
        assert result.shape[0] == 1
        assert result.is_correct[0] is None
        assert result.is_found_by_tokenizer.sum() == 0

    def test_evaluate_output_with_document(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        result = self.tokenizer.evaluate(self.document)
        assert result.shape[0] == 2
        assert result.is_correct.sum() == 1
        assert result.is_found_by_tokenizer.sum() == 0

    def test_evaluate_output_offsets_with_document(self):
        """Test offsets in output for the evaluate method with a Document with 1 Span."""
        result = self.tokenizer.evaluate(self.document)
        assert result.start_offset[0] == self.span.start_offset
        assert result.end_offset[0] == self.span.end_offset

    def test_evaluate_category_input(self):
        """Test input for the evaluate_category method."""
        with self.assertRaises(AssertionError):
            self.tokenizer.evaluate_category(self.project)

    def test_evaluate_category_output_without_test_documents(self):
        """Test evaluate a Category without test Documents."""
        project = Project(id_=None)
        category = Category(project=project)
        with self.assertRaises(ValueError):
            self.tokenizer.evaluate_category(category)

    def test_evaluate_category_output_with_test_documents(self):
        """Test evaluate a Category with a test Documents."""
        result = self.tokenizer.evaluate_category(self.category_2)
        assert result.shape[0] == 2
        assert result.category_id.dropna().unique() == self.category_2.id_

    def test_evaluate_project_input(self):
        """Test input for the evaluate_project method."""
        with self.assertRaises(AssertionError):
            self.tokenizer.evaluate_project(self.category_1)

    def test_evaluate_project_output_without_categories(self):
        """Test evaluate a Project without Categories."""
        project = Project(id_=None)
        with self.assertRaises(ValueError):
            self.tokenizer.evaluate_project(project)

    def test_evaluate_project_output_without_test_documents(self):
        """Test evaluate a Project without test Documents."""
        project = Project(id_=None)
        _ = Category(project=project)
        with self.assertRaises(ValueError):
            self.tokenizer.evaluate_project(project)

    def test_evaluate_project_output_with_test_documents(self):
        """Test evaluate a Project with test Documents."""
        result = self.tokenizer.evaluate_project(self.project)
        assert result.shape[0] == 4
        assert set(result.category_id.dropna().unique()) == set([self.category_1.id_, self.category_2.id_])


class TestListTokenizer(unittest.TestCase):
    """Test ListTokenizer."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the tokenizer and test setup."""
        cls.tokenizer_1 = RegexTokenizer(regex="Good")
        cls.tokenizer_2 = RegexTokenizer(regex="morning")
        cls.tokenizer = ListTokenizer(tokenizers=[cls.tokenizer_1, cls.tokenizer_2])

        cls.project = Project(id_=None)
        cls.category_1 = Category(project=cls.project, id_=1)
        cls.project.add_category(cls.category_1)
        cls.label_set = LabelSet(id_=2, project=cls.project, categories=[cls.category_1])
        cls.label = Label(id_=3, text='LabelName', project=cls.project, label_sets=[cls.label_set])

        cls.document = Document(project=cls.project, category=cls.category_1, text="Good morning.", dataset_status=3)
        annotation_set = AnnotationSet(id_=4, document=cls.document, label_set=cls.label_set)
        cls.span_1 = Span(start_offset=0, end_offset=4)
        cls.span_2 = Span(start_offset=5, end_offset=12)
        _ = Annotation(
            document=cls.document,
            is_correct=True,
            annotation_set=annotation_set,
            label=cls.label,
            label_set=cls.label_set,
            spans=[cls.span_1, cls.span_2],
        )

    def test_fit_input(self):
        """Test input for the fit of the tokenizer."""
        with self.assertRaises(AssertionError):
            self.tokenizer.fit(self.document)

    def test_fit_output(self):
        """Test output for the fit of the tokenizer."""
        self.assertIsNone(self.tokenizer.fit(self.category_1))

    def test_tokenize_input(self):
        """Test input for the tokenize method."""
        with self.assertRaises(AssertionError):
            self.tokenizer.tokenize(self.project)

    @pytest.mark.skip("It's possible to create an Annotation if not all Spans are equal to another Annotation")
    def test_tokenize_document_with_matching_spans(self):
        """
        Test tokenize a Document with Annotation with Spans that can be found by the tokenizer.

        This will result in 0 Spans created by the tokenizer.
        """
        document = Document(project=self.project, category=self.category_1, text="Good morning.")
        annotation_set = AnnotationSet(id_=1, document=document, label_set=self.label_set)
        span_1 = Span(start_offset=0, end_offset=4)
        span_2 = Span(start_offset=5, end_offset=12)
        _ = Annotation(
            id_=1,
            document=document,
            is_correct=True,
            annotation_set=annotation_set,
            label=self.label,
            label_set=self.label_set,
            spans=[span_1, span_2],
        )

        self.tokenizer.tokenize(document)
        no_label_annotations = document.annotations(use_correct=False, label=self.project.no_label)
        assert len(no_label_annotations) == 0

    def test_tokenize_document_no_matching_spans(self):
        """Test tokenize a Document with Annotation with Spans that cannot be found by the tokenizer."""
        document = Document(project=self.project, category=self.category_1, text="Good morning.")
        annotation_set = AnnotationSet(id_=1, document=document, label_set=self.label_set)
        span_1 = Span(start_offset=0, end_offset=3)
        span_2 = Span(start_offset=5, end_offset=11)
        _ = Annotation(
            document=document,
            is_correct=True,
            annotation_set=annotation_set,
            label=self.label,
            label_set=self.label_set,
            spans=[span_1, span_2],
        )

        self.tokenizer.tokenize(document)
        no_label_annotations = document.annotations(use_correct=False, label=self.project.no_label)
        assert len(no_label_annotations) == 2
        assert no_label_annotations[0].spans[0] != span_1
        assert no_label_annotations[1].spans[0] != span_2

    def test_tokenizer_1(self):
        """Test that tokenizer_1 has only 1 match."""
        result = self.tokenizer_1.evaluate(self.document)
        assert result.is_correct.sum() == 2
        assert result.is_found_by_tokenizer.sum() == 1

    def test_tokenizer_2(self):
        """Test that tokenizer_2 has only 1 match."""
        result = self.tokenizer_2.evaluate(self.document)
        assert result.is_correct.sum() == 2
        assert result.is_found_by_tokenizer.sum() == 1

    def test_evaluate_list_tokenizer(self):
        """Test that with the combination of the tokenizers is possible to find both correct Spans."""
        result = self.tokenizer.evaluate(self.document)
        assert result.is_correct.sum() == 2
        assert result.is_found_by_tokenizer.sum() == 2

    def test_evaluate_list_tokenizer_offsets(self):
        """Test offsets of the result of the tokenizer."""
        result = self.tokenizer.evaluate(self.document)
        assert result.start_offset[0] == self.span_1.start_offset
        assert result.end_offset[0] == self.span_1.end_offset
        assert result.start_offset[1] == self.span_2.start_offset
        assert result.end_offset[1] == self.span_2.end_offset
