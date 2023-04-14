"""Test base tokenizers."""
import logging
import unittest
from typing import List

import pandas as pd
import pytest
import time

from konfuzio_sdk.data import Project, Annotation, Document, Label, AnnotationSet, LabelSet, Span, Category
from konfuzio_sdk.tokenizer.base import AbstractTokenizer, ListTokenizer, ProcessingStep
from konfuzio_sdk.tokenizer.regex import RegexTokenizer, WhitespaceTokenizer
from tests.variables import OFFLINE_PROJECT, TEST_DOCUMENT_ID

logger = logging.getLogger(__name__)


class TestProcessingStep(unittest.TestCase):
    """Test ProcessingStep."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize a processing step."""
        cls.project = Project(id_=None)
        cls.category = Category(project=cls.project, id_=1)
        cls.document = Document(project=cls.project, category=cls.category, text="Good morning.", id_=1)
        cls.tokenizer = WhitespaceTokenizer()
        cls.processing_step = ProcessingStep(tokenizer=cls.tokenizer, document=cls.document, runtime=0.1)

    def test_eval_dict(self):
        """Test initialization."""
        processing_eval = self.processing_step.eval_dict()
        assert processing_eval['tokenizer_name'] == 'WhitespaceTokenizer: \'[^ \\\\n\\\\t\\\\f]+\''
        assert processing_eval['document_id'] == 1
        assert processing_eval['number_of_pages'] == 1
        assert processing_eval['runtime'] == 0.1


class TestAbstractTokenizer(unittest.TestCase):
    """Test create an instance of the AbstractTokenizer."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the tokenizer and test setup."""
        # DummyTokenizer definition

        class DummyTokenizer(AbstractTokenizer):
            def tokenize(self, document: Document) -> List[Span]:
                assert isinstance(document, Document)
                t0 = time.monotonic()
                self.processing_steps.append(ProcessingStep(self.__repr__(), document, time.monotonic() - t0))
                return []

            def found_spans(self, document: Document):
                pass

            def __eq__(self, other):
                pass

        cls.tokenizer = DummyTokenizer()

        cls.project = Project(id_=None)
        cls.category_1 = Category(project=cls.project, id_=1)
        cls.category_2 = Category(project=cls.project, id_=2)
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
        with pytest.raises(TypeError, match="Can't instantiate abstract class AbstractTokenizer"):
            _ = AbstractTokenizer()

    def test_string_representation(self):
        """Test string representation."""
        assert str(self.tokenizer) == "DummyTokenizer"

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

    def test_evaluate_output_results_with_empty_document(self):
        """Test output for the evaluate method with an empty Document."""
        document = Document(project=self.project, category=self.category_1, text="")
        result = self.tokenizer.evaluate(document)
        assert result.shape[0] == 1
        assert result.is_correct[0] is None
        assert result.tokenizer_true_positive.sum() == 0

    def test_processing_runtime_with_empty_document(self):
        """Test the information of the processing runtime with a Document without text."""
        document = Document(project=self.project, category=self.category_1, text="")
        self.tokenizer.processing_steps = []
        _ = self.tokenizer.evaluate(document)
        processing = self.tokenizer.get_runtime_info()
        assert processing.shape[0] == 1
        assert processing.tokenizer_name[0] == self.tokenizer.__repr__()
        self.assertIsNone(processing.document_id[0])
        assert processing.runtime[0] < 1e-3

    def test_evaluate_output_results_with_document(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        result = self.tokenizer.evaluate(self.document)
        assert result.shape[0] == 2
        assert result.is_correct.sum() == 1
        assert result.tokenizer_true_positive.sum() == 0

    def test_processing_runtime_with_document(self):
        """Test the information of the processing runtime with a Document with text."""
        self.tokenizer.processing_steps = []
        _ = self.tokenizer.evaluate(self.document)
        processing = self.tokenizer.get_runtime_info()
        assert processing.shape[0] == 1
        assert processing.tokenizer_name[0] == str(self.tokenizer)
        self.assertIsNone(processing.document_id[0])
        assert processing.runtime[0] < 1e-3

    def test_evaluate_output_offsets_with_document(self):
        """Test offsets in output for the evaluate method with a Document with 1 Span."""
        result = self.tokenizer.evaluate(self.document)
        assert result.start_offset[0] == self.span.start_offset
        assert result.end_offset[0] == self.span.end_offset

    def test_evaluate_dataset_input(self):
        """Test input for the evaluate_category method."""
        with pytest.raises(TypeError, match='is not iterable'):
            self.tokenizer.evaluate_dataset(self.project)
        with pytest.raises(AssertionError, match='Invalid document type'):
            self.tokenizer.evaluate_dataset([self.project])

    def test_evaluate_dataset_output_without_test_documents(self):
        """Test evaluate an empty list of Documents."""
        with pytest.raises(ValueError, match='No objects to concatenate'):
            self.tokenizer.evaluate_dataset([])

    def test_evaluate_dataset_output_with_test_documents(self):
        """Test evaluate a Category with a test Documents."""
        result = self.tokenizer.evaluate_dataset(self.category_2.test_documents())
        assert len(result.data) == 2
        assert result.data.loc[0]["category_id"] == self.category_2.id_

        # an empty span for the NO_LABEL_SET is always created
        assert result.data.loc[1]["category_id"] is None

    @unittest.skip(reason='removed narrow implementation to evaluate multiple Documents: evaluate_category')
    def test_evaluate_category_input(self):
        """Test input for the evaluate_category method."""
        with self.assertRaises(AssertionError):
            self.tokenizer.evaluate_category(self.project)

    @unittest.skip(reason='removed narrow implementation to evaluate multiple Documents: evaluate_category')
    def test_evaluate_category_output_without_test_documents(self):
        """Test evaluate a Category without test Documents."""
        project = Project(id_=None)
        category = Category(project=project)
        with self.assertRaises(ValueError):
            self.tokenizer.evaluate_category(category)

    @unittest.skip(reason='removed narrow implementation to evaluate multiple Documents: evaluate_category')
    def test_evaluate_category_output_with_test_documents(self):
        """Test evaluate a Category with a test Documents."""
        result = self.tokenizer.evaluate_category(self.category_2)
        assert result.shape[0] == 2
        assert result.category_id.dropna().unique() == self.category_2.id_

    def test_lose_weight(self):
        """Test lose weight."""
        self.tokenizer.processing_steps = []
        _ = self.tokenizer.evaluate(self.document)
        processing = self.tokenizer.get_runtime_info()
        assert processing.shape[0] == 1
        self.tokenizer.lose_weight()
        processing = self.tokenizer.get_runtime_info()
        assert processing.shape[0] == 0


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

    def test_tokenize_input(self):
        """Test input for the tokenize method."""
        with self.assertRaises(AssertionError):
            self.tokenizer.tokenize(self.project)

    @unittest.skip("It's possible to create an Annotation if not all Spans are equal to another Annotation")
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
        assert span_1.regex_matching == []
        assert span_2.regex_matching == []
        assert self.tokenizer.missing_spans(document) == document.spans(use_correct=True)

    def test_tokenize_with_empty_document(self):
        """Test tokenize a Document without text."""
        document = Document(project=self.project, category=self.category_1)

        with pytest.raises(NotImplementedError, match='be tokenized when text is None'):
            self.tokenizer.tokenize(document)

    def test_tokenizer_1(self):
        """Test that tokenizer_1 has only 1 match."""
        result = self.tokenizer_1.evaluate(self.document)
        assert result.is_correct.sum() == 2
        assert result.tokenizer_true_positive.sum() == 1

    def test_tokenizer_2(self):
        """Test that tokenizer_2 has only 1 match."""
        result = self.tokenizer_2.evaluate(self.document)
        assert result.is_correct.sum() == 2
        assert result.tokenizer_true_positive.sum() == 1

    def test_evaluate_list_tokenizer(self):
        """Test that with the combination of the tokenizers is possible to find both correct Spans."""
        result = self.tokenizer.evaluate(self.document)
        assert result.is_correct.sum() == 2
        assert result.tokenizer_true_positive.sum() == 2

    def test_evaluate_list_tokenizer_offsets(self):
        """Test offsets of the result of the tokenizer."""
        result = self.tokenizer.evaluate(self.document)
        assert result.start_offset[0] == self.span_1.start_offset
        assert result.end_offset[0] == self.span_1.end_offset
        assert result.start_offset[1] == self.span_2.start_offset
        assert result.end_offset[1] == self.span_2.end_offset

    def test_test_lose_weight(self):
        """Test reset processing steps for the ListTokenizer (lose_weight)."""
        self.tokenizer.processing_steps = []
        _ = self.tokenizer.evaluate(self.document)
        assert len(self.tokenizer.processing_steps) == 2
        self.tokenizer.lose_weight()
        assert len(self.tokenizer.processing_steps) == 0

    def test_equality_check(self):
        """Test Tokenizer comparison method."""
        whitespace_regex = RegexTokenizer(regex=r"[^ \n\t\f]+")
        list_tokenizer_1 = ListTokenizer(tokenizers=[WhitespaceTokenizer(), RegexTokenizer(regex="a")])
        list_tokenizer_2 = ListTokenizer(tokenizers=[whitespace_regex, RegexTokenizer(regex="a")])
        list_tokenizer_3 = ListTokenizer(tokenizers=[RegexTokenizer(regex="a"), WhitespaceTokenizer()])

        assert WhitespaceTokenizer() == whitespace_regex
        assert list_tokenizer_1 == list_tokenizer_2
        assert list_tokenizer_1 != list_tokenizer_3
        assert RegexTokenizer(regex="a") != RegexTokenizer(regex="b")

    def test_duplicate_check(self):
        """Test handling of Tokenizer duplicates in ListTokenizer."""
        test_tokenizer = ListTokenizer(
            tokenizers=[
                WhitespaceTokenizer(),
                RegexTokenizer(regex="a"),
                RegexTokenizer(regex="b"),
                WhitespaceTokenizer(),
                RegexTokenizer(regex="a"),
                RegexTokenizer(regex=r"[^ \n\t\f]+"),
            ]  # equivalent to WhitespaceTokenizer
        )

        assert len(test_tokenizer.tokenizers) == 3
        assert test_tokenizer.tokenizers == [
            WhitespaceTokenizer(),
            RegexTokenizer(regex="a"),
            RegexTokenizer(regex="b"),
        ]

    def test_processing_runtime_of_list_tokenizer(self):
        """Test that the information of the processing runtime refers to each Tokenizer in the list of Tokenizers."""
        self.tokenizer.processing_steps = []
        _ = self.tokenizer.evaluate(self.document)
        processing = self.tokenizer.get_runtime_info()
        assert processing.shape[0] == 2
        assert processing.tokenizer_name[0] == str(self.tokenizer_1)
        assert processing.tokenizer_name[1] == str(self.tokenizer_2)
        self.assertIsNone(processing.document_id[0])
        self.assertIsNone(processing.document_id[1])
        assert processing.runtime[0] < 1e-3


class TestTokenize(unittest.TestCase):
    """Find all Spans that cannot be detected by a Tokenizer."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the test Project."""
        project = Project(id_=None, project_folder=OFFLINE_PROJECT)
        cls.document = project.get_document_by_id(TEST_DOCUMENT_ID)
        assert len(cls.document.spans()) == 24

    def test_find_missing_spans(self):
        """Find all missing Spans in a Project."""
        tokenizer = RegexTokenizer(regex=r'\d')
        missing_spans = tokenizer.missing_spans(self.document)
        # Span 365 to 366 (Tax ID) can be found be Tokenizer
        # three Spans are not correct and don't need to be found
        # (1 revised and 2 unrevised)
        assert sum([not span.annotation.is_correct for span in self.document.spans()]) == 3
        self.assertEqual(len(missing_spans), 20)
