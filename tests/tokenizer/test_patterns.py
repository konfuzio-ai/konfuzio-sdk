"""Test regex tokenizers."""
import logging
import unittest

import pytest

from konfuzio_sdk.data import Annotation, AnnotationSet, Category, Document, Label, LabelSet, Project, Span
from konfuzio_sdk.samples import LocalTextProject
from konfuzio_sdk.tokenizer.base import ListTokenizer
from konfuzio_sdk.tokenizer.regex import (
    CapitalizedTextTokenizer,
    ColonPrecededTokenizer,
    ConnectedTextTokenizer,
    LineUntilCommaTokenizer,
    NonTextTokenizer,
    NumbersTokenizer,
    RegexTokenizer,
    WhitespaceNoPunctuationTokenizer,
    WhitespaceTokenizer,
)
from tests.variables import OFFLINE_PROJECT, TEST_DOCUMENT_ID

logger = logging.getLogger(__name__)


class TestRegexTokenizer(unittest.TestCase):
    """Test the RegexTokenizer."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the tokenizer and test setup."""
        cls.project = Project(id_=None)
        cls.category = Category(project=cls.project, id_=1)
        cls.category_2 = Category(project=cls.project, id_=2)
        cls.label_set = LabelSet(id_=2, project=cls.project, categories=[cls.category, cls.category_2])
        cls.label = Label(id_=3, text='LabelName', project=cls.project, label_sets=[cls.label_set])

        cls.document = Document(project=cls.project, category=cls.category, text='Good morning.', dataset_status=3)
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

        cls.document_2 = Document(project=cls.project, category=cls.category_2, text='Good day.', dataset_status=3)
        annotation_set_2 = AnnotationSet(id_=5, document=cls.document_2, label_set=cls.label_set)
        cls.span_2 = Span(start_offset=0, end_offset=4)
        _ = Annotation(
            document=cls.document_2,
            is_correct=True,
            annotation_set=annotation_set_2,
            label=cls.label,
            label_set=cls.label_set,
            spans=[cls.span_2],
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

    def test_tokenize_input(self):
        """Test input for the tokenize method."""
        with self.assertRaises(AssertionError):
            self.tokenizer.tokenize(self.category)

    def test_tokenize_document_with_matching_span(self):
        """
        Test tokenize a Document with Annotation that can be found by the tokenizer.

        This will result in 0 Spans created by the tokenizer.
        """
        document = Document(project=self.project, category=self.category, text='Good morning.')
        annotation_set = AnnotationSet(id_=1, document=document, label_set=self.label_set)
        span = Span(start_offset=0, end_offset=4)
        _ = Annotation(
            id_=1,
            document=document,
            is_correct=True,
            annotation_set=annotation_set,
            label=self.label,
            label_set=self.label_set,
            spans=[span],
        )

        self.tokenizer.tokenize(document)
        no_label_annotations = document.annotations(use_correct=False, label=self.project.no_label)
        assert len(no_label_annotations) == 0
        assert span.regex_matching == [self.tokenizer]
        assert self.tokenizer.span_match(span) is True

    def test_tokenize_document_no_matching_span(self):
        """Test tokenize a Document with Annotation that cannot be found by the tokenizer."""
        document = Document(project=self.project, category=self.category, text='Good morning.')
        annotation_set = AnnotationSet(id_=1, document=document, label_set=self.label_set)
        span = Span(start_offset=0, end_offset=3)
        _ = Annotation(
            id_=1,
            document=document,
            is_correct=True,
            annotation_set=annotation_set,
            label=self.label,
            label_set=self.label_set,
            spans=[span],
        )

        self.tokenizer.tokenize(document)
        no_label_annotations = document.annotations(use_correct=False, label=self.project.no_label)
        assert len(no_label_annotations) == len(document.annotations()) == 1
        assert no_label_annotations[0].spans[0] != span
        assert span.regex_matching == []
        # assert annotations[0].spans[0].created_by == self.__repr__()

    def test_tokenize_with_empty_document(self):
        """Test tokenize a Document without text."""
        document = Document(project=self.project, category=self.category)
        with pytest.raises(NotImplementedError, match='be tokenized when text is None'):
            self.tokenizer.tokenize(document)

    def test_evaluate_output_with_document(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        result = self.tokenizer.evaluate(self.document)
        assert result.shape[0] == 1
        assert result.is_correct.sum() == 1
        assert result.tokenizer_true_positive.sum() == 1

    def test_evaluate_output_offsets_with_document(self):
        """Test offsets in output for the evaluate method with a Document with 1 Span."""
        result = self.tokenizer.evaluate(self.document)
        assert result.start_offset[0] == self.span.start_offset
        assert result.end_offset[0] == self.span.end_offset

    def test_span_match(self):
        """Test span_match method to check if offset found by Regex Tokenizer."""
        document = Document(project=self.project, category=self.category, text='Good morning. Good afternoon.')

        annotation_set = AnnotationSet(id_=2, document=document, label_set=self.label_set)
        morning_span = Span(start_offset=5, end_offset=12)
        _ = Annotation(
            id_=2,
            document=document,
            is_correct=True,
            annotation_set=annotation_set,
            label=self.label,
            label_set=self.label_set,
            spans=[morning_span],
        )

        self.tokenizer.tokenize(document)
        no_label_annotations = document.annotations(use_correct=False, label=self.project.no_label)
        assert len(no_label_annotations) == 2
        assert len(document.annotations()) == 1
        assert len(document.annotations(use_correct=False)) == 3

        for span in document.spans(use_correct=False, label=self.project.no_label):
            assert self.tokenizer in span.regex_matching
            assert self.tokenizer.span_match(span)

        assert not self.tokenizer.span_match(morning_span)

        whitespace_tokenizer = WhitespaceTokenizer()

        assert whitespace_tokenizer.span_match(document.spans(use_correct=False)[0])
        assert not whitespace_tokenizer.span_match(morning_span)


class TestTemplateRegexTokenizer(unittest.TestCase):
    """Template for the testings of the tokenizers based on RegexTokenizer."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the tokenizer and test setup."""
        cls.project = Project(id_=None)
        cls.category = Category(project=cls.project, id_=1)
        cls.label_set = LabelSet(id_=2, project=cls.project, categories=[cls.category])
        cls.label = Label(id_=3, text='test', project=cls.project, label_sets=[cls.label_set])

    def _create_artificial_document(self, text, offsets):
        document = Document(project=self.project, category=self.category, text=text)
        annotation_set = AnnotationSet(document=document, label_set=self.label_set)
        spans = []
        for span_offsets in offsets:
            spans.append(Span(start_offset=span_offsets[0], end_offset=span_offsets[1]))
        _ = Annotation(
            document=document,
            is_correct=True,
            annotation_set=annotation_set,
            label=self.label,
            label_set=self.label_set,
            spans=spans,
        )

        return document


class TestWhitespaceTokenizer(TestTemplateRegexTokenizer):
    """Test the WhitespaceTokenizer."""

    tokenizer = WhitespaceTokenizer()

    def test_case_1_group_capitalized_words(self):
        """Test if tokenizer can find a group of words starting with a capitalized character."""
        document = self._create_artificial_document(text='Company A&B GmbH  ', offsets=[(0, 16)])
        assert document.annotations()[0].offset_string == ['Company A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_2_group_capitalized_words_in_the_middle_of_text(self):
        """Test if tokenizer can find a group of words starting with a capitalized character in the middle of text."""
        document = self._create_artificial_document(text='Company is Company A&B GmbH now', offsets=[(11, 27)])
        assert document.annotations()[0].offset_string == ['Company A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_3_group_capitalized_words_in_the_middle_of_text_without_period(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='Company is Company A&B GmbH.', offsets=[(11, 27)])
        assert document.annotations()[0].offset_string == ['Company A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_4_group_specific_capitalized_words(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='Company Company A&B GmbH', offsets=[(8, 24)])
        assert document.annotations()[0].offset_string == ['Company A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_5_group_words_excluding_non_word_characters(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='street Name 1-2b,', offsets=[(0, 11)])
        assert document.annotations()[0].offset_string == ['street Name']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_6_group_non_words(self):
        """Test if tokenizer can find a group of non-word characters."""
        document = self._create_artificial_document(text='To local C-1234 City Name', offsets=[(9, 15)])
        assert document.annotations()[0].offset_string == ['C-1234']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 1
        assert (
            result[result.tokenizer_true_positive == 1].start_offset[0]
            == document.annotations()[0].spans[0].start_offset
        )
        assert (
            result[result.tokenizer_true_positive == 1].end_offset[0] == document.annotations()[0].spans[0].end_offset
        )

    def test_case_7_non_words_excluding_comma_at_end(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='street Name 1-2b,', offsets=[(12, 16)])
        assert document.annotations()[0].offset_string == ['1-2b']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_8_non_words_excluding_period_at_end(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='date 1.2.2022.', offsets=[(5, 13)])
        assert document.annotations()[0].offset_string == ['1.2.2022']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_9_non_words_separated_by_whitespace(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='date 01. 01. 2022', offsets=[(5, 17)])
        assert document.annotations()[0].offset_string == ['01. 01. 2022']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_10_date_with_month_in_the_middle(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='date 1 Jan 2022 ', offsets=[(5, 15)])
        assert document.annotations()[0].offset_string == ['1 Jan 2022']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_11_date_with_month_in_the_beginning(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='date Jan 1, 2022 ', offsets=[(5, 16)])
        assert document.annotations()[0].offset_string == ['Jan 1, 2022']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_12_date_with_month_in_the_beginning(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='code AB 12-3:200', offsets=[(5, 16)])
        assert document.annotations()[0].offset_string == ['AB 12-3:200']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_13_paragraph(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(
            text='This is \na description. Occupies a paragraph.', offsets=[(0, 7), (9, 45)]
        )
        assert document.annotations()[0].offset_string == ['This is', 'a description. Occupies a paragraph.']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_14_sentence_single_line(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='This is a sentence.', offsets=[(0, 19)])
        assert document.annotations()[0].offset_string == ['This is a sentence.']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_15_sentence_multiline(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(
            text="This is \na sentence. It's 1 sentence only.", offsets=[(0, 7), (9, 20)]
        )
        assert document.annotations()[0].offset_string == ['This is', 'a sentence.']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_16_group_of_numbers(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='contact Tel 234 132 134 2', offsets=[(12, 25)])
        assert document.annotations()[0].offset_string == ['234 132 134 2']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_17_group_of_numbers(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='contact +12 234 234 132', offsets=[(8, 23)])
        assert document.annotations()[0].offset_string == ['+12 234 234 132']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0
        assert not self.tokenizer.span_match(document.annotations()[0].spans[0])

    def test_case_18_word_with_spatial_characters(self):
        """Test if tokenizer can find a word with a special character."""
        document = self._create_artificial_document(text='write to: person_name@company.com', offsets=[(10, 33)])
        assert document.annotations()[0].offset_string == ['person_name@company.com']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 1

        start_offset = document.annotations()[0].spans[0].start_offset
        end_offset = document.annotations()[0].spans[0].end_offset

        assert self.tokenizer.span_match(document.annotations()[0].spans[0])
        assert result[result.tokenizer_true_positive == 1].start_offset[0] == start_offset
        assert result[result.tokenizer_true_positive == 1].end_offset[0] == end_offset

    def test_case_19_word_preceded_by_colon(self):
        """Test if tokenizer can find a word preceded by colon."""
        document = self._create_artificial_document(text='write to: name', offsets=[(10, 14)])
        assert document.annotations()[0].offset_string == ['name']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 1
        assert (
            result[result.tokenizer_true_positive == 1].start_offset[0]
            == document.annotations()[0].spans[0].start_offset
        )
        assert (
            result[result.tokenizer_true_positive == 1].end_offset[0] == document.annotations()[0].spans[0].end_offset
        )

    def test_case_20_group_capitalized_words_preceded_by_whitespaces_and_followed_by_comma(self):
        """Test if tokenizer can find a group of capitalized words preceded by whitespace and followed by comma."""
        document = self._create_artificial_document(text='     Company A&B GmbH,', offsets=[(5, 21)])
        assert document.annotations()[0].offset_string == ['Company A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_21_group_of_numbers_preceded_by_capitalized_letter_and_punctuation(self):
        """Test if tokenizer can find a group of numbers preceded by a capitalized character and punctuation."""
        document = self._create_artificial_document(text='N. 1242022 123 ', offsets=[(3, 14)])
        assert document.annotations()[0].offset_string == ['1242022 123']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_22_number_preceded_by_non_capitalized_character(self):
        """Test if tokenizer can find a group of numbers preceded by characters and punctuation."""
        document = self._create_artificial_document(text='Nr. 1242022', offsets=[(4, 11)])
        assert document.annotations()[0].offset_string == ['1242022']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 1

        start_offset = document.annotations()[0].spans[0].start_offset
        end_offset = document.annotations()[0].spans[0].end_offset

        assert self.tokenizer.span_match(document.annotations()[0].spans[0])
        assert result[result.tokenizer_true_positive == 1].start_offset[0] == start_offset
        assert result[result.tokenizer_true_positive == 1].end_offset[0] == end_offset

    def test_case_23_word_preceded_by_colon(self):
        """Test if tokenizer can find a word preceded by colon."""
        document = self._create_artificial_document(text='Phone: 123-123-3', offsets=[(7, 16)])
        assert document.annotations()[0].offset_string == ['123-123-3']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 1
        assert (
            result[result.tokenizer_true_positive == 1].start_offset[0]
            == document.annotations()[0].spans[0].start_offset
        )
        assert (
            result[result.tokenizer_true_positive == 1].end_offset[0] == document.annotations()[0].spans[0].end_offset
        )

    def test_case_24_date_preceded_by_colon(self):
        """Test if tokenizer can find a date preceded by colon."""
        document = self._create_artificial_document(text='Date: 10. May 2020', offsets=[(6, 18)])
        assert document.annotations()[0].offset_string == ['10. May 2020']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_25_group_words_within_whitespace_and_comma(self):
        """Test if tokenizer can find a group of  words within whitespaces and comma."""
        document = self._create_artificial_document(text='\n     Company und A&B GmbH,\n', offsets=[(6, 26)])
        assert document.annotations()[0].offset_string == ['Company und A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0


class TestWhitespaceNoPunctuationTokenizer(TestTemplateRegexTokenizer):
    """Test the WhitespaceNoPunctuationTokenizer."""

    tokenizer = WhitespaceNoPunctuationTokenizer()

    def test_case_1_group_capitalized_words(self):
        """Test if tokenizer can find a group of words starting with a capitalized character."""
        document = self._create_artificial_document(text='Company A&B GmbH  ', offsets=[(0, 16)])
        assert document.annotations()[0].offset_string == ['Company A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_2_group_capitalized_words_in_the_middle_of_text(self):
        """Test if tokenizer can find a group of words starting with a capitalized character in the middle of text."""
        document = self._create_artificial_document(text='Company is Company A&B GmbH now', offsets=[(11, 27)])
        assert document.annotations()[0].offset_string == ['Company A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_3_group_capitalized_words_in_the_middle_of_text_without_period(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='Company is Company A&B GmbH.', offsets=[(11, 27)])
        assert document.annotations()[0].offset_string == ['Company A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_4_group_specific_capitalized_words(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='Company Company A&B GmbH', offsets=[(8, 24)])
        assert document.annotations()[0].offset_string == ['Company A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_5_group_words_excluding_non_word_characters(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='street Name 1-2b,', offsets=[(0, 11)])
        assert document.annotations()[0].offset_string == ['street Name']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_6_group_non_words(self):
        """Test if tokenizer can find a group of non-word characters."""
        document = self._create_artificial_document(text='To local C-1234 City Name', offsets=[(9, 15)])
        assert document.annotations()[0].offset_string == ['C-1234']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 1
        assert (
            result[result.tokenizer_true_positive == 1].start_offset[0]
            == document.annotations()[0].spans[0].start_offset
        )
        assert (
            result[result.tokenizer_true_positive == 1].end_offset[0] == document.annotations()[0].spans[0].end_offset
        )

    def test_case_7_non_words_excluding_comma_at_end(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='street Name 1-2b,', offsets=[(12, 16)])
        assert document.annotations()[0].offset_string == ['1-2b']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 1
        assert (
            result[result.tokenizer_true_positive == 1].start_offset[0]
            == document.annotations()[0].spans[0].start_offset
        )
        assert (
            result[result.tokenizer_true_positive == 1].end_offset[0] == document.annotations()[0].spans[0].end_offset
        )

    def test_case_8_non_words_excluding_period_at_end(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='date 1.2.2022.', offsets=[(5, 13)])
        assert document.annotations()[0].offset_string == ['1.2.2022']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_9_non_words_separated_by_whitespace(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='date 01. 01. 2022', offsets=[(5, 17)])
        assert document.annotations()[0].offset_string == ['01. 01. 2022']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_10_date_with_month_in_the_middle(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='date 1 Jan 2022 ', offsets=[(5, 15)])
        assert document.annotations()[0].offset_string == ['1 Jan 2022']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_11_date_with_month_in_the_beginning(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='date Jan 1, 2022 ', offsets=[(5, 16)])
        assert document.annotations()[0].offset_string == ['Jan 1, 2022']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_12_date_with_month_in_the_beginning(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='code AB 12-3:200', offsets=[(5, 16)])
        assert document.annotations()[0].offset_string == ['AB 12-3:200']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_13_paragraph(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(
            text='This is \na description. Occupies a paragraph.', offsets=[(0, 7), (9, 45)]
        )
        assert document.annotations()[0].offset_string == ['This is', 'a description. Occupies a paragraph.']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_14_sentence_single_line(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='This is a sentence.', offsets=[(0, 19)])
        assert document.annotations()[0].offset_string == ['This is a sentence.']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_15_sentence_multiline(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(
            text="This is \na sentence. It's 1 sentence only.", offsets=[(0, 7), (9, 20)]
        )
        assert document.annotations()[0].offset_string == ['This is', 'a sentence.']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_16_group_of_numbers(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='contact Tel 234 132 134 2', offsets=[(12, 25)])
        assert document.annotations()[0].offset_string == ['234 132 134 2']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_17_group_of_numbers(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='contact +12 234 234 132', offsets=[(8, 23)])
        assert document.annotations()[0].offset_string == ['+12 234 234 132']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_18_word_with_spatial_characters(self):
        """Test if tokenizer can find a word with a special character."""
        document = self._create_artificial_document(text='write to: person_name@company.com', offsets=[(10, 33)])
        assert document.annotations()[0].offset_string == ['person_name@company.com']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_19_word_preceded_by_colon(self):
        """Test if tokenizer can find a word preceded by colon."""
        document = self._create_artificial_document(text='write to: name', offsets=[(10, 14)])
        assert document.annotations()[0].offset_string == ['name']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 1
        assert (
            result[result.tokenizer_true_positive == 1].start_offset[0]
            == document.annotations()[0].spans[0].start_offset
        )
        assert (
            result[result.tokenizer_true_positive == 1].end_offset[0] == document.annotations()[0].spans[0].end_offset
        )

    def test_case_20_group_capitalized_words_preceded_by_whitespaces_and_followed_by_comma(self):
        """Test if tokenizer can find a group of capitalized words preceded by whitespace and followed by comma."""
        document = self._create_artificial_document(text='     Company A&B GmbH,', offsets=[(5, 21)])
        assert document.annotations()[0].offset_string == ['Company A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_21_group_of_numbers_preceded_by_capitalized_letter_and_punctuation(self):
        """Test if tokenizer can find a group of numbers preceded by a capitalized character and punctuation."""
        document = self._create_artificial_document(text='N. 1242022 123 ', offsets=[(3, 14)])
        assert document.annotations()[0].offset_string == ['1242022 123']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_22_number_preceded_by_non_capitalized_character(self):
        """Test if tokenizer can find a group of numbers preceded by characters and punctuation."""
        document = self._create_artificial_document(text='Nr. 1242022', offsets=[(4, 11)])
        assert document.annotations()[0].offset_string == ['1242022']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 1
        assert (
            result[result.tokenizer_true_positive == 1].start_offset[0]
            == document.annotations()[0].spans[0].start_offset
        )
        assert (
            result[result.tokenizer_true_positive == 1].end_offset[0] == document.annotations()[0].spans[0].end_offset
        )

    def test_case_23_word_preceded_by_colon(self):
        """Test if tokenizer can find a word preceded by colon."""
        document = self._create_artificial_document(text='Phone: 123-123-3', offsets=[(7, 16)])
        assert document.annotations()[0].offset_string == ['123-123-3']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 1
        assert (
            result[result.tokenizer_true_positive == 1].start_offset[0]
            == document.annotations()[0].spans[0].start_offset
        )
        assert (
            result[result.tokenizer_true_positive == 1].end_offset[0] == document.annotations()[0].spans[0].end_offset
        )

    def test_case_24_date_preceded_by_colon(self):
        """Test if tokenizer can find a date preceded by colon."""
        document = self._create_artificial_document(text='Date: 10. May 2020', offsets=[(6, 18)])
        assert document.annotations()[0].offset_string == ['10. May 2020']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_25_group_words_within_whitespace_and_comma(self):
        """Test if tokenizer can find a group of  words within whitespaces and comma."""
        document = self._create_artificial_document(text='\n     Company und A&B GmbH,\n', offsets=[(6, 26)])
        assert document.annotations()[0].offset_string == ['Company und A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0


class TestConnectedTextTokenizer(TestTemplateRegexTokenizer):
    """Test the ConnectedTextTokenizer."""

    tokenizer = ConnectedTextTokenizer()

    # Tokenizer can find
    def test_case_1_group_capitalized_words(self):
        """Test if tokenizer can find a group of words starting with a capitalized character."""
        document = self._create_artificial_document(text='Company A&B GmbH  ', offsets=[(0, 16)])
        assert document.annotations()[0].offset_string == ['Company A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 1
        assert (
            result[result.tokenizer_true_positive == 1].start_offset[0]
            == document.annotations()[0].spans[0].start_offset
        )
        assert (
            result[result.tokenizer_true_positive == 1].end_offset[0] == document.annotations()[0].spans[0].end_offset
        )

    # Tokenizer cannot find
    def test_case_2_group_capitalized_words_in_the_middle_of_text(self):
        """Test if tokenizer can find a group of words starting with a capitalized character in the middle of text."""
        document = self._create_artificial_document(text='Company is Company A&B GmbH now', offsets=[(11, 27)])
        assert document.annotations()[0].offset_string == ['Company A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_3_group_capitalized_words_in_the_middle_of_text_without_period(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='Company is Company A&B GmbH.', offsets=[(11, 27)])
        assert document.annotations()[0].offset_string == ['Company A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_4_group_specific_capitalized_words(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='Company Company A&B GmbH', offsets=[(8, 24)])
        assert document.annotations()[0].offset_string == ['Company A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_5_group_words_excluding_non_word_characters(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='street Name 1-2b,', offsets=[(0, 11)])
        assert document.annotations()[0].offset_string == ['street Name']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_6_group_non_words(self):
        """Test if tokenizer can find a group of non-word characters."""
        document = self._create_artificial_document(text='To local C-1234 City Name', offsets=[(9, 15)])
        assert document.annotations()[0].offset_string == ['C-1234']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_7_non_words_excluding_comma_at_end(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='street Name 1-2b,', offsets=[(12, 16)])
        assert document.annotations()[0].offset_string == ['1-2b']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_8_non_words_excluding_period_at_end(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='date 1.2.2022.', offsets=[(5, 13)])
        assert document.annotations()[0].offset_string == ['1.2.2022']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_9_non_words_separated_by_whitespace(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='date 01. 01. 2022', offsets=[(5, 17)])
        assert document.annotations()[0].offset_string == ['01. 01. 2022']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_10_date_with_month_in_the_middle(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='date 1 Jan 2022 ', offsets=[(5, 15)])
        assert document.annotations()[0].offset_string == ['1 Jan 2022']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_11_date_with_month_in_the_beginning(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='date Jan 1, 2022 ', offsets=[(5, 16)])
        assert document.annotations()[0].offset_string == ['Jan 1, 2022']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_12_date_with_month_in_the_beginning(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='code AB 12-3:200', offsets=[(5, 16)])
        assert document.annotations()[0].offset_string == ['AB 12-3:200']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_13_paragraph(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(
            text='This is \na description. Occupies a paragraph.', offsets=[(0, 7), (9, 45)]
        )
        assert document.annotations()[0].offset_string == ['This is', 'a description. Occupies a paragraph.']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 2
        assert (
            result[result.tokenizer_true_positive == 1].start_offset[0]
            == document.annotations()[0].spans[0].start_offset
        )
        assert (
            result[result.tokenizer_true_positive == 1].end_offset[0] == document.annotations()[0].spans[0].end_offset
        )

    def test_case_14_sentence_single_line(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='This is a sentence.', offsets=[(0, 19)])
        assert document.annotations()[0].offset_string == ['This is a sentence.']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 1
        assert (
            result[result.tokenizer_true_positive == 1].start_offset[0]
            == document.annotations()[0].spans[0].start_offset
        )
        assert (
            result[result.tokenizer_true_positive == 1].end_offset[0] == document.annotations()[0].spans[0].end_offset
        )

    def test_case_15_sentence_multiline(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(
            text="This is \na sentence. It's 1 sentence only.", offsets=[(0, 7), (9, 20)]
        )
        assert document.annotations()[0].offset_string == ['This is', 'a sentence.']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 1  # should be 2

    def test_case_16_group_of_numbers(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='contact Tel 234 132 134 2', offsets=[(12, 25)])
        assert document.annotations()[0].offset_string == ['234 132 134 2']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_17_group_of_numbers(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='contact +12 234 234 132', offsets=[(8, 23)])
        assert document.annotations()[0].offset_string == ['+12 234 234 132']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_18_word_with_spatial_characters(self):
        """Test if tokenizer can find a word with a special character."""
        document = self._create_artificial_document(text='write to: person_name@company.com', offsets=[(10, 33)])
        assert document.annotations()[0].offset_string == ['person_name@company.com']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_19_word_preceded_by_colon(self):
        """Test if tokenizer can find a word preceded by colon."""
        document = self._create_artificial_document(text='write to: name', offsets=[(10, 14)])
        assert document.annotations()[0].offset_string == ['name']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_20_group_capitalized_words_preceded_by_whitespaces_and_followed_by_comma(self):
        """Test if tokenizer can find a group of capitalized words preceded by whitespace and followed by comma."""
        document = self._create_artificial_document(text='     Company A&B GmbH,', offsets=[(5, 21)])
        assert document.annotations()[0].offset_string == ['Company A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_21_group_of_numbers_preceded_by_capitalized_letter_and_punctuation(self):
        """Test if tokenizer can find a group of numbers preceded by a capitalized character and punctuation."""
        document = self._create_artificial_document(text='N. 1242022 123 ', offsets=[(3, 14)])
        assert document.annotations()[0].offset_string == ['1242022 123']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_22_number_preceded_by_non_capitalized_character(self):
        """Test if tokenizer can find a group of numbers preceded by characters and punctuation."""
        document = self._create_artificial_document(text='Nr. 1242022', offsets=[(4, 11)])
        assert document.annotations()[0].offset_string == ['1242022']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_23_word_preceded_by_colon(self):
        """Test if tokenizer can find a word preceded by colon."""
        document = self._create_artificial_document(text='Phone: 123-123-3', offsets=[(7, 16)])
        assert document.annotations()[0].offset_string == ['123-123-3']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_24_date_preceded_by_colon(self):
        """Test if tokenizer can find a date preceded by colon."""
        document = self._create_artificial_document(text='Date: 10. May 2020', offsets=[(6, 18)])
        assert document.annotations()[0].offset_string == ['10. May 2020']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_25_group_words_within_whitespace_and_comma(self):
        """Test if tokenizer can find a group of  words within whitespaces and comma."""
        document = self._create_artificial_document(text='\n     Company und A&B GmbH,\n', offsets=[(6, 26)])
        assert document.annotations()[0].offset_string == ['Company und A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0


class TestColonPrecededTokenizer(TestTemplateRegexTokenizer):
    """Test the ColonPrecededTokenizer."""

    tokenizer = ColonPrecededTokenizer()

    # Tokenizer cannot find
    def test_case_1_group_capitalized_words(self):
        """Test if tokenizer can find a group of words starting with a capitalized character."""
        document = self._create_artificial_document(text='Company A&B GmbH  ', offsets=[(0, 16)])
        assert document.annotations()[0].offset_string == ['Company A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_2_group_capitalized_words_in_the_middle_of_text(self):
        """Test if tokenizer can find a group of words starting with a capitalized character in the middle of text."""
        document = self._create_artificial_document(text='Company is Company A&B GmbH now', offsets=[(11, 27)])
        assert document.annotations()[0].offset_string == ['Company A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_3_group_capitalized_words_in_the_middle_of_text_without_period(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='Company is Company A&B GmbH.', offsets=[(11, 27)])
        assert document.annotations()[0].offset_string == ['Company A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_4_group_specific_capitalized_words(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='Company Company A&B GmbH', offsets=[(8, 24)])
        assert document.annotations()[0].offset_string == ['Company A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_5_group_words_excluding_non_word_characters(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='street Name 1-2b,', offsets=[(0, 11)])
        assert document.annotations()[0].offset_string == ['street Name']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_6_group_non_words(self):
        """Test if tokenizer can find a group of non-word characters."""
        document = self._create_artificial_document(text='To local C-1234 City Name', offsets=[(9, 15)])
        assert document.annotations()[0].offset_string == ['C-1234']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_7_non_words_excluding_comma_at_end(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='street Name 1-2b,', offsets=[(12, 16)])
        assert document.annotations()[0].offset_string == ['1-2b']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_8_non_words_excluding_period_at_end(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='date 1.2.2022.', offsets=[(5, 13)])
        assert document.annotations()[0].offset_string == ['1.2.2022']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_9_non_words_separated_by_whitespace(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='date 01. 01. 2022', offsets=[(5, 17)])
        assert document.annotations()[0].offset_string == ['01. 01. 2022']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_10_date_with_month_in_the_middle(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='date 1 Jan 2022 ', offsets=[(5, 15)])
        assert document.annotations()[0].offset_string == ['1 Jan 2022']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_11_date_with_month_in_the_beginning(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='date Jan 1, 2022 ', offsets=[(5, 16)])
        assert document.annotations()[0].offset_string == ['Jan 1, 2022']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_12_date_with_month_in_the_beginning(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='code AB 12-3:200', offsets=[(5, 16)])
        assert document.annotations()[0].offset_string == ['AB 12-3:200']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_13_paragraph(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(
            text='This is \na description. Occupies a paragraph.', offsets=[(0, 7), (9, 45)]
        )
        assert document.annotations()[0].offset_string == ['This is', 'a description. Occupies a paragraph.']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_14_sentence_single_line(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='This is a sentence.', offsets=[(0, 19)])
        assert document.annotations()[0].offset_string == ['This is a sentence.']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_15_sentence_multiline(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(
            text="This is \na sentence. It's 1 sentence only.", offsets=[(0, 7), (9, 20)]
        )
        assert document.annotations()[0].offset_string == ['This is', 'a sentence.']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_16_group_of_numbers(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='contact Tel 234 132 134 2', offsets=[(12, 25)])
        assert document.annotations()[0].offset_string == ['234 132 134 2']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_17_group_of_numbers(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='contact +12 234 234 132', offsets=[(8, 23)])
        assert document.annotations()[0].offset_string == ['+12 234 234 132']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_18_word_with_spatial_characters(self):
        """Test if tokenizer can find a word with a special character."""
        document = self._create_artificial_document(text='write to: person_name@company.com', offsets=[(10, 33)])
        assert document.annotations()[0].offset_string == ['person_name@company.com']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    # Tokenizer can find
    def test_case_19_word_preceded_by_colon(self):
        """Test if tokenizer can find a word preceded by colon."""
        document = self._create_artificial_document(text='write to: name', offsets=[(10, 14)])
        assert document.annotations()[0].offset_string == ['name']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 1
        assert (
            result[result.tokenizer_true_positive == 1].start_offset[0]
            == document.annotations()[0].spans[0].start_offset
        )
        assert (
            result[result.tokenizer_true_positive == 1].end_offset[0] == document.annotations()[0].spans[0].end_offset
        )

    def test_case_20_group_capitalized_words_preceded_by_whitespaces_and_followed_by_comma(self):
        """Test if tokenizer can find a group of capitalized words preceded by whitespace and followed by comma."""
        document = self._create_artificial_document(text='     Company A&B GmbH,', offsets=[(5, 21)])
        assert document.annotations()[0].offset_string == ['Company A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_21_group_of_numbers_preceded_by_capitalized_letter_and_punctuation(self):
        """Test if tokenizer can find a group of numbers preceded by a capitalized character and punctuation."""
        document = self._create_artificial_document(text='N. 1242022 123 ', offsets=[(3, 14)])
        assert document.annotations()[0].offset_string == ['1242022 123']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_22_number_preceded_by_non_capitalized_character(self):
        """Test if tokenizer can find a group of numbers preceded by characters and punctuation."""
        document = self._create_artificial_document(text='Nr. 1242022', offsets=[(4, 11)])
        assert document.annotations()[0].offset_string == ['1242022']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_23_word_preceded_by_colon(self):
        """Test if tokenizer can find a word preceded by colon."""
        document = self._create_artificial_document(text='Phone: 123-123-3', offsets=[(7, 16)])
        assert document.annotations()[0].offset_string == ['123-123-3']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 1
        assert (
            result[result.tokenizer_true_positive == 1].start_offset[0]
            == document.annotations()[0].spans[0].start_offset
        )
        assert (
            result[result.tokenizer_true_positive == 1].end_offset[0] == document.annotations()[0].spans[0].end_offset
        )

    def test_case_24_date_preceded_by_colon(self):
        """Test if tokenizer can find a date preceded by colon."""
        document = self._create_artificial_document(text='Date: 10. May 2020', offsets=[(6, 18)])
        assert document.annotations()[0].offset_string == ['10. May 2020']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 1
        assert (
            result[result.tokenizer_true_positive == 1].start_offset[0]
            == document.annotations()[0].spans[0].start_offset
        )
        assert (
            result[result.tokenizer_true_positive == 1].end_offset[0] == document.annotations()[0].spans[0].end_offset
        )

    def test_case_25_group_words_within_whitespace_and_comma(self):
        """Test if tokenizer can find a group of  words within whitespaces and comma."""
        document = self._create_artificial_document(text='\n     Company und A&B GmbH,\n', offsets=[(6, 26)])
        assert document.annotations()[0].offset_string == ['Company und A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0


class TestCapitalizedTextTokenizer(TestTemplateRegexTokenizer):
    """Test the CapitalizedTextTokenizer."""

    tokenizer = CapitalizedTextTokenizer()

    def test_case_1_group_capitalized_words(self):
        """Test if tokenizer can find a group of words starting with a capitalized character."""
        document = self._create_artificial_document(text='Company A&B GmbH  ', offsets=[(0, 16)])
        assert document.annotations()[0].offset_string == ['Company A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 1

        start_offset = document.annotations()[0].spans[0].start_offset
        end_offset = document.annotations()[0].spans[0].end_offset

        assert self.tokenizer.span_match(document.annotations()[0].spans[0])
        assert result[result.tokenizer_true_positive == 1].start_offset[0] == start_offset
        assert result[result.tokenizer_true_positive == 1].end_offset[0] == end_offset

    def test_case_2_group_capitalized_words_in_the_middle_of_text(self):
        """Test if tokenizer can find a group of words starting with a capitalized character in the middle of text."""
        document = self._create_artificial_document(text='Company is Company A&B GmbH now', offsets=[(11, 27)])
        assert document.annotations()[0].offset_string == ['Company A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 1

        start_offset = document.annotations()[0].spans[0].start_offset
        end_offset = document.annotations()[0].spans[0].end_offset

        assert self.tokenizer.span_match(document.annotations()[0].spans[0])
        assert result[result.tokenizer_true_positive == 1].start_offset[0] == start_offset
        assert result[result.tokenizer_true_positive == 1].end_offset[0] == end_offset

    def test_case_3_group_capitalized_words_in_the_middle_of_text_without_period(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='Company is Company A&B GmbH.', offsets=[(11, 27)])
        assert document.annotations()[0].offset_string == ['Company A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 1

        start_offset = document.annotations()[0].spans[0].start_offset
        end_offset = document.annotations()[0].spans[0].end_offset

        assert self.tokenizer.span_match(document.annotations()[0].spans[0])
        assert result[result.tokenizer_true_positive == 1].start_offset[0] == start_offset
        assert result[result.tokenizer_true_positive == 1].end_offset[0] == end_offset

    # Tokenizer cannot find
    def test_case_4_group_specific_capitalized_words(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='Company Company A&B GmbH', offsets=[(8, 24)])
        assert document.annotations()[0].offset_string == ['Company A&B GmbH']
        result = self.tokenizer.evaluate(document)

        assert not self.tokenizer.span_match(document.annotations()[0].spans[0])
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_5_group_words_excluding_non_word_characters(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='street Name 1-2b,', offsets=[(0, 11)])
        assert document.annotations()[0].offset_string == ['street Name']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_6_group_non_words(self):
        """Test if tokenizer can find a group of non-word characters."""
        document = self._create_artificial_document(text='To local C-1234 City Name', offsets=[(9, 15)])
        assert document.annotations()[0].offset_string == ['C-1234']
        result = self.tokenizer.evaluate(document)

        assert not self.tokenizer.span_match(document.annotations()[0].spans[0])
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_7_non_words_excluding_comma_at_end(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='street Name 1-2b,', offsets=[(12, 16)])
        assert document.annotations()[0].offset_string == ['1-2b']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_8_non_words_excluding_period_at_end(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='date 1.2.2022.', offsets=[(5, 13)])
        assert document.annotations()[0].offset_string == ['1.2.2022']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_9_non_words_separated_by_whitespace(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='date 01. 01. 2022', offsets=[(5, 17)])
        assert document.annotations()[0].offset_string == ['01. 01. 2022']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_10_date_with_month_in_the_middle(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='date 1 Jan 2022 ', offsets=[(5, 15)])
        assert document.annotations()[0].offset_string == ['1 Jan 2022']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_11_date_with_month_in_the_beginning(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='date Jan 1, 2022 ', offsets=[(5, 16)])
        assert document.annotations()[0].offset_string == ['Jan 1, 2022']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_12_date_with_month_in_the_beginning(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='code AB 12-3:200', offsets=[(5, 16)])
        assert document.annotations()[0].offset_string == ['AB 12-3:200']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_13_paragraph(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(
            text='This is \na description. Occupies a paragraph.', offsets=[(0, 7), (9, 45)]
        )
        assert document.annotations()[0].offset_string == ['This is', 'a description. Occupies a paragraph.']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_14_sentence_single_line(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='This is a sentence.', offsets=[(0, 19)])
        assert document.annotations()[0].offset_string == ['This is a sentence.']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_15_sentence_multiline(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(
            text="This is \na sentence. It's 1 sentence only.", offsets=[(0, 7), (9, 20)]
        )
        assert document.annotations()[0].offset_string == ['This is', 'a sentence.']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_16_group_of_numbers(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='contact Tel 234 132 134 2', offsets=[(12, 25)])
        assert document.annotations()[0].offset_string == ['234 132 134 2']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_17_group_of_numbers(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='contact +12 234 234 132', offsets=[(8, 23)])
        assert document.annotations()[0].offset_string == ['+12 234 234 132']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_18_word_with_spatial_characters(self):
        """Test if tokenizer can find a word with a special character."""
        document = self._create_artificial_document(text='write to: person_name@company.com', offsets=[(10, 33)])
        assert document.annotations()[0].offset_string == ['person_name@company.com']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_19_word_preceded_by_colon(self):
        """Test if tokenizer can find a word preceded by colon."""
        document = self._create_artificial_document(text='write to: name', offsets=[(10, 14)])
        assert document.annotations()[0].offset_string == ['name']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_20_group_capitalized_words_preceded_by_whitespaces_and_followed_by_comma(self):
        """Test if tokenizer can find a group of capitalized words preceded by whitespace and followed by comma."""
        document = self._create_artificial_document(text='     Company A&B GmbH,', offsets=[(5, 21)])
        assert document.annotations()[0].offset_string == ['Company A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 1
        assert (
            result[result.tokenizer_true_positive == 1].start_offset[0]
            == document.annotations()[0].spans[0].start_offset
        )
        assert (
            result[result.tokenizer_true_positive == 1].end_offset[0] == document.annotations()[0].spans[0].end_offset
        )

    def test_case_21_group_of_numbers_preceded_by_capitalized_letter_and_punctuation(self):
        """Test if tokenizer can find a group of numbers preceded by a capitalized character and punctuation."""
        document = self._create_artificial_document(text='N. 1242022 123 ', offsets=[(3, 14)])
        assert document.annotations()[0].offset_string == ['1242022 123']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_22_number_preceded_by_non_capitalized_character(self):
        """Test if tokenizer can find a group of numbers preceded by characters and punctuation."""
        document = self._create_artificial_document(text='Nr. 1242022', offsets=[(4, 11)])
        assert document.annotations()[0].offset_string == ['1242022']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_23_word_preceded_by_colon(self):
        """Test if tokenizer can find a word preceded by colon."""
        document = self._create_artificial_document(text='Phone: 123-123-3', offsets=[(7, 16)])
        assert document.annotations()[0].offset_string == ['123-123-3']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_24_date_preceded_by_colon(self):
        """Test if tokenizer can find a date preceded by colon."""
        document = self._create_artificial_document(text='Date: 10. May 2020', offsets=[(6, 18)])
        assert document.annotations()[0].offset_string == ['10. May 2020']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_25_group_words_within_whitespace_and_comma(self):
        """Test if tokenizer can find a group of  words within whitespaces and comma."""
        document = self._create_artificial_document(text='\n     Company und A&B GmbH,\n', offsets=[(6, 26)])
        assert document.annotations()[0].offset_string == ['Company und A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0


class TestNonTextTokenizer(TestTemplateRegexTokenizer):
    """Test the NonTextTokenizer."""

    tokenizer = NonTextTokenizer()

    # Tokenizer can find
    def test_case_9_non_words_separated_by_whitespace(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='date 01. 01. 2022', offsets=[(5, 17)])
        assert document.annotations()[0].offset_string == ['01. 01. 2022']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 1
        assert (
            result[result.tokenizer_true_positive == 1].start_offset[0]
            == document.annotations()[0].spans[0].start_offset
        )
        assert (
            result[result.tokenizer_true_positive == 1].end_offset[0] == document.annotations()[0].spans[0].end_offset
        )

    # Tokenizer cannot find
    def test_case_1_group_capitalized_words(self):
        """Test if tokenizer can find a group of words starting with a capitalized character."""
        document = self._create_artificial_document(text='Company A&B GmbH  ', offsets=[(0, 16)])
        assert document.annotations()[0].offset_string == ['Company A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_2_group_capitalized_words_in_the_middle_of_text(self):
        """Test if tokenizer can find a group of words starting with a capitalized character in the middle of text."""
        document = self._create_artificial_document(text='Company is Company A&B GmbH now', offsets=[(11, 27)])
        assert document.annotations()[0].offset_string == ['Company A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_3_group_capitalized_words_in_the_middle_of_text_without_period(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='Company is Company A&B GmbH.', offsets=[(11, 27)])
        assert document.annotations()[0].offset_string == ['Company A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_4_group_specific_capitalized_words(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='Company Company A&B GmbH', offsets=[(8, 24)])
        assert document.annotations()[0].offset_string == ['Company A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_5_group_words_excluding_non_word_characters(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='street Name 1-2b,', offsets=[(0, 11)])
        assert document.annotations()[0].offset_string == ['street Name']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_6_group_non_words(self):
        """Test if tokenizer can find a group of non-word characters."""
        document = self._create_artificial_document(text='To local C-1234 City Name', offsets=[(9, 15)])
        assert document.annotations()[0].offset_string == ['C-1234']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_7_non_words_excluding_comma_at_end(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='street Name 1-2b,', offsets=[(12, 16)])
        assert document.annotations()[0].offset_string == ['1-2b']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_8_non_words_excluding_period_at_end(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='date 1.2.2022.', offsets=[(5, 13)])
        assert document.annotations()[0].offset_string == ['1.2.2022']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_10_date_with_month_in_the_middle(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='date 1 Jan 2022 ', offsets=[(5, 15)])
        assert document.annotations()[0].offset_string == ['1 Jan 2022']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_11_date_with_month_in_the_beginning(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='date Jan 1, 2022 ', offsets=[(5, 16)])
        assert document.annotations()[0].offset_string == ['Jan 1, 2022']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_12_date_with_month_in_the_beginning(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='code AB 12-3:200', offsets=[(5, 16)])
        assert document.annotations()[0].offset_string == ['AB 12-3:200']
        result = self.tokenizer.evaluate(document)
        assert (
            result[result.tokenizer_true_positive == 1].start_offset[0]
            == document.annotations()[0].spans[0].start_offset
        )
        assert (
            result[result.tokenizer_true_positive == 1].end_offset[0] == document.annotations()[0].spans[0].end_offset
        )

    def test_case_13_paragraph(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(
            text='This is \na description. Occupies a paragraph.', offsets=[(0, 7), (9, 45)]
        )
        assert document.annotations()[0].offset_string == ['This is', 'a description. Occupies a paragraph.']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_14_sentence_single_line(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='This is a sentence.', offsets=[(0, 19)])
        assert document.annotations()[0].offset_string == ['This is a sentence.']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_15_sentence_multiline(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(
            text="This is \na sentence. It's 1 sentence only.", offsets=[(0, 7), (9, 20)]
        )
        assert document.annotations()[0].offset_string == ['This is', 'a sentence.']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_16_group_of_numbers(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='contact Tel 234 132 134 2', offsets=[(12, 25)])
        assert document.annotations()[0].offset_string == ['234 132 134 2']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 1
        assert (
            result[result.tokenizer_true_positive == 1].start_offset[0]
            == document.annotations()[0].spans[0].start_offset
        )
        assert (
            result[result.tokenizer_true_positive == 1].end_offset[0] == document.annotations()[0].spans[0].end_offset
        )

    def test_case_17_group_of_numbers(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='contact +12 234 234 132', offsets=[(8, 23)])
        assert document.annotations()[0].offset_string == ['+12 234 234 132']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_18_word_with_spatial_characters(self):
        """Test if tokenizer can find a word with a special character."""
        document = self._create_artificial_document(text='write to: person_name@company.com', offsets=[(10, 33)])
        assert document.annotations()[0].offset_string == ['person_name@company.com']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_19_word_preceded_by_colon(self):
        """Test if tokenizer can find a word preceded by colon."""
        document = self._create_artificial_document(text='write to: name', offsets=[(10, 14)])
        assert document.annotations()[0].offset_string == ['name']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_20_group_capitalized_words_preceded_by_whitespaces_and_followed_by_comma(self):
        """Test if tokenizer can find a group of capitalized words preceded by whitespace and followed by comma."""
        document = self._create_artificial_document(text='     Company A&B GmbH,', offsets=[(5, 21)])
        assert document.annotations()[0].offset_string == ['Company A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_21_group_of_numbers_preceded_by_capitalized_letter_and_punctuation(self):
        """Test if tokenizer can find a group of numbers preceded by a capitalized character and punctuation."""
        document = self._create_artificial_document(text='N. 1242022 123 ', offsets=[(3, 14)])
        assert document.annotations()[0].offset_string == ['1242022 123']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_22_number_preceded_by_non_capitalized_character(self):
        """Test if tokenizer can find a group of numbers preceded by characters and punctuation."""
        document = self._create_artificial_document(text='Nr. 1242022', offsets=[(4, 11)])
        assert document.annotations()[0].offset_string == ['1242022']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 1
        assert (
            result[result.tokenizer_true_positive == 1].start_offset[0]
            == document.annotations()[0].spans[0].start_offset
        )
        assert (
            result[result.tokenizer_true_positive == 1].end_offset[0] == document.annotations()[0].spans[0].end_offset
        )

    def test_case_23_word_preceded_by_colon(self):
        """Test if tokenizer can find a word preceded by colon."""
        document = self._create_artificial_document(text='Phone: 123-123-3', offsets=[(7, 16)])
        assert document.annotations()[0].offset_string == ['123-123-3']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 1
        assert (
            result[result.tokenizer_true_positive == 1].start_offset[0]
            == document.annotations()[0].spans[0].start_offset
        )
        assert (
            result[result.tokenizer_true_positive == 1].end_offset[0] == document.annotations()[0].spans[0].end_offset
        )

    def test_case_24_date_preceded_by_colon(self):
        """Test if tokenizer can find a date preceded by colon."""
        document = self._create_artificial_document(text='Date: 10. May 2020', offsets=[(6, 18)])
        assert document.annotations()[0].offset_string == ['10. May 2020']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_25_group_words_within_whitespace_and_comma(self):
        """Test if tokenizer can find a group of  words within whitespaces and comma."""
        document = self._create_artificial_document(text='\n     Company und A&B GmbH,\n', offsets=[(6, 26)])
        assert document.annotations()[0].offset_string == ['Company und A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0


class TestNumbersTokenizer(TestTemplateRegexTokenizer):
    """Test the NumbersTokenizer."""

    tokenizer = NumbersTokenizer()

    def test_case_1_group_capitalized_words(self):
        """Test if tokenizer can find a group of words starting with a capitalized character."""
        document = self._create_artificial_document(text='Company A&B GmbH  ', offsets=[(0, 16)])
        assert document.annotations()[0].offset_string == ['Company A&B GmbH']
        result = self.tokenizer.evaluate(document)

        assert not self.tokenizer.span_match(document.annotations()[0].spans[0])
        assert result.tokenizer_true_positive.sum() == 0

    # Tokenizer cannot find
    def test_case_2_group_capitalized_words_in_the_middle_of_text(self):
        """Test if tokenizer can find a group of words starting with a capitalized character in the middle of text."""
        document = self._create_artificial_document(text='Company is Company A&B GmbH now', offsets=[(11, 27)])
        assert document.annotations()[0].offset_string == ['Company A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_3_group_capitalized_words_in_the_middle_of_text_without_period(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='Company is Company A&B GmbH.', offsets=[(11, 27)])
        assert document.annotations()[0].offset_string == ['Company A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_4_group_specific_capitalized_words(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='Company Company A&B GmbH', offsets=[(8, 24)])
        assert document.annotations()[0].offset_string == ['Company A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_5_group_words_excluding_non_word_characters(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='street Name 1-2b,', offsets=[(0, 11)])
        assert document.annotations()[0].offset_string == ['street Name']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_6_group_non_words(self):
        """Test if tokenizer can find a group of non-word characters."""
        document = self._create_artificial_document(text='To local C-1234 City Name', offsets=[(9, 15)])
        assert document.annotations()[0].offset_string == ['C-1234']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_7_non_words_excluding_comma_at_end(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='street Name 1-2b,', offsets=[(12, 16)])
        assert document.annotations()[0].offset_string == ['1-2b']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_8_non_words_excluding_period_at_end(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='date 1.2.2022.', offsets=[(5, 13)])
        assert document.annotations()[0].offset_string == ['1.2.2022']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_9_non_words_separated_by_whitespace(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='date 01. 01. 2022', offsets=[(5, 17)])
        assert document.annotations()[0].offset_string == ['01. 01. 2022']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_10_date_with_month_in_the_middle(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='date 1 Jan 2022 ', offsets=[(5, 15)])
        assert document.annotations()[0].offset_string == ['1 Jan 2022']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_11_date_with_month_in_the_beginning(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='date Jan 1, 2022 ', offsets=[(5, 16)])
        assert document.annotations()[0].offset_string == ['Jan 1, 2022']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_12_date_with_month_in_the_beginning(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='code AB 12-3:200', offsets=[(5, 16)])
        assert document.annotations()[0].offset_string == ['AB 12-3:200']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_13_paragraph(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(
            text='This is \na description. Occupies a paragraph.', offsets=[(0, 7), (9, 45)]
        )
        assert document.annotations()[0].offset_string == ['This is', 'a description. Occupies a paragraph.']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_14_sentence_single_line(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='This is a sentence.', offsets=[(0, 19)])
        assert document.annotations()[0].offset_string == ['This is a sentence.']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_15_sentence_multiline(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(
            text="This is \na sentence. It's 1 sentence only.", offsets=[(0, 7), (9, 20)]
        )
        assert document.annotations()[0].offset_string == ['This is', 'a sentence.']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_16_group_of_numbers(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='contact Tel 234 132 134 2', offsets=[(12, 25)])
        assert document.annotations()[0].offset_string == ['234 132 134 2']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_17_group_of_numbers(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='contact +12 234 234 132', offsets=[(8, 23)])
        assert document.annotations()[0].offset_string == ['+12 234 234 132']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_18_word_with_spatial_characters(self):
        """Test if tokenizer can find a word with a special character."""
        document = self._create_artificial_document(text='write to: person_name@company.com', offsets=[(10, 33)])
        assert document.annotations()[0].offset_string == ['person_name@company.com']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_19_word_preceded_by_colon(self):
        """Test if tokenizer can find a word preceded by colon."""
        document = self._create_artificial_document(text='write to: name', offsets=[(10, 14)])
        assert document.annotations()[0].offset_string == ['name']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_20_group_capitalized_words_preceded_by_whitespaces_and_followed_by_comma(self):
        """Test if tokenizer can find a group of capitalized words preceded by whitespace and followed by comma."""
        document = self._create_artificial_document(text='     Company A&B GmbH,', offsets=[(5, 21)])
        assert document.annotations()[0].offset_string == ['Company A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_21_group_of_numbers_preceded_by_capitalized_letter_and_punctuation(self):
        """Test if tokenizer can find a group of numbers preceded by a capitalized character and punctuation."""
        document = self._create_artificial_document(text='N. 1242022 123 ', offsets=[(3, 14)])
        assert document.annotations()[0].offset_string == ['1242022 123']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 1
        assert (
            result[result.tokenizer_true_positive == 1].start_offset[0]
            == document.annotations()[0].spans[0].start_offset
        )
        assert (
            result[result.tokenizer_true_positive == 1].end_offset[0] == document.annotations()[0].spans[0].end_offset
        )

    def test_case_22_number_preceded_by_non_capitalized_character(self):
        """Test if tokenizer can find a group of numbers preceded by characters and punctuation."""
        document = self._create_artificial_document(text='Nr. 1242022', offsets=[(4, 11)])
        assert document.annotations()[0].offset_string == ['1242022']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_23_word_preceded_by_colon(self):
        """Test if tokenizer can find a word preceded by colon."""
        document = self._create_artificial_document(text='Phone: 123-123-3', offsets=[(7, 16)])
        assert document.annotations()[0].offset_string == ['123-123-3']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_24_date_preceded_by_colon(self):
        """Test if tokenizer can find a date preceded by colon."""
        document = self._create_artificial_document(text='Date: 10. May 2020', offsets=[(6, 18)])
        assert document.annotations()[0].offset_string == ['10. May 2020']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_25_group_words_within_whitespace_and_comma(self):
        """Test if tokenizer can find a group of  words within whitespaces and comma."""
        document = self._create_artificial_document(text='\n     Company und A&B GmbH,\n', offsets=[(6, 26)])
        assert document.annotations()[0].offset_string == ['Company und A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0


class TestLineUntilCommaTokenizer(TestTemplateRegexTokenizer):
    """Test the LineUntilCommaTokenizer."""

    tokenizer = LineUntilCommaTokenizer()

    def test_case_1_group_capitalized_words(self):
        """Test if tokenizer can find a group of words starting with a capitalized character."""
        document = self._create_artificial_document(text='Company A&B GmbH  ', offsets=[(0, 16)])
        assert document.annotations()[0].offset_string == ['Company A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    # Tokenizer cannot find
    def test_case_2_group_capitalized_words_in_the_middle_of_text(self):
        """Test if tokenizer can find a group of words starting with a capitalized character in the middle of text."""
        document = self._create_artificial_document(text='Company is Company A&B GmbH now', offsets=[(11, 27)])
        assert document.annotations()[0].offset_string == ['Company A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_3_group_capitalized_words_in_the_middle_of_text_without_period(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='Company is Company A&B GmbH.', offsets=[(11, 27)])
        assert document.annotations()[0].offset_string == ['Company A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_4_group_specific_capitalized_words(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='Company Company A&B GmbH', offsets=[(8, 24)])
        assert document.annotations()[0].offset_string == ['Company A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_5_group_words_excluding_non_word_characters(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='street Name 1-2b,', offsets=[(0, 11)])
        assert document.annotations()[0].offset_string == ['street Name']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_6_group_non_words(self):
        """Test if tokenizer can find a group of non-word characters."""
        document = self._create_artificial_document(text='To local C-1234 City Name', offsets=[(9, 15)])
        assert document.annotations()[0].offset_string == ['C-1234']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_7_non_words_excluding_comma_at_end(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='street Name 1-2b,', offsets=[(12, 16)])
        assert document.annotations()[0].offset_string == ['1-2b']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_8_non_words_excluding_period_at_end(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='date 1.2.2022.', offsets=[(5, 13)])
        assert document.annotations()[0].offset_string == ['1.2.2022']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_9_non_words_separated_by_whitespace(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='date 01. 01. 2022', offsets=[(5, 17)])
        assert document.annotations()[0].offset_string == ['01. 01. 2022']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_10_date_with_month_in_the_middle(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='date 1 Jan 2022 ', offsets=[(5, 15)])
        assert document.annotations()[0].offset_string == ['1 Jan 2022']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_11_date_with_month_in_the_beginning(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='date Jan 1, 2022 ', offsets=[(5, 16)])
        assert document.annotations()[0].offset_string == ['Jan 1, 2022']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_12_date_with_month_in_the_beginning(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='code AB 12-3:200', offsets=[(5, 16)])
        assert document.annotations()[0].offset_string == ['AB 12-3:200']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_13_paragraph(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(
            text='This is \na description. Occupies a paragraph.', offsets=[(0, 7), (9, 45)]
        )
        assert document.annotations()[0].offset_string == ['This is', 'a description. Occupies a paragraph.']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_14_sentence_single_line(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='This is a sentence.', offsets=[(0, 19)])
        assert document.annotations()[0].offset_string == ['This is a sentence.']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_15_sentence_multiline(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(
            text="This is \na sentence. It's 1 sentence only.", offsets=[(0, 7), (9, 20)]
        )
        assert document.annotations()[0].offset_string == ['This is', 'a sentence.']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_16_group_of_numbers(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='contact Tel 234 132 134 2', offsets=[(12, 25)])
        assert document.annotations()[0].offset_string == ['234 132 134 2']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_17_group_of_numbers(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text='contact +12 234 234 132', offsets=[(8, 23)])
        assert document.annotations()[0].offset_string == ['+12 234 234 132']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_18_word_with_spatial_characters(self):
        """Test if tokenizer can find a word with a special character."""
        document = self._create_artificial_document(text='write to: person_name@company.com', offsets=[(10, 33)])
        assert document.annotations()[0].offset_string == ['person_name@company.com']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_19_word_preceded_by_colon(self):
        """Test if tokenizer can find a word preceded by colon."""
        document = self._create_artificial_document(text='write to: name', offsets=[(10, 14)])
        assert document.annotations()[0].offset_string == ['name']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_20_group_capitalized_words_preceded_by_whitespaces_and_followed_by_comma(self):
        """Test if tokenizer can find a group of capitalized words preceded by whitespace and followed by comma."""
        document = self._create_artificial_document(text='     Company A&B GmbH,', offsets=[(5, 21)])
        assert document.annotations()[0].offset_string == ['Company A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_21_group_of_numbers_preceded_by_capitalized_letter_and_punctuation(self):
        """Test if tokenizer can find a group of numbers preceded by a capitalized character and punctuation."""
        document = self._create_artificial_document(text='N. 1242022 123 ', offsets=[(3, 14)])
        assert document.annotations()[0].offset_string == ['1242022 123']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_22_number_preceded_by_non_capitalized_character(self):
        """Test if tokenizer can find a group of numbers preceded by characters and punctuation."""
        document = self._create_artificial_document(text='Nr. 1242022', offsets=[(4, 11)])
        assert document.annotations()[0].offset_string == ['1242022']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_23_word_preceded_by_colon(self):
        """Test if tokenizer can find a word preceded by colon."""
        document = self._create_artificial_document(text='Phone: 123-123-3', offsets=[(7, 16)])
        assert document.annotations()[0].offset_string == ['123-123-3']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_24_date_preceded_by_colon(self):
        """Test if tokenizer can find a date preceded by colon."""
        document = self._create_artificial_document(text='Date: 10. May 2020', offsets=[(6, 18)])
        assert document.annotations()[0].offset_string == ['10. May 2020']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 0

    def test_case_25_group_words_within_whitespace_and_comma(self):
        """Test if tokenizer can find a group of  words within whitespaces and comma."""
        document = self._create_artificial_document(text='\n     Company und A&B GmbH,\n', offsets=[(6, 26)])
        assert document.annotations()[0].offset_string == ['Company und A&B GmbH']
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 1
        assert (
            result[result.tokenizer_true_positive == 1].start_offset[0]
            == document.annotations()[0].spans[0].start_offset
        )
        assert (
            result[result.tokenizer_true_positive == 1].end_offset[0] == document.annotations()[0].spans[0].end_offset
        )


class TestAutomatedRegexTokenizer(TestTemplateRegexTokenizer):
    """Test the AutomatedRegexTokenizer."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the tokenizer and test setup."""
        cls.project = LocalTextProject()

    def setUp(self) -> None:
        """Build a ListTokenizer and rely on label.find_regex."""
        self.tokenizer = ListTokenizer(tokenizers=[WhitespaceTokenizer()])
        self.category = self.project.get_category_by_id(id_=1)
        for label in self.category.labels:
            for regex in label.find_regex(category=self.category):
                self.tokenizer.tokenizers.append(RegexTokenizer(regex=regex))

    def test_1_tokenizers_added(self):
        """Test to find one additional tokenizer in addition to the WhitespaceTokenizer."""
        assert len(self.tokenizer.tokenizers) == 4

    def test_2_find_what_default_tokenizer_misses(self):
        """Test if tokenizer can find what cannot be found by the WhitespaceTokenizer."""
        document = self.project.local_training_document
        result = self.tokenizer.evaluate(document)
        assert result.tokenizer_true_positive.sum() == 3
        assert (
            result[result.tokenizer_true_positive == 1].start_offset[0]
            == document.annotations()[0].spans[0].start_offset
        )
        assert (
            result[result.tokenizer_true_positive == 1].end_offset[0] == document.annotations()[0].spans[0].end_offset
        )

    def test_3_lose_weight(self):
        """Test Label lose_weight after regex search."""
        label = self.category.project.get_label_by_name('DefaultLabelName')

        _ = label.regex(categories=[self.category])

        assert label._evaluations != {}
        assert label._tokens != {}
        assert label._regex != {}

        label.lose_weight()

        label = self.category.project.get_label_by_name('DefaultLabelName')
        assert label._evaluations == {}
        assert label._tokens == {}
        assert label._regex == {}


class TestMissingSpans(unittest.TestCase):
    """Find all Spans that cannot be detected by a Tokenizer."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the test Project."""
        project = Project(id_=None, project_folder=OFFLINE_PROJECT)
        cls.document = project.get_document_by_id(TEST_DOCUMENT_ID)

    def test_find_missing_spans(self):
        """Find all missing Spans in a Project."""
        tokenizer = RegexTokenizer(regex=r'\d')
        tokenizer.tokenize(self.document)
