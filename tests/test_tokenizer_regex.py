"""Test regex tokenizers."""
import logging
import unittest

from konfuzio_sdk.data import Project, Annotation, Document, Label, AnnotationSet, LabelSet, Span, Category
from konfuzio_sdk.tokenizer.regex import (
    RegexTokenizer,
    WhitespaceTokenizer,
    ConnectedTextTokenizer,
    ColonPrecededTokenizer,
    CapitalizedTextTokenizer,
    NonTextTokenizer,
)

logger = logging.getLogger(__name__)


class TestRegexTokenizer(unittest.TestCase):
    """Test the RegexTokenizer."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the tokenizer and test setup."""
        cls.project = Project(id_=None)
        cls.category = Category(project=cls.project, id_=1)
        cls.category_2 = Category(project=cls.project, id_=2)
        cls.project.add_category(cls.category)
        cls.project.add_category(cls.category_2)
        cls.label_set = LabelSet(id_=2, project=cls.project, categories=[cls.category, cls.category_2])
        cls.label = Label(id_=3, text='LabelName', project=cls.project, label_sets=[cls.label_set])

        cls.document = Document(project=cls.project, category=cls.category, text="Good morning.", dataset_status=3)
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

        cls.document_2 = Document(project=cls.project, category=cls.category_2, text="Good day.", dataset_status=3)
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
        assert len(no_label_annotations) == len(document.annotations()) == 1
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
        assert  len(no_label_annotations) == len(document.annotations()) == 1
        assert no_label_annotations[0].spans[0] != span
        # assert annotations[0].spans[0].created_by == self.__repr__()

    def test_tokenize_with_empty_document(self):
        """Test tokenize a Document without text."""
        document = Document(project=self.project, category=self.category)
        self.tokenizer.tokenize(document)
        assert len(document.annotations) == 0

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

    def test_evaluate_category(self):
        """Test evaluate a Category with a Document with 1 Span that can be found by the tokenizer."""
        result = self.tokenizer.evaluate_category(self.category)
        assert result.is_correct.sum() == 1
        assert result.is_found_by_tokenizer.sum() == 1

    def test_evaluate_project(self):
        """Test evaluate a Project with Documents with 2 Spans that can be found by the tokenizer."""
        result = self.tokenizer.evaluate_project(self.project)
        assert result.is_correct.sum() == 2
        assert result.is_found_by_tokenizer.sum() == 2
        assert set(result.category_id.dropna().unique()) == {self.category.id_, self.category_2.id_}

    def test_evaluate_project_output_offsets(self):
        """Test evaluation offsets of a Project with Documents with 2 Spans that can be found by the tokenizer."""
        result = self.tokenizer.evaluate_project(self.project)
        result_doc_1 = result[result.document_id_local == self.document.id_local]
        result_doc_2 = result[result.document_id_local == self.document_2.id_local]
        assert result_doc_1.start_offset[0] == self.span.start_offset
        assert result_doc_1.end_offset[0] == self.span.end_offset
        assert result_doc_2.start_offset[0] == self.span_2.start_offset
        assert result_doc_2.end_offset[0] == self.span_2.end_offset


class TestWhitespaceTokenizer(unittest.TestCase):
    """Test the WhitespaceTokenizer."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the tokenizer and test setup."""
        cls.project = Project(id_=None)
        cls.category = Category(project=cls.project, id_=1)
        cls.project.add_category(cls.category)
        cls.label_set = LabelSet(id_=2, project=cls.project, categories=[cls.category])
        cls.label = Label(id_=3, text='test', project=cls.project, label_sets=[cls.label_set])

        cls.tokenizer = WhitespaceTokenizer()

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

    # Tokenizer can find
    def test_case_6_group_non_words(self):
        """Test if tokenizer can find a group of non-word characters."""
        document = self._create_artificial_document(text="To local C-1234 City Name", offsets=[(9, 15)])
        assert document.annotations()[0].offset_string == ["C-1234"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 1
        assert (
            result[result.is_found_by_tokenizer == 1].start_offset[0] == document.annotations()[0].spans[0].start_offset
        )
        assert result[result.is_found_by_tokenizer == 1].end_offset[0] == document.annotations()[0].spans[0].end_offset

    def test_case_18_word_with_spatial_characters(self):
        """Test if tokenizer can find a word with a special character."""
        document = self._create_artificial_document(text="write to: person_name@company.com", offsets=[(10, 33)])
        assert document.annotations()[0].offset_string == ["person_name@company.com"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 1
        assert (
            result[result.is_found_by_tokenizer == 1].start_offset[0] == document.annotations()[0].spans[0].start_offset
        )
        assert result[result.is_found_by_tokenizer == 1].end_offset[0] == document.annotations()[0].spans[0].end_offset

    def test_case_19_word_preceded_by_colon(self):
        """Test if tokenizer can find a word preceded by colon."""
        document = self._create_artificial_document(text="write to: name", offsets=[(10, 14)])
        assert document.annotations()[0].offset_string == ["name"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 1
        assert (
            result[result.is_found_by_tokenizer == 1].start_offset[0] == document.annotations()[0].spans[0].start_offset
        )
        assert result[result.is_found_by_tokenizer == 1].end_offset[0] == document.annotations()[0].spans[0].end_offset

    # Tokenizer cannot find
    def test_case_1_group_capitalized_words(self):
        """Test if tokenizer can find a group of words starting with a capitalized character."""
        document = self._create_artificial_document(text="Company A&B GmbH  ", offsets=[(0, 16)])
        assert document.annotations()[0].offset_string == ["Company A&B GmbH"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_2_group_capitalized_words_in_the_middle_of_text(self):
        """Test if tokenizer can find a group of words starting with a capitalized character in the middle of text."""
        document = self._create_artificial_document(text="Company is Company A&B GmbH now", offsets=[(11, 27)])
        assert document.annotations()[0].offset_string == ["Company A&B GmbH"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_3_group_capitalized_words_in_the_middle_of_text_without_period(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="Company is Company A&B GmbH.", offsets=[(11, 27)])
        assert document.annotations()[0].offset_string == ["Company A&B GmbH"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_4_group_specific_capitalized_words(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="Company Company A&B GmbH", offsets=[(8, 24)])
        assert document.annotations()[0].offset_string == ["Company A&B GmbH"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_5_group_words_excluding_non_word_characters(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="street Name 1-2b,", offsets=[(0, 11)])
        assert document.annotations()[0].offset_string == ["street Name"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_7_non_words_excluding_comma_at_end(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="street Name 1-2b,", offsets=[(12, 16)])
        assert document.annotations()[0].offset_string == ["1-2b"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_8_non_words_excluding_period_at_end(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="date 1.2.2022.", offsets=[(5, 13)])
        assert document.annotations()[0].offset_string == ["1.2.2022"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_9_non_words_separated_by_whitespace(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="date 01. 01. 2022", offsets=[(5, 17)])
        assert document.annotations()[0].offset_string == ["01. 01. 2022"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_10_date_with_month_in_the_middle(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="date 1 Jan 2022 ", offsets=[(5, 15)])
        assert document.annotations()[0].offset_string == ["1 Jan 2022"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_11_date_with_month_in_the_beginning(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="date Jan 1, 2022 ", offsets=[(5, 16)])
        assert document.annotations()[0].offset_string == ["Jan 1, 2022"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_12_date_with_month_in_the_beginning(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="code AB 12-3:200", offsets=[(5, 16)])
        assert document.annotations()[0].offset_string == ["AB 12-3:200"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_13_paragraph(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(
            text="This is \na description. Occupies a paragraph.", offsets=[(0, 7), (9, 45)]
        )
        assert document.annotations()[0].offset_string == ["This is", "a description. Occupies a paragraph."]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_14_sentence_single_line(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="This is a sentence.", offsets=[(0, 19)])
        assert document.annotations()[0].offset_string == ["This is a sentence."]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_15_sentence_multiline(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(
            text="This is \na sentence. It's 1 sentence only.", offsets=[(0, 7), (9, 20)]
        )
        assert document.annotations()[0].offset_string == ["This is", "a sentence."]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_16_group_of_numbers(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="contact Tel 234 132 134 2", offsets=[(12, 25)])
        assert document.annotations()[0].offset_string == ["234 132 134 2"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_17_group_of_numbers(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="contact +12 234 234 132", offsets=[(8, 23)])
        assert document.annotations()[0].offset_string == ["+12 234 234 132"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0


class TestConnectedTextTokenizer(unittest.TestCase):
    """Test the ConnectedTextTokenizer."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the tokenizer and test setup."""
        cls.project = Project(id_=None)
        cls.category = Category(project=cls.project, id_=1)
        cls.project.add_category(cls.category)
        cls.label_set = LabelSet(id_=2, project=cls.project, categories=[cls.category])
        cls.label = Label(id_=3, text='test', project=cls.project, label_sets=[cls.label_set])

        cls.tokenizer = ConnectedTextTokenizer()

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

    # Tokenizer can find
    def test_case_1_group_capitalized_words(self):
        """Test if tokenizer can find a group of words starting with a capitalized character."""
        document = self._create_artificial_document(text="Company A&B GmbH  ", offsets=[(0, 16)])
        assert document.annotations()[0].offset_string == ["Company A&B GmbH"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 1
        assert (
            result[result.is_found_by_tokenizer == 1].start_offset[0] == document.annotations()[0].spans[0].start_offset
        )
        assert result[result.is_found_by_tokenizer == 1].end_offset[0] == document.annotations()[0].spans[0].end_offset

    def test_case_13_paragraph(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(
            text="This is \na description. Occupies a paragraph.", offsets=[(0, 7), (9, 45)]
        )
        assert document.annotations()[0].offset_string == ["This is", "a description. Occupies a paragraph."]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 2
        assert (
            result[result.is_found_by_tokenizer == 1].start_offset[0] == document.annotations()[0].spans[0].start_offset
        )
        assert result[result.is_found_by_tokenizer == 1].end_offset[0] == document.annotations()[0].spans[0].end_offset

    def test_case_14_sentence_single_line(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="This is a sentence.", offsets=[(0, 19)])
        assert document.annotations()[0].offset_string == ["This is a sentence."]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 1
        assert (
            result[result.is_found_by_tokenizer == 1].start_offset[0] == document.annotations()[0].spans[0].start_offset
        )
        assert result[result.is_found_by_tokenizer == 1].end_offset[0] == document.annotations()[0].spans[0].end_offset

    # Tokenizer cannot find
    def test_case_2_group_capitalized_words_in_the_middle_of_text(self):
        """Test if tokenizer can find a group of words starting with a capitalized character in the middle of text."""
        document = self._create_artificial_document(text="Company is Company A&B GmbH now", offsets=[(11, 27)])
        assert document.annotations()[0].offset_string == ["Company A&B GmbH"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_3_group_capitalized_words_in_the_middle_of_text_without_period(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="Company is Company A&B GmbH.", offsets=[(11, 27)])
        assert document.annotations()[0].offset_string == ["Company A&B GmbH"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_4_group_specific_capitalized_words(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="Company Company A&B GmbH", offsets=[(8, 24)])
        assert document.annotations()[0].offset_string == ["Company A&B GmbH"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_5_group_words_excluding_non_word_characters(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="street Name 1-2b,", offsets=[(0, 11)])
        assert document.annotations()[0].offset_string == ["street Name"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_6_group_non_words(self):
        """Test if tokenizer can find a group of non-word characters."""
        document = self._create_artificial_document(text="To local C-1234 City Name", offsets=[(9, 15)])
        assert document.annotations()[0].offset_string == ["C-1234"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_7_non_words_excluding_comma_at_end(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="street Name 1-2b,", offsets=[(12, 16)])
        assert document.annotations()[0].offset_string == ["1-2b"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_8_non_words_excluding_period_at_end(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="date 1.2.2022.", offsets=[(5, 13)])
        assert document.annotations()[0].offset_string == ["1.2.2022"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_9_non_words_separated_by_whitespace(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="date 01. 01. 2022", offsets=[(5, 17)])
        assert document.annotations()[0].offset_string == ["01. 01. 2022"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_10_date_with_month_in_the_middle(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="date 1 Jan 2022 ", offsets=[(5, 15)])
        assert document.annotations()[0].offset_string == ["1 Jan 2022"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_11_date_with_month_in_the_beginning(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="date Jan 1, 2022 ", offsets=[(5, 16)])
        assert document.annotations()[0].offset_string == ["Jan 1, 2022"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_12_date_with_month_in_the_beginning(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="code AB 12-3:200", offsets=[(5, 16)])
        assert document.annotations()[0].offset_string == ["AB 12-3:200"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_15_sentence_multiline(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(
            text="This is \na sentence. It's 1 sentence only.", offsets=[(0, 7), (9, 20)]
        )
        assert document.annotations()[0].offset_string == ["This is", "a sentence."]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 1  # should be 2

    def test_case_16_group_of_numbers(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="contact Tel 234 132 134 2", offsets=[(12, 25)])
        assert document.annotations()[0].offset_string == ["234 132 134 2"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_17_group_of_numbers(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="contact +12 234 234 132", offsets=[(8, 23)])
        assert document.annotations()[0].offset_string == ["+12 234 234 132"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_18_word_with_spatial_characters(self):
        """Test if tokenizer can find a word with a special character."""
        document = self._create_artificial_document(text="write to: person_name@company.com", offsets=[(10, 33)])
        assert document.annotations()[0].offset_string == ["person_name@company.com"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_19_word_preceded_by_colon(self):
        """Test if tokenizer can find a word preceded by colon."""
        document = self._create_artificial_document(text="write to: name", offsets=[(10, 14)])
        assert document.annotations()[0].offset_string == ["name"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0


class TestColonPrecededTokenizer(unittest.TestCase):
    """Test the ColonPrecededTokenizer."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the tokenizer and test setup."""
        cls.project = Project(id_=None)
        cls.category = Category(project=cls.project, id_=1)
        cls.project.add_category(cls.category)
        cls.label_set = LabelSet(id_=2, project=cls.project, categories=[cls.category])
        cls.label = Label(id_=3, text='test', project=cls.project, label_sets=[cls.label_set])

        cls.tokenizer = ColonPrecededTokenizer()

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

    # Tokenizer can find
    def test_case_19_word_preceded_by_colon(self):
        """Test if tokenizer can find a word preceded by colon."""
        document = self._create_artificial_document(text="write to: name", offsets=[(10, 14)])
        assert document.annotations()[0].offset_string == ["name"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 1
        assert (
            result[result.is_found_by_tokenizer == 1].start_offset[0] == document.annotations()[0].spans[0].start_offset
        )
        assert result[result.is_found_by_tokenizer == 1].end_offset[0] == document.annotations()[0].spans[0].end_offset

    # Tokenizer cannot find
    def test_case_1_group_capitalized_words(self):
        """Test if tokenizer can find a group of words starting with a capitalized character."""
        document = self._create_artificial_document(text="Company A&B GmbH  ", offsets=[(0, 16)])
        assert document.annotations()[0].offset_string == ["Company A&B GmbH"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_2_group_capitalized_words_in_the_middle_of_text(self):
        """Test if tokenizer can find a group of words starting with a capitalized character in the middle of text."""
        document = self._create_artificial_document(text="Company is Company A&B GmbH now", offsets=[(11, 27)])
        assert document.annotations()[0].offset_string == ["Company A&B GmbH"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_3_group_capitalized_words_in_the_middle_of_text_without_period(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="Company is Company A&B GmbH.", offsets=[(11, 27)])
        assert document.annotations()[0].offset_string == ["Company A&B GmbH"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_4_group_specific_capitalized_words(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="Company Company A&B GmbH", offsets=[(8, 24)])
        assert document.annotations()[0].offset_string == ["Company A&B GmbH"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_5_group_words_excluding_non_word_characters(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="street Name 1-2b,", offsets=[(0, 11)])
        assert document.annotations()[0].offset_string == ["street Name"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_6_group_non_words(self):
        """Test if tokenizer can find a group of non-word characters."""
        document = self._create_artificial_document(text="To local C-1234 City Name", offsets=[(9, 15)])
        assert document.annotations()[0].offset_string == ["C-1234"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_7_non_words_excluding_comma_at_end(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="street Name 1-2b,", offsets=[(12, 16)])
        assert document.annotations()[0].offset_string == ["1-2b"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_8_non_words_excluding_period_at_end(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="date 1.2.2022.", offsets=[(5, 13)])
        assert document.annotations()[0].offset_string == ["1.2.2022"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_9_non_words_separated_by_whitespace(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="date 01. 01. 2022", offsets=[(5, 17)])
        assert document.annotations()[0].offset_string == ["01. 01. 2022"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_10_date_with_month_in_the_middle(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="date 1 Jan 2022 ", offsets=[(5, 15)])
        assert document.annotations()[0].offset_string == ["1 Jan 2022"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_11_date_with_month_in_the_beginning(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="date Jan 1, 2022 ", offsets=[(5, 16)])
        assert document.annotations()[0].offset_string == ["Jan 1, 2022"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_12_date_with_month_in_the_beginning(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="code AB 12-3:200", offsets=[(5, 16)])
        assert document.annotations()[0].offset_string == ["AB 12-3:200"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_13_paragraph(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(
            text="This is \na description. Occupies a paragraph.", offsets=[(0, 7), (9, 45)]
        )
        assert document.annotations()[0].offset_string == ["This is", "a description. Occupies a paragraph."]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_14_sentence_single_line(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="This is a sentence.", offsets=[(0, 19)])
        assert document.annotations()[0].offset_string == ["This is a sentence."]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_15_sentence_multiline(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(
            text="This is \na sentence. It's 1 sentence only.", offsets=[(0, 7), (9, 20)]
        )
        assert document.annotations()[0].offset_string == ["This is", "a sentence."]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_16_group_of_numbers(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="contact Tel 234 132 134 2", offsets=[(12, 25)])
        assert document.annotations()[0].offset_string == ["234 132 134 2"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_17_group_of_numbers(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="contact +12 234 234 132", offsets=[(8, 23)])
        assert document.annotations()[0].offset_string == ["+12 234 234 132"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_18_word_with_spatial_characters(self):
        """Test if tokenizer can find a word with a special character."""
        document = self._create_artificial_document(text="write to: person_name@company.com", offsets=[(10, 33)])
        assert document.annotations()[0].offset_string == ["person_name@company.com"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0


class TestCapitalizedTextTokenizer(unittest.TestCase):
    """Test the CapitalizedTextTokenizer."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the tokenizer and test setup."""
        cls.project = Project(id_=None)
        cls.category = Category(project=cls.project, id_=1)
        cls.project.add_category(cls.category)
        cls.label_set = LabelSet(id_=2, project=cls.project, categories=[cls.category])
        cls.label = Label(id_=3, text='test', project=cls.project, label_sets=[cls.label_set])

        cls.tokenizer = CapitalizedTextTokenizer()

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

    # Tokenizer can find
    def test_case_1_group_capitalized_words(self):
        """Test if tokenizer can find a group of words starting with a capitalized character."""
        document = self._create_artificial_document(text="Company A&B GmbH  ", offsets=[(0, 16)])
        assert document.annotations()[0].offset_string == ["Company A&B GmbH"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 1
        assert (
            result[result.is_found_by_tokenizer == 1].start_offset[0] == document.annotations()[0].spans[0].start_offset
        )
        assert result[result.is_found_by_tokenizer == 1].end_offset[0] == document.annotations()[0].spans[0].end_offset

    def test_case_2_group_capitalized_words_in_the_middle_of_text(self):
        """Test if tokenizer can find a group of words starting with a capitalized character in the middle of text."""
        document = self._create_artificial_document(text="Company is Company A&B GmbH now", offsets=[(11, 27)])
        assert document.annotations()[0].offset_string == ["Company A&B GmbH"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 1
        assert (
            result[result.is_found_by_tokenizer == 1].start_offset[0] == document.annotations()[0].spans[0].start_offset
        )
        assert result[result.is_found_by_tokenizer == 1].end_offset[0] == document.annotations()[0].spans[0].end_offset

    def test_case_3_group_capitalized_words_in_the_middle_of_text_without_period(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="Company is Company A&B GmbH.", offsets=[(11, 27)])
        assert document.annotations()[0].offset_string == ["Company A&B GmbH"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 1
        assert (
            result[result.is_found_by_tokenizer == 1].start_offset[0] == document.annotations()[0].spans[0].start_offset
        )
        assert result[result.is_found_by_tokenizer == 1].end_offset[0] == document.annotations()[0].spans[0].end_offset

    # Tokenizer cannot find
    def test_case_4_group_specific_capitalized_words(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="Company Company A&B GmbH", offsets=[(8, 24)])
        assert document.annotations()[0].offset_string == ["Company A&B GmbH"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_5_group_words_excluding_non_word_characters(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="street Name 1-2b,", offsets=[(0, 11)])
        assert document.annotations()[0].offset_string == ["street Name"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_6_group_non_words(self):
        """Test if tokenizer can find a group of non-word characters."""
        document = self._create_artificial_document(text="To local C-1234 City Name", offsets=[(9, 15)])
        assert document.annotations()[0].offset_string == ["C-1234"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_7_non_words_excluding_comma_at_end(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="street Name 1-2b,", offsets=[(12, 16)])
        assert document.annotations()[0].offset_string == ["1-2b"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_8_non_words_excluding_period_at_end(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="date 1.2.2022.", offsets=[(5, 13)])
        assert document.annotations()[0].offset_string == ["1.2.2022"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_9_non_words_separated_by_whitespace(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="date 01. 01. 2022", offsets=[(5, 17)])
        assert document.annotations()[0].offset_string == ["01. 01. 2022"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_10_date_with_month_in_the_middle(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="date 1 Jan 2022 ", offsets=[(5, 15)])
        assert document.annotations()[0].offset_string == ["1 Jan 2022"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_11_date_with_month_in_the_beginning(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="date Jan 1, 2022 ", offsets=[(5, 16)])
        assert document.annotations()[0].offset_string == ["Jan 1, 2022"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_12_date_with_month_in_the_beginning(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="code AB 12-3:200", offsets=[(5, 16)])
        assert document.annotations()[0].offset_string == ["AB 12-3:200"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_13_paragraph(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(
            text="This is \na description. Occupies a paragraph.", offsets=[(0, 7), (9, 45)]
        )
        assert document.annotations()[0].offset_string == ["This is", "a description. Occupies a paragraph."]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_14_sentence_single_line(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="This is a sentence.", offsets=[(0, 19)])
        assert document.annotations()[0].offset_string == ["This is a sentence."]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_15_sentence_multiline(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(
            text="This is \na sentence. It's 1 sentence only.", offsets=[(0, 7), (9, 20)]
        )
        assert document.annotations()[0].offset_string == ["This is", "a sentence."]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_16_group_of_numbers(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="contact Tel 234 132 134 2", offsets=[(12, 25)])
        assert document.annotations()[0].offset_string == ["234 132 134 2"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_17_group_of_numbers(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="contact +12 234 234 132", offsets=[(8, 23)])
        assert document.annotations()[0].offset_string == ["+12 234 234 132"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_18_word_with_spatial_characters(self):
        """Test if tokenizer can find a word with a special character."""
        document = self._create_artificial_document(text="write to: person_name@company.com", offsets=[(10, 33)])
        assert document.annotations()[0].offset_string == ["person_name@company.com"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_19_word_preceded_by_colon(self):
        """Test if tokenizer can find a word preceded by colon."""
        document = self._create_artificial_document(text="write to: name", offsets=[(10, 14)])
        assert document.annotations()[0].offset_string == ["name"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0


class TestNonTextTokenizer(unittest.TestCase):
    """Test the NonTextTokenizer."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the tokenizer and test setup."""
        cls.project = Project(id_=None)
        cls.category = Category(project=cls.project, id_=1)
        cls.project.add_category(cls.category)
        cls.label_set = LabelSet(id_=2, project=cls.project, categories=[cls.category])
        cls.label = Label(id_=3, text='test', project=cls.project, label_sets=[cls.label_set])

        cls.tokenizer = NonTextTokenizer()

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

    # Tokenizer can find
    def test_case_9_non_words_separated_by_whitespace(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="date 01. 01. 2022", offsets=[(5, 17)])
        assert document.annotations()[0].offset_string == ["01. 01. 2022"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 1
        assert (
            result[result.is_found_by_tokenizer == 1].start_offset[0] == document.annotations()[0].spans[0].start_offset
        )
        assert result[result.is_found_by_tokenizer == 1].end_offset[0] == document.annotations()[0].spans[0].end_offset

    def test_case_12_date_with_month_in_the_beginning(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="code AB 12-3:200", offsets=[(5, 16)])
        assert document.annotations()[0].offset_string == ["AB 12-3:200"]
        result = self.tokenizer.evaluate(document)
        assert (
            result[result.is_found_by_tokenizer == 1].start_offset[0] == document.annotations()[0].spans[0].start_offset
        )
        assert result[result.is_found_by_tokenizer == 1].end_offset[0] == document.annotations()[0].spans[0].end_offset

    def test_case_16_group_of_numbers(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="contact Tel 234 132 134 2", offsets=[(12, 25)])
        assert document.annotations()[0].offset_string == ["234 132 134 2"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 1
        assert (
            result[result.is_found_by_tokenizer == 1].start_offset[0] == document.annotations()[0].spans[0].start_offset
        )
        assert result[result.is_found_by_tokenizer == 1].end_offset[0] == document.annotations()[0].spans[0].end_offset

    # Tokenizer cannot find
    def test_case_1_group_capitalized_words(self):
        """Test if tokenizer can find a group of words starting with a capitalized character."""
        document = self._create_artificial_document(text="Company A&B GmbH  ", offsets=[(0, 16)])
        assert document.annotations()[0].offset_string == ["Company A&B GmbH"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_2_group_capitalized_words_in_the_middle_of_text(self):
        """Test if tokenizer can find a group of words starting with a capitalized character in the middle of text."""
        document = self._create_artificial_document(text="Company is Company A&B GmbH now", offsets=[(11, 27)])
        assert document.annotations()[0].offset_string == ["Company A&B GmbH"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_3_group_capitalized_words_in_the_middle_of_text_without_period(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="Company is Company A&B GmbH.", offsets=[(11, 27)])
        assert document.annotations()[0].offset_string == ["Company A&B GmbH"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_4_group_specific_capitalized_words(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="Company Company A&B GmbH", offsets=[(8, 24)])
        assert document.annotations()[0].offset_string == ["Company A&B GmbH"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_5_group_words_excluding_non_word_characters(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="street Name 1-2b,", offsets=[(0, 11)])
        assert document.annotations()[0].offset_string == ["street Name"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_6_group_non_words(self):
        """Test if tokenizer can find a group of non-word characters."""
        document = self._create_artificial_document(text="To local C-1234 City Name", offsets=[(9, 15)])
        assert document.annotations()[0].offset_string == ["C-1234"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_7_non_words_excluding_comma_at_end(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="street Name 1-2b,", offsets=[(12, 16)])
        assert document.annotations()[0].offset_string == ["1-2b"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_8_non_words_excluding_period_at_end(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="date 1.2.2022.", offsets=[(5, 13)])
        assert document.annotations()[0].offset_string == ["1.2.2022"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_10_date_with_month_in_the_middle(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="date 1 Jan 2022 ", offsets=[(5, 15)])
        assert document.annotations()[0].offset_string == ["1 Jan 2022"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_11_date_with_month_in_the_beginning(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="date Jan 1, 2022 ", offsets=[(5, 16)])
        assert document.annotations()[0].offset_string == ["Jan 1, 2022"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_13_paragraph(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(
            text="This is \na description. Occupies a paragraph.", offsets=[(0, 7), (9, 45)]
        )
        assert document.annotations()[0].offset_string == ["This is", "a description. Occupies a paragraph."]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_14_sentence_single_line(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="This is a sentence.", offsets=[(0, 19)])
        assert document.annotations()[0].offset_string == ["This is a sentence."]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_15_sentence_multiline(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(
            text="This is \na sentence. It's 1 sentence only.", offsets=[(0, 7), (9, 20)]
        )
        assert document.annotations()[0].offset_string == ["This is", "a sentence."]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_17_group_of_numbers(self):
        """Test output for the evaluate method with a Document with 1 Span."""
        document = self._create_artificial_document(text="contact +12 234 234 132", offsets=[(8, 23)])
        assert document.annotations()[0].offset_string == ["+12 234 234 132"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_18_word_with_spatial_characters(self):
        """Test if tokenizer can find a word with a special character."""
        document = self._create_artificial_document(text="write to: person_name@company.com", offsets=[(10, 33)])
        assert document.annotations()[0].offset_string == ["person_name@company.com"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0

    def test_case_19_word_preceded_by_colon(self):
        """Test if tokenizer can find a word preceded by colon."""
        document = self._create_artificial_document(text="write to: name", offsets=[(10, 14)])
        assert document.annotations()[0].offset_string == ["name"]
        result = self.tokenizer.evaluate(document)
        assert result.is_found_by_tokenizer.sum() == 0
