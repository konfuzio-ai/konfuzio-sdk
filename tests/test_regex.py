"""Test Regex functionality on strings."""
import logging
import os
import textwrap
import unittest
from timeit import timeit

import pytest
import regex as re

from konfuzio_sdk.data import Annotation, AnnotationSet, Category, Document, Label, LabelSet, Project, Span
from konfuzio_sdk.regex import (
    generic_candidate_function,
    get_best_regex,
    merge_regex,
    plausible_regex,
    regex_matches,
    suggest_regex_for_string,
)
from konfuzio_sdk.tokenizer.regex import RegexTokenizer
from konfuzio_sdk.utils import does_not_raise, is_file
from tests.variables import OFFLINE_PROJECT, TEST_DOCUMENT_ID

logger = logging.getLogger(__name__)

suggest_regex_for_string_data = [
    ({'string': ' '}, r'[ ]{1,2}', does_not_raise()),
    (
        {'string': 'VAG GmbH', 'replace_characters': True},
        r'[A-ZÄÖÜ]+[ ]{1,2}[A-ZÄÖÜ][a-zäöüß]+[A-ZÄÖÜ]',
        does_not_raise(),
    ),
    (
        {'string': 'a2bau GmbH', 'replace_characters': True},
        r'[a-zäöüß]\d[a-zäöüß]+[ ]{1,2}[A-ZÄÖÜ][a-zäöüß]+[A-ZÄÖÜ]',
        does_not_raise(),
    ),
    (
        {'string': 'Bau-Wohn-Gesellschaft', 'replace_characters': True},
        '[A-ZÄÖÜ][a-zäöüß]+[-][A-ZÄÖÜ][a-zäöüß]+[-][A-ZÄÖÜ][a-zäöüß]+',
        does_not_raise(),
    ),
    ({'string': 'Bau-Wohn-Gesellschaft', 'replace_characters': False}, 'Bau[-]Wohn[-]Gesellschaft', does_not_raise()),
    ({'string': '  '}, '[ ]{2,}', does_not_raise()),
    (
        {'string': ' $ hello @ ! ? # [ ]'},
        r'[ ]{1,2}\$[ ]{1,2}hello[ ]{1,2}\@[ ]{1,2}\![ ]{1,2}\?[ ]{1,2}\#[ ]{1,2}\[[ ]{1,2}\]',
        does_not_raise(),
    ),
    ({'string': '(hello)+'}, '\\(hello\\)[\\+]', does_not_raise()),
    ({'string': '(hello) +'}, '\\(hello\\)[ ]{1,2}[\\+]', does_not_raise()),
    (
        {'string': 'Große Bäume, haben Äpfel.', 'replace_characters': True},
        r'[A-ZÄÖÜ][a-zäöüß]+[ ]{1,2}[A-ZÄÖÜ][a-zäöüß]+\,[ ]{1,2}[a-zäöüß]+[ ]{1,2}[A-ZÄÖÜ][a-zäöüß]+\.',
        does_not_raise(),
    ),
    # check äöüß
    ({'string': 'Hello                  World'}, 'Hello[ ]{2,}World', does_not_raise()),
    ({'string': 'Hello\n\nWorld'}, 'Hello\n\nWorld', does_not_raise()),
    (
        {'string': '  GRUPPE\n\n   Industrie-Police  \\\n  Nachtrag Nr. '},
        '[ ]{2,}GRUPPE\n\n[ ]{2,}Industrie[-]Police[ ]{2,}\\\\\n[ ]{2,}Nachtrag[ ]{1,2}Nr\\.[ ]{1,2}',
        does_not_raise(),
    ),
    ({'string': 'Hello\nWorld'}, 'Hello\nWorld', does_not_raise()),
    ({'string': 'Am 22.05.2020'}, r'Am[ ]{1,2}\d\d\.\d\d\.\d\d\d\d', does_not_raise()),
    ({'string': 'Am 22.05.2020'}, r'Am[ ]{1,2}\d\d\.\d\d\.\d\d\d\d', does_not_raise()),
    ({'string': '[mailto:ch@helm-nagel.com]'}, '\\[mailto:ch\\@helm[-]nagel\\.com\\]', does_not_raise()),
    ({'string': '              \\       ifr'}, '[ ]{2,}\\\\[ ]{2,}ifr', does_not_raise()),
    (
        {'string': '22. Januar 2020', 'replace_characters': True},
        r'\d\d\.[ ]{1,2}[A-ZÄÖÜ][a-zäöüß]+[ ]{1,2}\d\d\d\d',
        does_not_raise(),
    ),
    (
        {'string': '22. \nJanuar 2020', 'replace_characters': True},
        '\\d\\d\\.[ ]{1,2}\n[A-ZÄÖÜ][a-zäöüß]+[ ]{1,2}\\d\\d\\d\\d',
        does_not_raise(),
    ),
    (
        {'string': '22. \n\nJanuar 2020', 'replace_characters': True},
        '\\d\\d\\.[ ]{1,2}\n\n[A-ZÄÖÜ][a-zäöüß]+[ ]{1,2}\\d\\d\\d\\d',
        does_not_raise(),
    ),
    # ('input', 'output', pytest.raises(ValueError)),
    ({'string': '{hello'}, '\\{hello', does_not_raise()),
    ({'string': 'he*llo'}, 'he\\*llo', does_not_raise()),
    ({'string': 'he|llo'}, 'he\\|llo', does_not_raise()),
    # The following three examples, show the main functionality of the suggest_regex_for_string function.
    (
        {'string': 'Achtung       Kontrolle 911    ', 'replace_numbers': False},
        'Achtung[ ]{2,}Kontrolle[ ]{1,2}911[ ]{2,}',
        does_not_raise(),
    ),
    ({'string': 'Achtung       Kontrolle 911    '}, r'Achtung[ ]{2,}Kontrolle[ ]{1,2}\d\d\d[ ]{2,}', does_not_raise()),
    # see issue https://gitlab.com/konfuzio/objectives/-/issues/1373
    # ({'string': 'Garage/n'}, r'Garage\/n', does_not_raise()),
    (
        {'string': 'Achtung       Kontrolle 911    ', 'replace_characters': True},
        r'[A-ZÄÖÜ][a-zäöüß]+[ ]{2,}[A-ZÄÖÜ][a-zäöüß]+[ ]{1,2}\d\d\d[ ]{2,}',
        does_not_raise(),
    ),
    (
        {'string': 'hello.mio@konfuzio.c', 'replace_characters': True},
        r'[a-zäöüß]+\.[a-zäöüß]+\@[a-zäöüß]+\.[a-zäöüß]',
        does_not_raise(),
    ),
]


@pytest.mark.parametrize('input_strings, output, expected_exception', suggest_regex_for_string_data)
def test_suggest_regex_for_string(input_strings, output, expected_exception):
    """Test different strings to represent them as a regex."""
    with expected_exception:
        if output is not None:
            assert suggest_regex_for_string(**input_strings) == output
        else:
            assert suggest_regex_for_string(**input_strings) is None


def test_merge_regex():
    """Test to merge multiple regex to one group."""
    my_regex = merge_regex([r'[a-z]+', r'\d+', r'[A-Z]+'])
    assert my_regex == r'(?:[a-z]+|[A-Z]+|\d+)'
    test_string = '39 gallons is the per capita consumption of softdrinks in US.'
    tokens = regex_matches(test_string, my_regex)
    assert len(tokens) == len(test_string.split(' '))


def test_regex_plausibility_compile_error():
    """Test the plausibility check for regex."""
    assert 'x' == plausible_regex(r'x', 'xxx')
    assert '' == plausible_regex(r'****', 'xxx')
    assert '' == plausible_regex(r'\d', 'xxx')


def test_regex_spans_with_invalid_regex_group_name():
    """Test to run regex_matches with an invalid group name."""
    result = regex_matches('I go home at 5 AM.', regex=r'(?P<9variable>\d)')
    expected_result = [
        {
            'regex_used': "'(?P<_9variable>\\\\d)'",
            'regex_group': '_9variable',
            'value': '5',
            'start_offset': 13,
            'end_offset': 14,
            'start_text': 0,
        },
        {
            'regex_used': "'(?P<_9variable>\\\\d)'",
            'regex_group': '0',
            'value': '5',
            'start_offset': 13,
            'end_offset': 14,
            'start_text': 0,
        },
    ]
    assert expected_result == result


def test_regex_spans_filtered_group():
    """Test to run regex_matches with an invalid group name."""
    result = regex_matches('Call me at 12 AM.', regex=r'(?P<variable1>\d)(?P<variable2>\d)', filtered_group='variable1')
    expected_result = [
        {
            'regex_used': "'(?P<variable1>\\\\d)(?P<variable2>\\\\d)'",
            'regex_group': 'variable1',
            'value': '1',
            'start_offset': 11,
            'end_offset': 12,
            'start_text': 0,
        }
    ]
    assert expected_result == result


def test_empty_get_best_regex():
    """Test to evaluate an empty list."""
    assert get_best_regex([]) == []


def test_get_best_regex_from_project():
    """Test to get evaluation."""
    # todo: https://gitlab.com/konfuzio/objectives/-/issues/9358
    project = Project(id_=None, project_folder=OFFLINE_PROJECT)
    document = project.get_document_by_id(TEST_DOCUMENT_ID)
    evaluations = document.evaluate_regex(r'(?P<Label_860>\d)', project.get_label_by_id(860))  # Tax
    assert evaluations['count_total_findings'] == 329
    assert evaluations['count_total_correct_findings'] == 1
    assert evaluations['count_correct_annotations'] == 1
    assert evaluations['count_correct_annotations_not_found'] == 0
    assert evaluations['doc_matched']
    assert evaluations['annotation_precision'] == 0.00303951367781155
    assert evaluations['document_recall'] == 0
    assert evaluations['annotation_recall'] == 1.0  # todo: evaluation on Span Level is missing
    assert evaluations['f1_score'] == 0.00606060606060606


class TestTokens(unittest.TestCase):
    """Create Tokens from example Data of the Konfuzio Host."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the Project."""
        cls.prj = Project(id_=None, project_folder=OFFLINE_PROJECT)

    @pytest.mark.xfail(reason='We force to have an annotation_set to init an Annotation.')
    def test_token_replacement_only_whitespace_2(self):
        """Create a regex where only whitespaces are replaced."""
        label_data = {
            'description': 'Betrag des jew. Bruttobezugs',
            'id_': 88888888,
            'shortcut': 'N/A',
            'text': 'Betrag',
            'get_data_type_display': 'Text',
            'text_clean': 'Betrag',
            'token_full_replacement': False,
            'token_whitespace_replacement': True,
            'token_number_replacement': False,
            'label_sets': self.prj.label_sets,
            'has_multiple_top_candidates': False,
        }
        new_label = Label(project=self.prj, **label_data)
        category = self.prj.get_category_by_id(63)
        doc = self.prj.documents[0]

        new_anno = Annotation(
            start_offset=127,
            end_offset=140,
            label=new_label.id_,
            label_set_id=new_label.label_sets[0].id_,  # hand selected Document label_set
            revised=True,
            is_correct=True,
            accuracy=0.98765431,
            document=doc,
            # annotation_set=doc.get_annotation_set_by_id()
        )

        assert len(new_label.annotations(categories=[category])) == 1

        assert new_anno.offset_string == ['Dezember 2018']
        assert len(new_anno.tokens()) == 1

        regex = new_anno.tokens()[0]['regex']
        assert '_W_' in regex
        assert '_F_' not in regex
        assert '_N_' not in regex
        assert regex == '(?P<Betrag_W_None_fallback>Dezember[ ]{1,2}2018)'

    def test_generic_candidate_function(self):
        """Test to create a function which applies a RegEx."""
        my_function = generic_candidate_function(r'\d\d.\d\d.\d\d\d\d')
        self.assertEqual(
            (['23.04.2055'], ['I was born at the ', '.'], [(18, 28)]), my_function('I was born at the 23.04.2055.')
        )

    def test_label_keyword_token(self):
        """Extract value for Steuerklasse."""
        label = next(x for x in self.prj.labels if x.name == 'Steuerklasse')
        category = self.prj.get_category_by_id(63)
        regexes = label.regex(categories=[category], update=True)
        assert len(sorted(regexes)) == 1
        assert is_file(os.path.join(self.prj.regex_folder, f'{category.name}_{label.name_clean}_tokens.json5'))

    def test_label_plz_token(self):
        """Extract all tax gross amounts of all payslips."""
        label = self.prj.get_label_by_name('Steuer-Brutto')
        category = self.prj.get_category_by_id(63)
        regexes = label.regex(categories=[category], update=True)[category.id_]
        assert len(regexes) == 1
        assert '(?P<Label_12503_N_' in regexes[0]
        assert is_file(os.path.join(self.prj.regex_folder, f'{category.name}_{label.name_clean}_tokens.json5'))

    def test_label_token_auszahlungsbetrag(self):
        """Return the summary of all regex needed to get the wage."""
        label = self.prj.get_label_by_name('Auszahlungsbetrag')
        category = self.prj.get_category_by_id(63)
        regexes = label.regex(categories=[category], update=True)[category.id_]

        assert len(regexes[0].split('|')) == 3
        assert '(?P<Label_858_' in regexes[0].split('|')[0]
        assert '(?P<Label_858_' in regexes[0].split('|')[1]
        assert '(?P<Label_858_' in regexes[0].split('|')[2]
        assert '>\\d\\d\\d\\,\\d\\d)' in regexes[0]
        assert '>\\d\\.\\d\\d\\d\\,\\d\\d)' in regexes[0]

    def test_label_empty_annotations(self):
        """Empty Annotations should not create regex."""
        category = self.prj.get_category_by_id(63)
        try:
            label = next(x for x in self.prj.labels if len(x.annotations(categories=[category])) == 0)
            regexes = label.regex(categories=[category])[category.id_]
            assert sorted(regexes) == []
            # File is created even if there are no Annotations
            assert is_file(os.path.join(self.prj.regex_folder, f'{category.name}_{label.name_clean}_tokens.json5'))
        except StopIteration:
            pass

    def test_iban_regex_from_offline_project(self):
        """Test to get evaluation."""
        project = Project(id_=None, project_folder=OFFLINE_PROJECT)
        label = project.get_label_by_name('Bank inkl. IBAN')
        category = project.get_category_by_id(63)
        tokens = label.base_regex(category)

        assert (
            '[A-ZÄÖÜ]+\\d\\d[ ]{1,2}\\d\\d\\d\\d[ ]{1,2}\\d\\d\\d\\d[ ]{1,2}'
            '\\d\\d\\d\\d[ ]{1,2}\\d\\d\\d\\d[ ]{1,2}\\d\\d' in tokens
        )

        regexes = label.find_regex(category=category)
        assert len(regexes) == 1
        l_tok = RegexTokenizer(regexes[0])
        missing_spans = label.spans_not_found_by_tokenizer(l_tok, categories=[category], use_correct=True)
        assert len(missing_spans) == 17


class TestTokensMultipleCategories(unittest.TestCase):
    """Create Tokens when Label is shared by different Categories."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the Project."""
        cls.project = Project(id_=None)
        cls.category = Category(project=cls.project, id_=1)
        cls.category_2 = Category(project=cls.project, id_=2)
        cls.label_set = LabelSet(id_=2, project=cls.project, categories=[cls.category, cls.category_2])
        cls.label = Label(id_=3, text='LabelName', project=cls.project, label_sets=[cls.label_set])

        # Category 1
        cls.document = Document(project=cls.project, category=cls.category, text='Hi all,', dataset_status=2)
        annotation_set = AnnotationSet(id_=4, document=cls.document, label_set=cls.label_set)
        cls.span = Span(start_offset=3, end_offset=6)
        cls.annotation = Annotation(
            id_=5,
            document=cls.document,
            is_correct=True,
            annotation_set=annotation_set,
            label=cls.label,
            label_set=cls.label_set,
            spans=[cls.span],
        )

        # Category 2
        cls.document_2 = Document(project=cls.project, category=cls.category_2, text='Morning.', dataset_status=2)
        annotation_set_2 = AnnotationSet(id_=10, document=cls.document_2, label_set=cls.label_set)
        cls.span_2 = Span(start_offset=0, end_offset=7)
        cls.annotation_2 = Annotation(
            id_=11,
            document=cls.document_2,
            is_correct=True,
            annotation_set=annotation_set_2,
            label=cls.label,
            label_set=cls.label_set,
            spans=[cls.span_2],
        )

    def test_tokens_single_category(self):
        """Test tokens created for a single Category."""
        regexes = self.label.regex(categories=[self.category], update=True)
        assert len(regexes) == 1
        assert '(?P<Label_3_W_5_3>all)' in regexes[self.category.id_][0]

    def test_find_tokens(self):
        """Test to find tokens a Category."""
        regexes = self.label.find_regex(category=self.category)
        # clean evaluations for other tests (this test creates 16 evaluations)
        self.label._evaluations = {}
        self.label._tokens = {}

        assert '(?P<Label_3_W_5_3>all)' in regexes[0]

    def test_find_regex(self):
        """Test to find regex for a Category."""
        regexes = self.label.regex(categories=[self.category], update=True)[self.category.id_]
        self.annotation._tokens = []  # reset after test
        # clean evaluations for other tests (this test creates 16 evaluations)
        self.label._evaluations = {}
        self.label._tokens = {}
        # we can have a different regex selected if the regexes are very similar because of slightly variations in
        # runtime
        assert regexes == ['[ ]{1,2}(?:(?P<Label_3_W_5_3>all))\\,']

    def test_annotation_tokens(self):
        """Test tokens created for an Annotation."""
        tokens = self.annotation_2.tokens()
        self.annotation_2._tokens = []  # reset after test
        assert '(?P<Label_3_W_11_0>Morning)' in [e['regex'] for e in tokens]
        assert '(?P<Label_3_F_11_0>[A-ZÄÖÜ][a-zäöüß]+)' in [e['regex'] for e in tokens]

    def test_token_append_to_annotation(self):
        """Test append token to Annotation."""
        tokens_before = self.annotation_2.tokens()
        self.annotation_2.token_append('(?P<None_W_11_0>Morning)', regex_quality=0)
        # no changes because token already exists
        tokens_after = self.annotation_2.tokens()
        assert tokens_before == tokens_after

    def test_tokens_multiple_categories(self):
        """Test tokens created based on multiple Categories."""
        regexes = self.label.regex(categories=[self.category, self.category_2], update=True)
        assert len(regexes) == 2
        assert '(?P<Label_3_W_5_3>all)' in regexes[self.category.id_][0]
        assert '(?P<Label_3_F_11_0>[A-ZÄÖÜ][a-zäöüß]+)' in regexes[self.category_2.id_][0]
        # regexes[self.category_2.id_][0] == '[ ]{1,2}(?:(?P<Label_3_W_5_3>all))\\,'

    def test_tokens_one_category_after_another(self):
        """
        Test tokens created for one Category after having created tokens for another Category.

        This could be the situation when running in a loop for multiple Categories in a Project.
        """
        tokens_1 = self.label.regex(categories=[self.category], update=True)
        tokens_2 = self.label.regex(categories=[self.category_2], update=True)
        assert len(tokens_1) == 1
        assert len(tokens_2) == 1
        assert '(?P<Label_3_W_5_3>all)' in tokens_1[self.category.id_][0]
        assert '(?P<Label_3_F_11_0>[A-ZÄÖÜ][a-zäöüß]+)' in tokens_2[self.category_2.id_][0]

    def test_tokens_evaluations_single_category(self):
        """Test if the number of evaluations is the expected after getting the tokens for a single Category."""
        all_annotations = self.label.annotations(categories=[self.category])

        proposals = []
        for annotation in all_annotations:
            annotation_proposals = annotation.tokens()
            proposals += annotation_proposals

        assert len(proposals) == 2
        assert 'regex' in proposals[0]
        assert 'annotation_recall' in proposals[0]
        assert 'annotation_precision' in proposals[0]

    def test_tokens_evaluations_multiple_categories(self):
        """Test if the number of evaluations is the expected after getting the tokens for a single Category."""
        _ = self.label.regex(categories=[self.category, self.category_2], update=True)
        print(len(self.label._evaluations[self.category.id_]))
        print(len(self.label._evaluations[self.category_2.id_]))
        assert len(self.label._evaluations[self.category.id_]) == 2
        assert len(self.label._evaluations[self.category_2.id_]) == 2
        assert '(?P<Label_3_W_5_3>all)' in [e['regex'] for e in self.label._evaluations[self.category.id_]]
        assert '(?P<Label_3_F_11_0>[A-ZÄÖÜ][a-zäöüß]+)' in [
            e['regex'] for e in self.label._evaluations[self.category_2.id_]
        ]


class TestRegexGenerator(unittest.TestCase):
    """Test to create regex based on data online."""

    document_count = 26
    correct_annotations = 24

    @classmethod
    def setUpClass(cls) -> None:
        """Load the Project data from the Konfuzio Host."""
        cls.prj = Project(id_=46)
        cls.category = cls.prj.get_category_by_id(63)
        assert len(cls.prj.documents) == cls.document_count
        assert len(cls.prj.get_label_by_id(867).annotations(categories=[cls.category])) == cls.correct_annotations

    @classmethod
    def tearDownClass(cls) -> None:
        """Check that no local data was changed by the tests."""
        assert len(cls.prj.documents) == cls.document_count
        assert len(cls.prj.get_label_by_id(867).annotations(categories=[cls.category])) == cls.correct_annotations
        # cls.prj.delete()

    def test_regex_single_annotation_in_row(self):
        """Build a simple extraction for an amount."""
        analyzed_label = self.prj.get_label_by_name('Auszahlungsbetrag')
        analyzed_label.regex(categories=[self.category], update=True)

        assert len(analyzed_label.regex(categories=[self.category])) == 1
        # we now use the f_score to balance the annotation_precision and the document_recall
        # thereby we find the top regex easily: we only need one regex to match all findings
        for document in self.category.documents():
            annos = document.annotations(label=analyzed_label)
            for auto_regex in analyzed_label.regex(categories=[self.category])[self.category.id_]:
                findings = re.findall(auto_regex, document.text)
                clean_findings = {item for sublist in findings for item in sublist}
                for anno in annos:
                    for span in anno.spans:
                        assert span.offset_string in clean_findings

    def test_annotation_to_regex(self):
        """Test to calculate a regex."""
        project = Project(id_=None)
        category = Category(project=project)
        label_set = LabelSet(id_=33, project=project, categories=[category])
        label = Label(
            id_=22, text='Label Name', text_clean='LabelName', project=project, label_sets=[label_set], threshold=0.5
        )
        document = Document(project=project, category=category, text='From 14.12.2021 to 1.1.2022.', dataset_status=2)
        span_1 = Span(start_offset=5, end_offset=15)
        annotation_set_1 = AnnotationSet(id_=1, document=document, label_set=label_set)
        _ = Annotation(
            document=document,
            is_correct=True,
            annotation_set=annotation_set_1,
            label=label,
            label_set=label_set,
            spans=[span_1],
        )
        regex = label.find_regex(category=category)
        self.assertEqual([r'[ ]{1,2}(?:(?P<Label_22_N_None_5>\d\d\.\d\d\.\d\d\d\d))[ ]{1,2}'], regex)

    # @unittest.skip(reason='Optimization does not work accurately at the moment. See "expected" result.')
    def test_two_annotation_of_one_label_to_regex(self):
        """Test to calculate a regex."""
        project = Project(id_=None)
        category = Category(project=project)
        label_set = LabelSet(id_=33, project=project, categories=[category], has_multiple_annotation_sets=True)
        label = Label(
            id_=22, text='Label Name', text_clean='LabelName', project=project, label_sets=[label_set], threshold=0.5
        )
        long_text = 'From 14.12.2021 to 1.1.2022 ' + 'From data to information by Konfuzio' * 1000
        document = Document(project=project, category=category, text=long_text, dataset_status=2)
        span_1 = Span(start_offset=5, end_offset=15)
        annotation_set_1 = AnnotationSet(id_=1, document=document, label_set=label_set)
        annotation_1 = Annotation(
            document=document,
            is_correct=True,
            annotation_set=annotation_set_1,
            label=label,
            label_set=label_set,
            spans=[span_1],
        )

        # we will only have to proposals as the full replacement and the number replacement are identical

        # second Annotation Set
        span_2 = Span(start_offset=19, end_offset=27)
        annotation_set_2 = AnnotationSet(id_=2, document=document, label_set=label_set)
        annotation_2 = Annotation(
            document=document,
            is_correct=True,
            annotation_set=annotation_set_2,
            label=label,
            label_set=label_set,
            spans=[span_2],
        )
        # we will only have to proposals as the full replacement and the number replacement are identical
        self.assertEqual(2, len(annotation_1.tokens()))
        self.assertEqual(2, len(annotation_2.tokens()))

        regex = label.find_regex(category=category)
        expected = (
            r'[ ]{1,2}(?:(?P<Label_22_N_None_5>\d\d\.\d\d\.\d\d\d\d)|(?P<Label_22_N_None_19>\d\.\d\.\d\d\d\d))[ ]{1,2}'
        )

        assert expected in regex[0]

    def test_regex_first_annotation_in_row(self):
        """Add a test if two annotated offsets are connected to each other "please pay <GrossAmount><Currency>."""
        # first name and last name are in one line in the document:
        first_names = self.prj.get_label_by_name('Vorname')
        last_name = self.prj.get_label_by_name('Nachname')
        category = self.category
        assert len(first_names.annotations(categories=[category])) == 25
        assert len(first_names.regex(categories=[category])) == 1

        proposals = first_names.find_regex(category=category)
        # assert len(proposals) == 2 + 3  # The default entity regexes

        first_name_proposal = proposals[0]

        regex_to_remove_groupnames_full = re.compile(r'\?P<.*?>')
        last_name_base = last_name.base_regex(category=category)
        last_name_base = re.sub(regex_to_remove_groupnames_full, '', last_name_base)

        assert '(?P<Label_865_' in first_name_proposal
        # assert '>[A-ZÄÖÜ][a-zäöüß]+[-][A-ZÄÖÜ][a-zäöüß]+[-][A-ZÄÖÜ][a-zäöüß]+)' in male_first_name
        assert '>[A-ZÄÖÜ][a-zäöüß]+[-][A-ZÄÖÜ][a-zäöüß]+)' in first_name_proposal
        assert last_name_base in first_name_proposal  # Nachname label
        textcorpus = ''.join([doc.text for doc in self.prj.documents])
        results = regex_matches(textcorpus, first_name_proposal, filtered_group='Label_865')
        assert [result['value'] for result in results] == [
            'Erna-Muster',
            'Daniel-Muster',
            'Inge-Muster',
            'Max-Muster',
            'Birgit-Muster',
            'Inge-Muster',
            'Max-Muster',
            'Xaver-Muster',
            'Ernst-Muster',
            'Heinz-Muster',
            'Max-Muster',
            'Ella-Muster',
            'Leo-Muster',
            'Thomas-Muster',
            'Hugo-Muster',
            'Heike-Muster',
            'Frank-Muster',
            'Berta-Muster',
            'Sparda-Bank',
            'Bruno-Muster',
            'Sparda-Bank',
            'Inge-Muster',
            'Hugo-Muster',
            'Leo-Muster',
            'Lydia-Muster',
            'Karl-Muster',
            'Rita-Muster',
            'Marie-Muster',
        ]

    def test_wage_regex(self):
        """Return the regex for the tax class regex."""
        tax = next(x for x in self.prj.labels if x.name == 'Steuerklasse')
        category = self.prj.get_category_by_id(63)
        regex = tax.find_regex(category=category)[0]
        assert '(?P<Label_860_N_' in regex

    def test_regex_second_annotation_in_row(self):
        """Delete the last character of the regex solution as only for some runs it will contain a line break."""
        first_names = self.prj.get_label_by_name('Vorname')
        last_name = self.prj.get_label_by_name('Nachname')

        category = self.prj.get_category_by_id(63)

        regex_to_remove_groupnames_full = re.compile(r'\?P<.*?>')

        last_name_base = last_name.base_regex(category=category)
        last_name_base = re.sub(regex_to_remove_groupnames_full, '', last_name_base)

        first_name_base = last_name.base_regex(category=category)
        first_name_base = re.sub(regex_to_remove_groupnames_full, '', first_name_base)

        assert len(first_names.find_regex(category=category)) == 1

        first_name_regex = first_names.find_regex(category=category)[0]

        assert '(?P<Label_865_' in first_name_regex
        assert 'Label_866' not in first_name_regex

        assert '>[A-ZÄÖÜ][a-zäöüß]+[-][A-ZÄÖÜ][a-zäöüß]+)' in first_name_regex
        assert last_name_base in first_name_regex

        assert len(last_name.find_regex(category=category)) == 2

        last_name_regex = last_name.find_regex(category=category)[0]

        assert '(?P<Label_866_' in last_name_regex
        assert 'Label_865' not in last_name_regex

        textcorpus = ''.join([doc.text for doc in self.prj.documents])
        results = regex_matches(textcorpus, last_name_regex, filtered_group='Label_866')
        assert [result['value'] for result in results] == [
            'Schmidt',
            'Schlosser',
            'Meier',
            'Schlosser-Azubi',
            'Xoxmann',
            'Egelmann',
            'Wichtig',
            'Schlosser',
            'Sorglos',
            'Leichtsinn',
            'Lau',
            'Huber',
            'Förster',
            'Lämmer',
            'Bauch',
            'Bauch',
            'Meier',
            'Baltus',
            'Winzig',
            'Waldmeister',
            'Sinner',
            'Sinnvoll',
            'Wald',
        ]


class Test_named_group_multi_match:
    """Test overlapping and multiple Annotation in one string."""

    # this test case was created as one issue in the PatternExtractionModel: The match of "Hansi repeats" would
    # overwrite all other Hansi ... matches, as Hansi was the key in the dictionary to aggregate his information (verbs)
    text = 'Hansi eats, Michael sleeps, Hansi works, Hansi repeats'
    rgx = r'(?P<Person>[A-Z][^ ]*) (?P<verb>[^ ,]*)'
    results = regex_matches(text, rgx)

    def test_number_of_matches(self):
        """Count the matches."""
        assert len(self.results) == 12

    def test_keys_in_first_match(self):
        """Check the keys after evaluating the regex."""
        for i, result_dict in enumerate(self.results):
            assert (
                list(result_dict.keys()).sort()
                == ['regex_used', 'name', 'regex_group', 'value', 'start_offset', 'end_offset', 'start_text'].sort()
            )

    def test_result_one(self):
        """Check the first regex group."""
        assert self.results[0] == {
            'regex_used': "'(?P<Person>[A-Z][^ ]*) (?P<verb>[^ ,]*)'",
            'regex_group': 'Person',
            'value': 'Hansi',
            'start_offset': 0,
            'end_offset': 5,
            'start_text': 0,
        }

    def test_result_two(self):
        """Check the second regex group."""
        assert self.results[1] == {
            'regex_used': "'(?P<Person>[A-Z][^ ]*) (?P<verb>[^ ,]*)'",
            'regex_group': 'verb',
            'value': 'eats',
            'start_offset': 6,
            'end_offset': 10,
            'start_text': 0,
        }

    def test_result_three(self):
        """Check the third regex group."""
        assert self.results[2] == {
            'regex_used': "'(?P<Person>[A-Z][^ ]*) (?P<verb>[^ ,]*)'",
            'value': 'Hansi eats',
            'regex_group': '0',
            'start_offset': 0,
            'end_offset': 10,
            'start_text': 0,
        }

    def test_result_four(self):
        """Check the fourth regex group."""
        assert self.results[3] == {
            'regex_used': "'(?P<Person>[A-Z][^ ]*) (?P<verb>[^ ,]*)'",
            'regex_group': 'Person',
            'value': 'Michael',
            'start_offset': 12,
            'end_offset': 19,
            'start_text': 0,
        }

    def test_result_five(self):
        """Check the fifth regex group."""
        assert self.results[4] == {
            'regex_used': "'(?P<Person>[A-Z][^ ]*) (?P<verb>[^ ,]*)'",
            'regex_group': 'verb',
            'value': 'sleeps',
            'start_offset': 20,
            'end_offset': 26,
            'start_text': 0,
        }

    def test_result_six(self):
        """Check the sixth regex group."""
        assert self.results[5] == {
            'regex_used': "'(?P<Person>[A-Z][^ ]*) (?P<verb>[^ ,]*)'",
            'value': 'Michael sleeps',
            'regex_group': '0',
            'start_offset': 12,
            'end_offset': 26,
            'start_text': 0,
        }

    def test_result_seven(self):
        """Check the seventh regex group."""
        assert self.results[6] == {
            'regex_used': "'(?P<Person>[A-Z][^ ]*) (?P<verb>[^ ,]*)'",
            'regex_group': 'Person',
            'value': 'Hansi',
            'start_offset': 28,
            'end_offset': 33,
            'start_text': 0,
        }

    def test_result_eight(self):
        """Check the eights regex group."""
        assert self.results[7] == {
            'regex_used': "'(?P<Person>[A-Z][^ ]*) (?P<verb>[^ ,]*)'",
            'regex_group': 'verb',
            'value': 'works',
            'start_offset': 34,
            'end_offset': 39,
            'start_text': 0,
        }

    def test_result_nine(self):
        """Check the ninths regex group."""
        assert self.results[8] == {
            'regex_used': "'(?P<Person>[A-Z][^ ]*) (?P<verb>[^ ,]*)'",
            'value': 'Hansi works',
            'regex_group': '0',
            'start_offset': 28,
            'end_offset': 39,
            'start_text': 0,
        }

    def test_result_ten(self):
        """Check the tenths regex group."""
        assert self.results[9] == {
            'regex_used': "'(?P<Person>[A-Z][^ ]*) (?P<verb>[^ ,]*)'",
            'regex_group': 'Person',
            'value': 'Hansi',
            'start_offset': 41,
            'end_offset': 46,
            'start_text': 0,
        }

    def test_result_eleven(self):
        """Check the eleventh regex group."""
        assert self.results[10] == {
            'regex_used': "'(?P<Person>[A-Z][^ ]*) (?P<verb>[^ ,]*)'",
            'regex_group': 'verb',
            'value': 'repeats',
            'start_offset': 47,
            'end_offset': 54,
            'start_text': 0,
        }

    def test_result_twelve(self):
        """Check the twelfths regex group."""
        assert self.results[11] == {
            'regex_used': "'(?P<Person>[A-Z][^ ]*) (?P<verb>[^ ,]*)'",
            'value': 'Hansi repeats',
            'regex_group': '0',
            'start_offset': 41,
            'end_offset': 54,
            'start_text': 0,
        }


class Test_match_by_named_regex:
    """Check overlapping and multiple regex groups."""

    text = 'Hansi eats, Michael sleeps'
    rgx = r'(?P<Person>[A-Z][^ ]*) (?P<verb>[^ ,]*)'
    results = regex_matches(text, rgx)

    def test_number_of_matches(self):
        """Check the number of returned groups before checking every individually."""
        assert len(self.results) == 6

    def test_keys_in_first_match(self):
        """Check the keys after evaluating the regex."""
        for i, result_dict in enumerate(self.results):
            assert (
                list(result_dict.keys()).sort()
                == ['regex_used', 'name', 'regex_group', 'value', 'start_offset', 'end_offset', 'start_text'].sort()
            )

    def test_result_one(self):
        """Check the first regex group."""
        assert self.results[0] == {
            'regex_used': "'(?P<Person>[A-Z][^ ]*) (?P<verb>[^ ,]*)'",
            'regex_group': 'Person',  # named regex groups don't get an index of the group
            'value': 'Hansi',
            'start_offset': 0,
            'end_offset': 5,
            'start_text': 0,
        }

    def test_result_two(self):
        """Check the second regex group."""
        assert self.results[1] == {
            'regex_used': "'(?P<Person>[A-Z][^ ]*) (?P<verb>[^ ,]*)'",
            'regex_group': 'verb',
            'value': 'eats',
            'start_offset': 6,
            'end_offset': 10,
            'start_text': 0,
        }

    def test_result_three(self):
        """Check the third regex group."""
        assert self.results[2] == {
            'regex_used': "'(?P<Person>[A-Z][^ ]*) (?P<verb>[^ ,]*)'",
            'value': 'Hansi eats',
            'regex_group': '0',
            'start_offset': 0,
            'end_offset': 10,
            'start_text': 0,
        }

    def test_result_four(self):
        """Check the fourth regex group."""
        assert self.results[3] == {
            'regex_used': "'(?P<Person>[A-Z][^ ]*) (?P<verb>[^ ,]*)'",
            'regex_group': 'Person',
            'value': 'Michael',
            'start_offset': 12,
            'end_offset': 19,
            'start_text': 0,
        }

    def test_result_five(self):
        """Check the fifth regex group."""
        assert self.results[4] == {
            'regex_used': "'(?P<Person>[A-Z][^ ]*) (?P<verb>[^ ,]*)'",
            'regex_group': 'verb',
            'value': 'sleeps',
            'start_offset': 20,
            'end_offset': 26,
            'start_text': 0,
        }

    def test_result_six(self):
        """Check the sixth regex group."""
        assert self.results[5] == {
            'regex_used': "'(?P<Person>[A-Z][^ ]*) (?P<verb>[^ ,]*)'",
            'value': 'Michael sleeps',
            'regex_group': '0',
            'start_offset': 12,
            'end_offset': 26,
            'start_text': 0,
        }


class Test_multi_match_by_unamed_regex:
    """Check overlapping and multiple regex groups."""

    text = 'Hansi eats, Michael sleeps, Hansi works, Hansi repeats'
    rgx = r'Hansi ([^ ]*),'
    results = regex_matches(text, rgx)

    def test_number_of_matches(self):
        """Check the number of returned groups before checking every individually."""
        assert len(self.results) == 4

    def test_keys_in_first_match(self):
        """Check the keys after evaluating the regex."""
        for i, result_dict in enumerate(self.results):
            assert (
                list(result_dict.keys()).sort()
                == ['regex_used', 'name', 'regex_group', 'value', 'start_offset', 'end_offset', 'start_text'].sort()
            )

    def test_result_one(self):
        """Check the first regex group."""
        assert self.results[0] == {
            'regex_used': "'Hansi ([^ ]*),'",
            'regex_group': '1',  # the full regex "regex_group:0" has one unnamed group, called 1
            'value': 'eats',
            'start_offset': 6,
            'end_offset': 10,
            'start_text': 0,
        }

    def test_result_two(self):
        """Check the second regex group."""
        assert self.results[1] == {
            'regex_used': "'Hansi ([^ ]*),'",
            'value': 'Hansi eats,',
            'regex_group': '0',
            'start_offset': 0,
            'end_offset': 11,
            'start_text': 0,
        }

    def test_result_three(self):
        """Check the third regex group."""
        assert self.results[2] == {
            'regex_used': "'Hansi ([^ ]*),'",
            'regex_group': '1',
            'value': 'works',
            'start_offset': 34,
            'end_offset': 39,
            'start_text': 0,
        }

    def test_result_four(self):
        """Check the fourth regex group."""
        assert self.results[3] == {
            'regex_used': "'Hansi ([^ ]*),'",
            'value': 'Hansi works,',
            'regex_group': '0',
            'start_offset': 28,
            'end_offset': 40,
            'start_text': 0,
        }


class Test_multi_match_by_ungrouped_regex:
    """Check overlapping and multiple regex groups."""

    text = 'Hansi eats, Michael sleeps, Hansi works, Hansi repeats'
    rgx = r'Hansi [^ ]*,'
    results = regex_matches(text, rgx)

    def test_number_of_matches(self):
        """Check the number of returned groups before checking every individually."""
        assert len(self.results) == 2

    def test_keys_in_first_match(self):
        """Check the keys after evaluating the regex."""
        for i, result_dict in enumerate(self.results):
            assert (
                list(result_dict.keys()).sort()
                == ['regex_used', 'name', 'regex_group', 'value', 'start_offset', 'end_offset', 'start_text'].sort()
            )

    def test_result_one(self):
        """Check the first regex group."""
        assert self.results[0] == {
            'regex_used': "'Hansi [^ ]*,'",
            'value': 'Hansi eats,',
            'regex_group': '0',  # index 0 means this if the match of the full regex
            'start_offset': 0,
            'end_offset': 11,
            'start_text': 0,
        }  # the text used for testing is not within a other text

    def test_result_two(self):
        """Check the second regex group."""
        assert self.results[1] == {
            'regex_used': "'Hansi [^ ]*,'",
            'value': 'Hansi works,',
            'regex_group': '0',  # index 0 means this if the match of the full regex
            'start_offset': 28,
            'end_offset': 40,
            'start_text': 0,
        }  # the text used for testing is not within a other text


def test_regex_spans_runtime():
    """Profile time to calculate regex."""
    setup_code = textwrap.dedent(
        """
    from konfuzio_sdk.regex import regex_matches

    text = 'Hansi eats, Michael sleeps, Hansi works, Hansi repeats, ' * 5000
    rgx = r'Hansi [^ ]*,'
    """
    )
    test_code = textwrap.dedent(
        """
    regex_matches(text, rgx)
    """
    )
    runtime = timeit(stmt=test_code, setup=setup_code, number=10) / 10
    logger.info(f'Runtime for 5.000 lines: {runtime:.5f} seconds.')
    assert runtime < 0.03 * 4  # multiply by 4 as Azure VM is 4 times slower than local machine
