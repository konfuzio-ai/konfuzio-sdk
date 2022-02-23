"""Test Regex functionality on strings."""
import textwrap
from timeit import timeit

from konfuzio_sdk.regex import (
    suggest_regex_for_string,
    merge_regex,
    generic_candidate_function,
    plausible_regex,
    get_best_regex,
)
from konfuzio_sdk.utils import does_not_raise

import logging
import re
import unittest

import pytest

from konfuzio_sdk.regex import regex_spans
from konfuzio_sdk.data import Project, Annotation, Label

logger = logging.getLogger(__name__)

suggest_regex_for_string_data = [
    ({'string': ' '}, r'[ ]+', does_not_raise()),
    ({'string': 'VAG GmbH', 'replace_characters': True}, r'[A-ZÄÖÜ]+[ ]+[A-ZÄÖÜ][a-zäöüß]+[A-ZÄÖÜ]', does_not_raise()),
    (
        {'string': 'a2bau GmbH', 'replace_characters': True},
        r'[a-zäöüß]\d[a-zäöüß]+[ ]+[A-ZÄÖÜ][a-zäöüß]+[A-ZÄÖÜ]',
        does_not_raise(),
    ),
    (
        {'string': 'Bau-Wohn-Gesellschaft', 'replace_characters': True},
        '[A-ZÄÖÜ][a-zäöüß]+[-][A-ZÄÖÜ][a-zäöüß]+[-][A-ZÄÖÜ][a-zäöüß]+',
        does_not_raise(),
    ),
    ({'string': 'Bau-Wohn-Gesellschaft', 'replace_characters': False}, 'Bau[-]Wohn[-]Gesellschaft', does_not_raise()),
    ({'string': '  '}, '[ ]{2,}', does_not_raise()),
    ({'string': ' $ hello @ ! ? # [ ]'}, r'[ ]+\$[ ]+hello[ ]+\@[ ]+\![ ]+\?[ ]+\#[ ]+\[[ ]+\]', does_not_raise()),
    ({'string': '(hello)+'}, '\\(hello\\)[\\+]', does_not_raise()),
    ({'string': '(hello) +'}, '\\(hello\\)[ ]+[\\+]', does_not_raise()),
    (
        {'string': 'Große Bäume, haben Äpfel.', 'replace_characters': True},
        r'[A-ZÄÖÜ][a-zäöüß]+[ ]+[A-ZÄÖÜ][a-zäöüß]+\,[ ]+[a-zäöüß]+[ ]+[A-ZÄÖÜ][a-zäöüß]+\.',
        does_not_raise(),
    ),
    # check äöüß
    ({'string': 'Hello                  World'}, 'Hello[ ]{2,}World', does_not_raise()),
    ({'string': 'Hello\n\nWorld'}, 'Hello\n\nWorld', does_not_raise()),
    (
        {'string': '  GRUPPE\n\n   Industrie-Police  \\\n  Nachtrag Nr. '},
        '[ ]{2,}GRUPPE\n\n[ ]{2,}Industrie[-]Police[ ]{2,}\\\\\n[ ]{2,}Nachtrag[ ]+Nr\\.[ ]+',
        does_not_raise(),
    ),
    ({'string': 'Hello\nWorld'}, 'Hello\nWorld', does_not_raise()),
    ({'string': 'Am 22.05.2020'}, r'Am[ ]+\d\d\.\d\d\.\d\d\d\d', does_not_raise()),
    ({'string': 'Am 22.05.2020'}, r'Am[ ]+\d\d\.\d\d\.\d\d\d\d', does_not_raise()),
    ({'string': '[mailto:ch@helm-nagel.com]'}, '\\[mailto:ch\\@helm[-]nagel\\.com\\]', does_not_raise()),
    ({'string': '              \\       ifr'}, '[ ]{2,}\\\\[ ]{2,}ifr', does_not_raise()),
    (
        {'string': '22. Januar 2020', 'replace_characters': True},
        r'\d\d\.[ ]+[A-ZÄÖÜ][a-zäöüß]+[ ]+\d\d\d\d',
        does_not_raise(),
    ),
    (
        {'string': '22. \nJanuar 2020', 'replace_characters': True},
        '\\d\\d\\.[ ]+\n[A-ZÄÖÜ][a-zäöüß]+[ ]+\\d\\d\\d\\d',
        does_not_raise(),
    ),
    (
        {'string': '22. \n\nJanuar 2020', 'replace_characters': True},
        '\\d\\d\\.[ ]+\n\n[A-ZÄÖÜ][a-zäöüß]+[ ]+\\d\\d\\d\\d',
        does_not_raise(),
    ),
    # ('input', 'output', pytest.raises(ValueError)),
    ({'string': '{hello'}, '\\{hello', does_not_raise()),
    ({'string': 'he*llo'}, 'he\\*llo', does_not_raise()),
    ({'string': 'he|llo'}, 'he\\|llo', does_not_raise()),
    # The following three examples, show the main functionality of the suggest_regex_for_string function.
    (
        {'string': 'Achtung       Kontrolle 911    ', 'replace_numbers': False},
        'Achtung[ ]{2,}Kontrolle[ ]+911[ ]{2,}',
        does_not_raise(),
    ),
    ({'string': 'Achtung       Kontrolle 911    '}, r'Achtung[ ]{2,}Kontrolle[ ]+\d\d\d[ ]{2,}', does_not_raise()),
    # see issue https://gitlab.com/konfuzio/objectives/-/issues/1373
    # ({'string': 'Garage/n'}, r'Garage\/n', does_not_raise()),
    (
        {'string': 'Achtung       Kontrolle 911    ', 'replace_characters': True},
        r'[A-ZÄÖÜ][a-zäöüß]+[ ]{2,}[A-ZÄÖÜ][a-zäöüß]+[ ]+\d\d\d[ ]{2,}',
        does_not_raise(),
    ),
    (
        {'string': 'hello.mio@konfuzio.c', 'replace_characters': True},
        r'[a-zäöüß]+\.[a-zäöüß]+\@[a-zäöüß]+\.[a-zäöüß]',
        does_not_raise(),
    ),
]


@pytest.mark.parametrize("input, output, expected_exception", suggest_regex_for_string_data)
def test_suggest_regex_for_string(input, output, expected_exception):
    """Test different strings to represent them as a regex."""
    with expected_exception:
        if output is not None:
            assert suggest_regex_for_string(**input) == output
        else:
            assert suggest_regex_for_string(**input) is None


def test_merge_regex():
    """Test to merge multiple regex to one group."""
    my_regex = merge_regex([r'[a-z]+', r'\d+', r'[A-Z]+'])
    assert my_regex == r'(?:[a-z]+|[A-Z]+|\d+)'
    test_string = "39 gallons is the per capita consumption of softdrinks in US."
    tokens = regex_spans(test_string, my_regex)
    assert len(tokens) == len(test_string.split(' '))


def test_regex_plausibility_compile_error():
    """Test the plausibility check for regex."""
    assert 'x' == plausible_regex(r'x', 'xxx')
    assert '' == plausible_regex(r'****', 'xxx')
    assert '' == plausible_regex(r'\d', 'xxx')


def test_regex_spans_with_invalid_regex_group_name():
    """Test to run regex_spans with an invalid group name."""
    result = regex_spans('I go home at 5 AM.', regex=r'(?P<9variable>\d)')
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


def test_get_best_regex():
    """Test to evaluate an empty list."""
    assert get_best_regex([]) == []


class TestTokens(unittest.TestCase):
    """Create Tokens from example Data of the Konfuzio Host."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the Project."""
        cls.prj = Project(id_=46)

    @pytest.mark.xfail(reason='We force to have an annotation_set to init an Annotation.')
    def test_token_replacement_only_whitespace_2(self):
        """Create a regex where only whitespaces are replaced."""
        label_data = {
            "description": "Betrag des jew. Bruttobezugs",
            "id_": 88888888,
            "shortcut": "N/A",
            "text": "Betrag",
            "get_data_type_display": "Text",
            "text_clean": "Betrag",
            "token_full_replacement": False,
            "token_whitespace_replacement": True,
            "token_number_replacement": False,
            "label_sets": self.prj.label_sets,
            "has_multiple_top_candidates": False,
        }
        new_label = Label(project=self.prj, **label_data)
        self.prj.add_label(new_label)
        doc = self.prj.documents[0]

        new_anno = Annotation(
            start_offset=127,
            end_offset=140,
            label=new_label.id_,
            label_set_id=new_label.label_sets[0].id_,  # hand selected Document label_set
            revised=True,
            is_correct=True,
            accuracy=0.98765431,
            document=doc
            # annotation_set=doc.get_annotation_set_by_id()
        )

        assert len(new_label.annotations) == 1

        assert new_anno.offset_string == ['Dezember 2018']
        assert len(new_anno.tokens()) == 1

        regex = new_anno.tokens()[0]['regex']
        assert '_W_' in regex
        assert '_F_' not in regex
        assert '_N_' not in regex
        assert regex == '(?P<Betrag_W_None_fallback>Dezember[ ]+2018)'

    def test_generic_candidate_function(self):
        """Test to create a function which applies a RegEx."""
        my_function = generic_candidate_function(r"\d\d.\d\d.\d\d\d\d")
        self.assertEqual(
            (['23.04.2055'], ['I was born at the ', '.'], [(18, 28)]), my_function('I was born at the 23.04.2055.')
        )

    def test_label_keyword_token(self):
        """Extract value for Steuerklasse."""
        label = next(x for x in self.prj.labels if x.name == 'Steuerklasse')
        assert label.tokens()
        tokens = sorted(label.tokens())
        assert len(tokens) == 1

    def test_label_plz_token(self):
        """Extract all tax gross amounts of all payslips."""
        label = self.prj.get_label_by_name('Steuer-Brutto')
        tokens = label.tokens()
        assert len(tokens) == 4
        assert '(?P<SteuerBrutto_N_' in tokens[0]

    def test_label_token_auszahlungsbetrag(self):
        """Return the summary of all regex needed to get the wage."""
        label = self.prj.get_label_by_name('Auszahlungsbetrag')

        tokens = sorted(label.tokens())
        assert len(tokens) == 3
        assert '(?P<Auszahlungsbetrag_' in tokens[0]
        assert '(?P<Auszahlungsbetrag_' in tokens[1]
        assert '(?P<Auszahlungsbetrag_' in tokens[2]
        assert '>\\d\\d\\d\\,\\d\\d)' in tokens[0] + tokens[1] + tokens[2]
        assert '>\\d\\.\\d\\d\\d\\,\\d\\d)' in tokens[0] + tokens[1] + tokens[2]
        assert '>\\d\\d\\,[ ]+\\d\\d[-])' in tokens[0] + tokens[1] + tokens[2]

    def test_label_empty_annotations(self):
        """Empty Annotations should not create regex."""
        try:
            label = next(x for x in self.prj.labels if len(x.annotations) == 0)
            assert sorted(label.tokens()) == []
        except StopIteration:
            pass

    @pytest.mark.skip("Line breaks are not supported.")
    def test_linebreaks_in_tokens(self):
        """Calculate a Annotation that has line breaks."""
        label = next(x for x in self.prj.labels if len(x.annotations) == 0)
        doc = self.prj.documents[0]
        new_anno_1 = Annotation(
            start_offset=177,
            end_offset=179,
            label=label.id_,
            label_set_id=label.label_sets[0].id_,  # hand selected Document label_set
            annotation_set_id=1,
            revised=True,
            is_correct=True,
            accuracy=0.98765431,
            document=doc,
        )
        tokens = new_anno_1.tokens()
        assert len(tokens) == 3
        assert new_anno_1.tokens()[0]['regex'] == '(?P<EMPTY_LABEL_W_None_fallback>\n[ ]+)'
        assert new_anno_1.tokens()[1]['regex'] == '(?P<EMPTY_LABEL_N_None_fallback>\n[ ]+)'
        assert new_anno_1.tokens()[2]['regex'] == '(?P<EMPTY_LABEL_F_None_fallback>\n[ ]+)'


class TestRegexGenerator(unittest.TestCase):
    """Test to create regex based on data online."""

    document_count = 27
    correct_annotations = 25

    @classmethod
    def setUpClass(cls) -> None:
        """Load the Project data from the Konfuzio Host."""
        cls.prj = Project(id_=46)
        assert len(cls.prj.documents) == cls.document_count
        assert len(cls.prj.labels[0].correct_annotations) == cls.correct_annotations

    @classmethod
    def tearDownClass(cls) -> None:
        """Check that no local data was changed by the tests."""
        assert len(cls.prj.documents) == cls.document_count
        assert len(cls.prj.labels[0].correct_annotations) == cls.correct_annotations
        # cls.prj.delete()

    def test_regex_single_annotation_in_row(self):
        """Build a simple extraction for an amount."""
        analyzed_label = self.prj.get_label_by_name('Auszahlungsbetrag')
        for label in self.prj.labels:
            label.regex(update=True)

        assert len(analyzed_label.regex()) == 1
        # we now use the f_score to balance the annotation_precision and the document_recall
        # thereby we find the top regex easily: we only need one regex to match all findings
        for document in self.prj.documents:
            annos = document.annotations(label=analyzed_label)
            for auto_regex in analyzed_label.regex():
                findings = re.findall(auto_regex, document.text)
                clean_findings = set([item for sublist in findings for item in sublist])
                for anno in annos:
                    for span in anno._spans:
                        assert span.offset_string in clean_findings

    def test_two_annotations_with_same_label_close_to_each_other(self):
        """Test that any text to regex subset works, even if the text contains two Annotations with the same label."""
        # previously this resulted in "sre_constants.error: redefinition of group name" error
        document = self.prj.documents[-2]
        assert len(document.annotations()) == 28
        # new functionality let you create a regex for a region in a document, from 950 to 1500 character
        regex = document.regex(start_offset=950, end_offset=1500)[0]
        # in this Document region we will use 2 times the regex tokens for Ort, each uses two tokens
        assert regex.count('(?P<Vorname_') == 1

    @unittest.skip('We do not support multiple Annotations in one offset for now')
    def test_regex_first_annotation_in_row(self):
        """Todo add a test if two annotated offsets are connected to each other "please pay <GrossAmount><Currency>."""
        # first name and last name are in one line in the document:
        first_names = self.prj.get_label_by_name('Vorname')
        assert len(first_names.annotations) == 27
        assert len(first_names.tokens()) == 1
        token_three_names = '>[A-ZÄÖÜ][a-zäöüß]+[-][A-ZÄÖÜ][a-zäöüß]+[-][A-ZÄÖÜ][a-zäöüß]+)'
        assert (token_three_names in s for s in first_names.tokens())
        token_two_names = '>[A-ZÄÖÜ][a-zäöüß]+[-][A-ZÄÖÜ][a-zäöüß]+)'

        assert (token_two_names in s for s in first_names.tokens())

        proposals = first_names.find_regex()
        assert len(proposals) == 2 + 3  # The default entity regexes

        male_first_name = proposals[0]
        assert 'Herrn' in male_first_name
        assert '(?P<Vorname_' in male_first_name
        assert '>[A-ZÄÖÜ][a-zäöüß]+[-][A-ZÄÖÜ][a-zäöüß]+[-][A-ZÄÖÜ][a-zäöüß]+)' in male_first_name
        assert '>[A-ZÄÖÜ][a-zäöüß]+[-][A-ZÄÖÜ][a-zäöüß]+)' in male_first_name
        assert '(?P<Nachname_' in male_first_name
        assert '>[A-ZÄÖÜ][a-zäöüß]+)' in male_first_name
        textcorpus = ''.join([doc.text for doc in self.prj.documents])
        results_male = regex_spans(textcorpus, male_first_name, filtered_group=first_names.name)
        assert [result['value'] for result in results_male] == [
            'Oskar-Muster',
            'Tillmannl-Muster',
            'Heinz-Muster',
            'Samuel-Muster',
            'Marco-Paul-Muster',
            'Walter-Muster',
            'Lukas-Muster',
            'Hugo-Muster',
        ]

        female_first_name = proposals[1]
        assert 'Frau' in female_first_name
        assert '(?P<Vorname_' in female_first_name
        assert '>[A-ZÄÖÜ][a-zäöüß]+[-][A-ZÄÖÜ][a-zäöüß]+[-][A-ZÄÖÜ][a-zäöüß]+)' in female_first_name
        assert '>[A-ZÄÖÜ][a-zäöüß]+[-][A-ZÄÖÜ][a-zäöüß]+)' in female_first_name
        assert '(?P<Nachname_' in female_first_name
        assert '>[A-ZÄÖÜ][a-zäöüß]+)' in female_first_name
        textcorpus = ''.join([doc.text for doc in self.prj.documents])
        results_female = regex_spans(textcorpus, female_first_name, filtered_group=first_names.name)
        assert [result['value'] for result in results_female] == [
            'Heike-Muster',
            'Cordula-Muster',
            'Valerie-Muster',
            'Franziska-Muster',
            'Dorothee-Muster',
            'Cedrica-Muster',
            'Sara-Muster',
        ]

    def test_wage_regex(self):
        """Return the regex for the tax class regex."""
        tax = next(x for x in self.prj.labels if x.name == 'Steuerklasse')
        regex = tax.find_regex()[0]
        assert '(?P<Steuerklasse_' in regex

    @unittest.skip('We do not support multiple Annotations in one offset for now')
    def test_regex_second_annotation_in_row(self):
        """Delete the last character of the regex solution as only for some runs it will contain a line break."""
        last_names = self.prj.get_label_by_name('Vorname')
        assert len(last_names.find_regex()) == 1
        last_name_regex = last_names.find_regex()[0]
        assert '(?P<Vorname_' in last_name_regex
        # assert '>[A-ZÄÖÜ][a-zäöüß]+[-][A-ZÄÖÜ][a-zäöüß]+[-][A-ZÄÖÜ][a-zäöüß]+)' in last_name_regex
        assert '>[A-ZÄÖÜ][a-zäöüß]+[-][A-ZÄÖÜ][a-zäöüß]+)' in last_name_regex
        assert '(?P<Nachname_' in last_name_regex
        assert '>[A-ZÄÖÜ][a-zäöüß]+)' in last_name_regex
        textcorpus = ''.join([doc.text for doc in self.prj.documents])
        results = regex_spans(textcorpus, last_name_regex, filtered_group=last_names.name)
        assert [result['value'] for result in results] == [
            'Förster',
            'Wissen',
            'Kork',
            'Morgen',
            'Brand',
            'Dekano',
            'Abend',
            'Tilly',
            'Rimmel',
            'Wallenstein',
            'Lichter',
            'Huber',
            'Garten',
            'Trimmer',
            'Mustermann',
        ]


class Test_named_group_multi_match:
    """Test overlapping and multiple Annotation in one string."""

    # this test case was created as one issue in the PatternExtractionModel: The match of "Hansi repeats" would
    # overwrite all other Hansi ... matches, as Hansi was the key in the dictionary to aggregate his information (verbs)
    text = 'Hansi eats, Michael sleeps, Hansi works, Hansi repeats'
    rgx = r'(?P<Person>[A-Z][^ ]*) (?P<verb>[^ ,]*)'
    results = regex_spans(text, rgx)

    def test_number_of_matches(self):
        """Count the matches."""
        assert len(self.results) == 12

    def test_keys_in_first_match(self):
        """Check the keys after evaluating the regex."""
        for i, dict in enumerate(self.results):
            assert (
                list(dict.keys()).sort()
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
    results = regex_spans(text, rgx)

    def test_number_of_matches(self):
        """Check the number of returned groups before checking every individually."""
        assert len(self.results) == 6

    def test_keys_in_first_match(self):
        """Check the keys after evaluating the regex."""
        for i, dict in enumerate(self.results):
            assert (
                list(dict.keys()).sort()
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
    results = regex_spans(text, rgx)

    def test_number_of_matches(self):
        """Check the number of returned groups before checking every individually."""
        assert len(self.results) == 4

    def test_keys_in_first_match(self):
        """Check the keys after evaluating the regex."""
        for i, dict in enumerate(self.results):
            assert (
                list(dict.keys()).sort()
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
    results = regex_spans(text, rgx)

    def test_number_of_matches(self):
        """Check the number of returned groups before checking every individually."""
        assert len(self.results) == 2

    def test_keys_in_first_match(self):
        """Check the keys after evaluating the regex."""
        for i, dict in enumerate(self.results):
            assert (
                list(dict.keys()).sort()
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
    from konfuzio_sdk.regex import regex_spans

    text = 'Hansi eats, Michael sleeps, Hansi works, Hansi repeats, ' * 5000
    rgx = r'Hansi [^ ]*,'
    """
    )
    test_code = textwrap.dedent(
        """
    regex_spans(text, rgx)
    """
    )
    runtime = timeit(stmt=test_code, setup=setup_code, number=10) / 10
    logger.info(f'Runtime for 5.000 lines: {runtime:.5f} seconds.')
    assert runtime < 0.03 * 4  # multiply by 4 as Azure VM is 4 times slower than local machine
