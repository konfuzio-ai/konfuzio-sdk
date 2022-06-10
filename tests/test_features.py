import math

import pytest
from konfuzio_sdk.pipelines.features import (num_count, digit_count, space_count, special_count, vowel_count,
                                             upper_count, duplicate_count, substring_count, unique_char_count,
                                             year_month_day_count, strip_accents, count_string_differences)


def test_feat_num_count():
    # Debug code for df: df[df[self.label_feature_list].isin([np.nan, np.inf, -np.inf]).any(1)]
    error_string_1 = '10042020200917e747'
    res = num_count(error_string_1)
    assert not math.isinf(res)

    error_string_2 = '26042020081513e749'
    res = num_count(error_string_2)
    assert not math.isinf(res)


def test_digit_count():
    result = digit_count("123456789ABC")
    assert result == 9


def test_num_count_wrong_format():
    num_count("word")


def test_space_count():
    result = space_count("1 2 3 4 5 ")
    assert result == 5


def test_space_count_with_tabs():
    result = space_count("\t")
    assert result == 4


def test_special_count():
    result = special_count("!_:ThreeSpecialChars")
    assert result == 3


def test_vowel_count():
    result = vowel_count("vowel")
    assert result == 2


def test_upper_count():
    result = upper_count("UPPERlower!")
    assert result == 5


def test_num_count():
    result = num_count("1.500,34")
    assert result == 1500.34


def test_duplicate_count():
    result = duplicate_count("AAABBCCDDE")
    assert result == 9


def test_substring_count():
    result = substring_count(["Apple", "Annaconda"], "a")
    assert result == [1, 3]


def test_unique_char_count():
    result = unique_char_count("12345678987654321")
    assert result == 9


def test_accented_char_strip_and_count():
    l_test = ['Hallà', 'àèìòùé', 'Nothing']

    l_stripped = [strip_accents(s) for s in l_test]
    assert l_stripped[0] == 'Halla'
    assert l_stripped[1] == 'aeioue'
    assert l_stripped[2] == 'Nothing'

    l_diff = [count_string_differences(s1, s2) for s1, s2 in zip(l_test, l_stripped)]
    assert l_diff[0] == 1
    assert l_diff[1] == 6
    assert l_diff[2] == 0


class TestAnnotation:
    offset_string = None
    translated_string = None
    is_correct = False


test_data_year_month_day_count = [
    (['1. November 2019'], ([2019], [11], [1]), 51453),
    (['1.Oktober2019 '], ([2019], [10], [1]), 51452),
    (['1. September 2019'], ([2019], [9], [1]), 51451),
    (['1.August2019'], ([2019], [8], [1]), 51450),
    (['23.0919'], ([2019], [9], [23]), 51449),
    (['011019'], ([2019], [10], [1]), 51449),
    (['0210.19'], ([2019], [10], [2]), 51449),
    (['1. Mai 2019'], ([2019], [5], [1]), 51448),
    (['16.122019'], ([2019], [12], [16]), 50954),
    (['07092012'], ([2012], [9], [7]), 0),
    (['14132020'], ([0], [0], [0]), 0),
    (['250785'], ([1985], [7], [25]), 0),
    (['1704.2020'], ([2020], [4], [17]), 0),
    (['/04.12.'], ([0], [12], [4]), 47776),
    (['04.12./'], ([0], [12], [4]), 47776),
    (['02-05-2019'], ([2019], [5], [2]), 54858),
    (['1. Oktober2019'], ([2019], [10], [1]), 0),
    (['13 Mar 2020'], ([2020], [3], [13]), 37527),
    (['30, Juni'], ([0], [6], [30]), 53921),
    (['2019-06-01'], ([2019], [6], [1]), 38217),
    (['30 Sep 2019'], ([2019], [9], [30]), 39970),
    (['July 1, 2019'], ([2019], [7], [1]), 38208),
    (['(29.03.2018)'], ([2018], [3], [29]), 51432),
    (['03,12.'], ([0], [12], [3]), 51439),
    (['23,01.'], ([0], [1], [23]), 51430),
    (['03,07,'], ([0], [0], [0]), 51435),  # (['03,07,'], ([0], [7], [3]), 51435),
    (['05.09;'], ([0], [9], [5]), 51436),
    (['24,01.'], ([0], [1], [24]), 51430),
    (['15.02.‚2019'], ([2019], [2], [15]), 54970)
]


@pytest.mark.parametrize("test_input, expected, document_id", test_data_year_month_day_count)
def test_dates(test_input, expected, document_id):
    res = year_month_day_count(test_input)
    assert res[0][0] == expected[0][0]
    assert res[1][0] == expected[1][0]
    assert res[2][0] == expected[2][0]


test_data_num = [
    ('3,444, 40+', 3444.4, 51438),
    ('5.473,04S', -5473.04, 51443),
    (' 362,85H', 362.85, 51443),
    ('3,288,50', 3288.50, 45551),
    ('1,635,74', 1635.74, 51426),
    ('0,00', 0, 514449),
    ('331.500', 331500, 57398),
    ('4.361.163', 4361163, 57268),
    ('4.361.163-', -4361163, 0),
    ('aghdabh', 0, 0),
    # ('2019-20-12', 0, 0)  # TODO: '2019-20-12' is being normalized to '20192012' should it be possible to normalize
    #                           a date to float?
]


@pytest.mark.parametrize("test_input, expected, document_id", test_data_num)
def test_num(test_input, expected, document_id):
    assert num_count(test_input) == expected
