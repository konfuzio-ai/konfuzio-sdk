"""Test for string normalization."""
import logging

import pytest

from konfuzio_sdk.normalize import (
    normalize,
    normalize_to_bool,
    normalize_to_date,
    normalize_to_float,
    normalize_to_percentage,
    normalize_to_positive_float,
)

logger = logging.getLogger(__name__)


def test_normalize_with_invalid_string():
    """Normalize should not raise an exception."""
    result = normalize('Woch.Arb.Zt', 'float')
    assert result is None


def test_date_with_only_one_digit_in_front_and_only_two_for_the_year():
    """Test to normalize years."""
    t1 = '1.01.01'
    t2 = '4.01.89'
    t3 = '2/03/05'
    t4 = '4/11/99'

    assert normalize_to_date(t1) == '2001-01-01'
    assert normalize_to_date(t2) == '1989-01-04'
    assert normalize_to_date(t3) == '2005-03-02'
    assert normalize_to_date(t4) == '1999-11-04'


def test_date_with_wrong_strings():
    """Test to refuse to normalize."""
    t1 = '01.A1.2001'
    t2 = '23.45.2020dasd'
    t3 = '20,20,3030'
    t4 = '20.90/9012'

    assert normalize_to_date(t1) is None
    assert normalize_to_date(t2) is None
    assert normalize_to_date(t3) is None
    assert normalize_to_date(t4) is None


def test_float_normalization():
    """Test to normalize floats."""
    t1 = '42.34-'
    assert normalize_to_float(t1) == -42.34


def test_empty_string():
    """Test to normalize empty string."""
    t1 = ''
    assert normalize_to_date(t1) is None
    assert normalize_to_float(t1) is None
    assert normalize_to_positive_float(t1) is None


def test_one_char():
    """Test to normalize negative number text."""
    t1 = '-2'
    assert normalize_to_date(t1) is None
    assert normalize_to_float(t1) == -2
    assert normalize_to_positive_float(t1) == 2


test_data_percentage = [
    ('12,34', 0.1234, None),
    ('12,3', 0.123, None),
    ('123,45', 1.2345, None),
    ('59,00-', 0.59, None),
    ('12,34 %', 0.1234, None),
    ('12,34 %.', 0.1234, None),
    ('12,34 % .', 0.1234, None),
    ('12,34 %;', 0.1234, 101925),
    ('12,34 % ;', 0.1234, 101925),
    ('12,34 %,', 0.1234, 101925),
    ('12,34 % ,', 0.1234, 101925),
    ('12,34  %  .', 0.1234, None),
    ('12,34  %  .', 0.1234, None),
    ('12.34 %', 0.1234, None),
    ('12.³4 %', None, None),
    ('12.34 %.', 0.1234, None),
    ('12.34 % .', 0.1234, None),
    ('12.34  %  .', 0.1234, None),
    ('12.34  %  .', 0.1234, None),
    ('12 34  %  .', 0.1234, None),
    ('12  34  %  .', 0.1234, None),
    ('12   34  %  .', None, None),
    ('434,27%,', 4.3427, None),
    ('100 %', 1, None),
    ('0 %', 0, None),
    ('0.00', 0, None),
    ('0.0', 0, None),
    ('0', 0, None),
    ('0 %', 0, None),
    ('0,00', 0, None),
    ('0,0³', None, None),
    ('0,0', 0, None),
    ('0', 0, None),
]


@pytest.mark.parametrize('test_input, expected, document_id', test_data_percentage)
def test_percentage(test_input, expected, document_id):
    """Test to normalize percentages."""
    assert normalize_to_percentage(test_input) == expected


test_data_positive_numbers = [
    ('59,00-', 59, 50945),
    ('585,87/-', 585.87, 50945),
    ("'786,71-", 786.71, 51429),
    ('7,375,009+ ', 7375009, 51429),
    (':2.000, 08 ', 2000.08, 51437),
    ('-2.759,7°', 2759.7, 51447),
    ('‚22,95', 22.95, 53920),
    ('1.967.', 1967.00, 110154),
    ('-1.800.00', 1800, 53921),
    ('“71,90', 71.90, 53921),
    ('-2.905.00', 2905, 53928),
    ('-O,51', 0.51, 53929),
    ('-3,000,00', 3000, 56252),
    ('+159,;03', 159.03, 56253),
    ('5,000,00', 5000, 45737),
    ('42, 975,38', 42975.38, 52385),
    ('4.187.184.13', 4187184.13, 110050),
    ('549.886.799.221', 549886799221, 110150),
    ('123,4567', 123.4567, None),
    ('4.2.', None, None),
    ('ein', 1, 99479),
    ('Drei', 3, None),
    ('vier', 4, None),
    ('One', 1, None),
    ('TWo', 2, None),
    ('eleven', 11, None),
    ('–100', 100, 109610),
    ('NIL', 0, None),
    ('StringThatIncludesNIL', None, None),
    ('kein', 0, None),
    ('KEin', 0, None),
    ('StringThatIncludeskein', None, None),
    ('keinen', 0, None),
    ('KEinen', 0, None),
    ('StringThatIncludeskeinen', None, None),
    ('keiner', 0, None),
    ('KEiner', 0, None),
    ('StringThatIncludeskeiner', None, None),
    ('none', 0, None),
    ('NoNe', 0, None),
    ('StringThatIncludesnone', None, None),
    ('54³', None, None),
    ('165a', None, None),
]


@pytest.mark.parametrize('test_input, expected, document_id', test_data_positive_numbers)
def test_positive_numbers(test_input, expected, document_id):
    """Test to normalize positive numbers."""
    assert normalize_to_positive_float(test_input) == expected


test_data_numbers = [
    ('3,444, 40+', 3444.4, 51438),
    ('5.473,04S', -5473.04, 51443),
    (' 362,85H', 362.85, 51443),
    ('3,288,50', 3288.50, 45551),
    ('1,635,74', 1635.74, 51426),
    ('0,00', 0, 514449),
    ('331.500', 331500, 57398),
    ('4.361.163', 4361163, 57268),
    ('4.361.163-', -4361163, 0),
    ('111144443333////111100008888////44440000000022226666', None, 0),
    ('(118.704)', -118704, 60168),
    ('10.225.717', 10225717, 58810),
    ('29.485.259', 29485259, 58810),
    ('129.485.259', 129485259, 0),
    ('331.500', 331500, 0),
    ('3³1.500', None, 0),
    ('3.000.000', 3000000, 0),
    ('56,430,681', 56430681, 0),
    ('43.34.34', None, 0),
    ('(51.901,99)', -51901.99, 0),
    ('2.662| ', 2662, 0),
    ("9'117, 30", 9117.3, 0),
    ('9’117, 20', 9117.2, 0),
    ('49’117', 49117, 0),
    ('9"117,10', 9117.1, 0),
    ('-,-', 0, 0),
    ('-', 0, 0),
    ('-,--', 0, 0),
    ('--,--', 0, 0),
    ('€1.010.296', 1010296, 93255),
    ('€3.372.097', 3372097, 93255),
    ('€(1.099)', -1099, 93255),
    ('€54.314', 54314, 93255),
    ('–100', -100, 109610),
    ('3.456,814,75', 3456814.75, 110383),
    ('NIL', 0, 0),
    ('abcdef', None, 0),
    ('I', 1, None),
    ('III', 3, None),
    (' XIV  ', 14, None),
    ('12³', None, 0),
    (' XL IV  ', 44, None),
    ('42 58', 42.58, None),
    ('42  58', 42.58, None),
    ('42   58', None, None),
    ('421 58', 421.58, None),
    ('4 258  23', 4258.23, None),
    ('4,258  23', 4258.23, None),
    ('4  258  23', 4258.23, None),
    ('42 58', 42.58, None),
    ('129.684,46*', 129684.46, None),
    ('*(118,704)', -118704, None),
    ('*118,704', None, None),
    ('*1*2*3*4', None, None),
    ('034466416.3105.832500.034', None, None),
    ('034,466416.3105.832500.034', None, None),
    ('034,466416.3105.832500.034.65', None, None),
    ('034,466416.3105.832', None, None),
    ('034,466416.31', 34466416.31, None),
    ('Woch.Arb.Zt.', None, None),
    ('0.30.2', None, None),
    ('14.55.43', None, None),
    ('12.23.34.545.23', None, None),
    ('123.24.123.444', None, None),
    ('0.1.222', None, None),
    ('123.123141.12.123', None, None),
    ('123 Mio', 123000000, None),
    ('123 Mio.', 123000000, None),
    ('100 Millionen', 100000000, None),
    ('12412..', None, None),
    ('165..', None, None),
    ('165,,', None, None),
    ('165,,,', None, None),
    (',,,23424,2,,,', None, None),
    ('...12124,123412,.12', None, None),
    ('..1..2.3..3333.', None, None),
    ('114433,8,60', None, None),
    ('165a', None, None),
    ('0,0-3,0', None, None),
    ('3,6-4.8', None, None),
    ('0,1–1,112', None, None),
    ('21231.41–124.4124,52', None, None),
    ('1.214141', 1.214141, None),
    ('22,1231', 22.1231, None),
    ('0,543255', 0.543255, None),
    ('3.141528743253920', 3.141528743253920, None),
    # ('12.', 12.0, 0), undefined test cases:
    # ('1.', 1.0, 0),
    # ('.', None, 0),
    # ('.1', 0.1, 0),
    # ('.123', 0.123, 0),
]


@pytest.mark.parametrize('test_input, expected, document_id', test_data_numbers)
def test_numbers(test_input, expected, document_id):
    """Test to normalize numbers."""
    assert normalize_to_float(test_input) == expected


test_data_dates = [
    ('1. November 2019', '2019-11-01', 51453),
    ('1.Oktober2019 ', '2019-10-01', 51452),
    ('1. September 2019', '2019-09-01', 51451),
    ('1.August2019', '2019-08-01', 51450),
    ('23.0919', '2019-09-23', 51449),
    ('011019', '2019-10-01', 51449),
    ('0210.19', '2019-10-02', 51449),
    ('1. Mai 2019', '2019-05-01', 51448),
    ('16.122019', '2019-12-16', 50954),
    ('07092012', '2012-09-07', 0),
    ('14132020', None, 0),
    ('250785', '1985-07-25', 0),
    ('1704.2020', '2020-04-17', 0),
    ('/04.12.', '0000-12-04', 47776),
    ('04.12./', '0000-12-04', 47776),
    ('02.06./02.06.', '0000-06-02', 57408),
    ('02.06./ 02.06.', '0000-06-02', 57408),
    ('02-05-2019', '2019-05-02', 54858),
    ('1. Oktober2019', '2019-10-01', 0),
    ('13 Mar 2020', '2020-03-13', 37527),
    ('30, Juni', '0000-06-30', 53921),
    ('2019-06-01', '2019-06-01', 38217),
    ('30 Sep 2019', '2019-09-30', 39970),
    ('July 1, 2019', '2019-07-01', 38208),
    ('(29.03.2018)', '2018-03-29', 51432),
    ('03,12.', '0000-12-03', 51439),
    ('23,01.', '0000-01-23', 51430),
    ('05.09;', '0000-09-05', 51436),
    ('24,01.', '0000-01-24', 51430),
    ('15.02.‚2019', '2019-02-15', 54970),
    ('1. Oktober2019', '2019-10-01', 0),
    ('1993-02-05T00:00:00', '1993-02-05', 0),
    ('July 31 ,2019', '2019-07-31', 0),
    ('23.0K.2010', None, 0),
    ('24.13.2020', None, 0),
    ('24.13.202³', None, 0),
    ('03,07,', None, 51435),
    ('30.07.2.90', None, 0),
    ('09/2002', '2002-09-01', 0),
    ('09.2002', '2002-09-01', 0),
    ('09/18', '2018-09-01', 0),
    ('Oktober 2011', '2011-10-01', 0),
    ('2001', '2001-01-01', 0),
    ('13.9.2021', '2021-09-13', 0),
]


@pytest.mark.parametrize('test_input, expected, document_id', test_data_dates)
def test_dates(test_input, expected, document_id):
    """Test to normalize dates."""
    assert normalize_to_date(test_input) == expected


test_data_bool = [
    ('nicht vorhanden', False, 0),
    ('nein', False, 0),
    ('nicht unterkellert', False, 0),
    ('ohne Rabattschutz', False, 0),
    ('mit Schutzbrief', True, 0),
    ('nicht versichert', False, 0),
    ('ja', True, 0),
    ('mit', True, 0),
    ('ohne', False, 0),
    ('', None, 0),
    ('alleinstehend ohne Kind', None, 0),
]


@pytest.mark.parametrize('test_input, expected, document_id', test_data_bool)
def test_bool(test_input, expected, document_id):
    """Test to normalize boolean."""
    assert normalize_to_bool(test_input) == expected
