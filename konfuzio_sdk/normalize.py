"""Convert the Span according to the data_type of the Annotation."""
import logging
import numpy as np
from typing import Dict, Optional
import regex as re


logger = logging.getLogger(__name__)

ROMAN_NUMS = {'M': 1000, 'D': 500, 'C': 100, 'L': 50, 'X': 10, 'V': 5, 'I': 1}


def _get_is_negative(offset_string: str) -> bool:
    """Check if a string has a negative sign."""
    is_negative = False
    if offset_string.count('-') > 0 or offset_string.count('–'):
        if offset_string.count('-') == 1 or offset_string.count('–') == 1:
            is_negative = True

    if offset_string.count('S') > 0:
        if offset_string.count('S') == 1 and offset_string[-1] == "S" and is_negative is False:
            is_negative = True

    if (
        offset_string[0] == '*'
        and offset_string.count('*') == 1
        and offset_string.count('(') == 1
        and offset_string.count(')') == 1
    ):
        is_negative = True

    offset_string_negative_check = (
        offset_string.replace(' ', '')
        .replace('"', '')
        .replace('„', '')
        .replace('+', '')
        .replace('-', '')
        .replace('–', '')
        .replace('€', '')
    )

    if len(offset_string_negative_check) > 2:
        if offset_string_negative_check[0] == '(' and offset_string_negative_check[-1] == ')':
            is_negative = True

    return is_negative


def normalize_to_float(offset_string: str) -> Optional[float]:
    """Given an offset_string: str this function tries to translate the offset-string to a number."""
    normalization = _normalize_string_to_absolute_float(offset_string)

    if normalization:
        is_negative = _get_is_negative(offset_string)
        normalization = normalization * (-1) ** int(is_negative)

    return normalization


def normalize_to_positive_float(offset_string: str) -> Optional[float]:
    """Given an offset_string this function tries to translate the offset-string to an absolute number (ignores +/-)."""
    return _normalize_string_to_absolute_float(offset_string)


def _normalize_string_to_absolute_float(offset_string: str) -> Optional[float]:
    """Given a string tries to translate that into an absolute float. SHOULD NOT BE CALLED DIRECTLY."""
    _float = None
    normalization = None

    if offset_string in ['-', '-,-', '-,--', '--,--', '--,-', '-.-', '-.--', '--.--', '--.-']:
        return 0.0

    if offset_string.lower() in ['nil', 'kein', 'keinen', 'keiner', 'none']:
        return 0

    if offset_string.lower() in ['ein', 'eine', 'einer', 'one']:
        return 1.0

    if offset_string.lower() in ['zwei', 'two']:
        return 2.0

    if offset_string.lower() in ['drei', 'three']:
        return 3.0

    if offset_string.lower() in ['vier', 'four']:
        return 4.0

    if offset_string.lower() in ['fünf', 'five']:
        return 5.0

    if offset_string.lower() in ['sechs', 'six']:
        return 6.0

    if offset_string.lower() in ['sieben', 'seven']:
        return 7.0

    if offset_string.lower() in ['acht', 'eight']:
        return 8.0

    if offset_string.lower() in ['neun', 'nine']:
        return 9.0

    if offset_string.lower() in ['zehn', 'ten']:
        return 10.0

    if offset_string.lower() in ['elf', 'eleven']:
        return 11.0

    if offset_string.lower() in ['zwölf', 'twelve']:
        return 12.0

    # check for major spaces
    if re.search(r'(\d)[ ]{3,}(\d)', offset_string):
        return None

    offset_string = re.sub(r"(\d)[ ]{1,2}(\d)", r'\1.\2', offset_string)

    if offset_string.count('*') > 1:
        return None

    if offset_string.count('*') == 1 and offset_string[0] == '*' and offset_string.count('(') == 0:
        return None

    offset_string = (
        offset_string.replace('O', '0')
        .replace('°', '')
        .replace(':', '')
        .replace('“', '')
        .replace("'", '')
        .replace('/', '')
        .replace('>', '')
        .replace('(', '')
        .replace(')', '')
        .replace('|', '')
        .replace(' ', '')
        .replace('"', '')
        .replace('„', '')
        .replace('+', '')
        .replace('-', '')
        .replace('–', '')
        .replace('€', '')
        .replace('*', '')
    )

    if len(offset_string) > 1:
        if (offset_string[-1] == 'S' or offset_string[-1] == 'H') and offset_string[-2].isdecimal():
            offset_string = offset_string[:-1]

    ln = len(offset_string)

    # check for 1.234,56
    if '.' in offset_string and offset_string.count(',') == 1 and offset_string.index('.') < offset_string.index(','):
        offset_string = offset_string.replace('.', '').replace(',', '.')  # => 1234.56
        if all(x.isdecimal() for x in offset_string.split('.')):
            _float = float(offset_string)
    # check for 1,234.56
    elif '.' in offset_string and ',' in offset_string and offset_string.index(',') < offset_string.index('.'):
        offset_string = offset_string.replace(',', '')  # => 1234.56
        if all(x.isdecimal() for x in offset_string.split('.')):
            _float = float(offset_string)
    # check for 1,234,56
    elif (
        ln > 6
        and offset_string.count(',') == 2
        and offset_string.count('.') == 0
        and offset_string[-3] == ','
        and offset_string[-7] == ','
    ):
        offset_string = offset_string[:-3] + '.' + offset_string[-2:]  # => 1,234.56
        offset_string = offset_string.replace(',', '')  # => 1234.56
        if all(x.isdecimal() for x in offset_string.split('.')):
            _float = float(offset_string)
    # check for 1.234.56
    elif ln > 6 and offset_string.count('.') >= 2 and offset_string[-3] == '.' and offset_string[-7] == '.':
        offset_string = offset_string.replace('.', '')  # => 123456
        offset_string = offset_string[:-2] + '.' + offset_string[-2:]  # => 1234.56
        if all(x.isdecimal() for x in offset_string.split('.')):
            _float = float(offset_string)
    # check for 1.967.
    elif ln > 5 and offset_string.count('.') == 2 and offset_string[-1] == '.' and offset_string[-5] == '.':
        offset_string = offset_string.replace('.', '')  # => 123456
        if offset_string.isdecimal():
            _float = float(offset_string)
    # check for 1.234.567
    elif ln > 7 and offset_string.count('.') >= 2 and offset_string[-4] == '.' and offset_string[-8] == '.':
        offset_string = offset_string.replace('.', '')  # => 1234567
        if offset_string.isdecimal():
            _float = float(offset_string)
    # check for 3.456,814,75
    elif ln > 7 and offset_string.count(',') == 2 and offset_string[-3] == ',' and offset_string[-7] == ',':
        offset_string = offset_string.replace(',', '').replace('.', '')  # => 1234567
        if offset_string.isdecimal():
            _float = float(offset_string) / 100.0
    # check for 1,234,567
    elif ln > 7 and offset_string.count(',') == 2 and offset_string[-4] == ',' and offset_string[-8] == ',':
        offset_string = offset_string.replace(',', '')  # => 1234567
        if offset_string.isdecimal():
            _float = float(offset_string)
    # check for 12,34 (comma is third last char).
    elif (
        ',' in offset_string
        and (len(offset_string) - offset_string.index(',')) == 3
        and offset_string.replace(',', '').isdecimal()
    ):
        offset_string = offset_string.replace(',', '.')  # => 12.34
        if all(x.isdecimal() for x in offset_string.split('.')):
            _float = float(offset_string)
    # check for 12.34 (dot is third last char).
    elif offset_string.count('.') == 1 and (len(offset_string) - offset_string.index('.')) == 3:
        if all(x.isdecimal() for x in offset_string.split('.')):
            _float = float(offset_string)  # => 12.34
    # check for 12,3 (comma is second last char).
    elif (
        ',' in offset_string
        and (len(offset_string) - offset_string.index(',')) == 2
        and offset_string.replace(',', '').isdecimal()
    ):
        _float = float(offset_string.replace(',', '.'))  # => 12.3
    # check for 12.3 (dot is second last char).
    elif offset_string.count('.') == 1 and (len(offset_string) - offset_string.index('.')) == 2:
        if all(x.isdecimal() for x in offset_string.split('.')):
            _float = float(offset_string)  # => 12.3
    # check for 500,000 (comma is forth last char).
    elif (
        ln > 0
        and ',' in offset_string
        and (len(offset_string) - offset_string.index(',')) == 4
        and offset_string.replace(',', '').isdecimal()
        and not offset_string[0] == ','
    ):
        _float = float(offset_string.replace(',', ''))  # => 500000
        _float = abs(_float)
        normalization = _float

    # check for 500.000
    elif (
        ln > 4
        and '.' in offset_string
        and offset_string[-4] == '.'
        and offset_string.replace('.', '').isdecimal()
        and offset_string.count('.') == 1
    ):
        normalization = abs(float(offset_string.replace('.', '')))
    # check for 5000 (only numbers)
    elif offset_string.isdecimal():
        _float = float(offset_string)
        _float = abs(_float)
        normalization = _float
    # check for 159,;03 (obscured edge case)
    elif (
        ln > 3
        and ';' in offset_string
        and ',' in offset_string
        and offset_string[-3] == ';'
        and offset_string[-4] == ','
    ):
        offset_string = offset_string.replace(',', '.').replace(';', '')  # => 159.03
        if all(x.isdecimal() for x in offset_string.split('.')):
            _float = float(offset_string)
    # # check for “71,90 (obscured edge case)
    # elif offset_string[0] == '“' and offset_string[-3] == ',':
    #     _float = float(offset_string.replace('“', '').replace(',','.'))  # => 71.90
    #     _float = abs(_float)
    #     normalization = _float
    # check for ,22,95 (obscured edge case)
    elif (
        ln > 2 and offset_string[0] == '‚' and offset_string[-3] == ','
    ):  # first comma is a very different comma ('‚' != ',')
        offset_string = offset_string[1:].replace(',', '.')  # => 22.95
        if all(x.isdecimal() for x in offset_string.split('.')):
            _float = float(offset_string)
    elif all(char in ROMAN_NUMS.keys() for char in offset_string):
        normalization = roman_to_float(offset_string)
    else:
        logger.debug(
            'Could not convert >>' + offset_string + '<< to positive/absolute float (no conversion case found)'
        )

    if _float is not None:
        _float = abs(_float)
        normalization = _float
    # handles the case when normalization is >float32 maximum size
    # TODO: handle the underflow case, i.e. normalization is a very small number
    if np.isinf(np.float32(normalization)):
        return None

    return normalization


def normalize_to_percentage(offset_string: str) -> Optional[float]:
    """Given an Annotation this function tries to translate the offset-string to an percentage -a float between 0 -1."""
    offset_string = offset_string.replace('+', '').replace('-', '').replace('"', '').replace('„', '')
    if len(offset_string) > 1 and offset_string[-1] in ['.', ';', ',']:
        offset_string = offset_string[:-1]

    if '%' in offset_string:
        percentage_detected = True
        offset_string = offset_string.replace('%', '')
    else:
        percentage_detected = False

    res = _normalize_string_to_absolute_float(offset_string)
    if res is not None and (percentage_detected or res > 1):
        res = res / 100

    if res is None:
        return None
    if res < 0:
        return None

    res = round(res, 6)
    return res


def normalize_to_date(offset_string: str) -> Optional[str]:
    """Given an Annotation this function tries to translate the offset-string to a date in the format 'DD.MM.YYYY'."""
    translation = None
    offset_string = (
        offset_string.replace(' ', '')
        .replace(':', '')
        .replace(',', '')
        .replace('[', '')
        .replace('(', '')
        .replace(')', '')
        .replace(';', '')
        .replace('‚', '')
    )
    org_str = offset_string

    # replace all written out months with the according numbers
    month_dict_de_en = {
        'JANUAR': '01.',
        'JANUARY': '01.',
        'FEBRUAR': '02.',
        'FEBRUARY': '02.',
        'MÄRZ': '03.',
        'MARCH': '03.',
        'APRIL': '04.',
        'MAI': '05.',
        'MAY': '05.',
        'JUNI': '06.',
        'JUNE': '06.',
        'JULI': '07.',
        'JULY': '07.',
        'AUGUST': '08.',
        'SEPTEMBER': '09.',
        'OKTOBER': '10.',
        'NOVEMBER': '11.',
        'DEZEMBER': '12.',
        'DECEMBER': '12.',
    }

    month_dict_short_eng_de = {
        'JAN': '01.',
        'FEB': '02.',
        'MAR': '03.',
        'APR': '04.',
        'MAY': '05.',
        'JUN': '06.',
        'JUL': '07.',
        'AUG': '08.',
        'SEP': '09.',
        'OCT': '10.',
        'OKT': '10.',
        'NOV': '11.',
        'DEC': '12.',
        'DEZ': '12.',
    }

    month_dict_de_en.update(month_dict_short_eng_de)

    for key in month_dict_de_en.keys():
        offset_string = offset_string.upper().replace(key, month_dict_de_en[key])

    # check first if we can find a date by using the "normal" date format
    translation = _check_for_dates_with_day_count(offset_string, org_str, month_dict_de_en)

    # if this doesn't yield results search for dates that only contain a month and year
    if not translation:
        translation = _check_for_dates_with_only_month_and_year(offset_string)

    # if still no translation is found just check if it is just a year number
    if not translation and offset_string.isdecimal() and len(offset_string) == 4:
        translation = '01.01.' + offset_string
        translation = _final_date_check(translation)

    if translation:
        translation = _convert_german_time_to_iso(translation)

    return translation


def _check_for_dates_with_day_count(offset_string: str, org_str: str, month_dict_de_en: Dict):
    """Convert any dates that have a day in them (with or without year)."""
    translation = None

    # adds a 0 in front of X.XX. or X/XX/
    if len(offset_string) < 4:
        return None

    if len(offset_string) > 4 and (
        (offset_string[1] == '.' and offset_string[4] == '.') or (offset_string[1] == '/' and offset_string[4] == '/')
    ):
        offset_string = '0' + offset_string

    no_white_space_raw = offset_string.replace(' ', '')

    # 0101.01 or 01.0101
    if len(offset_string.replace('.', '')) == 6 and (offset_string[2] == '.' or offset_string[-3] == '.'):
        offset_string_filtered = offset_string.replace('.', '')
        offset_string = (
            offset_string_filtered[:2] + '.' + offset_string_filtered[2:4] + '.' + offset_string_filtered[4:6]
        )
    # 010101
    elif len(offset_string) == 6 and offset_string.isdecimal():
        offset_string = offset_string[:2] + '.' + offset_string[2:4] + '.' + offset_string[4:6]
    # 01.012001 or 0101.2001
    elif len(offset_string.replace('.', '')) == 8 and (offset_string[2] == '.' or offset_string[-5] == '.'):
        offset_string_filtered = offset_string.replace('.', '')
        offset_string = (
            offset_string_filtered[:2] + '.' + offset_string_filtered[2:4] + '.' + offset_string_filtered[4:8]
        )
    # 01012001
    elif len(offset_string) == 8 and offset_string.isdecimal():
        offset_string = offset_string[:2] + '.' + offset_string[2:4] + '.' + offset_string[4:8]
    # /01.01.
    elif offset_string[0] == '/':
        offset_string = offset_string[1:]
    # 01.01/
    elif offset_string[-1] == '/' and not offset_string[-2].isdecimal():
        offset_string = offset_string[:-1]
    # 0101.
    elif offset_string[-1] == '.' and len(offset_string) == 5 and offset_string[:4].isdecimal():
        offset_string = offset_string[:2] + '.' + offset_string[2:]
    # 0101 (from 01,01,)
    elif len(offset_string) == 4 and offset_string.isdecimal() and offset_string.count(',') == 2:
        offset_string = offset_string[:2] + '.' + offset_string[2:] + '.'
    # 01.01/01.01
    elif (
        len(no_white_space_raw) == 13
        and no_white_space_raw[0:6] == no_white_space_raw[7:13]
        and not no_white_space_raw[6].isdecimal()
    ):
        offset_string = no_white_space_raw[0:6]
    # 1993-02-05T00:00:00
    elif (
        len(offset_string) >= 17
        and offset_string[0:4].isdecimal()
        and offset_string[5:7].isdecimal()
        and offset_string[8:10].isdecimal()
    ):
        offset_string = offset_string[0:10]

    # check for 2001-01-01
    if len(offset_string) == 10 and offset_string[4] == '-' and offset_string[7] == '-':
        _date = f'{offset_string[8:10]}.{offset_string[5:7]}.{offset_string[0:4]}'
        translation = _date
    # check for 01.01.2001
    elif len(offset_string) == 10 and offset_string[2] == '.' and offset_string[5] == '.':
        _date = offset_string  # => 01.01.2001
        translation = _date
    # check for 01/01/2001
    elif len(offset_string) == 10 and offset_string[2] == '/' and offset_string[5] == '/':
        _date = offset_string.replace('/', '.')  # => 01.01.2001
        translation = _date
    # check for 01-01-2001
    elif len(offset_string) == 10 and offset_string[2] == '-' and offset_string[5] == '-':
        _date = offset_string.replace('-', '.')  # => 01.01.2001
        translation = _date
    # check for 01.01.01
    elif (
        len(offset_string) == 8
        and offset_string[2] == '.'
        and offset_string[5] == '.'
        and offset_string[6:].isdecimal()
    ):
        year_num = int(offset_string[6:])

        if year_num > 50:
            cent_num = 19
        else:
            cent_num = 20

        translation = offset_string[:6] + str(cent_num) + offset_string[6:]
    # check for 01/01/01
    elif (
        len(offset_string) == 8
        and offset_string[2] == '/'
        and offset_string[5] == '/'
        and offset_string[6:].isdecimal()
    ):
        year_num = int(offset_string[6:])

        if year_num > 50:
            cent_num = 19
        else:
            cent_num = 20

        translation = (offset_string[:6] + str(cent_num) + offset_string[6:]).replace('/', '.')
    # check for 01.01
    elif len(offset_string) == 5 and offset_string[2] == '.':
        _date = offset_string + '.0000'  # => 01.01.0000
        translation = _date
    # check for 01.01.
    elif len(offset_string) == 6 and offset_string[2] == '.' and offset_string[5] == '.':
        _date = offset_string + '0000'  # => 01.01.0000
        translation = _date
    # check for 2001-01-01
    elif len(offset_string) == 10 and offset_string[-3] == '-' and offset_string[4] == '-':
        translation = offset_string[-2:] + '.' + offset_string[5:7] + '.' + offset_string[:4]
    else:
        logger.debug('Could not convert >>' + offset_string + '<< to date (no conversion case found)')

    translation = _final_date_check(translation)

    if not translation:
        # check for 'July 1, 2019' (stupid edge case)
        _str = org_str
        _year = None
        _month = None
        _day = None

        _year = _str[-4:]
        _str = _str[:-4]

        for key in month_dict_de_en.keys():
            if key in _str.upper():
                _str = _str.upper().replace(key, '')
                _month = month_dict_de_en[key]
                break

        try:
            _day = int(_str)
        except ValueError:
            _day = None

        if _day and _month and _year:
            translation = str(_day) + '.' + str(_month) + str(_year)

            if len(str(_day)) < 2:
                translation = '0' + translation

        translation = _final_date_check(translation)

    return translation


def _check_for_dates_with_only_month_and_year(offset_string: str):
    """Check for date formats that only include a month and year."""
    translation = None

    if len(offset_string) < 5:
        return None

    # 09/18
    if (
        offset_string[-3] == '/'
        and offset_string.count('/') == 1
        and offset_string.replace('/', '').isdecimal()
        and len(offset_string.replace('/', '')) == 4
    ):
        year_num = int(offset_string[-2:])

        if year_num > 50:
            cent_num = 19
        else:
            cent_num = 20

        translation = offset_string[:2] + '.' + str(cent_num) + offset_string[-2:]

    if not translation and len(offset_string) < 6:
        return None

    # 09.2002
    if (
        offset_string[-5] == '.'
        and offset_string.count('.') == 1
        and offset_string.replace('.', '').isdecimal()
        and len(offset_string.replace('.', '')) == 6
    ):
        translation = offset_string
    # 09/2002
    elif (
        offset_string[-5] == '/'
        and offset_string.count('/') == 1
        and offset_string.replace('/', '').isdecimal()
        and len(offset_string.replace('/', '')) == 6
    ):
        translation = offset_string[:2] + '.' + offset_string[3:]

    if translation:
        translation = '01.' + translation
        translation = _final_date_check(translation)

    return translation


def _convert_german_time_to_iso(date_string: str):
    """Convert a german date input as a string to ISO format (DD.MM.YYYY -> YYYY-MM-DD). ONLY DD.MM.YYYY FORMAT."""
    return date_string[-4:] + '-' + date_string[3:5] + '-' + date_string[:2]


def _final_date_check(date_string: str):
    """Validate the converted dates including appropriate errors."""
    if date_string:
        if not (
            len(date_string) == 10
            and date_string[2] == '.'
            and date_string[5] == '.'
            and date_string[-4:].isdecimal()
            and date_string[:2].isdecimal()
            and date_string[3:5].isdecimal()
        ):
            logger.debug('Could not convert >>' + date_string + '<< to date (date contains letters)')
            date_string = None
        elif (
            not ((1900 < int(date_string[-4:]) < 2100) or int(date_string[-4:]) == 0)
            or not (int(date_string[:2]) < 32)
            or not (int(date_string[3:5]) < 13)
        ):
            logger.debug('Could not convert >>' + date_string + '<< to date (invalid date)')
            date_string = None
    return date_string


def normalize_to_bool(offset_string: str):
    """Given an offset_string this function tries to translate the offset-string to a bool."""
    offset_string = offset_string.upper()
    offset_string_list = offset_string.split()

    no_list = ['NEIN', 'NICHT', 'KEIN', 'OHNE', 'NO']
    yes_list = ['VORHANDEN', 'JA', 'MIT', 'YES']

    # one and two word strings
    if len(offset_string_list) == 1 or len(offset_string_list) == 2:
        y_word = any(w in offset_string_list[0] for w in yes_list)
        n_word = any(w in offset_string_list[0] for w in no_list)

        if y_word and not n_word:
            return True
        elif not y_word and n_word:
            return False
        else:
            return None
    else:
        return None


def roman_to_float(offset_string: str) -> Optional[float]:
    """Convert a Roman numeral to an integer."""
    input = offset_string.upper()
    if len(offset_string) == 0:
        return None
    roman_sum = 0
    for i, char in enumerate(offset_string):
        try:
            value = ROMAN_NUMS[input[i]]
            # If the next place holds a larger number, this value is negative
            if i + 1 < len(input) and ROMAN_NUMS[input[i + 1]] > value:
                roman_sum -= value
            else:
                roman_sum += value
        except KeyError:
            return None
    return float(roman_sum)


def normalize(offset_string, data_type):
    """Wrap all normalize functionality."""
    try:
        if data_type in ['Positive Number', 'float_positive']:
            result = normalize_to_positive_float(offset_string)
        elif data_type in ['Number', 'float']:
            result = normalize_to_float(offset_string)
        elif data_type in ['Date', 'date']:
            result = normalize_to_date(offset_string)
        elif data_type in ['True/False', 'bool']:
            result = normalize_to_bool(offset_string)  # bool not implemented yet.
        elif data_type in ['percentage', 'Percentage']:
            result = normalize_to_percentage(offset_string)
        elif data_type in ['Text', 'str']:
            result = offset_string
        else:
            result = None
    except Exception as e:  # NOQA
        logger.debug('Text >>' + offset_string + f'<< with data type {data_type} cannot be converted')
        result = None
        pass

    return result
