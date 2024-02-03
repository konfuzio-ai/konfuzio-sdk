"""Generic way to build regex from examples."""
import logging
from typing import Dict, List

import pandas
import regex as re
from tabulate import tabulate

logger = logging.getLogger(__name__)


def merge_regex(regex_tokens: List[str]):
    """Merge a list of regex to one group."""
    tokens = r'|'.join(sorted(regex_tokens, key=len, reverse=True))
    return f'(?:{tokens})'


def harmonize_whitespaces(text):
    """Convert multiple whitespaces to one."""
    single_whitespace_replaced = re.sub(r'(?<! ) (?! )', r'[ ]{1,2}', text)
    suggestion = re.sub(r' {2,}', r'[ ]{2,}', single_whitespace_replaced)
    return suggestion


def escape(string: str):
    """Escape a string, so that it can still be used to create a regex."""
    escaped_original = (
        string.replace('\\', '\\\\')
        .replace('[', r'\[')
        .replace(']', r'\]')
        .replace('+', r'[\+]')
        .replace('*', r'\*')
        .replace('|', r'\|')
        .replace('\n', '\n')
        .replace('-', '[-]')
        .replace('.', r'\.')
        .replace('$', r'\$')
        .replace('(', r'\(')
        .replace(')', r'\)')
        .replace('@', r'\@')
        .replace('?', r'\?')
        .replace('!', r'\!')
        .replace(',', r'\,')
        .replace('#', r'\#')
        .replace('{', r'\{')
        .replace('}', r'\}')
    )
    return escaped_original


def plausible_regex(suggestion, string):
    """
    Test regex for plausibility.

    We keep those tests in production to collect edge cases and always return true.
    """
    try:
        re.compile(suggestion)
        plausibility_run = re.findall(suggestion, string)
        if not plausibility_run:
            logger.error(
                f'Using "{repr(string)}" we found the regex {repr(suggestion)}, which does not match the input.'
            )
            logger.error(
                'We are not able to able to convert your string to a valid regex. Please help to make it happen.'
            )
            result = ''
        else:
            result = suggestion

    except re.error as e:
        logger.exception(f'The proposed regex >>{repr(suggestion)}<< is not a valid regex of string: >>{string}<<')
        logger.error('We are not able to able to convert your string to a valid regex. Please help to make it happen.')
        logger.error(e)
        result = ''

    return result


def suggest_regex_for_string(string: str, replace_characters: bool = False, replace_numbers: bool = True):
    """Suggest regex for a given string."""
    escaped_original = escape(string)

    if replace_characters:
        # strict replace capital letters
        strict_escaped_capital_letters = re.sub(r'[A-Z\Ä\Ö\Ü]', r'[A-ZÄÖÜ]', escaped_original)
        # combine multiple capital letters in sequence
        combined_capital_letters = re.sub(r'(\[A-Z\Ä\Ö\Ü\]){2,}', r'[A-ZÄÖÜ]+', strict_escaped_capital_letters)
        # escape all lower case letters
        escaped_small_letters = re.sub(r'[a-zäöüß]', r'[a-zäöüß]', combined_capital_letters)
        # combine multiple lower case letters in sequence
        escaped_original = re.sub(r'(\[a-zäöüß\]){2,}', '[a-zäöüß]+', escaped_small_letters)

    if replace_numbers:
        escaped_original = re.sub('\\d', r'\\d', escaped_original)

    # replace multiple whitespaces with r' +'
    suggestion = harmonize_whitespaces(escaped_original)

    suggestion = plausible_regex(suggestion, string)
    return suggestion


def get_best_regex(evaluations: List, log_stats: bool = True) -> List:
    """Optimize selection of one regex in scenarios were we are unsure if all correct Annotations are Labeled."""
    df = pandas.DataFrame(evaluations)
    if df.empty:
        logger.error('We cannot find any regex!')
        return []

    df = df.loc[df['f1_score'] > 0]

    df = df.sort_values(
        [
            'total_correct_findings',
            'f1_score',
            'regex_quality',
            'annotation_precision',
            'runtime',  # take the fastest regex
        ],
        ascending=[0, 0, 0, 0, 1],
    ).reset_index(drop=True)

    df['correct_findings_id'] = df['correct_findings'].apply(lambda x: {y.id_local for y in x})
    df['all_matches_id'] = [set.union(*df.loc[0:i, 'correct_findings_id']) for i in range(len(df.index))]
    df['new_matches_id'] = df.all_matches_id - df.all_matches_id.shift(1)
    null_mask = df['new_matches_id'].isnull()
    df.loc[null_mask, 'new_matches_id'] = df.loc[null_mask]['correct_findings_id']
    df.insert(0, 'new_matches_count', df['new_matches_id'].str.len())
    df = df.drop(['correct_findings_id', 'correct_findings', 'all_matches_id', 'new_matches_id'], axis=1)

    # iterate over sorted df, mark any row if it adds no matching value compared to regex above, we used max windowsize
    # matched_document = df.filter(regex=r'document_\d+').rolling(min_periods=1, window=100000000).max()
    # any regex which matches more Documents that the regex before, is a good regex
    # relevant_regex = matched_document.sum(axis=1).diff()
    # df['matched_annotations_total'] = matched_document.sum(axis=1)
    # df['matched_annotations_additional'] = relevant_regex
    # get the index of all good regex
    index_of_regex = df[df['new_matches_count'] > 0].index

    if log_stats:
        stats = df.loc[index_of_regex][
            ['regex', 'runtime', 'annotation_recall', 'annotation_precision', 'f1_score', 'new_matches_count']
        ]
        logger.debug(f'\n\n{tabulate(stats, floatfmt=".4f", headers="keys", tablefmt="pipe")}\n')

    # best_regex = df.loc[index_of_regex, 'regex'].to_list()
    best_regex = df.loc[df['new_matches_count'] > 0, 'regex'].to_list()

    return best_regex


def regex_matches(
    doctext: str, regex: str, start_chr: int = 0, flags=0, overlapped=False, keep_full_match=True, filtered_group=None
) -> List[Dict]:
    """
    Convert a text with the help by one regex to text offsets.

    A result of results is a full regex match, matches or (named) groups are separated by keys within this result. The
    function regexinfo in konfuzio.wrapper standardizes the information we keep per match.

    :param filtered_group: Name of the regex group you want to return as results
    :param keep_full_match: Keep the information about the full regex even the regex contains groups
    :param overlapped: Allow regex to overlap, e.g. ' ([^ ]*) ' creates an overlap on ' my name '
    :param flags: Regex flag to compile regex
    :param doctext: A text you want to apply a rgx on
    :param regex: The regex, either with groups, named groups or just a regex
    :param start_chr: The start chr of the annotation_set, in case the text is a annotation_set within a text
    """
    results = []

    # compile regex pattern
    # will throw an error if the name of the group, ?P<GROUP_NAME>, is not a valid Python variable name,
    # e.g. GROUP_NAME starts with a numeric character.
    # we catch this error and then add a leading underscore to the group name, making it a valid Python variable name
    try:
        pattern = re.compile(regex, flags=flags)
    except re.error:
        logger.error(regex)
        # throws error if group name is an invalid Python variable
        match = re.search(r'\?P<.*?>', regex)  # match the invalid group name
        group_name = match.group(0)  # get the string representation
        group_name = group_name.replace('?P<', '?P<_')  # add a leading underscore
        regex = re.sub(r'\?P<.*?>', group_name, regex)  # replace invalid group name with new one
        pattern = re.compile(regex, flags=flags)  # try the compile again

    for match in pattern.finditer(doctext, overlapped=overlapped):
        # hold results per match
        _results = []

        if match.groups():
            # parse named groups, if available
            for group_name, group_index in match.re.groupindex.items():
                if match[group_index] is not None:
                    # if one regex group ( a annotation's token) does not match, it returns none
                    # https://stackoverflow.com/a/59120080
                    if match.regs[group_index][1] > match.regs[group_index][0]:  # Work on text sequences not indices
                        _results.append(
                            {
                                'regex_used': repr(regex),
                                'regex_group': group_name,
                                'value': match[group_index],
                                'start_offset': match.regs[group_index][0],
                                'end_offset': match.regs[group_index][1],
                                'start_text': start_chr,
                            }
                        )
            # find unnamed groups if available
            unnamed_groups = [x for x in range(1, match.re.groups + 1) if x not in match.re.groupindex.values()]
            for group_index in unnamed_groups:
                if match.regs[group_index][1] > match.regs[group_index][0]:  # Work on text sequences not indices
                    _results.append(
                        {
                            'regex_used': repr(regex),
                            'regex_group': str(group_index),
                            'value': match[group_index],
                            'start_offset': match.regs[group_index][0],
                            'end_offset': match.regs[group_index][1],
                            'start_text': start_chr,
                        }
                    )

        if match.groups() and keep_full_match or not match.groups():
            if match.span()[1] > match.span()[0]:  # Work on text sequences not indices
                _results.append(
                    {
                        'regex_used': repr(regex),
                        'regex_group': '0',
                        'value': match.group(),
                        'start_offset': match.span()[0],
                        'end_offset': match.span()[1],
                        'start_text': start_chr,
                    }
                )

        # add results per match to all results
        results.extend(_results)

    if filtered_group:
        # allow to use similar group names, you can use "Ort_" if the group name is "Ort_255_259"
        return [result for result in results if filtered_group in result['regex_group']]
    else:
        return results


def generic_candidate_function(regex, flags=0, overlapped=False, filtered_group=None):
    """Regex approach to build a candidate function by one regex.

    :param filtered_group: If a regex contains multiple named groups, you can filter the respective group by name
    :param overlapped: Indicate if regex matches can overlapp.
    :param regex: Regex to create a candidate_function.
    :param flags: Regex flag which should be considered.
    :return: An initialized candidate function.
    """

    # function to build candidates
    def candidate_function(doctext):
        """
        Split the text in candidates and other text chunks.

        :param doctext: Text of the candidate
        :return: Tuple of list of candidates and other text chunks
        """
        annotations = regex_matches(
            doctext=doctext,
            regex=regex,
            flags=flags,
            overlapped=overlapped,
            keep_full_match=False,
            filtered_group=filtered_group,
        )

        # reduce the available information to value, start_offset and end_offset:
        # Due to historical aim of the candidate function to only find regex matches
        matches_tuples = [(d['value'], (d['start_offset'], d['end_offset'])) for d in annotations]

        candidates = [x for x, y in matches_tuples]
        candidates_spans = [y for x, y in matches_tuples]

        # Calculate other text bases on spans.
        other_text = []
        previous = 0
        for span in candidates_spans:
            other_text.append(doctext[previous : span[0]])
            previous = span[1]
        other_text.append(doctext[previous:])

        return candidates, other_text, candidates_spans

    candidate_function.__name__ = f'regex_{regex}'
    return candidate_function
