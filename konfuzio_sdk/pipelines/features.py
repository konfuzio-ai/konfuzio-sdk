"""Multiclass classifier for document extraction."""
import difflib
import logging
import numpy as np
import pandas as pd
import random
import unicodedata
import functools

from collections import Counter
from heapq import nsmallest
from konfuzio_sdk.data import Annotation, Document
from konfuzio_sdk.utils import get_bbox, iter_before_and_after
from konfuzio_sdk.normalize import normalize_to_date, normalize_to_float, normalize
from tabulate import tabulate
from typing import Callable, Dict, List, Optional, Tuple, Union


logger = logging.getLogger(__name__)
CANDIDATES_CACHE_SIZE = 100


def convert_to_feat(offset_string_list: list, ident_str: str = '') -> pd.DataFrame:
    """Return a df containing all the features generated using the offset_string."""
    df = pd.DataFrame()

    # strip all accents
    offset_string_list_accented = offset_string_list
    offset_string_list = [strip_accents(s) for s in offset_string_list]

    # gets the return lists for all the features  # TODO remove duplicated code.
    df[ident_str + "feat_vowel_len"] = [vowel_count(s) for s in offset_string_list]
    df[ident_str + "feat_special_len"] = [special_count(s) for s in offset_string_list]
    df[ident_str + "feat_space_len"] = [space_count(s) for s in offset_string_list]
    df[ident_str + "feat_digit_len"] = [digit_count(s) for s in offset_string_list]
    df[ident_str + "feat_len"] = [len(s) for s in offset_string_list]
    df[ident_str + "feat_upper_len"] = [upper_count(s) for s in offset_string_list]
    df[ident_str + "feat_num_count"] = [num_count(s) for s in offset_string_list]

    # replaced by our normalize function.
    # df[ident_str + "feat_as_float"] = [normalize_to_python_float(offset_string) for offset_string in offset_string_list]
    df[ident_str + "feat_unique_char_count"] = [unique_char_count(s) for s in offset_string_list]
    df[ident_str + "feat_duplicate_count"] = [duplicate_count(s) for s in offset_string_list]
    df[ident_str + "accented_char_count"] = [count_string_differences(s1, s2) for s1, s2 in
                                             zip(offset_string_list, offset_string_list_accented)]

    df[ident_str + "feat_year_count"], df[ident_str + "feat_month_count"], df[ident_str + "feat_day_count"] = \
        year_month_day_count(offset_string_list)

    # TODO how to encode no valid year: dataframe['feature_isnull'] = 0 #null-tracking column
    # We can try: https://stackoverflow.com/questions/58971596/random-forest-make-null-values-always-have-their-own-branch-in-a-decision-tree

    df[ident_str + "feat_substring_count_slash"] = substring_count(offset_string_list, "/")
    df[ident_str + "feat_substring_count_percent"] = substring_count(offset_string_list, "%")
    df[ident_str + "feat_substring_count_e"] = substring_count(offset_string_list, "e")
    df[ident_str + "feat_substring_count_g"] = substring_count(offset_string_list, "g")
    df[ident_str + "feat_substring_count_a"] = substring_count(offset_string_list, "a")
    df[ident_str + "feat_substring_count_u"] = substring_count(offset_string_list, "u")
    df[ident_str + "feat_substring_count_i"] = substring_count(offset_string_list, "i")
    df[ident_str + "feat_substring_count_f"] = substring_count(offset_string_list, "f")
    df[ident_str + "feat_substring_count_s"] = substring_count(offset_string_list, "s")
    df[ident_str + "feat_substring_count_oe"] = substring_count(offset_string_list, "ö")  # TODO make it per language
    df[ident_str + "feat_substring_count_ae"] = substring_count(offset_string_list, "ä")  # TODO make it per
    df[ident_str + "feat_substring_count_ue"] = substring_count(offset_string_list, "ü")
    df[ident_str + "feat_substring_count_er"] = substring_count(offset_string_list, "er")
    df[ident_str + "feat_substring_count_str"] = substring_count(offset_string_list, "str")
    df[ident_str + "feat_substring_count_k"] = substring_count(offset_string_list, "k")
    df[ident_str + "feat_substring_count_r"] = substring_count(offset_string_list, "r")
    df[ident_str + "feat_substring_count_y"] = substring_count(offset_string_list, "y")
    df[ident_str + "feat_substring_count_en"] = substring_count(offset_string_list, "en")
    df[ident_str + "feat_substring_count_g"] = substring_count(offset_string_list, "g")
    df[ident_str + "feat_substring_count_ch"] = substring_count(offset_string_list, "ch")
    df[ident_str + "feat_substring_count_sch"] = substring_count(offset_string_list, "sch")
    df[ident_str + "feat_substring_count_c"] = substring_count(offset_string_list, "c")
    df[ident_str + "feat_substring_count_ei"] = substring_count(offset_string_list, "ei")
    df[ident_str + "feat_substring_count_on"] = substring_count(offset_string_list, "on")
    df[ident_str + "feat_substring_count_ohn"] = substring_count(offset_string_list, "ohn")
    df[ident_str + "feat_substring_count_n"] = substring_count(offset_string_list, "n")
    df[ident_str + "feat_substring_count_m"] = substring_count(offset_string_list, "m")
    df[ident_str + "feat_substring_count_j"] = substring_count(offset_string_list, "j")
    df[ident_str + "feat_substring_count_h"] = substring_count(offset_string_list, "h")

    df[ident_str + "feat_substring_count_plus"] = substring_count(offset_string_list, "+")
    df[ident_str + "feat_substring_count_minus"] = substring_count(offset_string_list, "-")
    df[ident_str + "feat_substring_count_period"] = substring_count(offset_string_list, ".")
    df[ident_str + "feat_substring_count_comma"] = substring_count(offset_string_list, ",")

    df[ident_str + "feat_starts_with_plus"] = starts_with_substring(offset_string_list, "+")
    df[ident_str + "feat_starts_with_minus"] = starts_with_substring(offset_string_list, "-")

    df[ident_str + "feat_ends_with_plus"] = ends_with_substring(offset_string_list, "+")
    df[ident_str + "feat_ends_with_minus"] = ends_with_substring(offset_string_list, "-")

    return df


def starts_with_substring(list: list, substring: str) -> list:
    """Given a list of strings return 1 if string starts with the given substring for each item."""
    return [1 if s.lower().startswith(substring) else 0 for s in list]


def ends_with_substring(list: list, substring: str) -> list:
    """Given a list of strings return 1 if string starts with the given substring for each item."""
    return [1 if s.lower().endswith(substring) else 0 for s in list]


def digit_count(s: str) -> int:
    """Return the number of digits in a string."""
    return sum(c.isdigit() for c in s)


def space_count(s: str) -> int:
    """Return the number of spaces in a string."""
    return sum(c.isspace() for c in s) + s.count('\t') * 3  # Tab is already counted as one whitespace


def special_count(s: str) -> int:
    """Return the number of special (non-alphanumeric) characters in a string."""
    return sum(not c.isalnum() for c in s)


def strip_accents(s) -> str:
    """
    Strip all accents from a string.

    Source: http://stackoverflow.com/a/518232/2809427
    """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def vowel_count(s: str) -> int:
    """Return the number of vowels in a string."""
    return sum(is_vowel(c) for c in s)


def count_string_differences(s1: str, s2: str) -> int:
    """Return the number of differences between two strings."""
    if len(s2) > len(s1):  # the longer string has to be s1 to catch all differences
        s1, s2 = s2, s1

    return len(''.join(x[2:] for x in difflib.ndiff(s1, s2) if x.startswith('- ')))


def is_vowel(c: str) -> bool:
    """Given a char this function returns a bool that represents if the char is a vowel or not."""
    return c.lower() in 'aeiou'


def upper_count(s: str) -> int:
    """Return the number of uppercase characters in a string."""
    return sum(c.isupper() for c in s)


def year_month_day_count(offset_string_list: list) -> Tuple[List[int], List[int], List[int]]:
    """Given a list of offset-strings extracts the according dates, months and years for each string."""
    year_list = []
    month_list = []
    day_list = []

    for s in offset_string_list:
        _normalization = normalize_to_date(s)

        if _normalization:
            year_list.append(int(_normalization[:4]))
            month_list.append(int(_normalization[5:7]))
            day_list.append(int(_normalization[8:10]))
        else:
            year_list.append(0) # TODO check default values? # https://stackoverflow.com/questions/30317119/classifiers-in-scikit-learn-that-handle-nan-null
            month_list.append(0)  # df.fillna(df.mean(), inplace=True)
            day_list.append(0)

    return year_list, month_list, day_list


# checks if the string is a number and gives the number a value
def num_count(s: str) -> float:
    """
    Given a string this function tries to read it as a number (if not possible returns 0).

    If possible it returns the number as a float.
    TODO: this function should return the number or the count (1 or 0)?
    """
    num = normalize(s, 'float')

    if num:
        return num
    else:
        return 0 # TODO what to with None value


def normalize_to_python_float(s: str) -> float:
    """
    Given a string this function tries to read it as a number using python float (if not possible returns 0).

    If possible it returns the number as a float.
    """
    try:
        f = float(s)
        if f < np.finfo('float32').max:
            return f
        else:
            return 0.0
    except (ValueError, TypeError):
        return 0.0


def duplicate_count(s: str) -> int:
    """Given a string this function returns the number of duplicate characters."""
    count = {}
    for c in s:
        if c in count:
            count[c] += 1
        else:
            count[c] = 1

    counter = 0
    for key in count:
        if count[key] > 1:
            counter += count[key]

    return counter


def substring_count(list: list, substring: str) -> list:
    """Given a list of strings returns the occurrence of a certain substring and returns the results as a list."""
    r_list = [0] * len(list)

    for index in range(len(list)):
        r_list[index] = list[index].lower().count(substring)

    return r_list


def unique_char_count(s: str) -> int:
    """Given a string returns the number of unique characters."""
    return len(set(list(s)))


def _convert_to_relative_dict(dict: dict):
    """Convert a dict with absolute numbers as values to the same dict with the relative probabilities as values."""
    return_dict = {}
    abs_num = sum(dict.values())
    for key, value in dict.items():
        return_dict[key] = value / abs_num
    return return_dict


def plot_label_distribution(df_list: list, df_name_list=None) -> None:
    """Plot the label-distribution of given DataFrames side-by-side."""
    # check if any of the input df are empty
    for df in df_list:
        if df.empty:
            logger.error('One of the Dataframes in df_list is empty.')
            return None

    # helper function
    def Convert(tup, di):
        for a, b in tup:
            di.setdefault(a, []).append(b)
        return di

    # plot the relative distributions
    logger.info('Percentage of total samples (per dataset) that have a certain label:')
    rel_dict_list = []
    for df in df_list:
        rel_dict_list.append(_convert_to_relative_dict(Counter(list(df['label_name']))))
    logger.info('\n' + tabulate(
        pd.DataFrame(rel_dict_list, index=df_name_list).transpose(),
        floatfmt=".1%", headers="keys", tablefmt="pipe") + '\n')

    # print the number of documents in total and in the splits given
    # total_count = 0
    for index, df in enumerate(df_list):
        doc_name = df_name_list[index] if df_name_list else str(index)
        doc_count = len(set(df['document_id']))
        logger.info(doc_name + ' contains ' + str(doc_count) + ' different documents.')
        # total_count += doc_count
    # logger.info(str(total_count) + ' documents in total.')

    # plot the number of documents with at least one of a certain label
    logger.info('Percentage of documents per split that contain a certain label at least once:')
    doc_count_dict_list = []
    for df in df_list:
        doc_count_dict = {}
        doc_count = len(set(df['document_id']))
        toup_list = list(zip(list(df['label_name']), list(df['document_id'])))
        list_dict = Convert(toup_list, {})
        for key, value in list_dict.items():
            doc_count_dict[key] = float(len(set(value)) / doc_count)
        doc_count_dict_list.append(doc_count_dict)
    logger.info('\n' + tabulate(
        pd.DataFrame(doc_count_dict_list, index=df_name_list).transpose(),
        floatfmt=".1%", headers="keys", tablefmt="pipe") + '\n')


def evaluate_split_quality(df_train: pd.DataFrame, df_val: pd.DataFrame, percentage: Optional[float] = None):
    """Evaluate if the split method used produces satisfactory results."""
    # check if df_train or df_val is empty
    if df_train.empty:
        logger.error('df_train is empty.')
        return None
    if df_val.empty:
        logger.error('df_val is empty.')
        return None
    logger.info('Start split quality tests.')
    n_train_examples = df_train.shape[0]
    n_val_examples = df_val.shape[0]
    n_total_examples = n_train_examples + n_val_examples

    # check if the splits in total numbers is ok
    if percentage and n_total_examples > 100:
        if abs(n_train_examples / (n_total_examples * percentage) - 1) > 0.05:
            logger.error(f'Splits differ from split percentage significantly. Percentage: {percentage}. '
                         + f'Real Percentage: {n_train_examples / n_total_examples}')

    train_dict = df_train['label_name'].value_counts().to_dict()
    val_dict = df_val['label_name'].value_counts().to_dict()
    total_dict = pd.concat([df_train['label_name'], df_val['label_name']]).value_counts().to_dict()

    train_dict_rel = df_train['label_name'].value_counts(normalize=True).to_dict()
    val_dict_rel = df_val['label_name'].value_counts(normalize=True).to_dict()
    total_dict_rel = pd.concat([df_train['label_name'], df_val['label_name']]).value_counts(normalize=True).to_dict()

    # checks the balance of the labels per split (and if there is at least one)
    for key, value in total_dict_rel.items():
        if key not in train_dict.keys():
            logger.error('No sample of label "' + key + '" found in training dataset.')
        elif total_dict[key] > 30 and abs(train_dict_rel[key] - value) > 0.05 * max(total_dict_rel[key], 0.01):
            logger.error('Unbalanced distribution of label "' + key + '" (Significant deviation in training set)')
        else:
            logger.info('Balanced distribution of label "' + key + '" in training set')

        if key not in val_dict.keys():
            logger.error('No sample of label "' + key + '" found in validation dataset.')
        elif total_dict[key] > 30 and abs(val_dict_rel[key] - value) > 0.05 * max(total_dict_rel[key], 0.01):
            logger.warning('Unbalanced distribution of label "' + key + '" (Significant deviation in validation set)')
        else:
            logger.info('Balanced distribution of label "' + key + '" in validation set')

    logger.info('Split quality test completed.')


def split_in_two_by_document_df(data: pd.DataFrame, percentage: float, check_imbalances=False)\
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the input df in two (by document) and return two dataframes of about the right size.

    The first item in the return tuple is of the percentage size.
    """
    logger.info('Split into test and training.')
    if data['document_id'].isnull().values.any():
        raise Exception('To split by document_id every annotation needs a non-NaN document_id!')

    df_list = [df_doc for k, df_doc in data.groupby('document_id')]

    return split_in_two_by_document_df_list(data_list=df_list, percentage=percentage,
                                            check_imbalances=check_imbalances)


def split_in_two_by_document_df_list(data_list: List[pd.DataFrame], percentage: float, check_imbalances=False)\
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a list of document df in to two concatenated df according to the percentage.

    The first item in the return tuple is of the percentage size.
    """
    logger.info('Split into test and training.')
    df_list = data_list
    total_sample_num = sum([len(df.index) for df in data_list])
    select_amount = int(total_sample_num * percentage)

    selected_count = 0
    selected_df = pd.DataFrame()
    rest_df = pd.DataFrame()

    random.Random(1).shuffle(df_list)

    # TODO: check for maximum deviation from the percentage specified
    for i, df_doc in enumerate(df_list):
        # Add first document to selected_df to avoid empty df
        if i == 0:
            selected_df = pd.concat([selected_df, df_doc])
            selected_count += len(df_doc.index)

        # Add second document to rest_df to avoid empty df
        if i == 1 and percentage < 1.0:
            rest_df = pd.concat([rest_df, df_doc])
            continue

        # Add further documents according to required percentage.
        if selected_count <= select_amount or percentage == 1.0:
            selected_df = pd.concat([selected_df, df_doc])
            selected_count += len(df_doc.index)
        else:
            rest_df = pd.concat([rest_df, df_doc])

    if selected_df.empty:
        raise Exception('Not enough data to train an AI model.')

    selected_df.reset_index(drop=True, inplace=True)
    rest_df.reset_index(drop=True, inplace=True)  # get labels used in each df

    if check_imbalances:
        selected_classes = set(selected_df['label_name'].unique())
        rest_classes = set(rest_df['label_name'].unique())
        # find labels that do not appear in both dfs
        non_overlapping_classes = selected_classes ^ rest_classes
        # remove non-overlapping examples
        selected_df = selected_df[~selected_df['label_name'].isin(non_overlapping_classes)]
        rest_df = rest_df[~rest_df['label_name'].isin(non_overlapping_classes)]
        logger.info(f'The following classes could not be split and have been removed: {non_overlapping_classes}')
    return selected_df, rest_df


# def annotation_to_dict(annotation: Annotation, include_pos: bool = False) -> dict:
#     """Convert an annotation to a dictionary."""
#     # Calculate area.
#     if annotation.x0 is None or annotation.y0 is None:
#         area = 0
#     else:
#         area = (annotation.x0 * annotation.y0)
#
#     # gets the data into a dict
#     annotation_dict = {
#         "id": annotation.id,
#         "document_id": annotation.document.id if annotation.document else None,
#         "offset_string": annotation.offset_string,
#         "normalized": annotation.normalized,
#         "label_name": annotation.label.name if annotation.label else None,
#         "revised": annotation.revised,
#         "is_correct": annotation.is_correct,
#         "accuracy": annotation.confidence,
#         "x0": annotation.x0,
#         "y0": annotation.y0,
#         "x1": annotation.x1,
#         "y1": annotation.y1,
#         "page_index": annotation.page_index,
#         "line_index": annotation.line_index,
#         "area": area,
#         "top": annotation.top,
#         "bottom": annotation.bottom,
#         "start_offset": annotation.start_offset,
#         "end_offset": annotation.end_offset,
#     }
#
#     for index, item in enumerate(annotation.l_list):
#         annotation_dict['l_dist' + str(index)] = item['dist']
#         annotation_dict['l_offset_string' + str(index)] = item['offset_string']
#         if include_pos:
#             annotation_dict['l_pos' + str(index)] = item['pos']
#     for index, item in enumerate(annotation.r_list):
#         annotation_dict['r_dist' + str(index)] = item['dist']
#         annotation_dict['r_offset_string' + str(index)] = item['offset_string']
#         if include_pos:
#             annotation_dict['r_pos' + str(index)] = item['pos']
#
#     # WIP: word on page feature
#     for index, item in enumerate(annotation.word_on_page_features):
#         annotation_dict['word_on_page_feat' + str(index)] = item
#
#     # if annotation.label and annotation.label.threshold:
#     #     annotation_dict["threshold"] = annotation.label.threshold
#     # else:
#     #     annotation_dict["threshold"] = 0.1
#
#     if hasattr(annotation, 'catchphrase_dict'):
#         for catchphrase, dist in annotation.catchphrase_dict.items():
#             annotation_dict['catchphrase_dist_' + catchphrase] = dist
#
#     return annotation_dict


# Use eval_dict function from SDK
# def span_to_dict(span: "Span") -> dict:
#     """Convert a span to a dictionary."""
#
#     annotation = span.annotation
#     res_dict = {
#         "id": span.id_,
#         "annotation_id": annotation.id_,
#         "document_id": annotation.document.id_ if annotation.document else None,
#         "offset_string": span.offset_string,
#         "normalized": span.normalized,
#         "revised": annotation.revised,
#         "is_correct": annotation.is_correct,
#         "confidence": annotation.confidence,
#         "page_index": span.page_index,  # TODO: we need page_index to create an annotation (issue 8757)
#         # "line_index": span.line_index,  # TODO: span does not have "line_index"
#         #  in addition in might be not the best feature
#         # "top": span.top,  # TODO: we need top to create an annotation from the extraction (issue 8757) I think its not needed (FZ)
#         # "bottom": span.bottom,    # TODO: we need bottom to create an annotation from the extraction (issue 8757) I think its not needed (FZ)
#         "start_offset": span.start_offset,
#         "end_offset": span.end_offset,
#     }
#     return res_dict

# Removed as untested
# def get_first_candidate(document_text, document_bbox, line_list):
#     """Get the first candidate in a document."""
#     for line_num, _line in enumerate(line_list):
#         line_start_offset = _line['start_offset']
#         line_end_offset = _line['end_offset']
#         for candidate in tokenize_fn(document_text[line_start_offset:line_end_offset]):
#             candidate_start_offset = candidate['start_offset'] + line_start_offset
#             candidate_end_offset = candidate['end_offset'] + line_start_offset
#             candidate_bbox = dict(
#                 **get_bbox(document_bbox, candidate_start_offset, candidate_end_offset),
#                 offset_string=document_text[candidate_start_offset: candidate_end_offset],
#                 start_offset=candidate_start_offset,
#                 end_offset=candidate_end_offset
#             )
#             return candidate_bbox



# Not needed anymore.
# def get_line_candidates(document_text, document_bbox, line_list, line_num, tokenize_fn, candidates_cache):
#     """Get the candidates from a given line_num."""
#     if line_num in candidates_cache:
#         return candidates_cache[line_num], candidates_cache
#     line = line_list[line_num]
#     line_start_offset = line['start_offset']
#     line_end_offset = line['end_offset']
#     line_candidates = []
#     for candidate in tokenize_fn(document_text[line_start_offset:line_end_offset]):
#         candidate_start_offset = candidate['start_offset'] + line_start_offset
#         candidate_end_offset = candidate['end_offset'] + line_start_offset
#         candidate_bbox = dict(
#             **get_bbox(document_bbox, candidate_start_offset, candidate_end_offset),
#             offset_string=document_text[candidate_start_offset: candidate_end_offset],
#             start_offset=candidate_start_offset,
#             end_offset=candidate_end_offset
#         )
#         line_candidates.append(candidate_bbox)
#     if len(candidates_cache) >= CANDIDATES_CACHE_SIZE:
#         earliest_line = min(candidates_cache.keys())
#         candidates_cache.pop(earliest_line)
#     candidates_cache[line_num] = line_candidates
#     return line_candidates, candidates_cache


def get_spatial_features(annotations: List[Annotation], abs_pos_feature_list: List, meta_information_list: List)\
        -> Tuple[pd.DataFrame, List]:
    """Get spatial features."""
    span_features: List[Dict] = []

    if not annotations:
        return pd.DataFrame(), span_features

    # annotations.sort(key=lambda x: x._spans[0].start_offset)
    for annotation in annotations:
        for span in annotation.spans:
            assert span.x0 is not None
            assert span.x1 is not None
            assert span.y0 is not None
            assert span.y1 is not None
            span_dict = {k: v for k, v in span.eval_dict().items() if k in abs_pos_feature_list + meta_information_list}
            span_features.append(span_dict)
            # span_dict = {}
            # span_dict["x0"] = span.x0
            # span_dict["y0"] = span.y0
            # span_dict["x1"] = span.x1
            # span_dict["y1"] = span.y1
            # # span_dict["page_index"] = span.page_index # page index is already present.
            # span_dict["area"] = span.x0 * span.y0
    df = pd.DataFrame(span_features)
    assert set(abs_pos_feature_list).issubset(df.columns.to_list())
    assert set(meta_information_list).issubset(df.columns.to_list())
    return df, abs_pos_feature_list


def get_y_train(annotations) -> List[str]:
    y_train = []
    for annotation in annotations:
        for span in annotation.spans:
            # TODO add test, extraction process can not handle this yet
            # if separate_labels:
            #     if span.annotation.label.name == 'NO_LABEL':
            #         y_train.append(span.annotation.label.name)
            #     else:
            #         y_train.append(span.annotation.label_set.name + '__' + span.annotation.label.name)
            # else:
            y_train.append(span.annotation.label.name)  # Label name should no always be set

    return y_train


# This is Span features
def get_span_features(
        document: Document,
        annotations: List[Annotation],
) -> Tuple[pd.DataFrame, List]:
    """
    Convert the json_data from one document to a DataFrame that can be used for training or prediction.

    Additionally returns the fake negatives, errors and conflicting annotations as a DataFrames and of course the
    column_order for training
    """
    logger.info(f'Start generating features for document {document}.')
    # span_features: List[Dict] = []

    # if document.text == '' or document.get_bbox() == {} or len(annotations) == 0:
    #     # if the document text is empty or if there are no ocr'd characters
    #     # then return an empty dataframe for the data, an empty feature list and an empty dataframe for the "error" data
    #     return pd.DataFrame(), []

    # line_list: List[Dict] = []
    # char_counter = 0
    # for line_text in document.text.replace('\f', '\n').split('\n'):
    #     n_chars_on_line = len(line_text)
    #     line_list.append({'start_offset': char_counter, 'end_offset': char_counter + n_chars_on_line})
    #     char_counter += n_chars_on_line + 1

    # comment as untested.
    # generate the Catchphrase-Dataframe
    # if catchphrase_features is not None:
    #     occurrence_dict = generate_catchphrase_occurrence_dict(line_list, catchphrase_features, document_text)

    # Untested
    # if first_word:
    #     first_candidate = get_first_candidate(document_text, document_bbox, line_list, tokenize_fn)
    #     first_word_string = first_candidate['offset_string']
    #     first_word_x0 = first_candidate['x0']
    #     first_word_y0 = first_candidate['y0']
    #     first_word_x1 = first_candidate['x1']
    #     first_word_y1 = first_candidate['y1']

    # annotations.sort(key=lambda x: x._spans[0].start_offset) # TODO sort in SDK

    # WIP: Word on page feature
    #  page_text_list = document_text.split('\f')

    # used to cache the catchphrase features
    # _line_num = -1
    # _catchphrase_dict = None
    # candidates_cache = dict()
    for annotation in annotations:
        for span in annotation.spans:
            # word_on_page_feature_list = []
            # word_on_page_feature_name_list = []

            # span.bbox()
            # span_dict = span.eval_dict()

            # WIP: Word on page feature, substring is untvested
            # if substring_features:
            #     for index, substring_feature in enumerate(substring_features):
            #         word_on_page_feature_list.append(substring_on_page(substring_feature, annotation, page_text_list))
            #         word_on_page_feature_name_list.append(f'word_on_page_feat{index}')
            # span.word_on_page_features = word_on_page_feature_list

            if annotation.id_:  # TODO should not be needed. However keep it for the moment as sanity check.
                # Annotation
                if annotation.is_correct or \
                        (not annotation.is_correct and annotation.revised) or \
                        (annotation.confidence and hasattr(annotation.label, 'threshold') and
                         annotation.confidence < annotation.label.threshold):
                    pass
                else:
                    logger.error(f'Annotation (ID {annotation.id_}) found that is not fit for the use in dataset!')

            # find the line containing the annotation
            # tokenize that line to get all candidates
            # convert each candidate into a bbox
            # append to line candidates
            # store the line_start_offset so if the next annotation is on the same line then we use the same
            # line_candidiates list and therefore saves us tokenizing the same line again
            # for line_num, line in enumerate(line_list):
            #     if line['start_offset'] <= span.end_offset and line['end_offset'] >= span.start_offset:
            #
            #         # get the catchphrase features, catchphrase untested.
            #         # if catchphrase_features is not None and len(catchphrase_features) != 0:
            #         #     if line_num == _line_num:
            #         #         span.catchphrase_dict = _catchphrase_dict
            #         #     else:
            #         #         _catchphrase_dict = generate_feature_dict_from_occurence_dict(
            #         #             occurrence_dict, catchphrase_features, line_num
            #         #         )
            #         #         span.catchphrase_dict = _catchphrase_dict
            #         #         _line_num = line_num
            #
            #         line_candidates, candidates_cache = get_line_candidates(
            #             document_text, document_bbox, line_list, line_num, tokenize_fn, candidates_cache
            #         )
            #         break

            # l_list = []
            # r_list = []
            #
            # for candidate in line_candidates:
            #     if candidate['end_offset'] <= span.start_offset:
            #         candidate['dist'] = span.x0 - candidate['x1']
            #         candidate['pos'] = 0
            #         l_list.append(candidate)
            #     elif candidate['start_offset'] >= span.end_offset:
            #         candidate['dist'] = candidate['x0'] - span.x1
            #         candidate['pos'] = 0
            #         r_list.append(candidate)
            #
            # if n_nearest_across_lines:
            #     prev_line_candidates = []
            #     i = 1
            #     while (line_num - i) >= 0:
            #         line_candidates, candidates_cache = get_line_candidates(document_text, document_bbox, line_list,
            #                                                                 line_num - i, tokenize_fn, candidates_cache)
            #         for candidate in line_candidates:
            #             candidate['dist'] = min(abs(span.x0 - candidate['x0']),
            #                                     abs(span.x0 - candidate['x1']),
            #                                     abs(span.x1 - candidate['x0']),
            #                                     abs(span.x1 - candidate['x1']))
            #             candidate['pos'] = -i
            #         prev_line_candidates.extend(line_candidates)
            #         if len(prev_line_candidates) >= n_left_nearest - len(l_list):
            #             break
            #         i += 1
            #
            #     next_line_candidates = []
            #     i = 1
            #     while line_num + i < len(line_list):
            #         line_candidates, candidates_cache = get_line_candidates(document_text, document_bbox, line_list,
            #                                                                 line_num + i, tokenize_fn, candidates_cache)
            #         for candidate in line_candidates:
            #             candidate['dist'] = min(abs(span.x0 - candidate['x0']),
            #                                     abs(span.x0 - candidate['x1']),
            #                                     abs(span.x1 - candidate['x0']),
            #                                     abs(span.x1 - candidate['x1']))
            #             candidate['pos'] = i
            #         next_line_candidates.extend(line_candidates)
            #         if len(next_line_candidates) >= n_right_nearest - len(r_list):
            #             break
            #         i += 1

            # # set first word features
            # if first_word:
            #     span.first_word_x0 = first_word_x0
            #     span.first_word_y0 = first_word_y0
            #     span.first_word_x1 = first_word_x1
            #     span.first_word_y1 = first_word_y1
            #     span.first_word_string = first_word_string



            # n_smallest_l_list = nsmallest(n_left_nearest, l_list, key=lambda x: x['dist'])
            # n_smallest_r_list = nsmallest(n_right_nearest, r_list, key=lambda x: x['dist'])
            #
            # if n_nearest_across_lines:
            #     n_smallest_l_list.extend(prev_line_candidates[::-1])
            #     n_smallest_r_list.extend(next_line_candidates)
            #
            # while len(n_smallest_l_list) < n_left_nearest:
            #     n_smallest_l_list.append({'offset_string': '', 'dist': 100000, 'pos': 0})
            #
            # while len(n_smallest_r_list) < n_right_nearest:
            #     n_smallest_r_list.append({'offset_string': '', 'dist': 100000, 'pos': 0})
            #
            # r_list = n_smallest_r_list[:n_right_nearest]
            # l_list = n_smallest_l_list[:n_left_nearest]

            # for index, item in enumerate(l_list):
            #     span_dict['l_dist' + str(index)] = item['dist']
            #     span_dict['l_offset_string' + str(index)] = item['offset_string']
            #     if n_nearest_across_lines:
            #         span_dict['l_pos' + str(index)] = item['pos']
            # for index, item in enumerate(r_list):
            #     span_dict['r_dist' + str(index)] = item['dist']
            #     span_dict['r_offset_string' + str(index)] = item['offset_string']
            #     if n_nearest_across_lines:
            #         span_dict['r_pos' + str(index)] = item['pos']

            # # checks for ERRORS
            # if span_dict["confidence"] is None \
            #         and not (span_dict["revised"] is False and span_dict["is_correct"] is True):
            #     raise Exception('Wrong annotation entered training.')
            #
            # # adds the sample_data to the list
            # if span_dict["page_index"] is None:
            #     raise Exception('Wrong annotation entered training.')
            #
            # span_features.append(span_dict)

    # creates the dataframe
    # df = pd.DataFrame(span_features)

    # # first word features
    # if first_word:
    #     df['first_word_x0'] = first_word_x0
    #     df['first_word_x1'] = first_word_x1
    #     df['first_word_y0'] = first_word_y0
    #     df['first_word_y1'] = first_word_y1
    #     df['first_word_string'] = first_word_string
    #
    #     # first word string features
    #     df_string_features_first = convert_to_feat(list(df["first_word_string"]), "first_word_")
    #     string_features_first_word = list(df_string_features_first.columns.values)  # NOQA
    #     df = df.join(df_string_features_first, lsuffix='_caller', rsuffix='_other')
    #     first_word_features = ['first_word_x0', 'first_word_y0', 'first_word_x1', 'first_word_y1']

    # creates all the features from the offset string
    offset_strings = []
    for annotation in annotations:
        for span in annotation.spans:
            offset_strings.append(span.offset_string)

    df = convert_to_feat(offset_strings)
    string_feature_column_order = list(df.columns.values)

    # feature_list = string_feature_column_order # + \
    #                # relative_string_feature_list + relative_pos_feature_list # + word_on_page_feature_name_list
    # if first_word:
    #     feature_list += first_word_features

    # append the catchphrase_features to the feature_list
    # if catchphrase_features is not None:
    #     for catchphrase in catchphrase_features:
    #         feature_list.append('catchphrase_dist_' + catchphrase)

    # joins it to the main DataFrame
    # df = df.join(df_string_features_real, lsuffix='_caller', rsuffix='_other')

    return df, string_feature_column_order


def substring_on_page(substring, annotation, page_text_list) -> bool:
    """Check if there is an occurrence of the word on the according page."""
    if not hasattr(annotation, "page_index"):
        logger.warning("Annotation has no page_index!")
        return False
    elif annotation.page_index > len(page_text_list) - 1:
        logger.warning("Annotation's page_index does not match given text.")
        return False
    else:
        return substring in page_text_list[annotation.page_index]


def generate_catchphrase_occurrence_dict(line_list, catchphrase_features, document_text) -> Dict:
    """Generate a dict that stores on which line certain catchphrases occurrence."""
    _dict = {catchphrase: [] for catchphrase in catchphrase_features}

    for line_num, _line in enumerate(line_list):
        line_text = document_text[_line['start_offset']: _line['end_offset']]
        for catchphrase in catchphrase_features:
            if catchphrase in line_text:
                _dict[catchphrase].append(line_num)

    return _dict


def generate_feature_dict_from_occurence_dict(occurence_dict, catchphrase_list, line_num) -> Dict:
    """Generate the fitting catchphrase features."""
    _dict = {catchphrase: None for catchphrase in catchphrase_list}

    for catchphrase in catchphrase_list:
        _dict[catchphrase] = next((i - line_num for i in occurence_dict[catchphrase] if i < line_num), -1)

    return _dict
