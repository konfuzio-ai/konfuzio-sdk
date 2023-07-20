"""Extract information from Documents.

Conventional template matching based approaches fail to generalize well to document images of unseen templates,
and are not robust against text recognition errors.

We follow the approach proposed by Sun et al. (2021) to encode both the visual and textual
features of detected text regions, and edges of which represent the spatial relations between neighboring text
regions. Their experiments validate that all information including visual features, textual
features and spatial relations can benefit key information extraction.

We reduce the hardware requirements from 1 NVIDIA Titan X GPUs with 12 GB memory to a 1 CPU and 16 GB memory by
replacing the end-to-end pipeline into two parts.

Sun, H., Kuang, Z., Yue, X., Lin, C., & Zhang, W. (2021). Spatial Dual-Modality Graph Reasoning for Key Information
Extraction. arXiv. https://doi.org/10.48550/ARXIV.2103.14470
"""
import collections
import difflib
import functools
import logging
import os
import time
import unicodedata

from copy import deepcopy
from heapq import nsmallest
from inspect import signature
from typing import Tuple, Optional, List, Union, Dict

import numpy
import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted

from konfuzio_sdk.data import Document, Annotation, Category, AnnotationSet, Label, LabelSet, Span
from konfuzio_sdk.trainer.base import BaseModel
from konfuzio_sdk.tokenizer.paragraph_and_sentence import ParagraphTokenizer, SentenceTokenizer

from konfuzio_sdk.normalize import (
    normalize_to_float,
    normalize_to_date,
    normalize_to_percentage,
    normalize_to_positive_float,
)
from konfuzio_sdk.regex import regex_matches
from konfuzio_sdk.utils import (
    get_timestamp,
    get_bbox,
    memory_size_of,
    sdk_isinstance,
)

from konfuzio_sdk.evaluate import ExtractionEvaluation

from konfuzio_sdk.tokenizer.base import ListTokenizer

logger = logging.getLogger(__name__)

"""Multiclass classifier for document extraction."""
CANDIDATES_CACHE_SIZE = 100


# # existent model classes
# MODEL_CLASSES = {'LabelSectionModel': LabelSectionModel,
#                  'DocumentModel': DocumentModel,
#                  'ParagraphModel': ParagraphModel,
#                  'CustomDocumentModel': CustomDocumentModel,
#                  'SentenceModel': SentenceModel
#                  }
#
# COMMON_PARAMETERS = ['tokenizer', 'text_vocab', 'model_type']
#
# label_section_components = ['label_vocab',
#                             'section_vocab',
#                             'label_classifier_config',
#                             'label_classifier_state_dict',
#                             'section_classifier_config',
#                             'section_classifier_state_dict',
#                             'extract_dicts']
#
# document_components = ['image_preprocessing',
#                        'image_augmentation',
#                        'category_vocab',
#                        'document_classifier_config',
#                        'document_classifier_state_dict']
#
# paragraph_components = ['tokenizer_mode',
#                         'paragraph_category_vocab',
#                         'paragraph_classifier_config',
#                         'paragraph_classifier_state_dict']
#
# sentence_components = ['sentence_tokenizer',
#                        'tokenizer_mode',
#                        'category_vocab',
#                        'classifier_config',
#                        'classifier_state_dict']
#
# label_section_components.extend(COMMON_PARAMETERS)
# document_components.extend(COMMON_PARAMETERS)
# paragraph_components.extend(COMMON_PARAMETERS)
# sentence_components.extend(COMMON_PARAMETERS)
# custom_document_model = deepcopy(document_components)
#
# # parameters that need to be saved with the model accordingly with the model type
# MODEL_PARAMETERS_TO_SAVE = {'LabelSectionModel': label_section_components,
#                             'DocumentModel': document_components,
#                             'ParagraphModel': paragraph_components,
#                             'CustomDocumentModel': custom_document_model,
#                             'SentenceModel': sentence_components,
#                             }
#
#
#
# def load_default_model(path: str):
#     """Load a model from default models."""
#     logger.info('loading model')
#
#     # load model dict
#     loaded_data = torch.load(path)
#
#     if 'model_type' not in loaded_data.keys():
#         model_type = path.split('_')[-1].split('.')[0]
#     else:
#         model_type = loaded_data['model_type']
#
#     model_class = MODEL_CLASSES[model_type]
#     model_args = MODEL_PARAMETERS_TO_SAVE[model_type]
#
#     # Verify if loaded data has all necessary components
#     assert all([arg in model_args for arg in loaded_data.keys()])
#
#     state_dict_name = [n for n in model_args if n.endswith('_state_dict')]
#
#     if len(state_dict_name) > 1:
#         # LabelSectionModel is a combination of 2 independent classifiers
#         assert model_type == 'LabelSectionModel'
#
#         label_classifier_state_dict = loaded_data['label_classifier_state_dict']
#         section_classifier_state_dict = loaded_data['section_classifier_state_dict']
#         extract_dicts = loaded_data['extract_dicts']
#
#         del loaded_data['label_classifier_state_dict']
#         del loaded_data['section_classifier_state_dict']
#         del loaded_data['extract_dicts']
#
#     else:
#         classifier_state_dict = loaded_data[state_dict_name[0]]
#         del loaded_data[state_dict_name[0]]
#
#     if 'model_type' in loaded_data.keys():
#         del loaded_data['model_type']
#
#     # create instance of the model class
#     model = model_class(projects=None, **loaded_data)
#
#     if model_type == 'LabelSectionModel':
#         # LabelSectionModel is a special case because it has 2 independent classifiers
#         # load parameters of the classifiers from saved parameters
#         model.label_classifier.load_state_dict(label_classifier_state_dict)
#         model.section_classifier.load_state_dict(section_classifier_state_dict)
#
#         # load extract dicts
#         model.extract_dicts = extract_dicts
#
#         # need to ensure classifiers start in evaluation mode
#         model.label_classifier.eval()
#         model.section_classifier.eval()
#
#     else:
#         # load parameters of the classifiers from saved parameters
#         model.classifier.load_state_dict(classifier_state_dict)
#
#         # need to ensure classifiers start in evaluation mode
#         model.classifier.eval()
#
#     return model

#
# def load_pickle(pickle_name: str, folder_path: str):
#     """
#     Load a pkl file or a pt (pytorch) file.
#
#   First check if the .pkl file exists at ./konfuzio.MODEL_ROOT/pickle_name, if not then assumes it is at ./pickle_name
#     Then, it assumes the .pkl file is compressed with bz2 and tries to extract and load it. If the pickle file is not
#     compressed with bz2 then it will throw an OSError and we then try and load the .pkl file will dill. This will then
#     throw an UnpicklingError if the file is not a pickle file, as expected.
#
#     :param pickle_name:
#     :return:
#     """
#     # https://stackoverflow.com/a/43006034/5344492
#     dill._dill._reverse_typemap['ClassType'] = type
#     pickle_path = os.path.join(folder_path, pickle_name)
#     if not os.path.isfile(pickle_path):
#         pickle_path = pickle_name

#     device = 'cpu'
#     if torch.cuda.is_available():
#         device = 'cuda'
#
#     if pickle_name.endswith('.pt'):
#         with open(pickle_path, 'rb') as f:
#             file_data = torch.load(pickle_path, map_location=torch.device(device))
#
#         if isinstance(file_data, dict):
#             # verification of str in path can be removed after all models being updated with the model_type
#             possible_names = [
#                 '_LabelSectionModel',
#                 '_DocumentModel',
#                 '_ParagraphModel',
#                 '_CustomDocumentModel',
#                 '_SentenceModel',
#             ]
#             if ('model_type' in file_data.keys() and file_data['model_type'] in MODEL_PARAMETERS_TO_SAVE.keys()) or
#               any([n in pickle_name for n in possible_names]):
#                 file_data = load_default_model(pickle_name)
#
#             else:
#                 raise NameError("Model type not recognized.")
#
#         else:
#             with open(pickle_path, 'rb') as f:
#                 file_data = torch.load(f, map_location=torch.device(device))
#     else:
#         try:
#             with bz2.open(pickle_path, 'rb') as f:
#                 file_data = dill.load(f)
#         except OSError:
#             with open(pickle_path, 'rb') as f:
#                 file_data = dill.load(f)
#
#     return file_data


def convert_to_feat(offset_string_list: list, ident_str: str = '') -> pandas.DataFrame:
    """Return a df containing all the features generated using the offset_string."""
    df = dict()  # pandas.DataFrame()

    # strip all accents
    offset_string_list_accented = offset_string_list
    offset_string_list = [strip_accents(s) for s in offset_string_list]

    # gets the return lists for all the features
    df[ident_str + "feat_vowel_len"] = [vowel_count(s) for s in offset_string_list]
    df[ident_str + "feat_special_len"] = [special_count(s) for s in offset_string_list]
    df[ident_str + "feat_space_len"] = [space_count(s) for s in offset_string_list]
    df[ident_str + "feat_digit_len"] = [digit_count(s) for s in offset_string_list]
    df[ident_str + "feat_len"] = [len(s) for s in offset_string_list]
    df[ident_str + "feat_upper_len"] = [upper_count(s) for s in offset_string_list]
    df[ident_str + "feat_date_count"] = [date_count(s) for s in offset_string_list]
    df[ident_str + "feat_num_count"] = [num_count(s) for s in offset_string_list]
    df[ident_str + "feat_as_float"] = [normalize_to_python_float(offset_string) for offset_string in offset_string_list]
    df[ident_str + "feat_unique_char_count"] = [unique_char_count(s) for s in offset_string_list]
    df[ident_str + "feat_duplicate_count"] = [duplicate_count(s) for s in offset_string_list]
    df[ident_str + "accented_char_count"] = [
        count_string_differences(s1, s2) for s1, s2 in zip(offset_string_list, offset_string_list_accented)
    ]

    (
        df[ident_str + "feat_year_count"],
        df[ident_str + "feat_month_count"],
        df[ident_str + "feat_day_count"],
    ) = year_month_day_count(offset_string_list)

    df[ident_str + "feat_substring_count_slash"] = substring_count(offset_string_list, "/")
    df[ident_str + "feat_substring_count_percent"] = substring_count(offset_string_list, "%")
    df[ident_str + "feat_substring_count_e"] = substring_count(offset_string_list, "e")
    df[ident_str + "feat_substring_count_g"] = substring_count(offset_string_list, "g")
    df[ident_str + "feat_substring_count_a"] = substring_count(offset_string_list, "a")
    df[ident_str + "feat_substring_count_u"] = substring_count(offset_string_list, "u")
    df[ident_str + "feat_substring_count_i"] = substring_count(offset_string_list, "i")
    df[ident_str + "feat_substring_count_f"] = substring_count(offset_string_list, "f")
    df[ident_str + "feat_substring_count_s"] = substring_count(offset_string_list, "s")
    df[ident_str + "feat_substring_count_oe"] = substring_count(offset_string_list, "ö")
    df[ident_str + "feat_substring_count_ae"] = substring_count(offset_string_list, "ä")
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

    df = pandas.DataFrame(df)

    return df


def substring_count(list: list, substring: str) -> list:
    """Given a list of strings returns the occurrence of a certain substring and returns the results as a list."""
    r_list = [0] * len(list)

    for index in range(len(list)):
        r_list[index] = list[index].lower().count(substring)

    return r_list


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


def date_count(s: str) -> int:
    """
    Given a string this function tries to read it as a date (if not possible returns 0).

    If possible it returns the relative difference to 01.01.2010 in days.
    """
    # checks the format
    if len(s) > 5:
        if (s[2] == '.' and s[5] == '.') or (s[2] == '/' and s[5] == '/'):
            date1 = pandas.to_datetime("01.01.2010", dayfirst=True)
            date2 = normalize_to_date(s)
            if not date2:
                return 0
            date2 = pandas.to_datetime(date2, errors='ignore')
            if date2 == s:
                return 0
            else:
                try:
                    diff = int((date2 - date1) / numpy.timedelta64(1, 'D'))
                except TypeError as e:
                    logger.debug(f'Could not substract for string {s} because of >>{e}<<.')
                    return 0

            if diff == 0:
                return 1
            else:
                return diff

        else:
            return 0
    return 0


def year_month_day_count(offset_string_list: list) -> Tuple[List[int], List[int], List[int]]:
    """Given a list of offset-strings extracts the according dates, months and years for each string."""
    year_list = []
    month_list = []
    day_list = []

    assert isinstance(offset_string_list, list)

    for s in offset_string_list:
        _normalization = normalize_to_date(s)
        if _normalization:
            year_list.append(int(_normalization[:4]))
            month_list.append(int(_normalization[5:7]))
            day_list.append(int(_normalization[8:10]))
        else:
            year_list.append(0)
            month_list.append(0)
            day_list.append(0)

    return year_list, month_list, day_list


# checks if the string is a number and gives the number a value
def num_count(s: str) -> float:
    """
    Given a string this function tries to read it as a number (if not possible returns 0).

    If possible it returns the number as a float.
    """
    num = normalize_to_float(s)

    if num:
        return num
    else:
        return 0


def normalize_to_python_float(s: str) -> float:
    """
    Given a string this function tries to read it as a number using python float (if not possible returns 0).

    If possible it returns the number as a float.
    """
    try:
        f = float(s)
        if f < numpy.finfo('float32').max:
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


def unique_char_count(s: str) -> int:
    """Given a string returns the number of unique characters."""
    return len(set(list(s)))


def get_first_candidate(document_text, document_bbox, line_list):
    """Get the first candidate in a document."""
    # todo allow to have mult tokenizers?
    for line_num, _line in enumerate(line_list):
        line_start_offset = _line['start_offset']
        line_end_offset = _line['end_offset']
        # todo
        tokenize_fn = functools.partial(regex_matches, regex='[^ \n\t\f]+')
        for candidate in tokenize_fn(document_text[line_start_offset:line_end_offset]):
            candidate_start_offset = candidate['start_offset'] + line_start_offset
            candidate_end_offset = candidate['end_offset'] + line_start_offset
            candidate_bbox = dict(
                **get_bbox(document_bbox, candidate_start_offset, candidate_end_offset),
                offset_string=document_text[candidate_start_offset:candidate_end_offset],
                start_offset=candidate_start_offset,
                end_offset=candidate_end_offset,
            )
            return candidate_bbox


def get_line_candidates(document_text, document_bbox, line_list, line_num, candidates_cache):
    """Get the candidates from a given line_num."""
    if line_num in candidates_cache:
        return candidates_cache[line_num], candidates_cache
    line = line_list[line_num]
    line_start_offset = line['start_offset']
    line_end_offset = line['end_offset']
    line_candidates = []
    # todo see get_first_candidate
    tokenize_fn = functools.partial(regex_matches, regex='[^ \n\t\f]+')
    for candidate in tokenize_fn(document_text[line_start_offset:line_end_offset]):
        candidate_start_offset = candidate['start_offset'] + line_start_offset
        candidate_end_offset = candidate['end_offset'] + line_start_offset
        # todo: the next line is memory heavy
        #  https://gitlab.com/konfuzio/objectives/-/issues/9342
        candidate_bbox = dict(
            **get_bbox(document_bbox, candidate_start_offset, candidate_end_offset),
            offset_string=document_text[candidate_start_offset:candidate_end_offset],
            start_offset=candidate_start_offset,
            end_offset=candidate_end_offset,
        )
        line_candidates.append(candidate_bbox)
    if len(candidates_cache) >= CANDIDATES_CACHE_SIZE:
        earliest_line = min(candidates_cache.keys())
        candidates_cache.pop(earliest_line)
    candidates_cache[line_num] = line_candidates
    return line_candidates, candidates_cache


def process_document_data(
    document: Document,
    spans: List[Span],
    n_nearest: Union[int, List, Tuple] = 2,
    first_word: bool = True,
    n_nearest_across_lines: bool = False,
) -> Tuple[pandas.DataFrame, List, pandas.DataFrame]:
    """
    Convert the json_data from one Document to a DataFrame that can be used for training or prediction.

    Additionally returns the fake negatives, errors and conflicting annotations as a DataFrames and of course the
    column_order for training
    """
    logger.info(f'Start generating features for document {document}.')

    assert spans == sorted(spans)  # should be already sorted

    file_error_data = []
    file_data_raw = []

    if isinstance(n_nearest, int):
        n_left_nearest = n_nearest
        n_right_nearest = n_nearest
    else:
        assert isinstance(n_nearest, (tuple, list)) and len(n_nearest) == 2
        n_left_nearest, n_right_nearest = n_nearest

    l_keys = ["l_dist" + str(x) for x in range(n_left_nearest)]
    r_keys = ["r_dist" + str(x) for x in range(n_right_nearest)]

    if n_nearest_across_lines:
        l_keys += ["l_pos" + str(x) for x in range(n_left_nearest)]
        r_keys += ["r_pos" + str(x) for x in range(n_right_nearest)]

    document_bbox = document.get_bbox()
    document_text = document.text
    document_n_pages = document.number_of_pages

    if document_text is None or document_bbox == {} or len(spans) == 0:
        # if the document text is empty or if there are no ocr'd characters
        # then return an empty dataframe for the data, an empty feature list and an empty dataframe for the "error" data
        raise NotImplementedError

    line_list: List[Dict] = []
    char_counter = 0
    for line_text in document_text.replace('\f', '\n').split('\n'):
        n_chars_on_line = len(line_text)
        line_list.append({'start_offset': char_counter, 'end_offset': char_counter + n_chars_on_line})
        char_counter += n_chars_on_line + 1

    if first_word:
        first_candidate = get_first_candidate(document_text, document_bbox, line_list)
        first_word_string = first_candidate['offset_string']
        first_word_x0 = first_candidate['x0']
        first_word_y0 = first_candidate['y0']
        first_word_x1 = first_candidate['x1']
        first_word_y1 = first_candidate['y1']

    candidates_cache = dict()
    for span in spans:

        # if span.annotation.id_:
        #     # Annotation
        #     logger.error(f'{span}')
        #     if (
        #         span.annotation.is_correct
        #         or (not span.annotation.is_correct and span.annotation.revised)
        #         or (
        #             span.annotation.confidence
        #             and hasattr(span.annotation.label, 'threshold')
        #             and span.annotation.confidence > span.annotation.label.threshold
        #         )
        #     ):
        #         pass
        #     else:
        #         logger.error(f'Annotation (ID {span.annotation.id_}) found that is not fit for the use in dataset!')

        # find the line containing the annotation
        # tokenize that line to get all candidates
        # convert each candidate into a bbox
        # append to line candidates
        # store the line_start_offset so if the next annotation is on the same line then we use the same
        # line_candidiates list and therefore saves us tokenizing the same line again

        line_num = span.line_index

        line_candidates, candidates_cache = get_line_candidates(
            document_text, document_bbox, line_list, line_num, candidates_cache
        )

        l_list = []
        r_list = []

        # todo add way to calculate distance features between spans consistently
        # https://gitlab.com/konfuzio/objectives/-/issues/9688
        for candidate in line_candidates:
            try:
                span.bbox()
                if candidate['end_offset'] <= span.start_offset:
                    candidate['dist'] = span.bbox().x0 - candidate['x1']
                    candidate['pos'] = 0
                    l_list.append(candidate)
                elif candidate['start_offset'] >= span.end_offset:
                    candidate['dist'] = candidate['x0'] - span.bbox().x1
                    candidate['pos'] = 0
                    r_list.append(candidate)
            except ValueError as e:
                logger.error(f'{candidate}: {str(e)}')

        if n_nearest_across_lines:
            prev_line_candidates = []
            i = 1
            while (line_num - i) >= 0:
                line_candidates, candidates_cache = get_line_candidates(
                    document_text,
                    document_bbox,
                    line_list,
                    line_num - i,
                    candidates_cache,
                )
                for candidate in line_candidates:
                    candidate['dist'] = min(
                        abs(span.bbox().x0 - candidate['x0']),
                        abs(span.bbox().x0 - candidate['x1']),
                        abs(span.bbox().x1 - candidate['x0']),
                        abs(span.bbox().x1 - candidate['x1']),
                    )
                    candidate['pos'] = -i
                prev_line_candidates.extend(line_candidates)
                if len(prev_line_candidates) >= n_left_nearest - len(l_list):
                    break
                i += 1

            next_line_candidates = []
            i = 1
            while line_num + i < len(line_list):
                line_candidates, candidates_cache = get_line_candidates(
                    document_text,
                    document_bbox,
                    line_list,
                    line_num + i,
                    candidates_cache,
                )
                for candidate in line_candidates:
                    candidate['dist'] = min(
                        abs(span.bbox().x0 - candidate['x0']),
                        abs(span.bbox().x0 - candidate['x1']),
                        abs(span.bbox().x1 - candidate['x0']),
                        abs(span.bbox().x1 - candidate['x1']),
                    )
                    candidate['pos'] = i
                next_line_candidates.extend(line_candidates)
                if len(next_line_candidates) >= n_right_nearest - len(r_list):
                    break
                i += 1

        n_smallest_l_list = nsmallest(n_left_nearest, l_list, key=lambda x: x['dist'])
        n_smallest_r_list = nsmallest(n_right_nearest, r_list, key=lambda x: x['dist'])

        if n_nearest_across_lines:
            n_smallest_l_list.extend(prev_line_candidates[::-1])
            n_smallest_r_list.extend(next_line_candidates)

        while len(n_smallest_l_list) < n_left_nearest:
            n_smallest_l_list.append({'offset_string': '', 'dist': 100000, 'pos': 0})

        while len(n_smallest_r_list) < n_right_nearest:
            n_smallest_r_list.append({'offset_string': '', 'dist': 100000, 'pos': 0})

        r_list = n_smallest_r_list[:n_right_nearest]
        l_list = n_smallest_l_list[:n_left_nearest]

        # set first word features
        if first_word:
            span.first_word_x0 = first_word_x0
            span.first_word_y0 = first_word_y0
            span.first_word_x1 = first_word_x1
            span.first_word_y1 = first_word_y1
            span.first_word_string = first_word_string

        span_dict = span.eval_dict()
        # span_to_dict(span=span, include_pos=n_nearest_across_lines)

        for index, item in enumerate(l_list):
            span_dict['l_dist' + str(index)] = item['dist']
            span_dict['l_offset_string' + str(index)] = item['offset_string']
            if n_nearest_across_lines:
                span_dict['l_pos' + str(index)] = item['pos']
        for index, item in enumerate(r_list):
            span_dict['r_dist' + str(index)] = item['dist']
            span_dict['r_offset_string' + str(index)] = item['offset_string']
            if n_nearest_across_lines:
                span_dict['r_pos' + str(index)] = item['pos']

        # checks for ERRORS
        if span_dict["confidence"] is None and not (span_dict["revised"] is False and span_dict["is_correct"] is True):
            file_error_data.append(span_dict)

        # adds the sample_data to the list
        if span_dict["page_index"] is not None:
            file_data_raw.append(span_dict)

    # creates the dataframe
    df = pandas.DataFrame(file_data_raw)
    df_errors = pandas.DataFrame(file_error_data)

    # first word features
    if first_word:
        df['first_word_x0'] = first_word_x0
        df['first_word_x1'] = first_word_x1
        df['first_word_y0'] = first_word_y0
        df['first_word_y1'] = first_word_y1
        df['first_word_string'] = first_word_string

        # first word string features
        df_string_features_first = convert_to_feat(list(df["first_word_string"]), "first_word_")
        string_features_first_word = list(df_string_features_first.columns.values)  # NOQA
        df = df.join(df_string_features_first, lsuffix='_caller', rsuffix='_other')
        first_word_features = ['first_word_x0', 'first_word_y0', 'first_word_x1', 'first_word_y1']
        first_word_features += string_features_first_word

    # creates all the features from the offset string
    df_string_features_real = convert_to_feat(list(df["offset_string"]))
    string_feature_column_order = list(df_string_features_real.columns.values)

    # joins it to the main DataFrame
    df = df.join(df_string_features_real, lsuffix='_caller', rsuffix='_other')
    relative_string_feature_list = []

    for index in range(n_left_nearest):
        df_string_features_l = convert_to_feat(list(df['l_offset_string' + str(index)]), 'l' + str(index) + '_')
        relative_string_feature_list += list(df_string_features_l.columns.values)
        df = df.join(df_string_features_l, lsuffix='_caller', rsuffix='_other')

    for index in range(n_right_nearest):
        df_string_features_r = convert_to_feat(list(df['r_offset_string' + str(index)]), 'r' + str(index) + '_')
        relative_string_feature_list += list(df_string_features_r.columns.values)
        df = df.join(df_string_features_r, lsuffix='_caller', rsuffix='_other')

    df["relative_position_in_page"] = df["page_index"] / document_n_pages

    abs_pos_feature_list = ["x0", "y0", "x1", "y1", "page_index", "area_quadrant_two", "area"]
    relative_pos_feature_list = [
        "x0_relative",
        "x1_relative",
        "y0_relative",
        "y1_relative",
        "relative_position_in_page",
    ]

    feature_list = (
        string_feature_column_order
        + abs_pos_feature_list
        + l_keys
        + r_keys
        + relative_string_feature_list
        + relative_pos_feature_list
    )
    if first_word:
        feature_list += first_word_features

    return df, feature_list, df_errors


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


class AbstractExtractionAI(BaseModel):
    """Parent class for all Extraction AIs, to extract information from unstructured human-readable text."""

    requires_text = True
    requires_images = False

    def __init__(self, category: Category, *args, **kwargs):
        """Initialize ExtractionModel."""
        # Go through keyword arguments, and either save their values to our
        # instance, or raise an error.
        super().__init__()
        self.category = category
        self.clf = None
        self.label_feature_list = None  # will be set later

        self.df_train = None

        self.evaluation = None

    @property
    def project(self):
        """Get RFExtractionAI Project."""
        if not self.category:
            raise AttributeError(f'{self} has no Category.')
        return self.category.project

    def check_is_ready(self):
        """
        Check if the ExtractionAI is ready for the inference.

        It is assumed that the model is ready if a Category is set, and is ready for extraction.

        :raises AttributeError: When no Category is specified.
        """
        logger.info(f"Checking if {self} is ready for extraction.")
        if not self.category:
            raise AttributeError(f'{self} requires a Category.')

    def fit(self):
        """Use as placeholder Function because the Abstract AI does not train a classifier."""
        logger.warning(f'{self} does not train a classifier.')
        pass

    def evaluate(self):
        """Use as placeholder Function."""
        logger.warning(f'{self} does not evaluate results.')
        pass

    def extract(self):
        """Use as placeholder Function."""
        logger.warning(f'{self} does not extract.')
        pass

    def extraction_result_to_document(self, document: Document, extraction_result: dict) -> Document:
        """Return a virtual Document annotated with AI Model output."""
        virtual_doc = deepcopy(document)
        virtual_annotation_set_id = 1  # counter for across mult. Annotation Set groups of a Label Set

        # define Annotation Set for the Category Label Set: todo: this is unclear from API side
        # default Annotation Set will be always added even if there are no predictions for it
        category_label_set = self.category.project.get_label_set_by_id(self.category.id_)
        virtual_default_annotation_set = AnnotationSet(
            document=virtual_doc, label_set=category_label_set, id_=virtual_annotation_set_id
        )
        virtual_annotation_set_id += 1
        for label_or_label_set_name, information in extraction_result.items():

            if isinstance(information, pandas.DataFrame):
                if information.empty:
                    continue

                # annotations belong to the default Annotation Set
                label = self.category.project.get_label_by_name(label_or_label_set_name)
                self.add_extractions_as_annotations(
                    document=virtual_doc,
                    extractions=information,
                    label=label,
                    label_set=category_label_set,
                    annotation_set=virtual_default_annotation_set,
                )
            # process multi Annotation Sets that are not part of the category Label Set
            else:
                label_set = self.category.project.get_label_set_by_name(label_or_label_set_name)

                if not isinstance(information, list):
                    information = [information]

                for entry in information:  # represents one of pot. multiple annotation-sets belonging of one LabelSet
                    if label_set is not category_label_set:
                        virtual_annotation_set = AnnotationSet(
                            document=virtual_doc, label_set=label_set, id_=virtual_annotation_set_id
                        )
                        virtual_annotation_set_id += 1
                    else:
                        virtual_annotation_set = virtual_default_annotation_set

                    for label_name, extractions in entry.items():
                        label = self.category.project.get_label_by_name(label_name)
                        self.add_extractions_as_annotations(
                            document=virtual_doc,
                            extractions=extractions,
                            label=label,
                            label_set=label_set,
                            annotation_set=virtual_annotation_set,
                        )

        return virtual_doc

    @staticmethod
    def add_extractions_as_annotations(
        extractions: pandas.DataFrame,
        document: Document,
        label: Label,
        label_set: LabelSet,
        annotation_set: AnnotationSet,
    ) -> None:
        """Add the extraction of a model to the document."""
        if not isinstance(extractions, pandas.DataFrame):
            raise TypeError(f'Provided extraction object should be a Dataframe, got a {type(extractions)} instead')
        if not extractions.empty:
            # TODO: define required fields
            required_fields = ['start_offset', 'end_offset', 'confidence']
            if not set(required_fields).issubset(extractions.columns):
                raise ValueError(
                    f'Extraction do not contain all required fields: {required_fields}.'
                    f' Extraction columns: {extractions.columns.to_list()}'
                )

            extracted_spans = extractions[required_fields].sort_values(by='confidence', ascending=False)

            for span in extracted_spans.to_dict('records'):
                try:
                    annotation = Annotation(
                        document=document,
                        label=label,
                        confidence=span['confidence'],
                        label_set=label_set,
                        annotation_set=annotation_set,
                        spans=[Span(start_offset=span['start_offset'], end_offset=span['end_offset'])],
                    )
                    if annotation.spans[0].offset_string is None:
                        raise NotImplementedError(
                            f"Extracted {annotation} does not have a correspondence in the " f"text of {document}."
                        )
                except ValueError as e:
                    if 'is a duplicate of' in str(e):
                        # Second duplicate Span is lower confidence since we sorted spans earlier, so we can ignore it
                        logger.warning(f'Could not add duplicated {span}: {str(e)}')
                    else:
                        raise e

    @classmethod
    def merge_horizontal(cls, res_dict: Dict, doc_text: str) -> Dict:
        """Merge contiguous spans with same predicted label.

        See more details at https://dev.konfuzio.com/sdk/explanations.html#horizontal-merge
        """
        logger.info("Horizontal merge.")
        merged_res_dict = dict()  # stores final results
        for label, items in res_dict.items():
            res_dicts = []
            buffer = []
            end = None

            for _, row in items.iterrows():  # iterate over the rows in the DataFrame
                # if they are valid merges then add to buffer
                if end and cls.is_valid_horizontal_merge(row, buffer, doc_text):
                    buffer.append(row)
                    end = row['end_offset']
                else:  # else, flush the buffer by creating a res_dict
                    if buffer:
                        res_dict = cls.flush_buffer(buffer, doc_text)
                        res_dicts.append(res_dict)
                    buffer = []
                    buffer.append(row)
                    end = row['end_offset']
            if buffer:  # flush buffer at the very end to clear anything left over
                res_dict = cls.flush_buffer(buffer, doc_text)
                res_dicts.append(res_dict)
            merged_df = pandas.DataFrame(
                res_dicts
            )  # convert the list of res_dicts created by `flush_buffer` into a DataFrame

            merged_res_dict[label] = merged_df

        return merged_res_dict

    @staticmethod
    def flush_buffer(buffer: List[pandas.Series], doc_text: str) -> Dict:
        """
        Merge a buffer of entities into a dictionary (which will eventually be turned into a DataFrame).

        A buffer is a list of pandas.Series objects.
        """
        assert 'label_name' in buffer[0]
        label = buffer[0]['label_name']

        starts = buffer[0]['start_offset']
        ends = buffer[-1]['end_offset']
        text = doc_text[starts:ends]

        res_dict = dict()
        res_dict['start_offset'] = starts
        res_dict['end_offset'] = ends
        res_dict['label_name'] = label
        res_dict['offset_string'] = text
        res_dict['confidence'] = numpy.mean([b['confidence'] for b in buffer])
        return res_dict

    @staticmethod
    def is_valid_horizontal_merge(
        row: pandas.Series,
        buffer: List[pandas.Series],
        doc_text: str,
        max_offset_distance: int = 5,
    ) -> bool:
        """
        Verify if the merging that we are trying to do is valid.

        A merging is valid only if:
          * All spans have the same predicted Label
          * Confidence of predicted Label is above the Label threshold
          * All spans are on the same line
          * No extraneous characters in between spans
          * A maximum of 5 spaces in between spans
          * The Label type is not one of the following: 'Number', 'Positive Number', 'Percentage', 'Date'
            OR the resulting merging create a span normalizable to the same type

        :param row: Row candidate to be merged to what is already in the buffer.
        :param buffer: Previous information.
        :param doc_text: Text of the document.
        :param max_offset_distance: Maximum distance between two entities that can be merged.
        :return: If the merge is valid or not.
        """
        if row['confidence'] < row['label_threshold']:
            return False

        # sanity checks
        if buffer[-1]['label_name'] != row['label_name']:
            return False
        elif buffer[-1]['confidence'] < buffer[-1]['label_threshold']:
            return False

        # Do not merge if any character in between the two Spans
        if not all([c == ' ' for c in doc_text[buffer[-1]['end_offset'] : row['start_offset']]]):
            return False

        # Do not merge if the difference in the offsets is bigger than the maximum offset distance
        if row['start_offset'] - buffer[-1]['end_offset'] > max_offset_distance:
            return False

        # only merge if text is on same line
        if '\n' in doc_text[buffer[0]['start_offset'] : row['end_offset']]:
            return False

        # Do not merge overlapping spans
        if row['start_offset'] < buffer[-1]['end_offset']:
            return False

        data_type = row['data_type']
        # always merge if not one of these data types
        if data_type not in {'Number', 'Positive Number', 'Percentage', 'Date'}:
            return True

        merge = None
        text = doc_text[buffer[0]['start_offset'] : row['end_offset']]

        # only merge percentages/dates/(positive) numbers if the result is still normalizable to the type
        if data_type == 'Percentage':
            merge = normalize_to_percentage(text)
        elif data_type == 'Date':
            merge = normalize_to_date(text)
        elif data_type == 'Number':
            merge = normalize_to_float(text)
        elif data_type == 'Positive Number':
            merge = normalize_to_positive_float(text)

        return merge is not None

    @staticmethod
    def has_compatible_interface(other) -> bool:
        """
        Validate that an instance of an Extraction AI implements the same interface as AbstractExtractionAI.

        An Extraction AI should implement methods with the same signature as:
        - AbstractExtractionAI.__init__
        - AbstractExtractionAI.fit
        - AbstractExtractionAI.extract
        - AbstractExtractionAI.check_is_ready

        :param other: An instance of an Extraction AI to compare with.
        """
        try:
            return (
                signature(other.__init__).parameters['category'].annotation.__name__ == 'Category'
                and signature(other.extract).parameters['document'].annotation.__name__ == 'Document'
                and signature(other.extract).return_annotation.__name__ == 'Document'
                and signature(other.fit)
                and signature(other.check_is_ready)
            )
        except KeyError:
            return False
        except AttributeError:
            return False

    @property
    def temp_pkl_file_path(self) -> str:
        """Generate a path for temporary pickle file."""
        temp_pkl_file_path = os.path.join(
            self.output_dir, f'{get_timestamp()}_{self.category.name.lower()}_{self.name_lower()}_tmp.cloudpickle'
        )
        return temp_pkl_file_path

    @property
    def pkl_file_path(self) -> str:
        """Generate a path for a resulting pickle file."""
        pkl_file_path = os.path.join(
            self.output_dir, f'{get_timestamp()}_{self.category.name.lower()}_' f'{self.name_lower()}_.pkl'
        )
        return pkl_file_path

    @staticmethod
    def load_model(pickle_path: str, max_ram: Union[None, str] = None):
        """
        Load the model and check if it has the interface compatible with the class.

        :param pickle_path: Path to the pickled model.
        :type pickle_path: str
        :raises FileNotFoundError: If the path is invalid.
        :raises OSError: When the data is corrupted or invalid and cannot be loaded.
        :raises TypeError: When the loaded pickle isn't recognized as a Konfuzio AI model.
        :return: Extraction AI model.
        """
        model = super(AbstractExtractionAI, AbstractExtractionAI).load_model(pickle_path, max_ram)
        if not AbstractExtractionAI.has_compatible_interface(model):
            raise TypeError(
                "Loaded model's interface is not compatible with any AIs. Please provide a model that has all the "
                "abstract methods implemented."
            )
        return model


class GroupAnnotationSets:
    """Groups Annotation into Annotation Sets."""

    def __init__(self):
        """Initialize TemplateClf."""
        self.n_nearest_template = 5
        self.max_depth = 100
        self.n_estimators = 100
        self.label_set_clf = None

    def fit_label_set_clf(self) -> Tuple[Optional[object], Optional[List['str']]]:
        """
        Fit classifier to predict start lines of Sections.

        :param documents:
        :return:
        """
        # Only train template clf is there are non default templates
        logger.info('Start training of LabelSet Classifier.')

        LabelSetInfo = collections.namedtuple(
            'LabelSetInfo', ['is_default', 'name', 'has_multiple_annotation_sets', 'target_names']
        )
        self.label_sets_info = [
            LabelSetInfo(
                **dict(
                    is_default=label_set.is_default,
                    name=label_set.name,
                    has_multiple_annotation_sets=label_set.has_multiple_annotation_sets,
                    target_names=label_set.get_target_names(self.use_separate_labels),
                )
            )
            for label_set in self.category.label_sets
        ]

        if not [lset for lset in self.category.label_sets if not lset.is_default]:
            # todo see https://gitlab.com/konfuzio/objectives/-/issues/2247
            # todo check for NO_LABEL_SET if we should keep it
            return
        logger.info('Start training of Multi-class Label Set Classifier.')
        # ignores the section count as it actually worsens results
        # todo check if no category labels should be ignored
        self.template_feature_list = list(self.clf.classes_)  # list of label classifier targets
        # logger.warning("template_feature_list:", self.template_feature_list)
        n_nearest = self.n_nearest_template  # if hasattr(self, 'n_nearest_template') else 0

        # Pretty long feature generation
        df_train_label = self.df_train

        df_train_label_list = [(document_id, df_doc) for document_id, df_doc in df_train_label.groupby('document_id')]

        df_train_template_list = []
        df_train_ground_truth_list = []
        for document_id, df_doc in df_train_label_list:
            document = self.category.project.get_document_by_id(document_id)
            df_train_template_list.append(self.convert_label_features_to_template_features(df_doc, document.text))
            df_train_ground_truth_list.append(self.build_document_template_feature(document))

        df_train_expanded_features_list = [
            self.generate_relative_line_features(n_nearest, pandas.DataFrame(df, columns=self.template_feature_list))
            for df in df_train_template_list
        ]

        df_train_ground_truth = pandas.DataFrame(
            pandas.concat(df_train_ground_truth_list), columns=self.template_feature_list + ['y']
        )

        self.template_expanded_feature_list = list(df_train_expanded_features_list[0].columns)

        df_train_expanded_features = pandas.DataFrame(
            pandas.concat(df_train_expanded_features_list), columns=self.template_expanded_feature_list
        )

        y_train = numpy.array(df_train_ground_truth['y']).astype('str')
        x_train = df_train_expanded_features[self.template_expanded_feature_list]

        # fillna(0) is used here as not every label is found in every document at least once
        x_train.fillna(0, inplace=True)

        # No features available
        if x_train.empty:
            logger.error(
                'No features available to train template classifier, ' 'probably because there are no annotations.'
            )
            return None, None

        label_set_clf = RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=420
        )
        label_set_clf.fit(x_train, y_train)

        self.label_set_clf = label_set_clf
        return self.label_set_clf, self.template_feature_list

    def generate_relative_line_features(self, n_nearest: int, df_features: pandas.DataFrame) -> pandas.DataFrame:
        """Add the features of the n_nearest previous and next lines."""
        if n_nearest == 0:
            return df_features

        min_row = 0
        max_row = len(df_features.index) - 1

        df_features_new_list = []

        for index, row in df_features.iterrows():
            row_dict = row.to_dict()

            # get a relevant lines and add them to the dict_list
            for i in range(n_nearest):
                if index + (i + 1) <= max_row:
                    d_next = df_features.iloc[index + (i + 1)].to_dict()
                else:
                    d_next = row.to_dict()
                    d_next = {k: 0 for k, v in d_next.items()}
                d_next = {f'next_line_{i + 1}_{k}': v for k, v in d_next.items()}

                if index - (i + 1) >= min_row:
                    d_prev = df_features.iloc[index - (i + 1)].to_dict()
                else:
                    d_prev = row.to_dict()
                    d_prev = {k: 0 for k, v in d_prev.items()}
                d_prev = {f'prev_line_{i + 1}_{k}': v for k, v in d_prev.items()}
                # merge the line into the row dict
                row_dict = {**row_dict, **d_next, **d_prev}

            df_features_new_list.append(row_dict)

        return pandas.DataFrame(df_features_new_list)

    def convert_label_features_to_template_features(
        self, feature_df_label: pandas.DataFrame, document_text
    ) -> pandas.DataFrame:
        """
        Convert the feature_df for the label_clf to a feature_df for the label_set_clf.

        The input is the Feature-Dataframe and text for one document.
        """
        # reset indices to avoid bugs with stupid NaN's as label_text
        feature_df_label.reset_index(drop=True, inplace=True)

        # predict and transform the DataFrame to be compatible with the other functions
        results = pandas.DataFrame(
            data=self.clf.predict_proba(X=feature_df_label[self.label_feature_list]), columns=self.clf.classes_
        )

        # Remove no_label predictions
        # if 'NO_LABEL' in results.columns:
        #     results = results.drop(['NO_LABEL'], axis=1)

        # if self.no_label_name in results.columns:
        #     results = results.drop([self.no_label_name], axis=1)

        # Store most likely prediction and its accuracy in separated columns
        feature_df_label['result_name'] = results.idxmax(axis=1)
        feature_df_label['confidence'] = results.max(axis=1)

        # convert the transformed df to the new template features
        feature_df_template = self.build_document_template_feature_X(document_text, feature_df_label).filter(
            self.template_feature_list, axis=1
        )
        feature_df_template = feature_df_template.reindex(columns=self.template_feature_list).fillna(0)

        return feature_df_template

    def build_document_template_feature(self, document) -> pandas.DataFrame():
        """Build document feature for template classifier given ground truth."""
        df = pandas.DataFrame()
        char_count = 0

        document_annotations = [
            annotation
            for annotation_set in document.annotation_sets()
            for annotation in annotation_set.annotations(use_correct=True)
        ]

        # Loop over lines
        for i, line in enumerate(document.text.replace('\f', '\n').split('\n')):
            matched_annotation_set = None
            new_char_count = char_count + len(line)
            assert line == document.text[char_count:new_char_count]
            # TODO: Currently we can't handle
            for annotation_set in document.annotation_sets():
                if annotation_set.start_offset and char_count <= annotation_set.start_offset < new_char_count:
                    matched_annotation_set: AnnotationSet = annotation_set
                    break

            line_annotations = [
                x for x in document_annotations if char_count <= x.spans[0].start_offset < new_char_count
            ]
            annotations_dict = dict((x.label.name, True) for x in line_annotations)
            counter_dict = dict(
                collections.Counter(annotation.annotation_set.label_set.name for annotation in line_annotations)
            )
            y = matched_annotation_set.label_set.name if matched_annotation_set else 'No'
            tmp_df = pandas.DataFrame(
                [{'line': i, 'y': y, 'document': document.id_, **annotations_dict, **counter_dict}]
            )
            df = pandas.concat([df, tmp_df], ignore_index=True)
            char_count = new_char_count + 1
        df['text'] = document.text.replace('\f', '\n').split('\n')
        return df.fillna(0)

    def build_document_template_feature_X(self, text, df) -> pandas.DataFrame():
        """
        Calculate features for a document given the extraction results.

        :param text:
        :param df:
        :return:
        """
        if self.category.name == 'NO_CATEGORY':
            raise AttributeError(f'{self} does not provide a Category.')

        global_df = pandas.DataFrame()
        char_count = 0
        # Using OptimalThreshold is a bad idea as it might defer between training (actual treshold from the label)
        # and runtime (default treshold.

        # df = df[df['confidence'] >= 0.1]  # df['OptimalThreshold']]
        lines = text.replace('\f', '\n').split('\n')
        for i, line in enumerate(lines):
            new_char_count = char_count + len(line)
            assert line == text[char_count:new_char_count]
            line_df = df[(char_count <= df['start_offset']) & (df['end_offset'] <= new_char_count)]
            spans = [row for index, row in line_df.iterrows()]
            spans_dict = dict((x['result_name'], True) for x in spans)
            # counter_dict = {}  # why?
            # annotations_accuracy_dict = defaultdict(lambda: 0)
            # for annotation in annotations:
            # annotations_accuracy_dict[f'{annotation["label"]}_accuracy'] += annotation['confidence']
            # try:

            #     label = next(x for x in self.category.project.labels if x.name == annotation['result_name'])
            # except StopIteration:
            #     continue
            # for label_set in self.label_sets:
            #     if label in label_set.labels:
            #         if label_set.name in counter_dict.keys():
            #             counter_dict[label_set.name] += 1
            #         else:
            #             counter_dict[label_set.name] = 1
            tmp_df = pandas.DataFrame([spans_dict])  # ([{**spans_dict, **counter_dict}])
            global_df = pandas.concat([global_df, tmp_df], ignore_index=True)
            char_count = new_char_count + 1
        global_df['text'] = lines
        return global_df.fillna(0)

    @classmethod
    def dict_to_dataframe(cls, res_dict):
        """Convert a Dict to Dataframe add label as column."""
        df = pandas.DataFrame()
        for name in res_dict.keys():
            label_df = res_dict[name]
            label_df['result_name'] = name
            df = df.append(label_df, sort=True)
        return df

    def extract_template_with_clf(self, text, res_dict):
        """Run LabelSet classifier to find AnnotationSets."""
        logger.info('Extract AnnotationSets.')
        if not res_dict:
            logger.warning('res_dict is empty')
            return res_dict
        n_nearest = self.n_nearest_template if hasattr(self, 'n_nearest_template') else 0
        feature_df = self.build_document_template_feature_X(text, self.dict_to_dataframe(res_dict)).filter(
            self.template_feature_list, axis=1
        )
        feature_df = feature_df.reindex(columns=self.template_feature_list).fillna(0)
        feature_df = self.generate_relative_line_features(n_nearest, feature_df)

        res_series = self.label_set_clf.predict(feature_df)
        res_templates = pandas.DataFrame(res_series)
        # res_templates['text'] = text.replace('\f', '\n').split('\n')  # Debug code.

        # TODO improve ordering. What happens if Annotations are not matched?
        logger.info('Building new res dict')
        new_res_dict = {}
        text_replaced = text.replace('\f', '\n')

        # Add extractions from non-default sections.
        for label_set in [x for x in self.label_sets_info if not x.is_default]:
            # Add Extraction from SectionLabels with multiple sections (as list).
            if label_set.has_multiple_annotation_sets:
                new_res_dict[label_set.name] = []
                detected_sections = res_templates[res_templates[0] == label_set.name]
                # List of tuples, e.g. [(1, DefaultSectionName), (14, DetailedSectionName), ...]
                # line_list = [(index, row[0]) for index, row in detected_sections.iterrows()]
                if not detected_sections.empty:
                    i = 0
                    # for each line of a certain section label
                    for line_number, section_name in detected_sections.iterrows():
                        section_dict = {}
                        # we try to find the labels that match that section
                        for target_label_name in label_set.target_names:
                            if target_label_name in res_dict.keys():

                                label_df = res_dict[target_label_name]
                                if label_df.empty:
                                    continue
                                # todo: the next line is memory heavy
                                #  https://gitlab.com/konfuzio/objectives/-/issues/9342
                                label_df['line'] = (
                                    label_df['start_offset'].apply(lambda x: text_replaced[: int(x)]).str.count('\n')
                                )
                                try:
                                    next_section_start: int = detected_sections.index[i + 1]  # line_list[i + 1][0]
                                except IndexError:  # ?
                                    next_section_start: int = text_replaced.count('\n') + 1
                                except Exception:
                                    raise

                                # we get the label df that is contained within the section
                                label_df = label_df[
                                    (line_number <= label_df['line']) & (label_df['line'] < next_section_start)
                                ]
                                if label_df.empty:
                                    continue
                                section_dict[target_label_name] = label_df  # Add to new result dict
                                # Remove from input dict
                                res_dict[target_label_name] = res_dict[target_label_name].drop(label_df.index)
                        i += 1
                        new_res_dict[label_set.name].append(section_dict)
            # Add Extraction from SectionLabels with single section (as dict).
            else:
                _dict = {}
                for target_label_name in label_set.target_names:
                    if target_label_name in res_dict.keys():
                        _dict[target_label_name] = res_dict[target_label_name]
                        del res_dict[target_label_name]  # ?
                if _dict:
                    new_res_dict[label_set.name] = _dict
                continue

        # Finally add remaining extractions to default section (if they are allowed to be there).
        for label_set in [x for x in self.label_sets_info if x.is_default]:
            for target_label_name in label_set.target_names:
                if target_label_name in res_dict.keys():
                    new_res_dict[target_label_name] = res_dict[target_label_name]
                    del res_dict[target_label_name]  # ?
            continue

        return new_res_dict


class RFExtractionAI(AbstractExtractionAI, GroupAnnotationSets):
    """Encode visual and textual features to extract text regions.

    Fit an extraction pipeline to extract linked Annotations.

    Both Label and Label Set classifiers are using a RandomForestClassifier from scikit-learn to run in a low memory and
    single CPU environment. A random forest classifier is a group of decision trees classifiers, see:
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

    The parameters of this class allow to select the Tokenizer, to configure the Label and Label Set classifiers and to
    select the type of features used by the Label and Label Set classifiers.

    They are divided in:
    - tokenizer selection
    - parametrization of the Label classifier
    - parametrization of the Label Set classifier
    - features for the Label classifier
    - features for the Label Set classifier

    By default, the text of the Documents is split into smaller chunks of text based on whitespaces
    ('WhitespaceTokenizer'). That means that all words present in the text will be shown to the AI. It is possible to
    define if the splitting of the text into smaller chunks should be done based on regexes learned from the
    Spans of the Annotations of the Category ('tokenizer_regex') or if to use a model from Spacy library for German
    language ('tokenizer_spacy'). Another option is to use a pre-defined list of tokenizers based on regexes
    ('tokenizer_regex_list') and, on top of the pre-defined list, to create tokenizers that match what is missed
    by those ('tokenizer_regex_combination').

    Some parameters of the scikit-learn RandomForestClassifier used for the Label and/or Label Set classifier
    can be set directly in Konfuzio Server ('label_n_estimators', 'label_max_depth', 'label_class_weight',
    'label_random_state', 'label_set_n_estimators', 'label_set_max_depth').

    Features are measurable pieces of data of the Annotation. By default, a combination of features is used that
    includes features built from the text of the Annotation ('string_features'), features built from the position of
    the Annotation in the Document ('spatial_features') and features from the Spans created by a WhitespaceTokenizer on
    the left or on the right of the Annotation ('n_nearest_left', 'n_nearest_right', 'n_nearest_across_lines).
    It is possible to exclude any of them ('spatial_features', 'string_features', 'n_nearest_left', 'n_nearest_right')
    or to specify the number of Spans created by a WhitespaceTokenizer to consider
    ('n_nearest_left', 'n_nearest_right').

    While extracting, the Label Set classifier takes the predictions from the Label classifier as input.
    The Label Set classifier groups them into Annotation sets.
    """

    def __init__(
        self,
        n_nearest: int = 2,
        first_word: bool = True,
        n_estimators: int = 100,
        max_depth: int = 100,
        no_label_limit: Union[int, float, None] = None,
        n_nearest_across_lines: bool = False,
        use_separate_labels: bool = True,
        category: Category = None,
        tokenizer=None,
        *args,
        **kwargs,
    ):
        """RFExtractionAI."""
        logger.info("Initializing RFExtractionAI.")
        super().__init__(category, *args, **kwargs)
        GroupAnnotationSets.__init__(self)

        self.label_feature_list = None

        logger.info("RFExtractionAI settings:")
        logger.info(f"{use_separate_labels=}")
        logger.info(f"{category=}")
        logger.info(f"{n_nearest=}")
        logger.info(f"{first_word=}")
        logger.info(f"{max_depth=}")
        logger.info(f"{n_estimators=}")
        logger.info(f"{no_label_limit=}")
        logger.info(f"{n_nearest_across_lines=}")

        self.use_separate_labels = use_separate_labels
        self.n_nearest = n_nearest
        self.first_word = first_word
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.no_label_limit = no_label_limit
        self.n_nearest_across_lines = n_nearest_across_lines

        self.tokenizer = tokenizer
        logger.info(f"{tokenizer=}")

        self.clf = None

        self.no_label_set_name = None
        self.no_label_name = None

        self.output_dir = None

    @property
    def requires_segmentation(self) -> bool:
        """Return True if the Extraction AI requires detectron segmentation results to process Documents."""
        if (
            sdk_isinstance(self.tokenizer, ParagraphTokenizer) or sdk_isinstance(self.tokenizer, SentenceTokenizer)
        ) and self.tokenizer.mode == 'detectron':
            return True
        elif self.tokenizer is None:
            logger.warning('Tokenizer is not set. Assuming no segmentation results is required.')
        return False

    def features(self, document: Document):
        """Calculate features using the best working default values that can be overwritten with self values."""
        logger.info(f"Starting {document} feature calculation.")
        if self.no_label_name is None or self.no_label_set_name is None:
            self.no_label_name = document.project.no_label.name_clean
            self.no_label_set_name = document.project.no_label_set.name_clean
        df, _feature_list, _temp_df_raw_errors = process_document_data(
            document=document,
            spans=document.spans(use_correct=False),
            n_nearest=self.n_nearest,
            first_word=self.first_word,
            # tokenize_fn=self.tokenizer.tokenize,  # todo: we are tokenizing the document multiple times
            # catchphrase_list=self.catchphrase_features,
            # substring_features=self.substring_features,
            n_nearest_across_lines=self.n_nearest_across_lines,
        )
        if self.use_separate_labels:
            df['target'] = df['label_set_name'] + '__' + df['label_name']
        else:
            df['target'] = df['label_name']
        return df, _feature_list, _temp_df_raw_errors

    def check_is_ready(self):
        """
        Check if the ExtractionAI is ready for the inference.

        It is assumed that the model is ready if a Tokenizer and a Category were set, Classifiers were set and trained.

        :raises AttributeError: When no Tokenizer is specified.
        :raises AttributeError: When no Category is specified.
        :raises AttributeError: When no Label Classifier has been provided.
        """
        super().check_is_ready()
        if self.tokenizer is None:
            raise AttributeError(f'{self} missing Tokenizer.')

        if self.clf is None:
            raise AttributeError(f'{self} does not provide a Label Classifier. Please add it.')
        else:
            check_is_fitted(self.clf)

        if self.label_set_clf is None:
            logger.warning(f'{self} does not provide a LabelSet Classfier.')

    def extract(self, document: Document) -> Document:
        """
        Infer information from a given Document.

        :param document: Document object
        :return: Document with predicted labels

        :raises:
         AttributeError: When missing a Tokenizer
         NotFittedError: When CLF is not fitted

        """
        logger.info(f"Starting extraction of {document}.")

        self.check_is_ready()

        # Main Logic -------------------------
        # 1. start inference with new document
        inference_document = deepcopy(document)

        # In case document category was changed after RFExtractionAI training
        inference_document._category = self.project.no_category
        inference_document.set_category(self.category)

        # 2. tokenize
        self.tokenizer.tokenize(inference_document)
        if not inference_document.spans():
            logger.error(f'{self.tokenizer} does not provide Spans for {document}')
            return inference_document

        # 3. preprocessing
        df, _feature_names, _raw_errors = self.features(inference_document)

        return self.extract_from_df(df, inference_document)

    def extract_from_df(self, df: pandas.DataFrame, inference_document: Document) -> Document:
        """Predict Labels from features."""
        try:
            independent_variables = df[self.label_feature_list]
        except KeyError:
            raise KeyError(f'Features of {inference_document} do not match the features of the pipeline.')
            # todo calculate features of Document as defined in pipeline and do not check afterwards
        # 4. prediction and store most likely prediction and its accuracy in separated columns
        results = pandas.DataFrame(data=self.clf.predict_proba(X=independent_variables), columns=self.clf.classes_)

        # Remove no_label predictions
        if self.no_label_name in results.columns:
            results = results.drop([self.no_label_name], axis=1)

        if self.no_label_set_name in results.columns:
            results = results.drop([self.no_label_set_name], axis=1)

        separate_no_label_target = self.no_label_set_name + '__' + self.no_label_name
        if separate_no_label_target in results.columns:
            results = results.drop([separate_no_label_target], axis=1)

        df['result_name'] = results.idxmax(axis=1)
        df['confidence'] = results.max(axis=1)

        # Main Logic -------------------------

        # Convert DataFrame to Dict with labels as keys and label dataframes as value.
        res_dict = {}
        for result_name in set(df['result_name']):
            result_df = df[(df['result_name'] == result_name) & (df['confidence'] >= df['label_threshold'])].copy()

            if not result_df.empty:
                res_dict[result_name] = result_df

        no_label_res_dict = {}
        for result_name in set(df['result_name']):
            result_df = df[(df['result_name'] == result_name) & (df['confidence'] < df['label_threshold'])].copy()

            if not result_df.empty:
                no_label_res_dict[result_name] = result_df

        # Filter results that are bellow the extract threshold
        # (helpful to reduce the size in case of many predictions/ big documents)

        # if hasattr(self, 'extract_threshold') and self.extract_threshold is not None:
        #     logger.info('Filtering res_dict')
        #     for result_name, value in res_dict.items():
        #         if isinstance(value, pandas.DataFrame):
        #             res_dict[result_name] = value[value['confidence'] > self.extract_threshold]

        res_dict = self.remove_empty_dataframes_from_extraction(res_dict)
        no_label_res_dict = self.remove_empty_dataframes_from_extraction(no_label_res_dict)

        # res_dict = self.filter_low_confidence_extractions(res_dict)
        if not sdk_isinstance(self.tokenizer, ParagraphTokenizer) and not sdk_isinstance(
            self.tokenizer, SentenceTokenizer
        ):
            # We assume that Paragraph or Sentence tokenizers have correctly tokenized the Document
            res_dict = self.merge_horizontal(res_dict, inference_document.text)

        # Try to calculate sections based on template classifier.
        if self.label_set_clf is not None and res_dict:  # todo smarter handling of multiple clf
            res_dict = self.extract_template_with_clf(inference_document.text, res_dict)
        res_dict[self.no_label_set_name] = no_label_res_dict

        if self.use_separate_labels:
            res_dict = self.separate_labels(res_dict)

        virtual_doc = self.extraction_result_to_document(inference_document, res_dict)

        self.tokenizer.found_spans(virtual_doc)

        if sdk_isinstance(self.tokenizer, ParagraphTokenizer) or sdk_isinstance(self.tokenizer, SentenceTokenizer):
            # When using the Paragraph or Sentence tokenizer, we restore the multi-line Annotations they created.
            virtual_doc = self.merge_vertical_like(virtual_doc, inference_document)
        else:
            # join document Spans into multi-line Annotation
            virtual_doc = self.merge_vertical(virtual_doc)

        return virtual_doc

    def merge_vertical(self, document: Document, only_multiline_labels=True):
        """
        Merge Annotations with the same Label.

        See more details at https://dev.konfuzio.com/sdk/explanations.html#vertical-merge

        :param document: Document whose Annotations should be merged vertically
        :param only_multiline_labels: Only merge if a multiline Label Annotation is in the Category Training set
        """
        logger.info("Vertical merging Annotations.")
        if not self.category:
            raise AttributeError(f'{self} merge_vertical requires a Category.')
        labels_dict = {}
        for label in self.category.labels:
            if not only_multiline_labels or label.has_multiline_annotations():
                labels_dict[label.name] = []

        for annotation in document.annotations(use_correct=False, ignore_below_threshold=True):
            if annotation.label.name in labels_dict:
                labels_dict[annotation.label.name].append(annotation)

        for label_id in labels_dict:
            buffer = []
            for annotation in labels_dict[label_id]:
                for span in annotation.spans:
                    # remove all spans in buffer more than 1 line apart
                    while buffer and span.line_index > buffer[0].line_index + 1:
                        buffer.pop(0)

                    if buffer and buffer[-1].page != span.page:
                        buffer = [span]
                        continue

                    if len(annotation.spans) > 1:
                        buffer.append(span)
                        continue

                    for candidate in buffer:
                        # only looking for elements in line above
                        if candidate.line_index == span.line_index:
                            break

                        # Merge if there is overlap in the horizontal direction or if only separated by a line break
                        # AND if the AnnotationSets are the same or if the Annotation is alone in its AnnotationSet
                        if (
                            (not (span.bbox().x0 > candidate.bbox().x1 or span.bbox().x1 < candidate.bbox().x0))
                            or document.text[candidate.end_offset : span.start_offset]
                            .replace(' ', '')
                            .replace('\n', '')
                            == ''
                        ) and (
                            span.annotation.annotation_set is candidate.annotation.annotation_set
                            or len(
                                span.annotation.annotation_set.annotations(
                                    use_correct=False, ignore_below_threshold=True
                                )
                            )
                            == 1
                        ):
                            span.annotation.delete(delete_online=False)
                            span.annotation = None
                            candidate.annotation.add_span(span)
                            buffer.remove(candidate)
                    buffer.append(span)
        return document

    def merge_vertical_like(self, document: Document, template_document: Document):
        """
        Merge Annotations the same way as in another copy of the same Document.

        All single-Span Annotations in the current Document (self) are matched with corresponding multi-line
        Spans in the given Document and are merged in the same way.
        The Label of the new multi-line Annotations is taken to be the most common Label among the original
        single-line Annotations that are being merged.

        :param document: Document with multi-line Annotations
        """
        logger.info(f"Vertical merging Annotations like {template_document}.")
        assert (
            document.text == template_document.text
        ), f"{self} and {template_document} need to have the same ocr text."
        span_to_annotation = {
            (span.start_offset, span.end_offset): hash(span.annotation)
            for span in template_document.spans(use_correct=False)
        }
        ann_to_anns = collections.defaultdict(list)
        for annotation in document.annotations(use_correct=False):
            assert (
                len(annotation.spans) == 1
            ), f"Cannot use merge_verical_like in {document} with multi-span {annotation}."
            span_offset_key = (annotation.spans[0].start_offset, annotation.spans[0].end_offset)
            if span_offset_key in span_to_annotation:
                ann_to_anns[span_to_annotation[span_offset_key]].append(annotation)
        for _, self_annotations in ann_to_anns.items():
            if len(self_annotations) == 1:
                continue
            else:
                self_annotations = sorted(self_annotations)
                keep_annotation = self_annotations[0]
                annotation_labels = [keep_annotation.label]
                for to_merge_annotation in self_annotations[1:]:
                    annotation_labels.append(to_merge_annotation.label)
                    span = to_merge_annotation.spans[0]
                    to_merge_annotation.delete(delete_online=False)
                    span.annotation = None
                    keep_annotation.add_span(span)
                most_common_label = collections.Counter(annotation_labels).most_common(1)[0][0]
                keep_annotation.label = most_common_label

        return document

    def separate_labels(self, res_dict: 'Dict') -> 'Dict':
        """
        Undo the renaming of the labels.

        In this way we have the output of the extraction in the correct format.
        """
        new_res = {}
        for key, value in res_dict.items():
            # if the value is a list, is because the key corresponds to a section label with multiple sections
            # the key has already the name of the section label
            # we need to go to each element of the list, which is a dictionary, and
            # rewrite the label name (remove the section label name) in the keys
            if isinstance(value, list):
                label_set = key
                if label_set not in new_res.keys():
                    new_res[label_set] = []

                for found_section in value:
                    new_found_section = {}
                    for label, df in found_section.items():
                        if '__' in label:
                            label = label.split('__')[1]
                            df.label_name = label
                            df.label = label
                        new_found_section[label] = df

                    new_res[label_set].append(new_found_section)

            # if the value is a dictionary, is because the key corresponds to a section label without multiple sections
            # we need to rewrite the label name (remove the section label name) in the keys
            elif isinstance(value, dict):
                label_set = key
                if label_set not in new_res.keys():
                    new_res[label_set] = {}

                for label, df in value.items():
                    if '__' in label:
                        label = label.split('__')[1]
                        df.label_name = label
                        df.label = label
                    new_res[label_set][label] = df

            # otherwise the value must be directly a dataframe and it will correspond to the default section
            # can also correspond to labels which the template clf couldn't attribute to any template.
            # so we still check if we have the changed label name
            elif '__' in key:
                label_set = key.split('__')[0]
                if label_set not in new_res.keys():
                    new_res[label_set] = {}
                key = key.split('__')[1]
                value.label_name = key
                value.label = key
                # if the section label already exists and allows multi sections
                if isinstance(new_res[label_set], list):
                    new_res[label_set].append({key: value})
                else:
                    new_res[label_set][key] = value
            else:
                new_res[key] = value

        return new_res

    def remove_empty_dataframes_from_extraction(self, result: Dict) -> Dict:
        """Remove empty dataframes from the result of an Extraction AI.

        The input is a dictionary where the values can be:
        - dataframe
        - dictionary where the values are dataframes
        - list of dictionaries  where the values are dataframes
        """
        for k in list(result.keys()):
            if isinstance(result[k], pandas.DataFrame) and result[k].empty:
                del result[k]
            elif isinstance(result[k], list):
                for e, element in enumerate(result[k]):
                    for sk in list(element.keys()):
                        if isinstance(element[sk], pandas.DataFrame) and element[sk].empty:
                            del result[k][e][sk]
            elif isinstance(result[k], dict):
                for ssk in list(result[k].keys()):
                    if isinstance(result[k][ssk], pandas.DataFrame) and result[k][ssk].empty:
                        del result[k][ssk]

        return result

    def filter_low_confidence_extractions(self, result: Dict) -> Dict:
        """Remove extractions with confidence below the threshold defined for the respective label.

        The input is a dictionary where the values can be:
        - dataframe
        - dictionary where the values are dataframes
        - list of dictionaries  where the values are dataframes

        :param result: Extraction results
        :returns: Filtered dictionary.
        """
        for k in list(result.keys()):
            if isinstance(result[k], pandas.DataFrame):
                filtered = self.filter_dataframe(result[k])
                if filtered.empty:
                    del result[k]
                else:
                    result[k] = filtered

            elif isinstance(result[k], list):
                for e, element in enumerate(result[k]):
                    for sk in list(element.keys()):
                        if isinstance(element[sk], pandas.DataFrame):
                            filtered = self.filter_dataframe(result[k][e][sk])
                            if filtered.empty:
                                del result[k][e][sk]
                            else:
                                result[k][e][sk] = filtered

            elif isinstance(result[k], dict):
                for ssk in list(result[k].keys()):
                    if isinstance(result[k][ssk], pandas.DataFrame):
                        filtered = self.filter_dataframe(result[k][ssk])
                        if filtered.empty:
                            del result[k][ssk]
                        else:
                            result[k][ssk] = filtered

        return result

    def filter_dataframe(self, df: pandas.DataFrame) -> pandas.DataFrame:
        """Filter dataframe rows accordingly with the confidence value.

        Rows (extractions) where the accuracy value is below the threshold defined for the label are removed.

        :param df: Dataframe with extraction results
        :returns: Filtered dataframe
        """
        filtered = df[df['confidence'] >= df['label_threshold']]
        return filtered

    def label_train_document(self, virtual_document: Document, original_document: Document):
        """Assign labels to Annotations in newly tokenized virtual training document."""
        doc_spans = original_document.spans(use_correct=True)
        s_i = 0
        for span in virtual_document.spans():
            while s_i < len(doc_spans) and span.start_offset > doc_spans[s_i].end_offset:
                s_i += 1
            if s_i >= len(doc_spans):
                break
            if span.end_offset < doc_spans[s_i].start_offset:
                continue

            r = range(doc_spans[s_i].start_offset, doc_spans[s_i].end_offset + 1)
            if span.start_offset in r and span.end_offset in r:
                span.annotation.label = doc_spans[s_i].annotation.label
                span.annotation.label_set = doc_spans[s_i].annotation.label_set
                span.annotation.annotation_set = doc_spans[s_i].annotation.annotation_set

    def feature_function(
        self,
        documents: List[Document],
        no_label_limit: Union[None, int, float] = None,
        retokenize: Optional[bool] = None,
        require_revised_annotations: bool = False,
    ) -> Tuple[List[pandas.DataFrame], list]:
        """Calculate features per Span of Annotations.

        :param documents: List of documents to extract features from.
        :param no_label_limit: Int or Float to limit number of new annotations to create during tokenization.
        :param retokenize: Bool for whether to recreate annotations from scratch or use already existing annotations.
        :param require_revised_annotations: Only allow calculation of features if no unrevised Annotation present.
        :return: Dataframe of features and list of feature names.
        """
        logger.info(f'Start generating features for {len(documents)} documents.')
        logger.info(f'{no_label_limit=}')
        logger.info(f'{retokenize=}')
        logger.info(f'{require_revised_annotations=}')

        if retokenize is None:
            if sdk_isinstance(self.tokenizer, ListTokenizer):
                retokenize = False
            else:
                retokenize = True
            logger.info(f'retokenize option set to {retokenize} with tokenizer {self.tokenizer}')

        df_real_list = []
        df_raw_errors_list = []
        feature_list = []

        # todo make regex Tokenizer optional as those will be saved by the Server
        # if not hasattr(self, 'regexes'):  # Can be removed for models after 09.10.2020
        #    self.regexes = [regex for label_model in self.labels for regex in label_model.label.regex()]

        for label in self.category.labels:
            label.has_multiline_annotations(categories=[self.category])

        for document in documents:
            # todo check for tokenizer: self.tokenizer.tokenize(document)  # todo: do we need it?
            # todo check removed  if x.x0 and x.y0
            # todo: use NO_LABEL for any Annotation that has no Label, instead of keeping Label = None
            for span in document.spans(use_correct=False):
                if span.annotation.id_:
                    # Annotation
                    # we use "<" below because we don't want to have unconfirmed annotations in the training set,
                    # and the ones below threshold wouldn't be considered anyway
                    if (
                        span.annotation.is_correct
                        or (not span.annotation.is_correct and span.annotation.revised)
                        or (
                            span.annotation.confidence
                            and hasattr(span.annotation.label, 'threshold')
                            and span.annotation.confidence < span.annotation.label.threshold
                        )
                    ):
                        pass
                    else:
                        if require_revised_annotations:
                            raise ValueError(
                                f"{span.annotation} is unrevised in this dataset and can't be used for training!"
                                f"Please revise it manually by either confirming it, rejecting it, or modifying it."
                            )
                        else:
                            logger.error(
                                f"{span.annotation} is unrevised in this dataset and may impact model "
                                f"performance! Please revise it manually by either confirming it, rejecting "
                                f"it, or modifying it."
                            )

            virtual_document = deepcopy(document)
            if retokenize:
                self.tokenizer.tokenize(virtual_document)
                self.label_train_document(virtual_document, document)
            else:
                for ann in document.annotations():
                    new_spans = []
                    for span in ann.spans:
                        new_span = Span(start_offset=span.start_offset, end_offset=span.end_offset)
                        new_spans.append(new_span)

                    new_ann = Annotation(
                        document=virtual_document,
                        annotation_set=virtual_document.no_label_annotation_set,
                        label=ann.label,
                        label_set=virtual_document.project.no_label_set,
                        category=self.category,
                        spans=new_spans,
                    )
                    new_ann.label_set = ann.label_set
                    new_ann.annotation_set = ann.annotation_set

                self.tokenizer.tokenize(virtual_document)

            no_label_annotations = virtual_document.annotations(
                use_correct=False, label=virtual_document.project.no_label
            )
            label_annotations = [x for x in virtual_document.annotations(use_correct=False) if x.label.id_ is not None]

            # We calculate features of documents as long as they have IDs, even if they are offline.
            # The assumption is that if they have an ID, then the data came either from the API or from the DB.
            if virtual_document.id_ is None and virtual_document.copy_of_id is None:
                # inference time todo reduce shuffled complexity
                assert (
                    not label_annotations
                ), "Documents that don't come from the server have no human revised Annotations."
                raise NotImplementedError(
                    f'{virtual_document} does not come from the server, please use process_document_data function.'
                )
            else:
                # training time: todo reduce shuffled complexity
                if isinstance(no_label_limit, int):
                    n_no_labels = no_label_limit
                elif isinstance(no_label_limit, float):
                    n_no_labels = int(len(label_annotations) * no_label_limit)
                else:
                    assert no_label_limit is None

                if no_label_limit is not None:
                    no_label_annotations = self.get_best_no_label_annotations(
                        n_no_labels, label_annotations, no_label_annotations
                    )
                    logger.info(
                        f'Document {virtual_document} NO_LABEL annotations reduced to {len(no_label_annotations)}'
                    )

            logger.info(f'Document {virtual_document} has {len(label_annotations)} labeled annotations')
            logger.info(f'Document {virtual_document} has {len(no_label_annotations)} NO_LABEL annotations')

            # todo: check if eq method of Annotation prevents duplicates
            # annotations = self._filter_annotations_for_duplicates(label_annotations + no_label_annotations)

            t0 = time.monotonic()

            temp_df_real, _feature_list, temp_df_raw_errors = self.features(virtual_document)

            logger.info(f'Document {virtual_document} processed in {time.monotonic() - t0:.1f} seconds.')

            virtual_document.delete(delete_online=False)  # reduce memory from virtual doc

            feature_list += _feature_list
            df_real_list.append(temp_df_real)
            df_raw_errors_list.append(temp_df_raw_errors)

        feature_list = list(dict.fromkeys(feature_list))  # remove duplicates while maintaining order

        if df_real_list:
            df_real_list = pandas.concat(df_real_list).reset_index(drop=True)
        else:
            raise NotImplementedError

        logger.info(f"Size of feature dict {memory_size_of(df_real_list)/1000} KB.")

        return df_real_list, feature_list

    def fit(self) -> RandomForestClassifier:
        """Given training data and the feature list this function returns the trained regression model."""
        logger.info('Start training of Multi-class Label Classifier.')

        # balanced gives every label the same weight so that the sample_number doesn't effect the results
        self.clf = RandomForestClassifier(
            class_weight="balanced", n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=420
        )

        self.clf.fit(self.df_train[self.label_feature_list], self.df_train['target'])

        logger.info(f"Size of Label classifier: {memory_size_of(self.clf)/1000} KB.")

        self.fit_label_set_clf()

        logger.info(f"Size of LabelSet classifier: {memory_size_of(self.label_set_clf)/1000} KB.")

        return self.clf

    def evaluate_full(
        self, strict: bool = True, use_training_docs: bool = False, use_view_annotations: bool = True
    ) -> ExtractionEvaluation:
        """
        Evaluate the full pipeline on the pipeline's Test Documents.

        :param strict: Evaluate on a Character exact level without any postprocessing.
        :param use_training_docs: Bool for whether to evaluate on the training documents instead of testing documents.
        :return: Evaluation object.
        """
        eval_list = []
        if not use_training_docs:
            eval_docs = self.test_documents
        else:
            eval_docs = self.documents

        for document in eval_docs:
            predicted_doc = self.extract(document=document)
            eval_list.append((document, predicted_doc))

        full_evaluation = ExtractionEvaluation(eval_list, strict=strict, use_view_annotations=use_view_annotations)

        return full_evaluation

    def evaluate_tokenizer(self, use_training_docs: bool = False) -> ExtractionEvaluation:
        """Evaluate the tokenizer."""
        if not use_training_docs:
            eval_docs = self.test_documents
        else:
            eval_docs = self.documents

        evaluation = self.tokenizer.evaluate_dataset(eval_docs)

        return evaluation

    def evaluate_clf(self, use_training_docs: bool = False) -> ExtractionEvaluation:
        """Evaluate the Label classifier."""
        eval_list = []
        if not use_training_docs:
            eval_docs = self.test_documents
        else:
            eval_docs = self.documents

        for document in eval_docs:
            virtual_doc = deepcopy(document)

            for ann in document.annotations():
                new_spans = []
                for span in ann.spans:
                    new_span = Span(start_offset=span.start_offset, end_offset=span.end_offset)
                    new_spans.append(new_span)

                _ = Annotation(
                    document=virtual_doc,
                    annotation_set=virtual_doc.no_label_annotation_set,
                    label=virtual_doc.project.no_label,
                    label_set=virtual_doc.project.no_label_set,
                    category=virtual_doc.category,
                    spans=new_spans,
                )

            feats_df, _, _ = self.features(virtual_doc)
            predicted_doc = self.extract_from_df(feats_df, virtual_doc)
            eval_list.append((document, predicted_doc))

        clf_evaluation = ExtractionEvaluation(eval_list, use_view_annotations=False)

        return clf_evaluation

    def evaluate_label_set_clf(self, use_training_docs: bool = False) -> ExtractionEvaluation:
        """Evaluate the LabelSet classifier."""
        if self.label_set_clf is None:
            raise AttributeError(f'{self} does not provide a LabelSet Classifier.')
        else:
            check_is_fitted(self.label_set_clf)

        eval_list = []
        if not use_training_docs:
            eval_docs = self.test_documents
        else:
            eval_docs = self.documents

        for document in eval_docs:
            df, _feature_names, _raw_errors = self.features(document)

            df['result_name'] = df['target']

            # Convert DataFrame to Dict with labels as keys and label dataframes as value.
            res_dict = {}
            for result_name in set(df['result_name']):
                result_df = df[(df['result_name'] == result_name)].copy()

                if not result_df.empty:
                    res_dict[result_name] = result_df

            res_dict = self.extract_template_with_clf(document.text, res_dict)

            if self.use_separate_labels:
                res_dict = self.separate_labels(res_dict)

            predicted_doc = self.extraction_result_to_document(document, res_dict)

            eval_list.append((document, predicted_doc))

        label_set_clf_evaluation = ExtractionEvaluation(eval_list, use_view_annotations=False)

        return label_set_clf_evaluation

    def reduce_model_weight(self):
        """Remove all non-strictly necessary parameters before saving."""
        super().reduce_model_weight()
        self.df_train = None
