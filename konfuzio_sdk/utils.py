"""Utils for the konfuzio sdk package."""
import copy
import datetime
import hashlib
import itertools
import logging
import os
import re
import unicodedata
import zipfile
from collections import defaultdict
from contextlib import contextmanager
from io import BytesIO
from random import randrange
from typing import Union, List, Tuple, Dict

import filetype
from PIL import Image

from konfuzio_sdk import IMAGE_FILE, PDF_FILE, OFFICE_FILE, SUPPORTED_FILE_TYPES

logger = logging.getLogger(__name__)


def get_id(a_string, include_time: bool = False) -> int:
    """
    Generate a unique ID.

    :param a_string: String used to generating the unique ID
    :param include_time: Bool to include the time in the unique ID
    :return: Unique ID
    """
    if include_time:
        unique_string = a_string + get_timestamp(konfuzio_format='%Y-%m-%d-%H-%M-%S.%f')
    else:
        unique_string = a_string
    try:
        return int(hashlib.md5(unique_string.encode()).hexdigest(), 16)
    except (UnicodeDecodeError, AttributeError):  # duck typing for bytes like objects
        return int(hashlib.md5(unique_string).hexdigest(), 16)


def is_file(file_path, raise_exception=True, maximum_size=100000000, allow_empty=False) -> bool:
    """
    Check if file is available or raise error if it does not exist.

    :param file_path: Path to the file to be checked
    :param raise_exception: Will raise an exception if file is not available
    :param maximum_size: Maximum size of the expected file, default < 100 mb
    :param allow_empty: Bool to allow empty files
    :return: True or false depending on the existence of the file
    """
    if os.path.isfile(file_path):
        file_size = os.path.getsize(file_path)
        if file_size > 0 or allow_empty:
            if file_size > maximum_size:
                logger.warning(f'Please check your BIG file with size {file_size / 1000000:.2f} MB at {file_path}.')
            with open(file_path, 'rb') as f:
                logger.debug(f"File expected and found at {file_path} with ID {get_id(f.read())}")
            return True
        else:
            if raise_exception:
                raise FileExistsError(f'Please check your file {file_path} with size {file_size} at {file_path}.')
            else:
                return False
    else:
        if raise_exception:
            raise FileNotFoundError(f'File expected but not found at: {file_path}')
        else:
            return False


def get_timestamp(konfuzio_format='%Y-%m-%d-%H-%M-%S') -> str:
    """
    Return formatted timestamp.

    :param konfuzio_format: Format of the timestamp (e.g. year-month-day-hour-min-sec)
    :return: Timestamp
    """
    now = datetime.datetime.now()
    timestamp = now.strftime(konfuzio_format)
    return timestamp


def load_image(input_file: Union[str, BytesIO]):
    """
    Load an image by path or via io.Bytes, e.g. via download by URL.

    :param input_file: Path to image or image in bytes format
    :return: Loaded image
    """
    if isinstance(input_file, str):
        assert (
            get_file_type(input_file) == IMAGE_FILE
        ), 'The image file you want to load, is not defined by us as an image.'
    image = Image.open(input_file)

    return image


def get_file_type(input_file: Union[str, BytesIO, bytes] = None) -> str:
    """
    Get the type of a file.

    :param input_file: Path to the file or file in bytes format
    :return: Name of file type
    """
    return get_file_type_and_extension(input_file=input_file)[0]


def get_file_type_and_extension(input_file: Union[str, BytesIO, bytes] = None) -> Tuple[str, str]:
    """
    Get the type of a file via the filetype library, which checks the magic bytes to see the internal format.

    :param input_file: Path to the file or file in bytes format
    :return: Name of file type
    """
    if isinstance(input_file, str):
        file_name = os.path.basename(input_file)
        file_path = input_file
        extension = filetype.guess_extension(input_file)
    elif isinstance(input_file, BytesIO):
        file_name = 'BytesIO'
        file_path = 'BytesIO'
        extension = filetype.guess_extension(input_file.getvalue())
    elif isinstance(input_file, bytes):
        file_name = 'bytes'
        file_path = 'bytes'
        extension = filetype.guess_extension(input_file)
    else:
        raise NotImplementedError(f'Unsupported type of argument file: {type(input_file)}.')

    def isdir(z, name):
        """Check zip file namelist."""
        return any(x.startswith("%s/" % name.rstrip("/")) for x in z.namelist())

    def isfile(z, name):
        """Check zip file namelist."""
        return any(x.endswith(name) for x in z.namelist())

    if extension is None:
        if isinstance(input_file, str):
            with open(input_file, 'rb') as f:
                data_bytes = f.read()
        elif isinstance(input_file, BytesIO):
            data_bytes = input_file.read()
        elif isinstance(input_file, bytes):
            data_bytes = input_file
        else:
            raise NotImplementedError(f'Unsupported type of argument file: {type(input_file)}.')
        if b'%PDF' in data_bytes:
            extension = filetype.guess_extension(b'%PDF' + data_bytes.split(b'%PDF')[-1])

    file_type = None
    if extension == 'pdf':
        file_type = PDF_FILE
    elif extension in ['png', 'tiff', 'tif', 'jpeg', 'jpg']:
        file_type = IMAGE_FILE
    elif extension == 'zip':
        r = zipfile.ZipFile(input_file)
        # check for office files
        if isdir(r, "docProps") or isdir(r, "_rels"):
            file_type = OFFICE_FILE
        # check for open office files
        if isdir(r, "META-INF") and isfile(r, 'meta.xml') and isfile(r, 'settings.xml'):
            file_type = OFFICE_FILE

    if file_type not in [PDF_FILE, IMAGE_FILE, OFFICE_FILE]:
        error_message = f'We do not support file {file_name} with extension {extension} to get text: {file_path}'
        logger.error(error_message)
        raise NotImplementedError(error_message)

    logger.debug(f'File {file_path} is of file type {SUPPORTED_FILE_TYPES[file_type]}')
    return file_type, extension


@contextmanager
def does_not_raise():
    """
    Serve a complement to raise, no-op context manager does_not_raise.

    docs.pytest.org/en/latest/example/parametrize.html#parametrizing-conditional-raising
    """
    yield


def convert_to_bio_scheme(text: str, annotations: List) -> List[Tuple[str, str]]:
    """
    Mark all the entities in the text as per the BIO scheme.

    The splitting is using the sequence of words, expecting some characters like "." a separate token.

    Hello O
    , O
    it O
    's O
    Konfuzio B-ORG
    . O

    The start and end offsets are considered having the origin in the beginning of the input text.
    If only part of the text of the Document is passed, the start and end offsets of the Annotations must be
    adapted first.

    :param text: text to be annotated in the bio scheme
    :param annotations: annotations in the Document with start and end offset and Label name
    :return: list of tuples with each word in the text an the respective Label
    """
    import nltk

    nltk.download('punkt')
    tagged_entities = []
    annotations.sort(key=lambda x: x[0])  # todo only spans can be sorted

    previous_start = 0
    end = 0
    if text:
        for start, end, label_name in annotations:
            prev_text = text[previous_start:start]
            for word in nltk.word_tokenize(prev_text):
                tagged_entities.append((word, 'O'))

            temp_str = text[start:end]
            tmp_list = nltk.word_tokenize(temp_str)

            if len(tmp_list) > 1:
                tagged_entities.append((tmp_list[0], 'B-' + label_name))
                for w in tmp_list[1:]:
                    tagged_entities.append((w, 'I-' + label_name))
            else:
                tagged_entities.append((tmp_list[0], 'B-' + label_name))

            previous_start = start

    if end < len(text):
        pos_text = text[end:]
        for word in nltk.word_tokenize(pos_text):
            tagged_entities.append((word, 'O'))

    return tagged_entities


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py.

    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\:\.\w\s-]', '', value.lower())
    return re.sub(r'[-\s\:\.]+', '-', value).replace('-_', '_')


def amend_file_name(file_name: str, append_text: str = '', new_extension: str = None) -> str:
    """
    Append text to a filename in front of extension.

    example found here: https://stackoverflow.com/a/37487898

    :param new_extension: Change the file extension
    :param file_path: Name of a file, e.g. file.pdf
    :param append_text: Text you you want to append between file name ane extension
    :return: extended path to file
    """
    if len(os.path.basename(file_name) + append_text) >= 255:
        raise OSError('The name of the file you want to generate is too long.')
    if file_name.strip():
        path, extension = os.path.splitext(file_name)
        path = slugify(path)

        if new_extension == '':
            extension = ''
        elif new_extension:
            extension = new_extension

        if append_text:
            append_text = f'_{append_text}'

        return f'{path}{append_text}{extension}'
    else:
        raise ValueError(f'Name of file cannot be: {file_name}')


#
# def get_paragraphs_by_line_space(
#     bbox: dict,
#     text: str,
#     height: Union[float, int] = None,
#     return_dataframe: bool = False,
#     line_height_ration: float = 0.8,
# ) -> Union[List[List[List[dict]]], Tuple[List[List[List[dict]]], pd.DataFrame]]:
#     """
#     Split a text into paragraphs considering the space between the lines.
#
#     A paragraph consists in a list of lines. Each line corresponds to a dictionary.
#
#     :param bbox: Bounding boxes of the characters in the  Document
#     :param text: Text of the document
#     :param height: Threshold value for the distance between lines
#     :param return_dataframe: If to return a dataframe with the paragraph text and page number
#     :param line_height_ration: Ratio of the result of median of the distance between lines as threshold
#     :return: List of with the paragraph information per page of the document.
#     """
#     # Add start_offset and end_offset to every bbox item.
#     bbox = dict((k, dict(**v, start_offset=int(k), end_offset=int(k) + 1)) for (k, v) in bbox.items())
#     page_numbers = set(int(box['page_number']) for box in bbox.values())
#     document_structure = []
#     data = []
#
#     if height is not None:
#         if not (isinstance(height, int) or isinstance(height, float)):
#             raise Exception(f'Parameter must be of type int or float. It is {type(height)}.')
#
#     for page_number in page_numbers:
#         previous_y0 = None
#         paragraphs = []
#
#         if height is None:
#             line_threshold = line_height_ration * median(
#                 box['y1'] - box['y0'] for box in bbox.values() if box['page_number'] == page_number
#             )
#         else:
#             line_threshold = height
#
#         line_numbers = set(int(box['line_number']) for box in bbox.values() if box['page_number'] == page_number)
#         for line_number in line_numbers:
#             line_bboxes = list(
#                 box for box in bbox.values() if box['page_number'] == page_number and box['line_number'] ==line_number
#             )
#             max_y1 = max([x['y1'] for x in line_bboxes])
#             min_y0 = min([x['y0'] for x in line_bboxes])
#
#             max_x1 = max([x['x1'] for x in line_bboxes])
#             min_x0 = min([x['x0'] for x in line_bboxes])
#
#             min_top = min([x['top'] for x in line_bboxes])
#             max_bottom = max([x['bottom'] for x in line_bboxes])
#
#             start_offset = min(x['start_offset'] for x in line_bboxes)
#             end_offset = max(x['end_offset'] for x in line_bboxes)
#             _text = text[start_offset:end_offset]
#             if _text.replace(' ', '') == '':
#                 continue
#
#             if previous_y0 and previous_y0 - max_y1 < line_threshold:
#                 paragraphs[-1].append(
#                     {
#                         'start_offset': start_offset,
#                         'end_offset': end_offset,
#                         'text': _text,
#                         'line_bbox': {
#                             'x0': min_x0,
#                             'x1': max_x1,
#                             'y0': min_y0,
#                             'y1': max_y1,
#                             'top': min_top,
#                             'bottom': max_bottom,
#                             'page_index': page_number - 1,
#                         },
#                     }
#                 )
#             else:
#                 paragraphs.append(
#                     [
#                         {
#                             'start_offset': start_offset,
#                             'end_offset': end_offset,
#                             'text': _text,
#                             'line_bbox': {
#                                 'x0': min_x0,
#                                 'x1': max_x1,
#                                 'y0': min_y0,
#                                 'y1': max_y1,
#                                 'top': min_top,
#                                 'bottom': max_bottom,
#                                 'page_index': page_number - 1,
#                             },
#                         }
#                     ]
#                 )
#
#             previous_y0 = min_y0
#
#         document_structure.append(paragraphs)
#
#         for paragraph_ in paragraphs:
#             paragraph_text = [line['text'] + "\n" for line in paragraph_]
#             paragraph_text = ''.join(paragraph_text)
#             data.append({"page_number": page_number, "paragraph_text": paragraph_text})
#
#     dataframe = pd.DataFrame(data=data)
#
#     if return_dataframe:
#         return document_structure, dataframe
#
#     else:
#         return document_structure


def get_sentences(text: str, offsets_map: Union[dict, None] = None, language: str = 'german') -> List[dict]:
    """
    Split a text into sentences using the sentence tokenizer from the package nltk.

    :param text: Text to split into sentences
    :param offsets_map: mapping between the position of the character in the input text and the offset in the text
    of the document
    :param language: language of the text
    :return: List with a dict per sentence with its text and its start and end offsets in the text of the document.
    """
    from nltk.tokenize import sent_tokenize

    sentences = set()
    tokens = sent_tokenize(text, language=language)

    for token_txt in tokens:
        try:
            matches = [(m.start(0), m.end(0)) for m in re.finditer(token_txt, text)]
        except:  # NOQA
            logger.warning(f'Not possible to find a match with sent_tokenize for token {token_txt}.')
            continue

        if len(matches) > 0:
            if offsets_map is not None:
                # get start and end offsets from the text in the document
                start_char = offsets_map[matches[0][0]]
                try:
                    end_char = offsets_map[matches[0][1]]
                except:  # NOQA
                    last_key_value = list(offsets_map.keys())[-1]
                    if matches[0][1] - last_key_value <= 1:
                        end_char = offsets_map[last_key_value]
                    else:
                        logger.warning('Not able to find matches due to mismatch between text and OCR.')
                        continue
            else:
                start_char = matches[0][0]
                end_char = matches[0][1]

            sentences.add((text[matches[0][0] : matches[0][1]], start_char, end_char))

    sentences = sorted(sentences, key=lambda x: x[1])  # sort by their start offsets

    # convert to list of dictionaries
    sentences = [{'offset_string': text, 'start_offset': start, 'end_offset': end} for text, start, end in sentences]

    return sentences


def map_offsets(characters_bboxes: list) -> dict:
    """
    Map the position of the character to its offset.

    E.g.:
    characters: x, y, z, w
    characters offsets: 2, 3, 20, 22

    The first character (x) has the offset 2.
    The fourth character (w) has the offset 22.
    ...

    offsets_map: {0: 2, 1: 3, 2: 20, 3: 22}

    :param characters_bboxes: Bounding boxes information of the characters.
    :returns: Mapping of the position of the characters and its offsets.
    """
    for character_bbox in characters_bboxes:
        character_bbox['string_offset'] = int(character_bbox['string_offset'])
    characters_bboxes.sort(key=lambda k: k['string_offset'])
    offsets_map = dict((i, x['string_offset']) for i, x in enumerate(characters_bboxes))

    return offsets_map


# def convert_segmentation_bbox(bbox: dict, page: dict) -> dict:
#     """
#     Convert bounding box from the segmentation result to the scale of the characters bboxes of the document.
#
#     :param bbox: Bounding box from the segmentation result
#     :param page: Page information
#     :return: Converted bounding box.
#     """
#     original_size = page['original_size']
#     image_size = page['size']
#     factor_y = original_size[1] / image_size[1]
#     factor_x = original_size[0] / image_size[0]
#     height = image_size[1]
#
#     temp_y0 = (height - bbox['y0']) * factor_y
#     temp_y1 = (height - bbox['y1']) * factor_y
#     bbox['y0'] = temp_y1
#     bbox['y1'] = temp_y0
#     bbox['x0'] = bbox['x0'] * factor_x
#     bbox['x1'] = bbox['x1'] * factor_x
#
#     return bbox

#
# def select_bboxes(selection_bbox: dict, page_bboxes: list, tolerance: int = 10) -> list:
#     """
#     Filter the characters bboxes of the Document page according to their x/y values.
#
#     The result only includes the characters that are inside the selection bbox.
#
#     :param selection_bbox: Bounding box used to select the characters bboxes.
#     :param page_bboxes: Bounding boxes of the characters in the Document page.
#     :param tolerance: Tolerance for the coordinates values.
#     :return: Selected characters bboxes.
#     """
#     selected_char_bboxes = [
#         char_bbox
#         for char_bbox in page_bboxes
#         if int(selection_bbox["x0"]) - tolerance <= char_bbox["x0"]
#         and int(selection_bbox["x1"]) + tolerance >= char_bbox["x1"]
#         and int(selection_bbox["y0"]) - tolerance <= char_bbox["y0"]
#         and int(selection_bbox["y1"]) + tolerance >= char_bbox["y1"]
#     ]
#
#     return selected_char_bboxes

#
# def group_bboxes_per_line(char_bboxes: dict, page_index: int) -> list:
#     """
#     Group characters bounding boxes per line.
#
#     A line will have a single bounding box.
#
#     :param char_bboxes: Bounding boxes of the characters.
#     :param page_index: Index of the page in the document.
#     :return: List with 1 bounding box per line.
#     """
#     lines_bboxes = []
#
#     # iterate over each line_number and all of the character bboxes with that line number
#     for line_number, line_char_bboxes in itertools.groupby(char_bboxes, lambda x: x['line_number']):
#         # set the default values which we overwrite with the actual character bbox values
#         x0 = 100000000
#         top = 10000000
#         y0 = 10000000
#         x1 = 0
#         y1 = 0
#         bottom = 0
#         start_offset = 100000000
#         end_offset = 0
#
#         # remove space chars from the line selection so they don't interfere with the merging of bboxes
#         # (a bbox should never start with a space char)
#         trimmed_line_char_bboxes = [char for char in line_char_bboxes if not char['text'].isspace()]
#
#         if len(trimmed_line_char_bboxes) == 0:
#             continue
#
#         # merge characters bounding boxes of the same line
#         for char_bbox in trimmed_line_char_bboxes:
#             x0 = min(char_bbox['x0'], x0)
#             top = min(char_bbox['top'], top)
#             y0 = min(char_bbox['y0'], y0)
#
#             x1 = max(char_bbox['x1'], x1)
#             bottom = max(char_bbox['bottom'], bottom)
#             y1 = max(char_bbox['y1'], y1)
#
#             start_offset = min(int(char_bbox['string_offset']), start_offset)
#             end_offset = max(int(char_bbox['string_offset']), end_offset)
#
#         line_bbox = {
#             'bottom': bottom,
#             'page_index': page_index,
#             'top': top,
#             'x0': x0,
#             'x1': x1,
#             'y0': y0,
#             'y1': y1,
#             'start_offset': start_offset,
#             'end_offset': end_offset + 1,
#             'line_number': line_number,
#         }
#
#         lines_bboxes.append(line_bbox)
#
#     return lines_bboxes


# def merge_bboxes(bboxes: list):
#     """
#     Merge bounding boxes.
#
#     :param bboxes: Bounding boxes to be merged.
#     :return: Merged bounding box.
#     """
#     merge_bbox = {
#         "x0": min([b['x0'] for b in bboxes]),
#         "x1": max([b['x1'] for b in bboxes]),
#         "y0": min([b['y0'] for b in bboxes]),
#         "y1": max([b['y1'] for b in bboxes]),
#         "top": min([b['top'] for b in bboxes]),
#         "bottom": max([b['bottom'] for b in bboxes]),
#         "page_index": bboxes[0]['page_index'],
#     }
#
#     return merge_bbox
#


def get_bbox(bbox, start_offset: int, end_offset: int) -> Dict:
    """
    Get single bbox for offset_string.

    Given a `bbox` (a dictionary containing a bbox for every character in a document) and a start/end_offset into that
    document, create a new bbox which covers every character bbox between the given start and end offset.

    Pages are zero indexed, i.e. the first page has page_number = 0.
    """
    # get the index of every character bbox in the Document between the start and end offset
    char_bbox_ids = [str(char_bbox_id) for char_bbox_id in range(start_offset, end_offset) if str(char_bbox_id) in bbox]

    # exit early if no bboxes are found between the start/end offset
    if not char_bbox_ids:
        logger.error(f"Between start {start_offset} and {end_offset} we do not find the bboxes of the characters.")
        return {'bottom': None, 'top': None, 'page_index': None, 'x0': None, 'x1': None, 'y0': None, 'y1': None}

    # set the default values which we overwrite with the actual character bbox values
    x0 = 100000000
    top = 10000000
    y0 = 10000000
    x1 = 0
    y1 = 0
    bottom = 0
    pdf_page_index = None
    line_indexes = []

    # combine all of the found character bboxes and calculate their combined x0, x1, etc. values
    for char_bbox_id in char_bbox_ids:
        x0 = min(bbox[char_bbox_id]['x0'], x0)
        top = min(bbox[char_bbox_id]['top'], top)
        y0 = min(bbox[char_bbox_id]['y0'], y0)

        x1 = max(bbox[char_bbox_id]['x1'], x1)
        bottom = max(bbox[char_bbox_id]['bottom'], bottom)
        y1 = max(bbox[char_bbox_id]['y1'], y1)
        line_indexes.append(bbox[char_bbox_id]['page_number'])

        if pdf_page_index is not None:
            try:
                assert pdf_page_index == bbox[char_bbox_id]['page_number'] - 1
            except AssertionError:
                logger.warning(
                    "We don't support bounding boxes over page breaks yet, and will return the bounding box"
                    "on the first page of the match."
                )
                break
        pdf_page_index = bbox[char_bbox_id]['page_number'] - 1

    res = {'bottom': bottom, 'page_index': pdf_page_index, 'top': top, 'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1}
    if len(set(line_indexes)) == 1:
        res['line_index'] = line_indexes[0]
    return res


def get_default_label_set_documents(
    documents: List, selected_default_label_sets: List, project_label_sets: list, merge_multi_default: bool
):
    """
    For each default Label_set in a prj get a list of Documents to be used for that default Label_set.

    For each default Label_set we collect the Labels that belong to that Label_set.

    Then, for each document, we verify the Category Label_set. If if matches the default Label_set, we add the
    Document to the default Label_set list of Documents.

    If merge_multi_default is False we discard any Documents with a different default Label_set.

    If the Category Label_set of the Document does not match, but we still want to use Labels that are shared between
    default Label_sets (merge_multi_default=True), then we check for the Labels in each annotation_set of the document
    that mach Labels in the default Label_set that we are analysing.

    We rename the Labels that are not shared as "NO_LABEL".

    Format of dict is: {default Label_set.id_: list of Documents)
    """
    # keys are default Label_set ids, values are list of Documents
    default_label_set_documents = defaultdict(list)
    # keys are default Label_set names, values are list of Label names that appear in that default label_set
    default_labels = defaultdict(set)

    # filter label_sets of the project that belong to the selected default label_sets
    selected_ids = [x.id_ for x in selected_default_label_sets]
    selected_label_sets = []
    for label_set in project_label_sets:
        if label_set.is_default and label_set.id_ in selected_ids:
            selected_label_sets.append(label_set)
            continue
        if len(list(set([x.id_ for x in label_set.categories if x is not None]) & set(selected_ids))) > 0:
            selected_label_sets.append(label_set)
            continue

    # get the Labels which appear in each default label_set
    for label_set in selected_label_sets:
        if label_set.is_default:
            # if the label_set is default, get the Label directly
            _default_label_sets = [label_set]
        else:
            # if not, it is a child label_set, get the default from its parent (default_label_set)
            _default_label_sets = label_set.categories

        # add the Labels which appear in that default label_set and that contain Annotations
        label_set_labels = []
        for label in label_set.labels:
            if len(label.annotations) > 0:
                label_set_labels.append(label)

        for _default_label_set in _default_label_sets:
            default_labels[_default_label_set.id_] |= set(label_set_labels)

    # for each Document label_set in the project
    for default_label_set in [x for x in selected_default_label_sets if x.is_default]:
        # copy Documents so we only edit a new copy of them
        _documents = copy.deepcopy(documents) if merge_multi_default else documents
        # for each document
        for document in _documents:
            # if the default label_set matches the Category label_set, simply add to Documents
            # we can't simply check if default_label_set.id_
            # is in document_annotation_sets because document_annotation_sets
            # can contain annotation_sets from multiple default label_sets
            if len(document.annotations()) == 0:
                continue
            if document.category == default_label_set:
                default_label_set_documents[default_label_set.id_].append(document)
            # if not, then we need to edit it before adding
            # but only if merge_multi_default is True
            # if merge_multi_default is False we discard any Documents with a different default label_set
            elif merge_multi_default:
                # loop over the annotation_sets
                for i, annotation_set in enumerate(document.annotation_sets):
                    # we need to check if the annotation_set belongs to the "wrong" default label_set
                    # if it belongs to a default label_set, get the id_ from the default label_set as id_ = None
                    # if not a default, get the label_set id_
                    if annotation_set.label_set.default_label_set:
                        document_label_set_id = annotation_set.label_set.default_label_set.id_
                    else:
                        document_label_set_id = annotation_set.label_set.id_
                    # if it does not match the current default label_set
                    if document_label_set_id != default_label_set.id_:
                        # get the Labels that do not overlap with the current default label_set
                        non_overlapping_labels = (
                            set(annotation_set.label_set.labels) - default_labels[default_label_set]
                        )  # NOQA
                        # if the Labels do not overlap with the current default label_set, change to NO_LABEL
                        for label in non_overlapping_labels:
                            label.name = 'NO_LABEL'
                # append Document to the default_label_set_documents
                default_label_set_documents[default_label_set.id_].append(document)
    # return all default label_set Documents
    # hotfix removed typing " -> Tuple[Dict[int, List], Dict[int, List]]" as it is unclear what should be returned
    return default_label_set_documents, default_labels


def separate_labels(project, default_label_sets: List = None):
    """
    Create separated Labels for Labels which are shared between Label Sets.

    This should be used only for the training purpose.

    For all Documents in the project (training + test) for each Category, we check all Annotations in annotation_sets
    that do not belong to the Category label_set.

    For each Label that we find, we rewrite the name of the label, adding the label_set name, followed by "__" and
    the original name of the Label
    E.g.: label_set: Shipper, Label: Name -> Label: Shipper__Name

    Notes:
    When using this method, the Labels in the project are changed. This should be used in combination with the model
    models_labels_multiclass.SeparateLabelsAnnotationMultiClassModel so that these changes are undone in the extract
    and the output contains the correct Labels names.

    If the Labels of the project should be used in the original format for other tasks, for example, for the
    business evaluation, the project should be reloaded after the training.

    """
    from konfuzio_sdk.data import Label

    if not default_label_sets:
        default_label_sets = [x for x in project.label_sets if x.is_default]

    # Group Documents by default label_set and prepare for training.
    default_label_set_documents_dict, _ = get_default_label_set_documents(
        documents=project.documents + project.test_documents,
        selected_default_label_sets=default_label_sets,
        project_label_sets=project.label_sets,
        merge_multi_default=False,
    )

    for default_label_set in default_label_sets:
        try:
            # Use patched Documents to also use knowledge from other Document types which share some labels.
            _documents = default_label_set_documents_dict[default_label_set.id_]

            if len(_documents) == 0:
                logger.error(f'There are no documents for {default_label_set.name}.')
                continue

            for document in _documents:
                # Should we move this to a separate function?
                if len(document.annotations()) == 0:
                    continue
                document_default_annotation_sets = [
                    x for x in document.annotation_sets if x.label_set.is_default and x.label_set == document.category
                ]
                if len(document_default_annotation_sets) != 1:
                    raise Exception(
                        f'Exactly 1 default annotation_set is expected. '
                        f'There is {len(document_default_annotation_sets)} in Document {document.id_}'
                    )
                for annotation_set in document.annotation_sets:
                    label_set = annotation_set.label_set
                    prj_label_set = project.get_label_set_by_id(label_set.id_)
                    if label_set.is_default is False:
                        for annotation in annotation_set.annotations:
                            new_label_name = label_set.name + '__' + annotation.label.name
                            new_label_name_clean = label_set.name_clean + '__' + annotation.label.name_clean
                            if new_label_name in [x.name for x in project.labels]:
                                new_label = next(x for x in project.labels if new_label_name == x.name)
                            else:
                                # Sender__FirstName and Receiver__FirstName
                                new_label = Label(
                                    id_=randrange(-999999, -1),  # hotfix: identify separate Labels by ID < 0
                                    text=new_label_name,
                                    text_clean=new_label_name_clean,
                                    get_data_type_display=annotation.label.data_type,
                                    description=annotation.label.description,
                                    project=annotation.label.project,
                                    token_full_replacement=annotation.label.token_full_replacement,
                                    token_whitespace_replacement=annotation.label.token_whitespace_replacement,
                                    token_number_replacement=annotation.label.token_number_replacement,
                                    has_multiple_top_candidates=annotation.label.has_multiple_top_candidates,
                                )
                                prj_label_set.add_label(new_label)
                            annotation.label = new_label

        except Exception as e:
            logger.error(f'Separate Labels for {default_label_set} failed because of >>{e}<<.')
            return None

    return project


def get_missing_offsets(start_offset: int, end_offset: int, annotated_offsets: List[range]):
    """
    Calculate the missing characters.

    :param start_offset: Start of the overall text as index
    :param end_offset: End of the overall text as index
    :param: A list integers, where one character presents a character. It may be outside the start and end offset.

    :type start_offset: int
    :type end_offset:
    :type annotated_offsets: List[int]

    ..todo: How do we handle tokens that are smaller / larger than the correct Annotations?
            https://docs.google.com/document/d/1bxUgvX1OGG_fbQvDXW7gDVcgfKto1dgP94uTp5srFP4/edit

     :Example:

    >>> get_missing_offsets(start_offset=0, end_offset=170, annotated_offsets=[range(66, 78), range(159, 169)])
    [range(0, 65), range(78, 158), range(169, 170)]

    """
    # range makes sure that uvalid ranges are ignored: list(range(4,2)) == []
    annotated_characters: List[int] = sum([list(span) for span in annotated_offsets], [])

    # Create boolean list of size high-low+1, each index i representing whether (i+low)th element found or not.
    points_of_range = [False] * (end_offset - start_offset + 1)
    for i in range(len(annotated_characters)):
        # if ith element of arr is in range low to high then mark corresponding index as true in array
        if start_offset <= annotated_characters[i] <= end_offset:
            points_of_range[annotated_characters[i] - start_offset] = True

    # Traverse through the range and create all Spans where the character is not included, i.e. False.
    missing_characters = []
    for x in range(end_offset - start_offset + 1):
        if not points_of_range[x]:
            missing_characters.append(start_offset + x)

    start_span = 0
    spans: List[range] = []
    for before, missing_character in zip(missing_characters, missing_characters[1:]):
        if before == start_offset:
            start_span = before  # enter the offset
        elif before == missing_characters[0] and before + 1 == missing_character:
            start_span = before  # later start as sequence starts with a labeled offset
        elif before == missing_characters[0] and before + 1 < missing_character:
            spans.append(range(before, before + 1))  # we found a single missing character, list(range(5,6)) == [5]
        elif before + 1 < missing_character and start_span < before:
            spans.append(range(start_span, before + 1))  # add intermediate
            start_span = missing_character
        elif before + 1 < missing_character and start_span == before:
            spans.append(range(start_span, before + 1))  # add intermediate for a single chracter
            start_span = missing_character
        elif missing_character == end_offset:
            spans.append(range(start_span, missing_character))  # exit the offset
        elif missing_character == missing_characters[-1]:
            spans.append(range(start_span, missing_character + 1))  # earlier end as sequence ends with a labeled offset

    return spans


def iter_before_and_after(iterable, before=1, after=None, fill=None):
    """Iterate and provide before and after element. Generalized from http://stackoverflow.com/a/1012089."""
    if after is None:
        after = before

    iterators = itertools.tee(iterable, 1 + before + after)

    new = []

    for i, iterator in enumerate(iterators[:before]):
        new.append(itertools.chain([fill] * (before - i), iterator))

    new.append(iterators[before])

    if after > 0:
        for i, iterator in enumerate(iterators[-after:]):
            new.append(itertools.chain(itertools.islice(iterator, i + 1, None), [fill] * (i + 1)))

    return zip(*new)
