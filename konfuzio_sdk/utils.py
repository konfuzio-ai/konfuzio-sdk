"""Utils for the konfuzio sdk package."""

import datetime
import hashlib
import logging
import os
import re
import zipfile
from contextlib import contextmanager
from io import BytesIO
from statistics import median
from typing import Union, List, Tuple

import filetype
import nltk
import pandas as pd
from PIL import Image
from konfuzio_sdk import IMAGE_FILE, PDF_FILE, OFFICE_FILE, SUPPORTED_FILE_TYPES
from nltk.tokenize import sent_tokenize

logger = logging.getLogger(__name__)


def get_id(a_string, include_time: bool = False) -> int:
    """
    Generate a unique ID.

    :param a_string: String used to generating the unique ID
    :param include_time: Bool to include the time in the unique ID
    :return: Unique ID
    """
    if include_time:
        unique_string = a_string + get_timestamp(format='%Y-%m-%d-%H-%M-%S.%f')
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
                raise FileExistsError(f'Please check your file with size {file_size} at {file_path}.')
            else:
                return False
    else:
        if raise_exception:
            raise FileNotFoundError(f'File expected but not found at: {file_path}')
        else:
            return False


def get_timestamp(format='%Y-%m-%d-%H-%M-%S') -> str:
    """
    Return formatted timestamp.

    :param format: Format of the timestamp (e.g. year-month-day-hour-min-sec)
    :return: Timestamp
    """
    now = datetime.datetime.now()
    timestamp = now.strftime(format)
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
    try:
        image = Image.open(input_file)
    except OSError:
        # in case of corrupted images
        return None

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
        return any(x.endswith(name) for x in r.namelist())

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
        r = zipfile.ZipFile(input_file, "r")
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

    The start and end offsets are considered having the origin in the begining of the input text.
    If only part of the text of the document is passed, the start and end offsets of the annotations must be
    adapted first.

    :param text: text to be annotated in the bio scheme
    :param annotations: annotations in the document with start and end offset and label name
    :return: list of tuples with each word in the text an the respective label
    """
    if len(text) == 0:
        logger.error('No text to be converted to the BIO-scheme.')
        return None

    nltk.download('punkt')
    tagged_entities = []
    annotations.sort(key=lambda x: x[0])

    if len(annotations) == 0:
        logger.info('No annotations in the converstion to the BIO-scheme.')
        for word in nltk.word_tokenize(text):
            tagged_entities.append((word, 'O'))
        return tagged_entities

    previous_start = 0

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


def get_paragraphs_by_line_space(
    bbox: dict,
    text: str,
    height: Union[float, int] = None,
    return_dataframe: bool = False,
    line_height_ration: float = 0.8,
) -> Union[List[List[List[dict]]], Tuple[List[List[List[dict]]], pd.DataFrame]]:
    """
    Split a text into paragraphs considering the space between the lines.

    A paragraph consists in a list of lines. Each line corresponds to a dictionary.

    :param bbox: Bounding boxes of the characters in the document
    :param text: Text of the document
    :param height: Threshold value for the distance between lines
    :param return_dataframe: If to return a dataframe with the paragraph text and page number
    :param line_height_ration: Ratio of the result of median of the distance between lines to be considered as threshold
    :return: List of with the paragraph information per page of the document.
    """
    # Add start_offset and end_offset to every bbox item.
    bbox = dict((k, dict(**v, start_offset=int(k), end_offset=int(k) + 1)) for (k, v) in bbox.items())
    page_numbers = set(int(box['page_number']) for box in bbox.values())
    document_structure = []
    data = []

    if height is not None:
        if not (isinstance(height, int) or isinstance(height, float)):
            raise Exception(f'Parameter must be of type int or float. It is {type(height)}.')

    for page_number in page_numbers:
        previous_y0 = None
        paragraphs = []

        if height is None:
            line_threshold = line_height_ration * median(
                box['y1'] - box['y0'] for box in bbox.values() if box['page_number'] == page_number
            )
        else:
            line_threshold = height

        line_numbers = set(int(box['line_number']) for box in bbox.values() if box['page_number'] == page_number)
        for line_number in line_numbers:
            line_bboxes = list(
                box for box in bbox.values() if box['page_number'] == page_number and box['line_number'] == line_number
            )
            max_y1 = max([x['y1'] for x in line_bboxes])
            min_y0 = min([x['y0'] for x in line_bboxes])

            max_x1 = max([x['x1'] for x in line_bboxes])
            min_x0 = min([x['x0'] for x in line_bboxes])

            min_top = min([x['top'] for x in line_bboxes])
            max_bottom = max([x['bottom'] for x in line_bboxes])

            start_offset = min(x['start_offset'] for x in line_bboxes)
            end_offset = max(x['end_offset'] for x in line_bboxes)
            _text = text[start_offset:end_offset]
            if _text.replace(' ', '') == '':
                continue

            if previous_y0 and previous_y0 - max_y1 < line_threshold:
                paragraphs[-1].append(
                    {
                        'start_offset': start_offset,
                        'end_offset': end_offset,
                        'text': _text,
                        'line_bbox': {
                            'x0': min_x0,
                            'x1': max_x1,
                            'y0': min_y0,
                            'y1': max_y1,
                            'top': min_top,
                            'bottom': max_bottom,
                            'page_index': page_number - 1,
                        },
                    }
                )
            else:
                paragraphs.append(
                    [
                        {
                            'start_offset': start_offset,
                            'end_offset': end_offset,
                            'text': _text,
                            'line_bbox': {
                                'x0': min_x0,
                                'x1': max_x1,
                                'y0': min_y0,
                                'y1': max_y1,
                                'top': min_top,
                                'bottom': max_bottom,
                                'page_index': page_number - 1,
                            },
                        }
                    ]
                )

            previous_y0 = min_y0

        document_structure.append(paragraphs)

        for paragraph_ in paragraphs:
            paragraph_text = [line['text'] + "\n" for line in paragraph_]
            paragraph_text = ''.join(paragraph_text)
            data.append({"page_number": page_number, "paragraph_text": paragraph_text})

    dataframe = pd.DataFrame(data=data)

    if return_dataframe:
        return document_structure, dataframe

    else:
        return document_structure


def get_sentences(text: str, offsets_map: Union[dict, None] = None, language: str = 'german') -> List[dict]:
    """
    Split a text into sentences using the sentence tokenizer from the package nltk.

    :param text: Text to split into sentences
    :param offsets_map: mapping between the position of the character in the input text and the offset in the text
    of the document
    :param language: language of the text
    :return: List with a dict per sentence with its text and its start and end offsets in the text of the document.
    """
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
