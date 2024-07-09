"""Utils for the konfuzio sdk package."""
import datetime
import itertools
import logging
import operator
import os
import unicodedata
import uuid
import zipfile
from contextlib import contextmanager
from io import BytesIO
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Type, Union
from warnings import warn

import filetype
import pkg_resources
import regex as re
from PIL import Image
from pympler import asizeof

from konfuzio_sdk import IMAGE_FILE, OFFICE_FILE, PDF_FILE, SUPPORTED_FILE_TYPES

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # only import when type checking to prevent circular import errors
    from konfuzio_sdk.data import Bbox, Span


def sdk_isinstance(instance, klass):
    """
    Implement a custom isinstance which is compatible with cloudpickle saving by value.

    When using cloudpickle with "register_pickle_by_value" the classes of "konfuzio.data" will be loaded in the
    "types" module. For this case the builtin method "isinstance" will return False because it tries to compare
    "types.Document" with "konfuzio_sdk.data.Document".
    """
    # TODO: Update test cases to use sdk_isinstance
    result = type(instance).__name__ == klass.__name__
    return result


def exception_or_log_error(
    msg: str,
    handler: str = 'sdk',
    fail_loudly: Optional[bool] = True,
    exception_type: Optional[Type[Exception]] = ValueError,
) -> None:
    """
    Log error or raise an exception.

    This function is needed to control error handling in production. If `fail_loudly` is set to `True`, the function
    raises an exception to type `exception_type` with a message and handler in the format `{"message" : msg,
                                                                                            "handler" : handler}`.
    If `fail_loudly` is set to `False`, the function logs an error with `msg` using the logger.

    :param msg: (str): The error message to be logged or raised.
    :param handler: (str): The handler associated with the error. Defaults to "sdk"
    :param fail_loudly: A flag indicating whether to raise an exception or log the error. Defaults to `True`.
    :param exception_type: The type of exception to be raised. Defaults to `ValueError`.
    :return: None
    """
    # Raise whatever exception while specifying the handler
    if fail_loudly:
        raise exception_type({'message': msg, 'handler': handler})
    else:
        logger.error(msg)


def get_id(include_time: bool = False) -> str:
    """
    Generate a unique ID.

    :param include_time: Bool to include the time in the unique ID
    :return: Unique ID
    """
    unique_id = str(uuid.uuid4())
    if include_time:
        return unique_id + get_timestamp(konfuzio_format='%Y-%m-%d-%H-%M-%S.%f')
    else:
        return unique_id


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
            with open(file_path, 'rb'):
                logger.debug(f'File expected and found at {file_path} with ID {get_id()}')
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


def memory_size_of(obj) -> int:
    """Return memory size of object in bytes."""
    size = asizeof.asizeof(obj)
    return size


def normalize_memory(memory: Union[None, str]) -> Union[int, None]:
    """
    Return memory size in human-readable form to int of number of bytes.

    :param memory: Memory size in human readable form (e.g. "50MB").
    :return: int of bytes if valid, else None
    """
    if memory is not None:
        if isinstance(memory, int) or memory.isdigit():
            memory = int(memory)
        else:
            # Check if the first part of the string (before the unit) is numeric
            if not memory[:-3].isdigit() and not memory[:-2].isdigit():
                logger.error(f'memory value {memory} invalid.')
                return None
            else:
                mem_val = int(memory[:-3] if memory[-3:].isalpha() else memory[:-2])

            # Check the unit and convert to bytes
            if memory[-2:].lower() == 'gb':
                memory = mem_val * 1e9
            elif memory[-2:].lower() == 'mb':
                memory = mem_val * 1e6
            elif memory[-2:].lower() == 'kb':
                memory = mem_val * 1e3
            elif memory[-3:].lower() == 'gib':
                memory = mem_val * 2**30
            elif memory[-3:].lower() == 'mib':
                memory = mem_val * 2**20
            elif memory[-3:].lower() == 'kib':
                memory = mem_val * 2**10
            else:
                logger.error(f'memory value {memory} invalid.')
                return None
    return memory


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
        return any(x.startswith('%s/' % name.rstrip('/')) for x in z.namelist())

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
        if isdir(r, 'docProps') or isdir(r, '_rels'):
            file_type = OFFICE_FILE
        # check for open office files
        if isdir(r, 'META-INF') and isfile(r, 'meta.xml') and isfile(r, 'settings.xml'):
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


def convert_to_bio_scheme(document) -> List[Tuple[str, str]]:
    """
    Mark all the entities in the text as per the BIO scheme.

    The splitting is using the sequence of words, expecting some characters like "." a separate token.

    Hello O
    , O
    it O
    's O
    Helm B-ORG
    und I-ORG
    Nagel I-ORG
    . O

    :param document: Document to be converted into the bio scheme
    :return: list of tuples with each word in the text an the respective Label
    """
    import nltk

    nltk.download('punkt')

    spans_in_doc = []
    for annotation in document.view_annotations():
        for span in annotation.spans:
            spans_in_doc.append((span.start_offset, span.end_offset, annotation.label.name))
    text = document.text

    tagged_entities = []
    spans_in_doc.sort(key=lambda x: x[0])

    previous_end = 0
    end = 0
    if text:
        for start, end, label_name in spans_in_doc:
            prev_text = text[previous_end:start]
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

            previous_end = end

    if end < len(text):
        pos_text = text[end:]
        for word in nltk.word_tokenize(pos_text):
            tagged_entities.append((word, 'O'))

    return tagged_entities


def slugify(value):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py.

    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\:\.\w\s-]', '', value.lower())
    return re.sub(r'[-\s\:\.]+', '-', value).replace('-_', '_')


def normalize_name(value: str) -> str:
    """
    Normalize names for different Konfuzio concepts by removing slashes and checking for non-ascii symbols.

    :param value: A name to be normalized.
    """
    if value:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
        return re.sub('/', '', value)
    else:
        return


def amend_file_name(
    file_name: str, append_text: str = '', append_separator: str = '_', new_extension: str = None
) -> str:
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
            append_text = f'{append_separator}{append_text}'

        return f'{path}{append_text}{extension}'
    else:
        raise ValueError(f'Name of file cannot be: {file_name}')


def amend_file_path(file_path: str, append_text: str = '', append_separator: str = '_', new_extension: str = None):
    """
    Similar to amend_file_name however the file_name is interpreted as a full path.

    :param new_extension: Change the file extension
    :param file_path: Name of a file, e.g. file.pdf
    :param append_text: Text you you want to append between file name ane extension

    :return: extended path to file
    """
    split_file_path, split_file_name = os.path.split(file_path)
    new_filename = amend_file_name(
        file_name=split_file_name,
        append_text=append_text,
        append_separator=append_separator,
        new_extension=new_extension,
    )
    return os.path.join(split_file_path, new_filename)


def get_sentences(text: str, offsets_map: Union[dict, None] = None, language: str = 'german') -> List[dict]:
    """
    Split a text into sentences using the sentence tokenizer from the package nltk.

    :param text: Text to split into sentences
    :param offsets_map: mapping between the position of the character in the input text and the offset in the text
    of the document
    :param language: language of the text
    :return: List with a dict per sentence with its text and its start and end offsets in the text of the document.
    """
    warn('Method needs testing and revision. Please create a Ticket if you use it.', DeprecationWarning, stacklevel=2)
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
    offsets_map = {i: x['string_offset'] for i, x in enumerate(characters_bboxes)}

    return offsets_map


def detectron_get_paragraph_bboxes(detectron_document_results: List[List[Dict]], document) -> List[List['Bbox']]:
    """Call detectron Bbox corresponding to each paragraph."""
    from konfuzio_sdk.data import Bbox

    assert len(detectron_document_results) == document.number_of_pages

    paragraph_document_bboxes: List[List['Bbox']] = []

    for page_index, detectron_page_results in enumerate(detectron_document_results):
        paragraph_page_bboxes = []
        for detectron_result in detectron_page_results:
            paragraph_bbox = Bbox.from_image_size(
                x0=detectron_result['x0'],
                x1=detectron_result['x1'],
                y1=detectron_result['y1'],
                y0=detectron_result['y0'],
                page=document.get_page_by_index(page_index=page_index),
            )
            label_name = detectron_result['label']
            paragraph_bbox._label_name = label_name
            paragraph_page_bboxes.append(paragraph_bbox)
        paragraph_document_bboxes.append(paragraph_page_bboxes)
    assert len(document.pages()) == len(paragraph_document_bboxes)

    return paragraph_document_bboxes


def select_bboxes(selection_bbox: dict, page_bboxes: list, tolerance: int = 10) -> list:
    """
    Filter the characters bboxes of the Document page according to their x/y values.

    The result only includes the characters that are inside the selection bbox.

    :param selection_bbox: Bounding box used to select the characters bboxes.
    :param page_bboxes: Bounding boxes of the characters in the Document page.
    :param tolerance: Tolerance for the coordinates values.
    :return: Selected characters bboxes.
    """
    warn('Method needs testing and revision. Please create a Ticket if you use it.', DeprecationWarning, stacklevel=2)
    selected_char_bboxes = [
        char_bbox
        for char_bbox in page_bboxes
        if int(selection_bbox['x0']) - tolerance <= char_bbox['x0']
        and int(selection_bbox['x1']) + tolerance >= char_bbox['x1']
        and int(selection_bbox['y0']) - tolerance <= char_bbox['y0']
        and int(selection_bbox['y1']) + tolerance >= char_bbox['y1']
    ]

    return selected_char_bboxes


def merge_bboxes(bboxes: list):
    """
    Merge bounding boxes.

    :param bboxes: Bounding boxes to be merged.
    :return: Merged bounding box.
    """
    warn('Method needs testing and revision. Please create a Ticket if you use it.', DeprecationWarning, stacklevel=2)
    # the issue is there: https://git.konfuzio.com/konfuzio/objectives/-/issues/9333
    merge_bbox = {
        'x0': min([b['x0'] for b in bboxes]),
        'x1': max([b['x1'] for b in bboxes]),
        'y0': min([b['y0'] for b in bboxes]),
        'y1': max([b['y1'] for b in bboxes]),
        'top': min([b['top'] for b in bboxes]),
        'bottom': max([b['bottom'] for b in bboxes]),
        'page_index': bboxes[0]['page_index'],
    }

    return merge_bbox


def get_bbox(bbox, start_offset: int, end_offset: int) -> Dict:
    """
    Get single bbox for offset_string.

    Given a `bbox` (a dictionary containing a bbox for every character in a document) and a start/end_offset into that
    document, create a new bbox which covers every character bbox between the given start and end offset.

    Pages are zero indexed, i.e. the first page has page_number = 0.

    :raises ValueError None of the characters provides a bounding box.
    """
    warn('This method is deprecated. Please use Spans in Documents or Bbox in Pages.', DeprecationWarning, stacklevel=2)
    # get the index of every character bbox in the Document between the start and end offset
    char_bbox_ids = [str(char_bbox_id) for char_bbox_id in range(start_offset, end_offset) if str(char_bbox_id) in bbox]

    # exit early if no bboxes are found between the start/end offset
    if not char_bbox_ids:
        logger.error(f'Between start {start_offset} and {end_offset} we do not find the bboxes of the characters.')
        raise ValueError(f'Characters from {start_offset} to {end_offset} do not provide any bounding box.')

    # set the default values which we overwrite with the actual character bbox values
    x0 = 100000000
    top = 10000000
    y0 = 10000000
    x1 = 0
    y1 = 0
    bottom = 0
    pdf_page_index = None
    line_indexes = []  # todo create one bounding box per line.

    # combine all the found character bboxes and calculate their combined x0, x1, etc. values
    for char_bbox_id in char_bbox_ids:
        # conditions for backward compatibility
        x0 = min(bbox[char_bbox_id]['x0'], x0)
        if 'top' in bbox[char_bbox_id].keys():
            top = min(bbox[char_bbox_id]['top'], top)
        y0 = min(bbox[char_bbox_id]['y0'], y0)

        x1 = max(bbox[char_bbox_id]['x1'], x1)
        if 'bottom' in bbox[char_bbox_id].keys():
            bottom = max(bbox[char_bbox_id]['bottom'], bottom)
        y1 = max(bbox[char_bbox_id]['y1'], y1)
        if 'page_number' in bbox[char_bbox_id]:
            line_indexes.append(bbox[char_bbox_id]['page_number'])

        if pdf_page_index is not None:
            try:
                assert pdf_page_index == bbox[char_bbox_id]['page_number'] - 1
            except AssertionError:
                logger.warning(
                    "We don't support bounding boxes over page breaks yet, and will return the bounding box"
                    'on the first page of the match.'
                )
                break
        if 'page_number' in bbox[char_bbox_id]:
            pdf_page_index = bbox[char_bbox_id]['page_number'] - 1

    res = {'bottom': bottom, 'page_index': pdf_page_index, 'top': top, 'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1}
    if len(set(line_indexes)) == 1:
        res['line_index'] = line_indexes[0]
    return res


def get_missing_offsets(start_offset: int, end_offset: int, annotated_offsets: List[range]):
    """
    Calculate the missing characters.

    :param start_offset: Start of the overall text as index
    :param end_offset: End of the overall text as index
    :param: A list integers, where one character presents a character. It may be outside the start and end offset.

    :type start_offset: int
    :type end_offset:
    :type annotated_offsets: List[int]

    .. todo::
        How do we handle tokens that are smaller / larger than the correct Annotations? See
        `link <https://docs.google.com/document/d/1bxUgvX1OGG_fbQvDXW7gDVcgfKto1dgP94uTp5srFP4/edit>`_

    >>> get_missing_offsets(start_offset=0, end_offset=170, annotated_offsets=[range(66, 78), range(159, 169)])
    [range(0, 66), range(78, 159), range(169, 170)]

    """
    # range makes sure that invalid ranges are ignored: list(range(4,2)) == []
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


def get_merged_bboxes(doc_bbox: Dict, bboxes: Union[Dict, List], doc_text: Optional[str] = None) -> List[Dict]:
    """
    Merge Bboxes.

    Given a document bbox in dict format and a dict or list of bboxes (the selection), get the smallest
    possible set of bboxes that represents the characters of the selection, divided by lines.

    Similar to `get_bbox` and `get_bboxes`, but takes bboxes as input instead of offsets.

    :param doc_bbox: a dict representing all the characters in the document
    :param bboxes: a list or dict of bboxes representing the user's selection
    :param doc_text: an optional string containing the document's text

    Returns a list of bboxes.
    """
    warn('Method needs testing and revision. Please create a Ticket if you use it.', DeprecationWarning, stacklevel=2)
    # initialize the list of bboxes that will later be returned
    line_bboxes = []

    # convert string indexes to int
    # this is very expensive but can't be moved from here for now since we need int indexes multiple times
    doc_bbox = {int(index): char_bbox for index, char_bbox in doc_bbox.items()}

    # convert selection dict if not already a list
    if isinstance(bboxes, dict):
        bboxes = list(bboxes.values())

    # sort selected bboxes by y first, then x
    bboxes.sort(key=operator.itemgetter('y0', 'x0'))

    # iterate through every bbox of the selection
    for selection_bbox in bboxes:
        selected_bboxes = [
            # the index of the character is its offset, i.e. the number of chars before it in the document's text
            {'string_offset': index, **char_bbox}
            for index, char_bbox in doc_bbox.items()
            if selection_bbox['page_index'] == char_bbox['page_number'] - 1
            # filter the characters of the document according to their x/y values, so that we only include the
            # characters that are inside the selection
            and selection_bbox['x0'] <= char_bbox['x0']
            and selection_bbox['x1'] >= char_bbox['x1']
            and selection_bbox['y0'] <= char_bbox['y0']
            and selection_bbox['y1'] >= char_bbox['y1']
        ]

        # decide what we are going to group the character bboxes by in order to group those on the same line
        # either group by 'line_number' (if they all have a 'line_number' attribute) or 'bottom'
        if all('line_number' in selected_bbox.keys() for selected_bbox in selected_bboxes):
            group_by = 'line_number'
        else:
            group_by = 'bottom'

        # iterate over each line_number (or bottom, depending on group_by) and all of the character
        # bboxes that have the same line_number (or bottom)
        for line_number, line_char_bboxes in itertools.groupby(selected_bboxes, lambda x: x[group_by]):
            # set the defaut values which we overwrite with the actual character bbox values
            x0 = 100000000
            top = 10000000
            y0 = 10000000
            x1 = 0
            y1 = 0
            bottom = 0
            pdf_page_index = None
            start_offset = 100000000
            end_offset = 0

            # remove space chars from the line selection so they don't interfere with the merging of bboxes
            # (a bbox should never start with a space char)
            trimmed_line_char_bboxes = [char for char in line_char_bboxes if not char['text'].isspace()]

            if len(trimmed_line_char_bboxes) == 0:
                continue

            # combine all of the found character bboxes on a given line and calculate their combined x0, x1, etc. values
            for char_bbox in trimmed_line_char_bboxes:
                x0 = min(char_bbox['x0'], x0)
                top = min(char_bbox['top'], top)
                y0 = min(char_bbox['y0'], y0)

                x1 = max(char_bbox['x1'], x1)
                bottom = max(char_bbox['bottom'], bottom)
                y1 = max(char_bbox['y1'], y1)

                start_offset = min(char_bbox['string_offset'], start_offset)
                end_offset = max(char_bbox['string_offset'], end_offset)

                if pdf_page_index is not None:
                    try:
                        assert pdf_page_index == char_bbox['page_number'] - 1
                    except AssertionError:
                        logger.warning(
                            "We don't support bounding boxes over page breaks yet, and will return the bounding box"
                            'on the first page of the match.'
                        )
                        break

                pdf_page_index = char_bbox['page_number'] - 1

            line_bbox = {
                'bottom': bottom,
                'page_index': pdf_page_index,
                'top': top,
                'x0': x0,
                'x1': x1,
                'y0': y0,
                'y1': y1,
                'start_offset': start_offset,
                'end_offset': end_offset + 1,
                'line_number': line_number,
            }

            # if we're being passed a bbox with an offset string, for example when saving a new annotation with a
            # custom string, keep it in the data structure so it can be returned and saved later
            if 'offset_string' in selection_bbox:
                line_bbox['offset_string'] = selection_bbox['offset_string']

            line_bboxes.append(line_bbox)

    # When receiving multiple selection bboxes, we want to merge those we can (consecutive ones)
    # into a single bbox.
    # For example, if I have the string "THIS IS SOME TEXT", and I want to click-select "THIS",
    # "SOME" and "TEXT", the function should return two bboxes: one for "THIS" and one for
    # "SOME TEXT" (since they are consecutive).
    # The way we do this is by sorting our `line_bboxes` by `start_offset` and checking for
    # characters between two consecutive normalized bboxes; if there isn't any, we assume they
    # can be merged, otherwise they stay the same.

    line_bboxes.sort(key=lambda k: k['start_offset'])

    # initialize the list where merged bboxes will be saved
    merged_bboxes = []

    for line_bbox in line_bboxes:
        # if this is the first bbox we're checking, just add it
        if len(merged_bboxes) < 1:
            merged_bboxes.append(line_bbox)
            continue

        # if the last bbox we added has a different line number, it should not be merged
        if merged_bboxes[-1]['line_number'] != line_bbox['line_number']:
            merged_bboxes.append(line_bbox)
            continue

        # determine whether there are characters between this bbox's start and the previous bbox's end.
        # the generator returns as soon as it finds a positive result.
        has_chars_in_between = any(
            True
            for key, char in doc_bbox.items()
            # the index of a doc_bbox char is its offset, so we can use the start/end offset from the
            # bboxes we're comparing to see if there are chars between them
            if (merged_bboxes[-1]['end_offset'] - 1) < key < line_bbox['start_offset']
            # a "space" char counts as no character for this check
            and not char['text'].isspace()
        )

        # if there are no chars in between, merge this bbox into the previous one
        if not has_chars_in_between:
            # we know that there are no characters between these two bounding boxes, but what about spaces?
            # we calculate the amt of spaces between these two bounding boxes so it can be replicated when building the
            # offset_string.
            amount_of_spaces_in_between = line_bbox['start_offset'] - merged_bboxes[-1]['end_offset']

            merged_bboxes[-1]['x0'] = min(merged_bboxes[-1]['x0'], line_bbox['x0'])
            merged_bboxes[-1]['top'] = min(merged_bboxes[-1]['top'], line_bbox['top'])
            merged_bboxes[-1]['y0'] = min(merged_bboxes[-1]['y0'], line_bbox['y0'])
            merged_bboxes[-1]['x1'] = max(merged_bboxes[-1]['x1'], line_bbox['x1'])
            merged_bboxes[-1]['bottom'] = max(merged_bboxes[-1]['bottom'], line_bbox['bottom'])
            merged_bboxes[-1]['y1'] = max(merged_bboxes[-1]['y1'], line_bbox['y1'])
            merged_bboxes[-1]['start_offset'] = min(merged_bboxes[-1]['start_offset'], line_bbox['start_offset'])
            merged_bboxes[-1]['end_offset'] = max(merged_bboxes[-1]['end_offset'], line_bbox['end_offset'])

            # if both bboxes have the offset string property, merge them
            if merged_bboxes[-1].get('offset_string') and line_bbox.get('offset_string'):
                merged_bboxes[-1]['offset_string'] += ' ' * amount_of_spaces_in_between + line_bbox['offset_string']

        # otherwise, just add the bbox to the list
        else:
            merged_bboxes.append(line_bbox)

    # if we're passed a doc_text, add an offset_string to each bbox
    if doc_text:
        for bbox in merged_bboxes:
            offset_string = doc_text[bbox['start_offset'] : bbox['end_offset']]
            # don't override offset_string if already set
            if not bbox.get('offset_string'):
                bbox['offset_string'] = offset_string
            bbox['offset_string_original'] = offset_string

    return merged_bboxes


def get_sdk_version():
    """Get a version of current Konfuzio SDK used."""
    return pkg_resources.get_distribution('konfuzio-sdk').version


def get_spans_from_bbox(selection_bbox: 'Bbox') -> List['Span']:
    """Get a list of Spans for all the text contained within a Bbox."""
    from konfuzio_sdk.data import Span

    selected_bboxes = [
        char_bbox for _, char_bbox in selection_bbox.page.get_bbox().items() if selection_bbox.check_overlap(char_bbox)
    ]
    selected_bboxes = sorted(selected_bboxes, key=lambda x: x['char_index'])

    # iterate over each line_number (or bottom, depending on group_by) and all the character
    # bboxes that have the same line_number (or bottom)
    spans = []
    if selected_bboxes:
        # condition for backward compatibility
        line_key = 'line_index' if 'line_index' in selected_bboxes[0].keys() else 'line_number'
        for _, line_char_bboxes in itertools.groupby(selected_bboxes, lambda x: x[line_key]):
            # remove space chars from the line selection, so they don't interfere with the merging of bboxes
            # (a bbox should never start with a space char)
            trimmed_line_char_bboxes = [char for char in line_char_bboxes if not char['text'].isspace()]

            if len(trimmed_line_char_bboxes) == 0:
                continue

            # combine all the found character bboxes on a given line and calculate their combined x0, x1, etc. values
            start_offset = min(char_bbox['char_index'] for char_bbox in trimmed_line_char_bboxes)
            end_offset = max(char_bbox['char_index'] for char_bbox in trimmed_line_char_bboxes)
            span = Span(start_offset=start_offset, end_offset=end_offset + 1, document=selection_bbox.page.document)
            spans.append(span)

    return spans


def log_subprocess_output(pipe, breaking_point: str = None):
    for line in iter(pipe.readline, b''):  # b'\n'-separated lines
        logging.info('got line from subprocess: %r', line)
        if breaking_point in str(line):
            break


def logging_from_subprocess(process, breaking_point: str = None):
    """Get detailed logs from the subprocess for easier debugging."""
    with process.stdout:
        log_subprocess_output(pipe=process.stdout, breaking_point=breaking_point)
