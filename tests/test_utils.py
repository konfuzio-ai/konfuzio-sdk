"""Validate utils functions."""
import importlib.metadata
import os
import unittest

import pytest

from konfuzio_sdk import IMAGE_FILE, OFFICE_FILE, PDF_FILE
from konfuzio_sdk.data import Annotation, AnnotationSet, Bbox, Category, Document, Label, LabelSet, Project, Span
from konfuzio_sdk.utils import (
    amend_file_name,
    amend_file_path,
    convert_to_bio_scheme,
    does_not_raise,
    exception_or_log_error,
    get_bbox,
    get_file_type,
    get_id,
    get_missing_offsets,
    get_sdk_version,
    get_sentences,
    get_spans_from_bbox,
    get_timestamp,
    is_file,
    iter_before_and_after,
    load_image,
    map_offsets,
    normalize_memory,
    normalize_name,
    slugify,
)
from tests.variables import (
    OFFLINE_PROJECT,
    TEST_DOCUMENT_ID,
)

FOLDER_ROOT = os.path.dirname(os.path.realpath(__file__))
TEST_PDF_FILE = os.path.join(FOLDER_ROOT, 'test_data', 'pdf.pdf')
TEST_IMAGE_FILE = os.path.join(FOLDER_ROOT, 'test_data', 'png.png')
TEST_ZIP_FILE = os.path.join(FOLDER_ROOT, 'test_data', 'docx.docx')


class TestUtils(unittest.TestCase):
    """Test utility functions."""

    def test_exception_or_log_error(self):
        """Test switching to Log error or raise an exception."""
        with pytest.raises(NotImplementedError, match='test exception msg'):
            exception_or_log_error(msg='test exception msg', fail_loudly=True, exception_type=NotImplementedError)
        exception_or_log_error(msg='test log error msg', fail_loudly=False)

    def test_get_id(self):
        """Test if the returned unique id_ is an instance of String."""
        assert isinstance(get_id(), str)

    def test_get_timestamp(self):
        """Test if the returned timestamp is an instance of String."""
        assert isinstance(get_timestamp(), str)

    def test_is_file(self):
        """Test if_file function with empty files,existing files and not existing files."""
        open('new_empty_file.txt', 'a').close()
        self.assertEqual(is_file('new_empty_file', False), False)
        os.remove('new_empty_file.txt')
        self.assertEqual(is_file('not_existing_file.txt', False), False)
        self.assertEqual(is_file(TEST_PDF_FILE, False), True)

    def test_get_file_type(self):
        """Test if the returned filetype is correct."""
        self.assertEqual(get_file_type(TEST_PDF_FILE), PDF_FILE)
        self.assertEqual(get_file_type(TEST_IMAGE_FILE), IMAGE_FILE)
        self.assertEqual(get_file_type(TEST_ZIP_FILE), OFFICE_FILE)

    def test_load_image(self):
        """Test if the image is read correctly."""
        assert load_image(TEST_IMAGE_FILE) is not None

    def test_convert_to_bio_scheme(self):
        """Test conversion to BIO scheme."""
        project = Project(None)
        category = Category(project=project, id_=1)
        label_set = LabelSet(id_=2, project=project, categories=[category])
        label = Label(text='Organization', project=project, label_sets=[label_set])
        text = "Hello, it's Konfuzio."
        document = Document(project, text=text, category=category)
        annotation_set = AnnotationSet(id_=1, project=project, document=document, label_set=label_set)
        _ = Annotation(
            id_=20,
            document=document,
            label=label,
            label_set=label_set,
            annotation_set=annotation_set,
            spans=[Span(start_offset=12, end_offset=20)],
        )

        converted_text = convert_to_bio_scheme(document)

        assert converted_text == [
            ('Hello', 'O'),
            (',', 'O'),
            ('it', 'O'),
            ("'s", 'O'),
            ('Konfuzio', 'B-Organization'),
            ('.', 'O'),
        ]

    def test_convert_to_bio_scheme_no_annotations(self):
        """Test conversion to BIO scheme without Annotations."""
        proj = Project(None)
        category = Category(project=proj, id_=1)
        text = "Hello, it's Konfuzio."
        doc = Document(proj, text=text, category=category)

        converted_text = convert_to_bio_scheme(doc)
        assert len(converted_text) == 6
        assert all(annotation[1] == 'O' for annotation in converted_text)

    def test_convert_to_bio_scheme_no_text(self):
        """Test conversion to BIO scheme without text."""
        project = Project(None)
        category = Category(project=project, id_=1)
        label_set = LabelSet(id_=2, project=project, categories=[category])
        label = Label(text='Organization', project=project, label_sets=[label_set])
        text = ''
        document = Document(project, text=text, category=category)
        annotation_set = AnnotationSet(id_=1, project=project, document=document, label_set=label_set)
        _ = Annotation(
            id_=20,
            document=document,
            label=label,
            label_set=label_set,
            annotation_set=annotation_set,
            spans=[Span(start_offset=12, end_offset=20)],
        )
        converted_text = convert_to_bio_scheme(document)

        self.assertEqual(converted_text, [])

    def test_map_offsets(self):
        """Test creation of mapping between the position of the character and its offset."""
        characters_bboxes = [
            {
                'string_offset': '1000',
                'adv': 2.58,
                'bottom': 128.13,
                'doctop': 118.13,
                'fontname': 'GlyphLessFont',
                'height': 10.0,
                'line_number': 14,
                'object_type': 'char',
                'page_number': 1,
                'size': 10.0,
                'text': 'n',
                'top': 118.13,
                'upright': 1,
                'width': 2.58,
                'x0': 481.74,
                'x1': 484.32,
                'y0': 713.55,
                'y1': 723.55,
            },
            {
                'string_offset': '1002',
                'adv': 2.64,
                'bottom': 128.13,
                'doctop': 118.13,
                'fontname': 'GlyphLessFont',
                'height': 10.0,
                'line_number': 14,
                'object_type': 'char',
                'page_number': 1,
                'size': 10.0,
                'text': 'S',
                'top': 118.13,
                'upright': 1,
                'width': 2.64,
                'x0': 486.72,
                'x1': 489.36,
                'y0': 713.55,
                'y1': 723.55,
            },
        ]

        my_map = map_offsets(characters_bboxes)

        expected_map = {0: 1000, 1: 1002}

        assert my_map == expected_map

    def test_get_sentences(self):
        """Test get sentences."""
        text = 'Hello world. This is a test.'

        sentences = get_sentences(text, language='english')

        assert len(sentences) == 2
        assert sentences[0]['offset_string'] == 'Hello world.'
        assert sentences[0]['start_offset'] == 0
        assert sentences[0]['end_offset'] == 12
        assert sentences[1]['offset_string'] == 'This is a test.'
        assert sentences[1]['start_offset'] == 13
        assert sentences[1]['end_offset'] == 28

    def test_get_sentences_with_offsets_mapping(self):
        """Test get sentences while using a map for the offsets."""
        text = 'Hello world. This is a test.'

        offsets_map = {
            0: 0,
            1: 1,
            2: 1,
            3: 3,
            4: 4,
            5: 5,
            6: 6,
            7: 10,
            8: 11,
            9: 12,
            10: 13,
            11: 14,
            12: 15,
            13: 50,
            14: 51,
            15: 52,
            16: 53,
            17: 54,
            18: 55,
            19: 56,
            20: 57,
            21: 58,
            22: 59,
            23: 80,
            24: 81,
            25: 82,
            26: 83,
            27: 84,
            28: 85,
        }

        sentences = get_sentences(text, offsets_map=offsets_map, language='english')

        assert len(sentences) == 2
        assert sentences[0]['offset_string'] == 'Hello world.'
        assert sentences[0]['start_offset'] == 0
        assert sentences[0]['end_offset'] == 15
        assert sentences[1]['offset_string'] == 'This is a test.'
        assert sentences[1]['start_offset'] == 50
        assert sentences[1]['end_offset'] == 85

    def test_slugify(self):
        """Test slugifying a value with the default parameters."""
        value = 'Category / name 123456***:'
        assert slugify(value=value) == 'category-name-123456-'

    def test_normalize_name(self):
        """Test normalizing names for Konfuzio concepts."""
        value = 'Category/name'
        assert normalize_name(value) == 'Categoryname'


file_name_append_data = [
    # text embeddings all over the text
    ('/tmp/text_embeddings_0639187398.pdf', 'tmptext_embeddings_0639187398_ocr.pdf', does_not_raise()),
    # text embeddings only on some pages of the text
    ('only_some_pages_have_embeddings.tiff', 'only_some_pages_have_embeddings_ocr.tiff', does_not_raise()),
    # two dots in a file name
    ('only_some_pages._have_embeddings.tiff', 'only_some_pages_have_embeddings_ocr.tiff', does_not_raise()),
    ('2022-02-13 19:23:06.168728.tiff', '2022-02-13-19-23-06-168728_ocr.tiff', does_not_raise()),
    # empty file path
    (' ', False, pytest.raises(ValueError)),
    # Current file name is already too long, 255 chr
    (
        '1qgjpzndwlawckhpjzpvlwxhwqywsjkixlnphihvwlfxtifjvzbqajcjlxdfclbtmstievnepcxubvmgc'
        'uyhpvpujkinqahmxjxhbsdejuqvmzcsaqlmynykgaznembeuuhtjwzudsigfukdnkiatqpmgvxfsonthd'
        'kbisiojrkulngipzxojkxgetqhgrrnigucneirfxothiekhxplfbbjxlxxyohdavatzoxuultcthjmdtt'
        'qxyuvzfyddao',
        False,
        pytest.raises(OSError),
    ),
    # File name which will be generated is to long: 254 character
    (
        'qgjpzndwlawckhpjzpvlwxhwqywsjkixlnphihvwlfxtifjvzbqajcjlxdfclbtmstievnepcxubvmgcu'
        'yhpvpujkinqahmxjxhbsdejuqvmzcsaqlmynykgaznembeuuhtjwzudsigfukdnkiatqpmgvxfsonthdk'
        'bisiojrkulngipzxojkxgetqhgrrnigucneirfxothiekhxplfbbjxlxxyohdavatzoxuultcthjmdttq'
        'xyuvzfyddao',
        False,
        pytest.raises(OSError),
    ),
]


@pytest.mark.parametrize('file_path, expected_result, expected_error', file_name_append_data)
def test_append_text_to_filename(file_path, expected_result, expected_error):
    """Test if we detect the correct file name."""
    with expected_error:
        assert amend_file_name(file_path, 'ocr') == expected_result


file_path_append_data = [
    # text embeddings all over the text
    (
        '/tmp/text_embeddings_0639187398.pdf',
        os.path.join('/', 'tmp', 'text_embeddings_0639187398_ocr.pdf'),
        does_not_raise(),
    ),
    # text embeddings only on some pages of the text
    ('only_some_pages_have_embeddings.tiff', 'only_some_pages_have_embeddings_ocr.tiff', does_not_raise()),
    # two dots in a file name
    (
        'only/_some_pages._have_embeddings.tiff',
        os.path.join('only', '_some_pages_have_embeddings_ocr.tiff'),
        does_not_raise(),
    ),
    ('2022/-02-13 19:23:06.168728.tiff', os.path.join('2022', '-02-13-19-23-06-168728_ocr.tiff'), does_not_raise()),
]


@pytest.mark.parametrize('file_path, expected_result, expected_error', file_path_append_data)
def test_append_text_to_amend_file_path(file_path, expected_result, expected_error):
    """Test if we detect the correct file path."""
    with expected_error:
        assert amend_file_path(file_path, 'ocr') == expected_result


def test_corrupted_name():
    """Test to convert an invalide file name to a valid file name."""
    assert amend_file_name('2022-02-13 19:23:06.168728.tiff') == '2022-02-13-19-23-06-168728.tiff'


def test_get_bbox():
    """Test to raise Value Error if character cannot provide a bbox."""
    with pytest.raises(ValueError):
        get_bbox(bbox={}, start_offset=1, end_offset=5)


@pytest.mark.parametrize(
    'memory, expected',
    [
        (None, None),
        (50, 50),
        ('50MB', 50000000),
        ('50mb', 50000000),
        ('5GB', 5000000000),
        ('5gb', 5000000000),
        ('5KB', 5000),
        ('5kb', 5000),
        ('5gib', 5368709120),
        ('5mib', 5242880),
        ('5kib', 5120),
    ],
)
def test_normalize_memory(memory, expected):
    """Test the memory normalization method."""
    assert normalize_memory(memory) == expected


# @pytest.mark.skip('Need implementation of Line and Paragraph first.')
# class TestParagraphByLine(unittest.TestCase):
#     """Test paragraph splitting by line height."""
#
#     text = 'a\nb'
#     bboxes = {
#         '0': {'x0': 0, 'x1': 10, 'y0': 200, 'y1': 210, 'top': 10, 'bottom': 290, 'line_number': 1, 'page_number': 1},
#         '2': {'x0': 0, 'x1': 10, 'y0': 0, 'y1': 10, 'top': 210, 'bottom': 90, 'line_number': 3, 'page_number': 1},
#     }
#     invalid_bboxes = {
#         '0': {'x0': 0, 'x1': 10, 'y0': 0, 'y1': 10, 'top': 210, 'bottom': 90, 'line_number': 1, 'page_number': 1},
#         '2': {'x0': 0, 'x1': 10, 'y0': 200, 'y1': 210, 'top': 10, 'bottom': 290, 'line_number': 3, 'page_number': 1},
#     }
#
#     def test_get_paragraphs_by_line_space(self):
#         """Test split paragraphs by line space."""
#         paragraphs = get_paragraphs_by_line_space(text=self.text, bbox=self.bboxes)
#         assert len(paragraphs) == 1  # One page in document.
#
#         paragraph_first_page = paragraphs[0]
#         assert len(paragraph_first_page) == 2  # Two paragraphs on first page.
#
#         assert len(paragraph_first_page[0]) == 1
#         assert paragraph_first_page[0][0]['start_offset'] == 0
#         assert paragraph_first_page[0][0]['end_offset'] == 1
#
#         assert len(paragraph_first_page[1]) == 1
#         assert paragraph_first_page[1][0]['start_offset'] == 2
#         assert paragraph_first_page[1][0]['end_offset'] == 3
#
#         assert len(paragraph_first_page) == 2
#
#     def test_get_paragraphs_by_line_space_custom_height(self):
#         """Test split paragraphs by line space."""
#         # Low height for splitting will result in a single paragraph.
#         paragraphs = get_paragraphs_by_line_space(text=self.text, bbox=self.bboxes, height=500)
#         assert len(paragraphs) == 1  # One page in document.
#
#         paragraph_first_page = paragraphs[0]
#         assert len(paragraph_first_page) == 1  # Two paragraphs on first page.
#
#         assert paragraph_first_page[0][0]['start_offset'] == 0
#         assert paragraph_first_page[0][0]['end_offset'] == 1
#         assert paragraph_first_page[0][1]['start_offset'] == 2
#         assert paragraph_first_page[0][1]['end_offset'] == 3
#
#     def test_get_paragraphs_by_line_space_with_invalid_bbox(self):
#         """Test split paragraphs by line space."""
#         # Low height for splitting will result in a single paragraph.
#         with self.assertRaises(ValueError):
#             _ = get_paragraphs_by_line_space(text=self.text, bbox=self.invalid_bboxes)


class TestMissingOffsets(unittest.TestCase):
    """Test to detect not labeled sequences."""

    def test_find_missing_characters(self):
        """Find the character offsets that are not annotated."""
        prj = Project(46)
        doc = prj.get_document_by_id(44823)
        offsets = []
        for annotation in doc.annotations(start_offset=0, end_offset=2000):
            for span in annotation.spans:
                offsets.append(range(span.start_offset, span.end_offset))

        # character 66:78 and 159:169 belong to a multiline annotation, so it spans
        # [66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168]
        missing_offsets = get_missing_offsets(start_offset=0, end_offset=170, annotated_offsets=offsets)
        assert missing_offsets == [range(0, 66), range(78, 159), range(169, 170)]

    def test_find_missing_spans_unordered(self):
        """Find the character offsets independent of the order."""
        ordered = get_missing_offsets(
            start_offset=0, end_offset=170, annotated_offsets=[range(66, 78), range(159, 169)]
        )
        unordered = get_missing_offsets(
            start_offset=0, end_offset=170, annotated_offsets=[range(159, 169), range(66, 78)]
        )
        assert ordered == unordered

    def test_overlapping_annotation(self):
        """Test to incorporate a labeled sequence which is only partly included."""
        missing = get_missing_offsets(start_offset=0, end_offset=16, annotated_offsets=[range(15, 17), range(6, 7)])
        assert missing == [range(0, 6), range(7, 15)]

    def test_annotation_start(self):
        """Test to on a sequence which starts with an annotation."""
        missing = get_missing_offsets(start_offset=0, end_offset=10, annotated_offsets=[range(5, 7), range(0, 3)])
        assert missing == [range(3, 5), range(7, 10)]

    def test_annotation_ends(self):
        """
        Test to on a sequence which ends with an annotation.

        :Example:

        # >>> "0123456789"[0:7]
        '0123456'
        # >>> "0123456789"[7:10]
        '789'
        """
        missing = get_missing_offsets(start_offset=0, end_offset=10, annotated_offsets=[range(7, 10)])
        assert missing == [range(0, 7)]

    def test_empty_annotations(self):
        """Test on an unlabeled sequence."""
        missing = get_missing_offsets(start_offset=0, end_offset=160, annotated_offsets=[])
        assert missing == [range(0, 160)]

    def test_fully_labeled(self):
        """Test on an labeled sequence."""
        missing = get_missing_offsets(start_offset=0, end_offset=160, annotated_offsets=[range(0, 160)])
        assert missing == []

    def test_find_missing_spans_unordered_start_later(self):
        """Find the character offsets independent of the order."""
        ordered = get_missing_offsets(
            start_offset=10, end_offset=170, annotated_offsets=[range(66, 78), range(159, 169)]
        )
        unordered = get_missing_offsets(
            start_offset=10, end_offset=170, annotated_offsets=[range(159, 169), range(66, 78)]
        )
        assert ordered == unordered

    def test_overlapping_annotation_start_later(self):
        """Test to incorporate a labeled sequence which is only partly included."""
        missing = get_missing_offsets(
            start_offset=10, end_offset=160, annotated_offsets=[range(159, 169), range(66, 78)]
        )
        assert missing == [range(10, 66), range(78, 159)]

    def test_annotation_start_start_later(self):
        """Test to on a sequence which starts with an annotation."""
        missing = get_missing_offsets(start_offset=2, end_offset=10, annotated_offsets=[range(5, 7), range(0, 3)])
        assert missing == [range(3, 5), range(7, 10)]

    def test_annotation_ends_start_later(self):
        """Test to on a sequence which ends with an annotation."""
        missing = get_missing_offsets(start_offset=2, end_offset=10, annotated_offsets=[range(7, 10)])
        assert missing == [range(2, 7)]

    def test_empty_annotations_start_later(self):
        """Test on an unlabeled sequence."""
        missing = get_missing_offsets(start_offset=100, end_offset=160, annotated_offsets=[])
        assert missing == [range(100, 160)]

    def test_fully_labeled_start_later(self):
        """Test on an labeled sequence."""
        missing = get_missing_offsets(start_offset=50, end_offset=160, annotated_offsets=[range(0, 160)])
        assert missing == []

    def test_one_character_missing(self):
        """Test on an labeled sequence."""
        missing = get_missing_offsets(start_offset=0, end_offset=10, annotated_offsets=[range(0, 5), range(6, 10)])
        assert missing == [range(5, 6)]


def test_iter_before_and_after():
    """Test to get before and after element."""
    for before, i, after in iter_before_and_after([1, 2, 3, 4, 5, 6]):
        if before:
            assert before + 1 == i
        elif after:
            assert after - 1 == i


def test_get_spans_from_bbox():
    """Test to get Spans in a bounding box."""
    project = Project(id_=None, project_folder=OFFLINE_PROJECT)
    document = project.get_document_by_id(document_id=TEST_DOCUMENT_ID)
    page = document.get_page_by_index(0)
    bbox = Bbox(x0=0, y0=0, x1=500, y1=100, page=page)

    spans = get_spans_from_bbox(selection_bbox=bbox)

    assert len(spans) == 7
    assert spans[0].start_offset == 3588
    assert spans[0].end_offset == 3677
    assert spans[6].start_offset == 4322
    assert spans[6].end_offset == 4427

    bbox_2 = Bbox(x0=300, y0=400, x1=400, y1=550, page=page)
    spans_2 = get_spans_from_bbox(selection_bbox=bbox_2)

    assert len(spans_2) == 2
    assert spans_2[0].start_offset == 1694
    assert spans_2[0].end_offset == 1717
    assert spans_2[1].start_offset == 1926
    assert spans_2[1].end_offset == 1932


def test_get_sdk_version():
    """Test to get a current SDK version."""
    version = get_sdk_version()
    assert isinstance(version, str)
    assert version == importlib.metadata.version('konfuzio_sdk')
