"""Validate utils functions."""
import os
import unittest

import pytest
from konfuzio_sdk import IMAGE_FILE, PDF_FILE, OFFICE_FILE
from konfuzio_sdk.data import Project
from konfuzio_sdk.utils import (
    get_id,
    get_timestamp,
    is_file,
    get_file_type,
    load_image,
    convert_to_bio_scheme,
    map_offsets,
    get_sentences,
    amend_file_name,
    does_not_raise,
    get_missing_offsets,
)

TEST_STRING = "sample string"
FOLDER_ROOT = os.path.dirname(os.path.realpath(__file__))
TEST_PDF_FILE = os.path.join(FOLDER_ROOT, 'test_data', 'pdf.pdf')
TEST_IMAGE_FILE = os.path.join(FOLDER_ROOT, 'test_data', 'png.png')
TEST_ZIP_FILE = os.path.join(FOLDER_ROOT, 'test_data', 'docx.docx')


@pytest.mark.local
class TestUtils(unittest.TestCase):
    """Test utility functions."""

    def test_get_id(self):
        """Test if the returned unique id_ is an instance of String."""
        assert isinstance(get_id(TEST_STRING), int)

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
        text = "Hello, it's Konfuzio."
        annotations = [(12, 20, 'Organization')]
        converted_text = convert_to_bio_scheme(text=text, annotations=annotations)
        assert converted_text == [
            ('Hello', 'O'),
            (',', 'O'),
            ('it', 'O'),
            ("'s", 'O'),
            ('Konfuzio', 'B-Organization'),
            ('.', 'O'),
        ]

    def test_convert_to_bio_scheme_no_annotations(self):
        """Test conversion to BIO scheme without annotations."""
        text = "Hello, it's Konfuzio."
        annotations = []
        converted_text = convert_to_bio_scheme(text=text, annotations=annotations)
        assert len(converted_text) == 6
        assert all([annotation[1] == 'O' for annotation in converted_text])

    def test_convert_to_bio_scheme_no_text(self):
        """Test conversion to BIO scheme without text."""
        text = ''
        annotations = [(12, 20, 'Organization')]
        converted_text = convert_to_bio_scheme(text=text, annotations=annotations)
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


file_name_append_data = [
    # text embeddings all over the text
    ('/tmp/text_embeddings_0639187398.pdf', '/tmp/text_embeddings_0639187398_ocr.pdf', does_not_raise()),
    # text embeddings only on some pages of the text
    ('only_some_pages_have_embeddings.tiff', 'only_some_pages_have_embeddings_ocr.tiff', does_not_raise()),
    # two sots in a file name
    ('only_some_pages._have_embeddings.tiff', 'only_some_pages._have_embeddings_ocr.tiff', does_not_raise()),
    ('2022-02-13 19:23:06.168728.tiff', '2022-02-13 19-23-06.168728_ocr.tiff', does_not_raise()),
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


@pytest.mark.parametrize("file_path, expected_result, expected_error", file_name_append_data)
def test_append_text_to_filename(file_path, expected_result, expected_error):
    """Test if we detect the correct file name."""
    with expected_error:
        assert amend_file_name(file_path, 'ocr') == expected_result


def test_corrupted_name():
    """Test to convert an invalide file name to a valid file name."""
    assert amend_file_name('2022-02-13 19:23:06.168728.tiff') == '2022-02-13 19-23-06.168728.tiff'


@pytest.mark.local
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
        assert missing_offsets == [(0, 65), (78, 158), (169, 170)]

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
        assert missing == [(0, 5), (7, 14)]

    def test_annotation_start(self):
        """Test to on a sequence which starts with an annotation."""
        missing = get_missing_offsets(start_offset=0, end_offset=10, annotated_offsets=[range(5, 7), range(0, 3)])
        assert missing == [(4, 4), (7, 10)]

    def test_annotation_ends(self):
        """Test to on a sequence which ends with an annotation."""
        missing = get_missing_offsets(start_offset=0, end_offset=10, annotated_offsets=[range(7, 10)])
        assert missing == [(0, 6)]

    def test_empty_annotations(self):
        """Test on an unlabeled sequence."""
        missing = get_missing_offsets(start_offset=0, end_offset=160, annotated_offsets=[])
        assert missing == [(0, 160)]

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
        assert missing == [(10, 65), (78, 158)]

    def test_annotation_start_start_later(self):
        """Test to on a sequence which starts with an annotation."""
        missing = get_missing_offsets(start_offset=2, end_offset=10, annotated_offsets=[range(5, 7), range(0, 3)])
        assert missing == [(4, 4), (7, 10)]

    def test_annotation_ends_start_later(self):
        """Test to on a sequence which ends with an annotation."""
        missing = get_missing_offsets(start_offset=2, end_offset=10, annotated_offsets=[range(7, 10)])
        assert missing == [(2, 6)]

    def test_empty_annotations_start_later(self):
        """Test on an unlabeled sequence."""
        missing = get_missing_offsets(start_offset=100, end_offset=160, annotated_offsets=[])
        assert missing == [(100, 160)]

    def test_fully_labeled_start_later(self):
        """Test on an labeled sequence."""
        missing = get_missing_offsets(start_offset=50, end_offset=160, annotated_offsets=[range(0, 160)])
        assert missing == []
