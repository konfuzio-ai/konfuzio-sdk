"""Validate utils functions."""
import os
import unittest

import pytest
from konfuzio_sdk import IMAGE_FILE, PDF_FILE, OFFICE_FILE
from konfuzio_sdk.utils import (
    get_id,
    get_timestamp,
    is_file,
    get_file_type,
    load_image,
    convert_to_bio_scheme,
    map_offsets,
    get_sentences,
)

TEST_STRING = "sample string"
FOLDER_ROOT = os.path.dirname(os.path.realpath(__file__))
TEST_PDF_FILE = os.path.join(FOLDER_ROOT, 'test_data/pdf/1_test.pdf')
TEST_IMAGE_FILE = os.path.join(FOLDER_ROOT, 'test_data/images/Konfuzio logo square.png')
TEST_ZIP_FILE = os.path.join(FOLDER_ROOT, 'test_data/file-sample.docx')


@pytest.mark.local
class TestUtils(unittest.TestCase):
    """Test utility functions."""

    def test_get_id(self):
        """Test if the returned unique id is an instance of String."""
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
        """Test convertion to BIO scheme."""
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
        """Test convertion to BIO scheme without annotations."""
        text = "Hello, it's Konfuzio."
        annotations = []
        converted_text = convert_to_bio_scheme(text=text, annotations=annotations)
        assert len(converted_text) == 6
        assert all([annot[1] == 'O' for annot in converted_text])

    def test_convert_to_bio_scheme_no_text(self):
        """Test convertion to BIO scheme without text."""
        text = ''
        annotations = [(12, 20, 'Organization')]
        converted_text = convert_to_bio_scheme(text=text, annotations=annotations)
        self.assertIsNone(converted_text)

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

        map = map_offsets(characters_bboxes)

        expected_map = {0: 1000, 1: 1002}

        assert map == expected_map

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
