import os
import unittest

import pytest

from konfuzio_sdk import IMAGE_FILE, PDF_FILE, OFFICE_FILE
from konfuzio_sdk.utils import get_id, get_timestamp, is_file, get_file_type, load_image

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
