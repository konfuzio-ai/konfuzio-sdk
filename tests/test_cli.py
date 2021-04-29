"""Validate cli functions."""

import logging
import os
import unittest
from unittest.mock import patch

import pytest

from konfuzio_sdk.cli import init_env, data, init_settings

logger = logging.getLogger(__name__)
TEST_DOCUMENT = 44823
ENV_FILE_EXAMPLE = '''KONFUZIO_HOST = https://app.konfuzio.com
KONFUZIO_USER = test@gmail.com
KONFUZIO_TOKEN = ""
KONFUZIO_DATA_FOLDER = data
KONFUZIO_PROJECT_ID = 46'''


class TestCLItoDownloadData(unittest.TestCase):
    """Test CLI functions."""

    @pytest.mark.skip(reason='Needs to generate a token with real credentials.')
    @patch("builtins.input", side_effect=['test_email@gmail.com', 'https://app.konfuzio.com', 'data', '46'])
    @patch("getpass.getpass", side_effect=['password'])
    @pytest.mark.local
    def test_init_env(self, mock_input1, mock_pass):
        """Test the generation of .env file."""
        generated_env = init_env(os.getcwd())
        self.assertEqual(generated_env, ENV_FILE_EXAMPLE)
        os.remove('.env')

    @pytest.mark.local
    def test_init_settings(self):
        """Test the generation of settings.py file."""
        init_settings(os.getcwd())
        assert os.path.isfile('settings.py')
        os.remove('settings.py')

    def test_data(self):
        """
        Test the creation of folders for a specific document.

        Creates temporarily a settings.py in the tests folder.
        """
        data()
        assert os.path.isdir(os.path.join('data/pdf/44823'))
