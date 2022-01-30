"""Validate cli functions."""
#
# import logging
# import os
# import unittest
# from unittest.mock import patch
#
# import pytest
#
# from konfuzio_sdk.cli import init_env, data, init_settings
#
# logger = logging.getLogger(__name__)
# TEST_DOCUMENT = 44823
# ENV_FILE_EXAMPLE = '''KONFUZIO_HOST = https://app.konfuzio.com
# KONFUZIO_USER = test@gmail.com
# KONFUZIO_TOKEN = ""
# KONFUZIO_DATA_FOLDER = data
# KONFUZIO_PROJECT_ID = 46'''
#
#
# class TestCLItoDownloadData(unittest.TestCase):
#     """Test CLI functions."""
#
#     def test_data(self):
#         """
#         Test the creation of folders for a specific document.
#
#         Creates temporarily a settings.py in the tests folder.
#         """
#         data()
#         assert os.path.isdir(os.path.join('data/pdf/44823'))
