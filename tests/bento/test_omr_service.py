"""Test interfaces created for containerization of Information Extraction AIs."""
import subprocess
import time
import unittest

import pytest
import requests

from konfuzio_sdk.data import Project
from konfuzio_sdk.settings_importer import is_dependency_installed
from konfuzio_sdk.trainer.omr import CheckboxDetector
from konfuzio_sdk.utils import logging_from_subprocess
from tests.variables import OFFLINE_PROJECT


@pytest.mark.skip(reason='Testing of the components requires starting a subprocess')
@pytest.mark.skipif(
    not is_dependency_installed('torch'),
    reason='Required dependencies not installed.',
)
class TestOMRCheckboxBento(unittest.TestCase):
    """Test that Bento-based OMR checkbox detector works."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create a model and its Bento instance of Extraction AI."""
        cls.pipeline = CheckboxDetector()
        cls.project = Project(id_=None, project_folder=OFFLINE_PROJECT)
        bento, path = cls.pipeline.save_bento()
        cls.bento_name = bento.tag.name + ':' + bento.tag.version
        cls.bento_process = subprocess.Popen(
            ['bentoml', 'serve', cls.bento_name], stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        time.sleep(5)
        print('served bento')
        cls.request_url = 'http://0.0.0.0:3000/extract'

    def test_extract(self):
        """Test that only a schema-adhering response is accepted by extract method of service."""
        data = {
            'text': self.test_document.text,
            'pages': [
                {
                    'number': 1,
                    'original_size': [self.test_document.pages()[0].width, self.test_document.pages()[0].height],
                    'image': None,
                }
            ],
            'bboxes': self.test_document.get_bbox(),
        }
        response = requests.post(url=self.request_url, json=data)
        logging_from_subprocess(process=self.bento_process, breaking_point='status=')
        assert len(response.json()) == 5
        assert sum([len(element['annotations']) for element in response.json()]) == 19

    def test_wrong_input(self):
        """Test that it's impossible to send a request with a structure not adhering to schema."""
        data = {'pages': 1234, 'new_field': 'ffff'}
        responses = requests.post(url=self.request_url, data=data)
        assert responses.status_code == 400
        assert 'Invalid JSON' in responses.text

    @classmethod
    def tearDownClass(cls) -> None:
        """Kill process."""
        cls.bento_process.kill()


if __name__ == '__main__':
    tester = TestOMRCheckboxBento()
    tester.setUpClass()
    tester.test_extract()
