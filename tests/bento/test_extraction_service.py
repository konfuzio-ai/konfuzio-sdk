"""Test interfaces created for containerization of Information Extraction AIs."""
import subprocess
import time
import unittest

import requests

from konfuzio_sdk.data import Project
from konfuzio_sdk.tokenizer.regex import WhitespaceTokenizer
from konfuzio_sdk.trainer.information_extraction import RFExtractionAI
from konfuzio_sdk.utils import logging_from_subprocess
from tests.variables import OFFLINE_PROJECT


class TestExtractionAIBento(unittest.TestCase):
    """Test that Bento-based Extraction AI works."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create a model and its Bento instance of Extraction AI."""
        cls.pipeline = RFExtractionAI()
        cls.project = Project(id_=None, project_folder=OFFLINE_PROJECT)
        cls.pipeline.tokenizer = WhitespaceTokenizer()
        cls.pipeline.category = cls.project.get_category_by_id(id_=63)
        cls.pipeline.documents = cls.pipeline.category.documents()
        cls.pipeline.test_documents = cls.pipeline.category.test_documents()
        cls.pipeline.df_train, cls.pipeline.label_feature_list = cls.pipeline.feature_function(
            documents=cls.pipeline.documents, require_revised_annotations=False
        )
        cls.pipeline.fit()
        cls.test_document = cls.project.get_document_by_id(44823)
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
