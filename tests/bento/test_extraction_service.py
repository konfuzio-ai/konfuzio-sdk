"""Test interfaces created for containerization of Information Extraction AIs."""
import subprocess
import time
import unittest

import pytest
import requests

from konfuzio_sdk.data import Project
from konfuzio_sdk.settings_importer import is_dependency_installed
from konfuzio_sdk.tokenizer.regex import WhitespaceTokenizer
from konfuzio_sdk.trainer.information_extraction import RFExtractionAI
from konfuzio_sdk.utils import logging_from_subprocess
from tests.variables import OFFLINE_PROJECT


# @pytest.mark.skip(reason='Too lengthy for regular test pipelines')
@pytest.mark.skipif(
    not is_dependency_installed('torch'),
    reason='Required dependencies not installed.',
)
class TestExtractionAIBento(unittest.TestCase):
    """Test that Bento-based Extraction AI works."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create a model and its Bento instance of Extraction AI."""
        cls.project = Project(id_=None, project_folder=OFFLINE_PROJECT)
        cls.test_document = cls.project.get_document_by_id(44823)
        cls.pipeline = RFExtractionAI()
        cls.pipeline.tokenizer = WhitespaceTokenizer()
        cls.pipeline.category = cls.project.get_category_by_id(id_=63)
        cls.pipeline.documents = cls.pipeline.category.documents()
        cls.pipeline.test_documents = cls.pipeline.category.test_documents()
        cls.pipeline.df_train, cls.pipeline.label_feature_list = cls.pipeline.feature_function(
            documents=cls.pipeline.documents, require_revised_annotations=False
        )
        cls.pipeline.fit()
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
            'bboxes': {
                str(bbox_id): {
                    'x0': bbox['x0'],
                    'x1': bbox['x1'],
                    'y0': bbox['y0'],
                    'y1': bbox['y1'],
                    'page': {
                        'number': bbox['page_number'],
                        'original_size': tuple(
                            self.test_document.get_page_by_index(bbox['page_number'] - 1)._original_size
                        ),
                        'image': self.test_document.get_page_by_index(bbox['page_number'] - 1).image_bytes,
                    },
                    'text': bbox['text'],
                }
                for bbox_id, bbox in self.test_document.get_bbox().items()
            },
            'pages': [
                {'number': page.number, 'original_size': page._original_size, 'image': page.image_bytes}
                for page in self.test_document.pages()
            ],
        }
        response = requests.post(url=self.request_url, json=data)
        logging_from_subprocess(process=self.bento_process, breaking_point='status=')
        assert len(response.json()['annotation_sets']) == 5
        assert sum([len(element['annotations']) for element in response.json()['annotation_sets']]) == 19

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
