"""Test interfaces created for containerization of Information Extraction AIs."""

import io
import subprocess
import time
import unittest

import pytest
import requests
from PIL import Image

from konfuzio_sdk.bento.extraction.schemas import ExtractRequest20240117, ExtractResponse20240117
from konfuzio_sdk.bento.extraction.utils import convert_document_to_request, convert_response_to_annotations
from konfuzio_sdk.data import Document, Project
from konfuzio_sdk.settings_importer import is_dependency_installed
from konfuzio_sdk.tokenizer.regex import WhitespaceTokenizer
from konfuzio_sdk.trainer.information_extraction import RFExtractionAI
from konfuzio_sdk.utils import logging_from_subprocess
from tests.variables import OFFLINE_PROJECT


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
            ['bentoml', 'serve', '-p 3001', cls.bento_name], stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        time.sleep(5)
        print('served bento')
        cls.request_url = 'http://0.0.0.0:3001/extract'

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
                    'page_number': bbox['page_number'],
                    'text': bbox['text'],
                }
                for bbox_id, bbox in self.test_document.get_bbox().items()
            },
            'pages': [
                {
                    'number': page.number,
                    'original_size': page._original_size,
                    'image': page.image_bytes,
                    'segmentation': page._segmentation,
                }
                for page in self.test_document.pages()
            ],
        }
        response = requests.post(url=self.request_url, json=data)
        logging_from_subprocess(process=self.bento_process, breaking_point='status=')
        assert len(response.json()['annotation_sets']) == 4
        assert sum([len(element['annotations']) for element in response.json()['annotation_sets']]) == 19

    def test_extract_converted(self):
        """Test that only a schema-adhering response is accepted by extract method of service."""
        prepared = convert_document_to_request(document=self.test_document, schema=ExtractRequest20240117)
        response = requests.post(url=self.request_url, json=prepared.dict())
        # logging_from_subprocess(process=bento_process, breaking_point='status=')
        assert len(response.json()['annotation_sets']) == 4
        assert sum([len(element['annotations']) for element in response.json()['annotation_sets']]) == 19
        response_schema = ExtractResponse20240117(annotation_sets=response.json()['annotation_sets'])
        empty_document = Document(project=self.project, category=self.pipeline.category)
        document_with_annotations = convert_response_to_annotations(response=response_schema, document=empty_document)
        assert len(document_with_annotations.annotations(use_correct=False)) == 19

    def test_extract_with_image_bytes(self):
        """Test that serializing image bytes works."""

        def to_bytes(image_path: str) -> bytes:
            with Image.open(image_path) as img:
                with io.BytesIO() as output:
                    img.save(output, format='PNG')
                    image_bytes = output.getvalue()
            return image_bytes

        pages = [
            {
                'number': page.number,
                'original_size': page._original_size,
                'image': to_bytes(page.image_path).hex(),
                'segmentation': page._segmentation,
            }
            for page in self.test_document.pages()
        ]
        data = {
            'text': self.test_document.text,
            'bboxes': {
                str(bbox_id): {
                    'x0': bbox['x0'],
                    'x1': bbox['x1'],
                    'y0': bbox['y0'],
                    'y1': bbox['y1'],
                    'page_number': bbox['page_number'],
                    'text': bbox['text'],
                }
                for bbox_id, bbox in self.test_document.get_bbox().items()
            },
            'pages': pages,
        }
        response = requests.post(url=self.request_url, json=data)
        # logging_from_subprocess(process=self.bento_process, breaking_point='status=')
        assert len(response.json()['annotation_sets']) == 4
        assert sum([len(element['annotations']) for element in response.json()['annotation_sets']]) == 19

    def test_wrong_input(self):
        """Test that it's impossible to send a request with a structure not adhering to schema."""
        data = {'pages': 1234, 'new_field': 'ffff'}
        response = requests.post(url=self.request_url, data=data)
        assert response.status_code == 400
        assert 'validation error' in response.text

    def test_service_error(self):
        """Test that Python errors occurring in the service are caught and returned as a response."""
        data = {'text': 'test', 'bboxes': {}, 'pages': []}
        response = requests.post(url=self.request_url, json=data)
        assert response.status_code == 500
        assert response.json()['error'] == 'NotImplementedError'
        assert len(response.json()['traceback']) > 0

    @classmethod
    def tearDownClass(cls) -> None:
        """Kill process."""
        cls.bento_process.kill()
