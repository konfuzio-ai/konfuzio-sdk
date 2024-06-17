"""Test interfaces created for containerization of Information Extraction AIs."""

import base64
import os
import subprocess
import time
import unittest
from io import BytesIO

import numpy as np
import pytest
import requests
from PIL import Image

from konfuzio_sdk.bento.extraction.schemas import CheckboxRequest20240523, CheckboxResponse20240523
from konfuzio_sdk.bento.extraction.utils import convert_document_to_request
from konfuzio_sdk.data import Project
from konfuzio_sdk.settings_importer import is_dependency_installed
from konfuzio_sdk.trainer.omr import CheckboxDetector


# @pytest.mark.skip(reason='Testing of the components requires starting a subprocess')
@pytest.mark.skipif(not is_dependency_installed('torch'), reason='Required dependencies not installed.')
class TestOMRCheckboxBento(unittest.TestCase):
    """Test that Bento-based OMR checkbox detector works."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create a model and its Bento instance of Extraction AI."""

        # Define used schemas
        cls.RequestSchema = CheckboxRequest20240523
        cls.ResponseSchema = CheckboxResponse20240523
        # Initialize pipeline
        cls.pipeline = CheckboxDetector()
        # Save and build Bento
        bento, path = cls.pipeline.save_bento()
        cls.bento_name = bento.tag.name + ':' + bento.tag.version

        # Initialize and create test data

        cls.project = Project(id_=15217, update=True)
        cls.test_doc = cls.project.get_document_by_id(5930563)  # musterformular-by.pdf

        dummy_image = Image.fromarray(np.full((100, 100, 3), (255, 0, 0), dtype=np.uint8))
        buffered = BytesIO()
        dummy_image.save(buffered, format='PNG')
        encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        cls.test_request = {
            'pages': [
                {'page_id': 8795115, 'width': 595, 'height': 841, 'image': encoded_image},
                {'page_id': 8795116, 'width': 595, 'height': 841, 'image': encoded_image},
            ],
            'annotations': [
                {'page_id': 8795115, 'annotation_id': 36015161, 'bbox': {'x0': 114, 'x1': 196, 'y0': 761, 'y1': 769}},
                {'page_id': 8795115, 'annotation_id': 36015153, 'bbox': {'x0': 114, 'x1': 164, 'y0': 733, 'y1': 744}},
            ],
        }
        cls.test_response = {
            'metadata': [
                {
                    'annotation_id': 36015161,
                    'checkbox': {
                        'is_checked': True,
                        'bbox': {'x0': 70, 'x1': 82, 'y0': 758, 'y1': 771},
                        'confidence': 0.952959418296814,
                    },
                },
                {
                    'annotation_id': 36015153,
                    'checkbox': {
                        'is_checked': False,
                        'bbox': {'x0': 70, 'x1': 82, 'y0': 733, 'y1': 745},
                        'confidence': 0.9631607532501221,
                    },
                },
            ]
        }
        # Serve Bento

        # kill process on port 3000 if exists
        os.system('lsof -t -i:3000 | xargs kill -9')

        cls.bento_process = subprocess.Popen(
            ['bentoml', 'serve', cls.bento_name], stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        time.sleep(5)
        print('served bento')

        cls.request_url = 'http://0.0.0.0:3000/extract'

    def test_testdocument(self):
        """Test that the test document contains all relevant Annotations and Checkboxes for the following tests."""
        assert self.test_doc is not None
        assert len(self.test_doc.pages()) == 2
        # Check, that the test document contains 17 Annotations with attribute 'is_linked_to_checkbox'
        # TODO: Uncomment this line when the linked_to_checkbox attribute is implemented by Server
        # assert len([a for a in self.test_doc.annotations() if a.label.is_linked_to_checkbox]) == 17, f"{self.test_doc} contains {len(self.test_doc.annotations())} Annotations with attribute 'is_linked_to_checkbox', it should be 17."

    def test_request_schema(self):
        request_schema = self.RequestSchema(**self.test_request)
        assert isinstance(request_schema, self.RequestSchema)

    def test_response_schema(self):
        response_schema = self.ResponseSchema(**self.test_response)
        assert isinstance(response_schema, self.ResponseSchema)

    def test_convert_document_to_request(self):
        """Test that the conversion of a document to a request adheres to the schema."""
        req = convert_document_to_request(document=self.test_doc, schema=self.RequestSchema)

        # check request pages
        assert len(req.pages) == 2
        assert req.pages[0].page_id == self.test_doc.pages()[0].id_
        assert req.pages[0].width == int(self.test_doc.pages()[0].width)
        assert req.pages[0].height == int(self.test_doc.pages()[0].height)
        assert isinstance(req.pages[0].image, str)

        # check request annotations
        assert req.annotations[0].page_id == self.test_doc.pages()[0].id_
        assert req.annotations[0].annotation_id == self.test_doc.annotations()[0].id_
        assert req.annotations[0].bbox.x0 == int(self.test_doc.annotations()[0].bboxes[0]['x0'])
        assert req.annotations[0].bbox.x1 == int(self.test_doc.annotations()[0].bboxes[0]['x1'])
        assert req.annotations[0].bbox.y0 == int(self.test_doc.annotations()[0].bboxes[0]['y0'])
        assert req.annotations[0].bbox.y1 == int(self.test_doc.annotations()[0].bboxes[0]['y1'])

        # check for number of annotations which are linked to checkboxes
        assert (
            len(req.annotations) == 17
        ), f'The prepared request contains {len(req.annotations)} Annotations, but should be 17.'

    def test_detector_dummydata(self):
        """Test that only a schema-adhering response is accepted by extract method of service."""
        # Create a dummy image (e.g., a 100x100 red image)
        response = requests.post(url=self.request_url, json=self.test_request)
        assert response.status_code == 200
        assert response.json() == {'metadata': []}  # dummy data does not contain checkboxes, hence empty response

    def test_detector_realdata(self):
        """Test that only a schema-adhering response is accepted by extract method of service."""
        request = convert_document_to_request(document=self.test_doc, schema=self.RequestSchema)
        response = requests.post(url=self.request_url, json=request.dict())
        ckboxes = response.json()['metadata']
        assert (
            len(ckboxes) == 17
        ), f'Test {self.test_doc} contains 17 checkboxes, but just {len(ckboxes)} have been detected.'

        # TODO: check for all 17 checkboxes, if they are checked and the position of the bounding box (overlap with the Document Annotation)
        # {'is_checked': True, 'bbox': {'x0': 70, 'x1': 82, 'y0': 758, 'y1': 771}

    def test_detector_wrongdata(self):
        """Test that it's impossible to send a request with a structure not adhering to schema."""
        data = {'pages': 1234, 'new_field': 'ffff'}
        responses = requests.post(url=self.request_url, data=data)
        assert responses.status_code == 400
        assert 'validation error' in responses.text

    @classmethod
    def tearDownClass(cls) -> None:
        """Kill process."""
        cls.bento_process.kill()
