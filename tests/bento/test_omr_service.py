"""Test OMR functionality."""

import os
import subprocess
import time
import unittest
from io import BytesIO

import bentoml
import numpy as np
import pytest
import requests
from PIL import Image

from konfuzio_sdk.bento.omr.schemas import CheckboxRequest20240523, CheckboxResponse20240523
from konfuzio_sdk.bento.omr.utils import convert_document_to_request
from konfuzio_sdk.data import Project
from konfuzio_sdk.settings_importer import is_dependency_installed
from konfuzio_sdk.trainer.omr import CheckboxDetector


@pytest.mark.skipif(
    not is_dependency_installed('torch') and not is_dependency_installed('torchvision'),
    reason='Required dependencies not installed.',
)
class TestOMRCheckboxBento(unittest.TestCase):
    """Test Bento-based OMR checkbox detector functionality."""

    @classmethod
    def setUpClass(cls) -> None:
        """Get latest Bento instance of checkbox detection service and initialize test data."""

        bento_version = os.getenv('CHECKBOX_DETECTOR_VERSION', 'latest')
        bento_tag = f'checkboxdetector:{bento_version}'

        # Define used schemas
        cls.RequestSchema = CheckboxRequest20240523
        cls.ResponseSchema = CheckboxResponse20240523
        # Initialize pipeline
        cls.pipeline = CheckboxDetector()
        # Load Bento checkboxdetector model
        try:
            # try to get the model from the local store
            bento = bentoml.bentos.get(bento_tag)
        except bentoml.exceptions.NotFound:
            try:
                # try to get the model from the yatai store
                bentoml.bentos.pull(bento_tag)
                bento = bentoml.bentos.get(bento_tag)
            except bentoml.exceptions.NotFound:
                print(f'The Bento {bento_tag} is not available in the local store or the Yatai store.')
        cls.bento_name = bento.tag.name + ':' + bento.tag.version

        # Initialize and create test data
        cls.project = Project(id_=15217, update=True)
        cls.test_doc = cls.project.get_document_by_id(5930563)  # musterformular-by.pdf
        # Ground truth for the checkboxes in the test document
        cls.ckboxes_gt_doc5930563 = [
            {'is_checked': True, 'bbox': {'x0': 70, 'x1': 82, 'y0': 758, 'y1': 771}},
            {'is_checked': False, 'bbox': {'x0': 70, 'x1': 82, 'y0': 733, 'y1': 745}},
            {'is_checked': False, 'bbox': {'x0': 85, 'x1': 97, 'y0': 481, 'y1': 493}},
            {'is_checked': False, 'bbox': {'x0': 85, 'x1': 97, 'y0': 431, 'y1': 443}},
            {'is_checked': True, 'bbox': {'x0': 85, 'x1': 97, 'y0': 367, 'y1': 380}},
            {'is_checked': False, 'bbox': {'x0': 85, 'x1': 97, 'y0': 343, 'y1': 355}},
            {'is_checked': True, 'bbox': {'x0': 85, 'x1': 97, 'y0': 317, 'y1': 329}},
            {'is_checked': True, 'bbox': {'x0': 85, 'x1': 97, 'y0': 254, 'y1': 266}},
            {'is_checked': False, 'bbox': {'x0': 113, 'x1': 125, 'y0': 229, 'y1': 241}},
            {'is_checked': True, 'bbox': {'x0': 113, 'x1': 125, 'y0': 216, 'y1': 228}},
            {'is_checked': True, 'bbox': {'x0': 141, 'x1': 153, 'y0': 203, 'y1': 216}},
            {'is_checked': False, 'bbox': {'x0': 141, 'x1': 153, 'y0': 191, 'y1': 203}},
            {'is_checked': True, 'bbox': {'x0': 85, 'x1': 97, 'y0': 115, 'y1': 127}},
            {'is_checked': False, 'bbox': {'x0': 85, 'x1': 97, 'y0': 796, 'y1': 808}},
            {'is_checked': True, 'bbox': {'x0': 85, 'x1': 97, 'y0': 746, 'y1': 758}},
            {'is_checked': False, 'bbox': {'x0': 85, 'x1': 97, 'y0': 696, 'y1': 707}},
            {'is_checked': True, 'bbox': {'x0': 70, 'x1': 82, 'y0': 352, 'y1': 363}},
        ]

        dummy_image = Image.fromarray(np.full((100, 100, 3), (255, 0, 0), dtype=np.uint8))
        buffered = BytesIO()
        dummy_image.save(buffered, format='PNG')
        encoded_image = buffered.getvalue()

        cls.test_request = {
            'pages': [
                {'page_id': 8795115, 'width': 595, 'height': 841, 'image': encoded_image},
                {'page_id': 8795116, 'width': 595, 'height': 841, 'image': encoded_image},
            ],
            'annotations': [
                {'page_id': 8795115, 'annotation_id': 36015161, 'bbox': {'x0': 114, 'x1': 196, 'y0': 761, 'y1': 769}},
                {'page_id': 8795115, 'annotation_id': 36015153, 'bbox': {'x0': 114, 'x1': 164, 'y0': 733, 'y1': 744}},
            ],
            'detection_threshold': 0.5,
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
        os.system('lsof -t -i:3000 | fuser -k 3000/tcp')

        cls.bento_process = subprocess.Popen(
            ['bentoml', 'serve', cls.bento_name], stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        # Debug the bento process in with breakpoints and cls.bento_process.stdout.readline()
        time.sleep(10)  # time needed to start up the bento server
        print('served bento')

        cls.request_url = 'http://0.0.0.0:3000/extract'

    def test_testdocument(self) -> None:
        """Test that the test document contains all relevant Annotations and Checkboxes for the following tests."""
        assert self.test_doc is not None
        assert len(self.test_doc.pages()) == 2
        assert (
            len(
                [
                    a
                    for a in self.test_doc.annotations()
                    if getattr(a.label, 'is_linked_to_checkbox', True) or a.label.is_linked_to_checkbox is None
                ]
            )
            == 17
        ), f"{self.test_doc} should contain 17 Annotations with attribute 'is_linked_to_checkbox'==True."

    def test_request_schema(self) -> None:
        """Test that the request schema and test_request is correctly initialized."""
        request_schema = self.RequestSchema(**self.test_request)
        assert isinstance(request_schema, self.RequestSchema)

    def test_response_schema(self) -> None:
        """Test that the response schema and test_response is correctly initialized."""
        response_schema = self.ResponseSchema(**self.test_response)
        assert isinstance(response_schema, self.ResponseSchema)

    def test_convert_document_to_request(self) -> None:
        """Test that the conversion of the test document to a request adheres to the correct schema."""
        req = convert_document_to_request(document=self.test_doc, schema=self.RequestSchema)

        # check request pages
        assert len(req.pages) == 2
        assert req.pages[0].page_id == self.test_doc.pages()[0].id_
        assert req.pages[0].width == int(self.test_doc.pages()[0].width)
        assert req.pages[0].height == int(self.test_doc.pages()[0].height)
        assert isinstance(req.pages[0].image, bytes)
        # check request annotations
        assert req.annotations[0].page_id == self.test_doc.pages()[0].id_
        assert req.annotations[0].annotation_id == self.test_doc.annotations()[0].id_
        assert req.annotations[0].bbox.x0 == int(self.test_doc.annotations()[0].bboxes[0]['x0'])
        assert req.annotations[0].bbox.x1 == int(self.test_doc.annotations()[0].bboxes[0]['x1'])
        assert req.annotations[0].bbox.y0 == int(self.test_doc.annotations()[0].bboxes[0]['y0'])
        assert req.annotations[0].bbox.y1 == int(self.test_doc.annotations()[0].bboxes[0]['y1'])

    def test_detector_dummydata(self) -> None:
        """Test the checkbox detection service with dummy data (no checkboxes in dummy data)."""
        # Create a dummy image (e.g., a 100x100 red image)
        request = self.RequestSchema(**self.test_request)
        response = requests.post(url=self.request_url, json=request.model_dump())
        assert response.status_code == 200
        assert response.json() == {'metadata': []}  # check empty response due to lack of checkboxes in dummy data

    def test_detector_realdata(self) -> None:
        """Test the checkbox detection service with a real test document (end2end)."""
        request = convert_document_to_request(document=self.test_doc, schema=self.RequestSchema)
        response = requests.post(url=self.request_url, json=request.model_dump())
        ckboxes = response.json()['metadata']

        assert (
            len(ckboxes) == 17
        ), f'Test {self.test_doc} contains 17 checkboxes, but just {len(ckboxes)} have been detected.'

        def bboxes_overlap(bbox1, bbox2) -> bool:
            """Check if two bounding boxes have any overlap, if so, it is considered valid."""
            # Check if there is any overlap on the x-axis
            if bbox1['x1'] < bbox2['x0'] or bbox2['x1'] < bbox1['x0']:
                return False
            # Check if there is any overlap on the y-axis
            if bbox1['y1'] < bbox2['y0'] or bbox2['y1'] < bbox1['y0']:
                return False
            return True

        # check each detected checkbox for overlap and same value as ground truth
        valid_box_ids = []
        # for all detected checkboxes
        for ckbox in ckboxes:
            # and each ground truth checkbox
            for ckbox_gt in self.ckboxes_gt_doc5930563:
                # check overlap
                if bboxes_overlap(ckbox['checkbox']['bbox'], ckbox_gt['bbox']):
                    # check value
                    if ckbox['checkbox']['is_checked'] == ckbox_gt['is_checked']:
                        # if both are true, the detection is valid
                        valid_box_ids.append(ckbox['annotation_id'])
                        break

        # check if all valid checkboxes are within the detections
        valid_box_ids = set(valid_box_ids)
        detected_box_ids = {ckbox['annotation_id'] for ckbox in ckboxes}
        assert (
            valid_box_ids == detected_box_ids
        ), f'Not all checkboxes in {self.test_doc} have been detected correctly, either the value or the position is wrong.'

    def test_detector_max_threshold(self) -> None:
        """Test the checkbox detection service with modified detection threshold parameter."""
        request = convert_document_to_request(document=self.test_doc, schema=self.RequestSchema)
        request.detection_threshold = 1.0
        response = requests.post(url=self.request_url, json=request.model_dump())
        assert response.status_code == 200
        assert (
            response.json() == {'metadata': []}
        ), f'Threshold is set to 1.0, as the confidence should not be higher than one, the response should be empty, but is: {response.json()}.'

    def test_detector_wrongdata(self) -> None:
        """Test that it's impossible to send a request with a structure not adhering to schema."""
        data = {'pages': 1234, 'new_field': 'ffff'}
        responses = requests.post(url=self.request_url, data=data)
        assert responses.status_code == 400
        assert 'validation error' in responses.text

    @classmethod
    def tearDownClass(cls) -> None:
        """Kill process."""
        cls.bento_process.kill()
