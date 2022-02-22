"""Validate API functions."""
import datetime
import logging
import os
import json
import sys
import unittest
from unittest.mock import patch

import pytest

from konfuzio_sdk import BASE_DIR
from konfuzio_sdk.api import (
    get_meta_of_files,
    download_file_konfuzio_api,
    upload_file_konfuzio_api,
    post_document_annotation,
    delete_document_annotation,
    delete_file_konfuzio_api,
    get_results_from_segmentation,
    update_file_konfuzio_api,
    get_project_list,
    get_document_details,
    get_project_details,
    upload_ai_model,
    init_env,
    _get_auth_token,
    create_new_project,
    create_label,
)
from konfuzio_sdk.utils import is_file

FOLDER_ROOT = os.path.dirname(os.path.realpath(__file__))

TEST_DOCUMENT_ID = 44823
TEST_PROJECT_ID = 46


@pytest.mark.serial
class TestKonfuzioSDKAPI(unittest.TestCase):
    """Test API with payslip example project."""

    def test_projects_details(self):
        """Test to get Document details."""
        data = get_project_list()
        assert TEST_PROJECT_ID in [prj["id"] for prj in data]
        assert data[0].keys() == {
            'id',
            'name',
            'labels',
            'section_labels',
            'storage_name',
            'priority_processing',
            'ocr_method',
        }

    def test_project_details(self):
        """Test to get Document details."""
        data = get_project_details(project_id=TEST_PROJECT_ID)
        assert data.keys() == {
            'id',
            'name',
            'labels',
            'section_labels',
            'storage_name',
            'priority_processing',
            'ocr_method',
        }

    def test_documents_list(self):
        """Test to get Documents details."""
        data = get_meta_of_files(project_id=TEST_PROJECT_ID)
        assert data[0].keys() == {
            'id',
            'number_of_pages',
            'callback_url',
            'callback_status_code',
            'category_template',
            'category_confidence',
            'file_url',
            'data_file_name',
            'data_file_producer',
            'data_file',
            'ocr_time',
            'extraction_time',
            'workflow_start_time',
            'workflow_end_time',
            'dataset_status',
            'status',
            'status_data',
            'updated_at',
        }

    def test_document_details(self):
        """Test to get Document details."""
        data = get_document_details(document_id=TEST_DOCUMENT_ID, project_id=TEST_PROJECT_ID)
        assert data.keys() == {
            'id',
            'number_of_pages',
            'callback_url',
            'callback_status_code',
            'file_url',
            'data_file_name',
            'text',
            # 'bbox',  removed from default to reduce loading time
            # 'hocr',  removed from default to reduce loading time
            'data_file_producer',
            'data_file',
            'ocr_time',
            'extraction_time',
            'workflow_start_time',
            'workflow_end_time',
            'status',
            'updated_at',
            'annotations',
            'sections',
            'pages',
            'category_template',
        }

    def test_long_document_details(self):
        """Test to get Document details."""
        data = get_document_details(document_id=216836, project_id=TEST_PROJECT_ID)
        assert data.keys() == {
            'id',
            'number_of_pages',
            'callback_url',
            'callback_status_code',
            'file_url',
            'data_file_name',
            'text',
            # 'bbox',  removed from default to reduce loading time
            # 'hocr',  removed from default to reduce loading time
            'data_file_producer',
            'data_file',
            'ocr_time',
            'extraction_time',
            'workflow_start_time',
            'workflow_end_time',
            'status',
            'updated_at',
            'annotations',
            'sections',
            'pages',
            'category_template',
        }

    def test_get_list_of_files(self):
        """Get meta information from Documents in the project."""
        sorted_documents = get_meta_of_files(project_id=TEST_PROJECT_ID)
        sorted_dataset_documents = [x for x in sorted_documents if x['dataset_status'] in [2, 3]]
        self.assertEqual(27 + 3, len(sorted_dataset_documents))

    def test_upload_file_konfuzio_api(self):
        """Test upload of a file through API and its removal."""
        file_path = os.path.join(FOLDER_ROOT, 'test_data', 'pdf.pdf')
        doc = upload_file_konfuzio_api(file_path, project_id=TEST_PROJECT_ID)
        assert doc.status_code == 201
        document_id = json.loads(doc.text)['id']
        assert delete_file_konfuzio_api(document_id)

    def test_download_file_with_ocr(self):
        """Test to download the OCR version of a document."""
        document_id = 215906
        downloaded_file = download_file_konfuzio_api(document_id=document_id)
        logging.info(f'Size of file {document_id}: {sys.getsizeof(downloaded_file)}')

    def test_download_file_without_ocr(self):
        """Test to download the original version of a document."""
        document_id = 215906
        downloaded_file = download_file_konfuzio_api(document_id=document_id, ocr=False)
        logging.info(f'Size of file {document_id}: {sys.getsizeof(downloaded_file)}')

    def test_download_file_not_available(self):
        """Test to download the original version of a document."""
        document_id = 15631000000000000000000000000
        with pytest.raises(FileNotFoundError):
            download_file_konfuzio_api(document_id=document_id)

    def test_get_annotations(self):
        """Download Annotations and the Text from API for a Document and check their offset alignment."""
        annotations = get_document_details(TEST_DOCUMENT_ID, project_id=TEST_PROJECT_ID)['annotations']
        self.assertEqual(len(annotations), 22)

    def test_post_document_annotation_multiline_as_bboxes(self):
        """Create a multiline Annotation via API."""
        label_id = 862  # just for testing
        label_set_id = 64  # just for testing

        bboxes = [
            {"page_index": 0, "x0": 198, "x1": 300, "y0": 508, "y1": 517},
            {"page_index": 0, "x0": 197.76, "x1": 233, "y0": 495, "y1": 508},
        ]

        response = post_document_annotation(
            document_id=TEST_DOCUMENT_ID,
            project_id=TEST_PROJECT_ID,
            start_offset=24,
            end_offset=1000,
            confidence=None,
            label_id=label_id,
            label_set_id=label_set_id,
            revised=False,
            is_correct=True,
            bboxes=bboxes,
        )

        assert response.status_code == 201
        annotation = json.loads(response.text)
        assert delete_document_annotation(TEST_DOCUMENT_ID, annotation['id'], project_id=TEST_PROJECT_ID)

    @unittest.skip(reason='Not supported by Server: https://gitlab.com/konfuzio/objectives/-/issues/8663')
    def test_post_document_annotation_multiline_as_offsets(self):
        """Create a multiline Annotation via API."""
        label_id = 862  # just for testing
        label_set_id = 64  # just for testing

        bboxes = [
            {"page_index": 0, "start_offset": 1868, "end_offset": 1883},
            {"page_index": 0, "start_offset": 1909, "end_offset": 1915},
        ]

        response = post_document_annotation(
            document_id=TEST_DOCUMENT_ID,
            project_id=TEST_PROJECT_ID,
            start_offset=24,
            end_offset=1000,
            accuracy=None,
            label_id=label_id,
            label_set_id=label_set_id,
            revised=False,
            is_correct=False,
            bboxes=bboxes,
        )

        assert response.status_code == 201
        annotation = json.loads(response.text)
        assert delete_document_annotation(
            document_id=TEST_DOCUMENT_ID, project_id=TEST_PROJECT_ID, annotation_id=annotation['id']
        )

    def test_post_document_annotation(self):
        """Create an Annotation via API."""
        start_offset = 60
        end_offset = 63
        confidence = 0.0001
        label_id = 863  # Refers to Label Betrag (863)
        label_set_id = 64  # Refers to LabelSet Brutto-Bezug (allows multiple Annotation Sets)
        # create a revised annotation, so we can verify its existence via get_document_annotations
        response = post_document_annotation(
            document_id=TEST_DOCUMENT_ID,
            project_id=TEST_PROJECT_ID,
            start_offset=start_offset,
            end_offset=end_offset,
            confidence=confidence,
            label_id=label_id,
            label_set_id=label_set_id,
            revised=False,
            is_correct=False,
        )
        annotation = json.loads(response.text)
        # check if the update has been received by the server
        annotations = get_document_details(TEST_DOCUMENT_ID, project_id=TEST_PROJECT_ID)['annotations']
        assert annotation['id'] in [annotation['id'] for annotation in annotations]
        # delete the annotation, i.e. change it's status from feedback required to negative
        negative_id = delete_document_annotation(TEST_DOCUMENT_ID, annotation['id'], project_id=TEST_PROJECT_ID)
        # delete it a second time to remove this Annotation from the feedback stored as negative
        assert delete_document_annotation(TEST_DOCUMENT_ID, negative_id, project_id=TEST_PROJECT_ID)

    def test_get_project_labels(self):
        """Download Labels from API for a Project."""
        label_ids = [label["id"] for label in get_project_details(project_id=TEST_PROJECT_ID)['labels']]
        assert set(label_ids) == {
            858,
            859,
            860,
            861,
            862,
            863,
            864,
            865,
            866,
            867,
            964,
            12444,
            12453,
            12470,
            12482,
            12483,
            12484,
            12503,
        }

    def test_download_office_file(self):
        """Test to download the original version of an Office file."""
        download_file_konfuzio_api(219912, ocr=False)

    def test_get_results_from_segmentation(self):
        """Download segmentation results."""
        result = get_results_from_segmentation(doc_id=TEST_DOCUMENT_ID, project_id=TEST_PROJECT_ID)
        assert len(result[0]) == 5  # on the first page 5 elements can be found
        assert set([box["label"] for box in result[0]]) == {"text", "figure", "table", "title"}

    def test_update_file_konfuzio_api(self):
        """Update the name of a document."""
        timestamp = str(datetime.datetime.now())
        result = update_file_konfuzio_api(document_id=214414, file_name=timestamp, dataset_status=0)
        assert result['data_file_name'] == timestamp

    def test_create_label(self):
        """Create a label."""
        # mock session
        class _Session:
            """Mock requests POST response."""

            status_code = 201

            def json(self):
                """Mock valid return."""
                return {"id": 420}

            def post(self, *arg, **kwargs):
                """Empty return value."""
                return self

        create_label(project_id=0, label_name='', label_sets=[], session=_Session())

    @unittest.skip(reason="Skip to iterate version of meta information of files, unclear when it paginates.")
    def test_meta_file_pagination(self):
        """Iterate over Urls with a next page and for empty projects without document."""
        pass

    def test_create_new_project(self):
        """Test to create new project."""
        # mock session
        class _Session:
            """Mock requests POST response."""

            status_code = 201

            def json(self):
                """Mock valid return."""
                return {"id": 420}

            def post(self, *arg, **kwargs):
                """Empty return value."""
                return self

        assert create_new_project('test', session=_Session()) == 420

    def test_create_new_project_permission_error(self):
        """Test to create new project."""
        # mock session
        class _Session:
            """Mock requests POST response."""

            status_code = 403

            def json(self):
                """Mock valid return."""
                return {"id": 420}

            def post(self, *arg, **kwargs):
                """Empty return value."""
                return self

        with self.assertRaises(PermissionError) as e:
            create_new_project('test', session=_Session())
            assert 'was not created' in str(e.exception)

    def test_download_file_konfuzio_api_with_whitespace_name_file(self):
        """Test to download a file which includes a whitespace in the name."""
        download_file_konfuzio_api(document_id=44860)

    def test_upload_ai_model(self):
        """Test to upload an AI model."""
        path = os.path.join(os.getcwd(), 'lohnabrechnung.pkl')
        if is_file(file_path=path, raise_exception=False):
            upload_ai_model(ai_model_path=path, category_ids=[63])

    @patch("requests.post")
    def test_get_auth_token(self, function):
        """Test to run CLI."""
        # mock response
        class _Response:
            """Mock requests POST response."""

            status_code = 200

            def json(self):
                """Mock valid return."""
                return {"token": "faketoken"}

        function.return_value = _Response()
        _get_auth_token('test', 'test')

    @patch("requests.post")
    def test_patched_init_env(self, function):
        """Test to run CLI."""
        # mock response
        class _Response:
            """Mock requests POST response."""

            status_code = 200

            def json(self):
                """Mock valid return."""
                return {"token": "faketoken"}

        function.return_value = _Response()
        env_file = ".testenv"
        assert init_env(user='me', password='pw', file_ending=env_file)
        os.remove(os.path.join(os.getcwd(), env_file))


def test_init_env():
    """Test to write env file."""
    with pytest.raises(PermissionError, match="Your credentials are not correct"):
        init_env(user="user", password="ABCD", working_directory=BASE_DIR, file_ending="x.env")
