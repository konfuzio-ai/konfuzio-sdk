"""Validate API functions."""

import datetime
import json
import logging
import os
import sys
import unittest
from unittest.mock import patch

import pytest
from requests import HTTPError

from konfuzio_sdk import BASE_DIR
from konfuzio_sdk.api import (
    TimeoutHTTPAdapter,
    _get_auth_token,
    change_document_annotation,
    create_label,
    create_new_project,
    delete_document_annotation,
    delete_file_konfuzio_api,
    delete_project,
    download_file_konfuzio_api,
    get_all_project_ais,
    get_document_annotations,
    get_document_details,
    get_meta_of_files,
    get_page_image,
    get_project_details,
    get_project_label_sets,
    get_project_labels,
    get_project_list,
    get_results_from_segmentation,
    init_env,
    post_document_annotation,
    restore_snapshot,
    update_document_konfuzio_api,
    upload_file_konfuzio_api,
)
from konfuzio_sdk.data import Bbox, Project, Span
from konfuzio_sdk.utils import get_spans_from_bbox
from tests.variables import TEST_DOCUMENT_ID, TEST_PROJECT_ID

FOLDER_ROOT = os.path.dirname(os.path.realpath(__file__))


class TestKonfuzioSDKAPI(unittest.TestCase):
    """Test API with payslip example Project."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the test Project."""
        cls.RESTORED_PROJECT_ID = restore_snapshot(snapshot_id=65)
        cls.project = Project(id_=cls.RESTORED_PROJECT_ID, update=True)
        original_document_text = Project(id_=46).get_document_by_id(TEST_DOCUMENT_ID).text
        cls.test_document = cls.project.get_document_by_id(
            [document for document in cls.project.documents if document.text == original_document_text][0].id_
        )
        original_document_text_2 = Project(id_=46).get_document_by_id(TEST_DOCUMENT_ID + 11).text
        cls.test_document_2 = cls.project.get_document_by_id(
            [document for document in cls.project.documents if document.text == original_document_text_2][0].id_
        )
        cls.test_category_id = cls.project.categories[0].id_

    def test_projects_details(self):
        """Test to get Document details."""
        data = get_project_list()
        new_var = self.RESTORED_PROJECT_ID
        assert new_var in [prj['id'] for prj in data['results']]
        assert set(data['results'][0]) == {
            'id',
            'name',
            'storage_name',
            'priority_processing',
            'ocr_method',
            'auto_rotate_documents',
            'category_ai_model_parameters',
            'category_ai',
            'decimal_separator',
            'auto_delete_documents_after_days',
            'enable_translated_strings',
        }

    def test_project_details(self):
        """Test to get Document details."""
        data = get_project_details(project_id=self.RESTORED_PROJECT_ID)
        assert set(data.keys()) == {
            'id',
            'name',
            'storage_name',
            'priority_processing',
            'ocr_method',
            'auto_rotate_documents',
            'category_ai_model_parameters',
            'category_ai',
            'decimal_separator',
            'auto_delete_documents_after_days',
            'enable_translated_strings',
        }

    def test_get_meta_of_files_multiple_pages(self):
        """Get the meta information of Document in a Project."""
        get_meta_of_files(project_id=self.RESTORED_PROJECT_ID, pagination_limit=10)

    @patch('requests.post')
    def test_empty_project(self, function):
        """Get the meta information of Documents if the Project is empty."""
        function.return_value = {'count': 0, 'next': None, 'previous': None, 'results': []}
        get_meta_of_files(project_id=self.RESTORED_PROJECT_ID, pagination_limit=10)

    def test_get_meta_of_files_one_page(self):
        """Get the meta information of Documents in a Project."""
        get_meta_of_files(project_id=self.RESTORED_PROJECT_ID, pagination_limit=1000000000)

    def test_get_meta_of_files_limited(self):
        """Get the meta information of a limited number of Documents."""
        meta = get_meta_of_files(project_id=self.RESTORED_PROJECT_ID, limit=10)
        assert len(meta) == 10

    def test_documents_list(self):
        """Test to get Documents details."""
        data = get_meta_of_files(project_id=self.RESTORED_PROJECT_ID)
        assert set(data[0].keys()) == {
            'id',
            'project',
            'document_set',
            'number_of_pages',
            'data_file_name',
            'data_file_producer',
            'file_url',
            'thumbnail_url',
            'ocr_time',
            'categorization_time',
            'extraction_time',
            'processing_time',
            'extraction_url',
            'callback_url',
            'callback_status_code',
            'category_confidence',
            'category_is_revised',
            'status_data',
            'labeling_available',
            'is_reviewed',
            'category',
            'dataset_status',
            'assignee',
            'created_by',
            'created_at',
            'updated_at',
            'proposed_split',
            'split_is_revised',
            'enable_translated_strings',
        }

    def test_document_details_document_not_available(self):
        """Test to get Document that does not exist."""
        with pytest.raises(HTTPError) as e:
            get_document_details(document_id=0)
        assert '404 Not Found' in str(e.value)

    def test_document_details_document_not_available_but_project_exists(self):
        """Test to get Document that does not exist."""
        with pytest.raises(HTTPError) as e:
            get_document_details(document_id=99999999999999999999)
        assert '404 Not Found' in str(e.value)

    def test_document_details(self):
        """Test to get Document details."""
        data = get_document_details(document_id=self.test_document.id_)
        assert set(data.keys()) == {
            'id',
            'project',
            'document_set',
            'number_of_pages',
            'data_file_name',
            'data_file_producer',
            'file_url',
            'thumbnail_url',
            'ocr_time',
            'categorization_time',
            'extraction_time',
            'processing_time',
            'extraction_url',
            'callback_url',
            'callback_status_code',
            'category_confidence',
            'category_is_revised',
            'status_data',
            'labeling_available',
            'is_reviewed',
            'text',
            'data_file_producer',
            'category',
            'dataset_status',
            'pages',
            'annotation_sets',
            'assignee',
            'created_by',
            'created_at',
            'updated_at',
            'proposed_split',
            'split_is_revised',
            'enable_translated_strings',
        }

    def test_long_document_details(self):
        """Test to get Document details."""
        data = get_document_details(document_id=216836)
        assert set(data.keys()) == {
            'id',
            'project',
            'document_set',
            'number_of_pages',
            'data_file_name',
            'data_file_producer',
            'file_url',
            'thumbnail_url',
            'ocr_time',
            'categorization_time',
            'extraction_time',
            'processing_time',
            'extraction_url',
            'callback_url',
            'callback_status_code',
            'category_confidence',
            'category_is_revised',
            'status_data',
            'labeling_available',
            'is_reviewed',
            'text',
            'data_file_producer',
            'category',
            'dataset_status',
            'pages',
            'annotation_sets',
            'assignee',
            'created_by',
            'created_at',
            'updated_at',
            'proposed_split',
            'split_is_revised',
            'enable_translated_strings',
        }

    def test_get_list_of_files(self):
        """Get meta information from Documents in the Project."""
        sorted_documents = get_meta_of_files(project_id=self.RESTORED_PROJECT_ID)
        sorted_dataset_documents = [x for x in sorted_documents if x['dataset_status'] in [2, 3]]

        self.assertEqual(26 + 3, len(sorted_dataset_documents))

    def test_upload_file_konfuzio_api_1(self):
        """Test upload of a file through API and its removal."""
        file_path = os.path.join(FOLDER_ROOT, 'test_data', 'pdf.pdf')
        doc = upload_file_konfuzio_api(file_path, project_id=self.RESTORED_PROJECT_ID)
        assert doc.status_code == 201
        document_id = json.loads(doc.text)['id']
        assert delete_file_konfuzio_api(document_id)

    def test_upload_file_konfuzio_api_invalid_callback_url(self):
        """Test upload of a file through API and its removal."""
        file_path = os.path.join(FOLDER_ROOT, 'test_data', 'pdf.pdf')
        with pytest.raises(HTTPError, match='Enter a valid URL.'):
            _ = upload_file_konfuzio_api(file_path, project_id=self.RESTORED_PROJECT_ID, callback_url='invalid url')

    def test_download_file_with_ocr(self):
        """Test to download the OCR version of a document."""
        document_id = self.test_document.id_
        downloaded_file = download_file_konfuzio_api(document_id=document_id)
        logging.info(f'Size of file {document_id}: {sys.getsizeof(downloaded_file)}')

    def test_download_file_without_ocr(self):
        """Test to download the original version of a document."""
        document_id = self.test_document.id_
        downloaded_file = download_file_konfuzio_api(document_id=document_id, ocr=False)
        logging.info(f'Size of file {document_id}: {sys.getsizeof(downloaded_file)}')

    def test_download_file_not_available(self):
        """Test to download the original version of a document."""
        document_id = 15631000000000000000000000000
        with pytest.raises(HTTPError):
            download_file_konfuzio_api(document_id=document_id)

    def test_get_annotations(self):
        """Download Annotations and the Text from API for a Document and check their offset alignment."""
        annotations = get_document_annotations(self.test_document.id_)['results']
        self.assertEqual(len(annotations), 21)

    def test_post_document_annotation_multiline_as_bboxes(self):
        """Create a multiline Annotation via API."""
        project = Project(id_=self.RESTORED_PROJECT_ID, strict_data_validation=False)
        label_id = project.get_label_by_name('Bezeichnung').id_  # just for testing
        label_set_id = project.get_label_set_by_name('Brutto-Bezug').id_  # just for testing

        bboxes = [
            {'page_index': 0, 'x0': 198, 'x1': 300, 'y0': 508, 'y1': 517},
            {'page_index': 0, 'x0': 197.76, 'x1': 233, 'y0': 495, 'y1': 508},
        ]
        document = project.get_document_by_id(self.test_document_2.id_)
        document.update()
        spans = get_spans_from_bbox(
            selection_bbox=Bbox(
                x0=bboxes[0]['x0'],
                x1=bboxes[0]['x1'],
                y0=bboxes[0]['y0'],
                y1=bboxes[0]['y1'],
                page=document.pages()[0],
            )
        ) + get_spans_from_bbox(
            selection_bbox=Bbox(
                x0=bboxes[1]['x0'],
                x1=bboxes[1]['x1'],
                y0=bboxes[1]['y0'],
                y1=bboxes[1]['y1'],
                page=document.pages()[0],
            )
        )
        response = post_document_annotation(
            document_id=document.id_,
            confidence=0.01,
            label_id=label_id,
            label_set_id=label_set_id,
            revised=False,
            is_correct=True,
            spans=spans,
        )

        assert response.status_code == 201
        annotation = json.loads(response.text)
        assert delete_document_annotation(annotation['id'], delete_from_database=True)

    @unittest.skip(reason='Not supported by Server: https://gitlab.com/konfuzio/objectives/-/issues/8663')
    def test_post_document_annotation_multiline_as_offsets(self):
        """Create a multiline Annotation via API."""
        project = Project(id_=self.RESTORED_PROJECT_ID, strict_data_validation=False)
        label_id = project.get_label_by_name('Bezeichnung').id_  # just for testing
        label_set_id = project.get_label_set_by_name('Brutto-Bezug').id_

        bboxes = [
            {'page_index': 0, 'start_offset': 1868, 'end_offset': 1883},
            {'page_index': 0, 'start_offset': 1909, 'end_offset': 1915},
        ]
        response = post_document_annotation(
            document_id=self.test_document_2.id_,
            project_id=self.RESTORED_PROJECT_ID,
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
        assert delete_document_annotation(annotation_id=annotation['id'])

    def test_post_document_annotation(self):
        """Create an Annotation via API."""
        project = Project(id_=self.RESTORED_PROJECT_ID, strict_data_validation=False, update=True)
        confidence = 0.0001
        label_id = label_id = project.get_label_by_name('Betrag').id_
        label_set_id = project.get_label_set_by_name('Brutto-Bezug').id_
        # create a revised annotation, so we can verify its existence via get_document_annotations
        project.init_or_update_document(from_online=True)
        document = project.get_document_by_id(self.test_document.id_)
        document.update()
        span = Span(document=document, start_offset=1002, end_offset=1010)
        # span.bbox()
        response = post_document_annotation(
            document_id=self.test_document.id_,
            confidence=confidence,
            label_id=label_id,
            label_set_id=label_set_id,
            revised=False,
            is_correct=False,
            spans=[span],
        )
        annotation = json.loads(response.text)
        # check if the update has been received by the server
        annotations = get_document_annotations(self.test_document.id_)['results']
        assert annotation['id'] in [annotation['id'] for annotation in annotations]
        # delete the annotation, i.e. change its status from feedback required to negative
        negative_id = delete_document_annotation(annotation['id'])
        # delete it a second time to remove this Annotation from the feedback stored as negative
        assert delete_document_annotation(negative_id, delete_from_database=True).status_code == 204

    def post_document_annotation_as_bboxes(self):
        """Test creating an Annotation that is based only on Bbox coordinates."""
        label_id = self.project.get_label_by_name('Bezeichnung').id_
        label_set_id = self.project.get_label_set_by_name('Brutto-Bezug').id_

        bboxes = [
            {'page_index': 0, 'x0': 198, 'x1': 300, 'y0': 508, 'y1': 517},
        ]
        document = self.project.get_document_by_id(self.test_document_2.id_)
        document.update()

        response = post_document_annotation(
            document_id=self.test_document_2.id_,
            confidence=0.01,
            label_id=label_id,
            label_set_id=label_set_id,
            revised=False,
            is_correct=True,
            spans=bboxes,
        )

        assert response.status_code == 201
        annotation = json.loads(response.text)
        assert delete_document_annotation(annotation['id'])

    def test_change_annotation(self):
        """Test modifying an existing Annotation."""
        annotation_with_steuerklasse = None
        for document in self.project.documents:
            for annotation in document.annotations():
                if annotation.label.name == 'Steuerklasse':
                    annotation_with_steuerklasse = annotation
                    break
        label_id = self.project.get_label_by_name('Steuerklasse')
        r = change_document_annotation(annotation_id=annotation_with_steuerklasse.id_, label_id=label_id)
        assert r.status_code == 200

    def test_get_project_labels(self):
        """Download Labels from API for a Project."""
        label_ids = [label['id'] for label in get_project_labels(project_id=TEST_PROJECT_ID)['results']]
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

    def test_get_project_label_sets(self):
        """Test getting all Label Sets of a Project."""
        label_set_ids = [label_set['id'] for label_set in get_project_label_sets(project_id=TEST_PROJECT_ID)['results']]
        assert label_set_ids == [64, 3706, 3686, 3707]

    def test_download_office_file(self):
        """Test to download the original version of an Office file."""
        download_file_konfuzio_api(257244, ocr=False)

    def test_download_image(self):
        """Test to download an image of a Page."""
        assert isinstance(get_page_image(document_id=TEST_DOCUMENT_ID, page_number=1), bytes)

    def test_get_results_from_segmentation(self):
        """Download segmentation results."""
        result = get_results_from_segmentation(doc_id=TEST_DOCUMENT_ID, project_id=TEST_PROJECT_ID)
        assert len(result[0]) == 5  # on the first page 5 elements can be found
        assert {box['label'] for box in result[0]} == {'text', 'figure', 'table', 'title'}

    def test_update_document_konfuzio_api(self):
        """Update the name and assignee of a Document."""
        timestamp = str(datetime.datetime.now())
        result = update_document_konfuzio_api(document_id=214414, file_name=timestamp, dataset_status=0)
        assert result['data_file_name'] == timestamp

    def test_update_document_konfuzio_api_no_changes(self):
        """Update a document without providing information."""
        data = get_document_details(document_id=214414)
        file_name = data['data_file_name']
        result = update_document_konfuzio_api(document_id=214414)
        assert result['data_file_name'] == file_name

    def test_create_label(self):
        """Create a label."""

        # mock session
        class _Session:
            """Mock requests POST response."""

            status_code = 201
            host = None

            def json(self):
                """Mock valid return."""
                return {'id': 420}

            def post(self, *arg, **kwargs):
                """Empty return value."""
                return self

        create_label(project_id=0, label_name='', label_sets=[], session=_Session())

    def test_create_new_project(self):
        """Test to create new Project."""

        # mock session
        class _Session:
            """Mock requests POST response."""

            status_code = 201

            def json(self):
                """Mock valid return."""
                return {'id': 420}

            def post(self, *arg, **kwargs):
                """Empty return value."""
                return self

        assert create_new_project('test', session=_Session()) == 420

    def test_create_new_project_permission_error(self):
        """Test to create new Project."""

        # mock session
        class _Session:
            """Mock requests POST response."""

            status_code = 403

            def json(self):
                """Mock valid return."""
                return {'id': 420}

            def post(self, *arg, **kwargs):
                """Empty return value."""
                return self

        with self.assertRaises(PermissionError) as e:
            create_new_project('test', session=_Session())
            assert 'was not created' in str(e.exception)

    def test_download_file_konfuzio_api_with_whitespace_name_file(self):
        """Test to download a file which includes a whitespace in the name."""
        download_file_konfuzio_api(document_id=44860)

    @patch('requests.post')
    def test_get_auth_token(self, function):
        """Test to run CLI."""

        # mock response
        class _Response:
            """Mock requests POST response."""

            status_code = 200

            def json(self):
                """Mock valid return."""
                return {'token': 'faketoken'}

        function.return_value = _Response()
        _get_auth_token('test', 'test')

    def test_permission_error_with_none_token(self):
        """Test to raise PermissionError."""
        adapter = TimeoutHTTPAdapter(timeout=1)
        # mock request

        class _Request:
            """Mock Request."""

            headers = {'Authorization': 'Token None'}

        with self.assertRaises(PermissionError) as context:
            adapter.send(request=_Request())
            assert 'is missing' in context.exception

    @patch('requests.post')
    def test_get_auth_token_connection_error(self, function):
        """Test to run CLI."""

        # mock response
        class _Response:
            """Mock requests POST response."""

            status_code = 500

            def json(self):
                """Mock valid return."""
                return {'token': 'faketoken'}

            def text(self):
                """Mock the text in the response."""
                return 'Error'

        function.return_value = _Response()
        with self.assertRaises(ConnectionError) as context:
            _get_auth_token('test', 'test')
            assert 'HTTP Status 500' in context.exception

    @patch('requests.post')
    def test_patched_init_env(self, function):
        """Test to run CLI."""

        # mock response
        class _Response:
            """Mock requests POST response."""

            status_code = 200

            def json(self):
                """Mock valid return."""
                return {'token': 'faketoken'}

        function.return_value = _Response()
        env_file = '.testenv'
        assert init_env(user='me', password='pw', file_ending=env_file)
        os.remove(os.path.join(os.getcwd(), env_file))

    @patch('konfuzio_sdk.api.konfuzio_session')
    @patch('konfuzio_sdk.api.get_extraction_ais_list_url')
    @patch('konfuzio_sdk.api.get_splitting_ais_list_url')
    @patch('konfuzio_sdk.api.get_categorization_ais_list_url')
    @patch('konfuzio_sdk.api.json.loads')
    def test_get_all_project_ais(
        self,
        mock_json_loads,
        mock_get_categorization_url,
        mock_get_splitting_url,
        mock_get_extraction_url,
        mock_session,
    ):
        """Retrieve all AIs from a Project."""
        # Setup
        sample_data = {'AI_DATA': 'AI_SAMPLE_DATA'}

        mock_session.return_value.get.return_value.status_code = 200
        mock_json_loads.return_value = sample_data

        # Action
        result = get_all_project_ais(project_id=1)

        # Assertions
        self.assertEqual(
            result,
            {
                'extraction': sample_data,
                'filesplitting': sample_data,
                'categorization': sample_data,
            },
        )

        from konfuzio_sdk.api import konfuzio_session

        # Ensure the mock methods were called with the correct arguments
        mock_get_extraction_url.assert_called_once_with(1, konfuzio_session().host)
        mock_get_splitting_url.assert_called_once_with(1, konfuzio_session().host)
        mock_get_categorization_url.assert_called_once_with(1, konfuzio_session().host)

    @patch('konfuzio_sdk.api.konfuzio_session')
    @patch('konfuzio_sdk.api.get_extraction_ais_list_url')
    @patch('konfuzio_sdk.api.get_splitting_ais_list_url')
    @patch('konfuzio_sdk.api.get_categorization_ais_list_url')
    @patch('konfuzio_sdk.api.json.loads')
    def test_get_all_project_ais_with_invalid_permissions(
        self,
        mock_json_loads,
        mock_get_categorization_url,
        mock_get_splitting_url,
        mock_get_extraction_url,
        mock_session,
    ):
        """Assert that despite not having permissions, the function can still be called without exception."""
        # Setup
        exception_message = '403 Client Error: Forbidden'
        sample_data = {'error': exception_message}  # direct string, not HTTPError

        mock_session.return_value.get.side_effect = HTTPError(exception_message)
        mock_json_loads.return_value = sample_data

        # Action
        result = get_all_project_ais(project_id=1)

        self.assertEqual(result['extraction']['error'].__str__(), exception_message)
        self.assertEqual(result['categorization']['error'].__str__(), exception_message)
        self.assertEqual(result['filesplitting']['error'].__str__(), exception_message)

    def test_restore_snapshot(self):
        """Test restoring a snapshot using snapshotrestores endpoint."""
        project_id = restore_snapshot(snapshot_id=65)
        all_projects = get_project_list()
        assert project_id in [project['id'] for project in all_projects['results']]
        new_project = Project(id_=project_id)
        for document in new_project.documents + new_project.test_documents:
            document.dataset_status = 0
            document.save_meta_data()
            document.delete(delete_online=True)
        r = delete_project(project_id=project_id)
        assert r.status_code == 204

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove the project created specifically for this test pipeline."""
        cls.project = Project(id_=cls.RESTORED_PROJECT_ID, update=True)
        for document in cls.project.documents + cls.project.test_documents:
            document.dataset_status = 0
            document.save_meta_data()
            document.delete(delete_online=True)
        response = delete_project(project_id=cls.RESTORED_PROJECT_ID)
        assert response.status_code == 204


def test_init_env():
    """Test to write env file."""
    with pytest.raises(PermissionError, match='Your credentials are not correct'):
        init_env(user='user', password='ABCD', working_directory=BASE_DIR, file_ending='x.env')
