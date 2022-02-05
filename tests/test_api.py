"""Validate API functions."""
import datetime
import logging
import os
import json
import sys
import unittest

import pytest
from konfuzio_sdk import KONFUZIO_PROJECT_ID, KONFUZIO_TOKEN, KONFUZIO_HOST
from konfuzio_sdk.api import (
    get_document_text,
    get_meta_of_files,
    download_file_konfuzio_api,
    get_project_labels,
    upload_file_konfuzio_api,
    get_document_annotations,
    post_document_annotation,
    delete_document_annotation,
    delete_file_konfuzio_api,
    get_results_from_segmentation,
    update_file_konfuzio_api,
    get_document_hocr,
    get_project_list,
)

# Change project root to tests folder
from konfuzio_sdk.data import Project

FOLDER_ROOT = os.path.dirname(os.path.realpath(__file__))

TEST_DOCUMENT_ID = 44823


@pytest.mark.serial
class TestKonfuzioSDKAPI(unittest.TestCase):
    """Test API with payslip example project."""

    def test_download_text(self):
        """Test get text for a document."""
        assert get_document_text(document_id=TEST_DOCUMENT_ID) is not None

    def test_get_list_of_files(self):
        """Get meta information from documents in the project."""
        sorted_documents = get_meta_of_files()
        sorted_dataset_documents = [x for x in sorted_documents if x['dataset_status'] in [2, 3]]
        self.assertEqual(24 + 4, len(sorted_dataset_documents))

    @unittest.skip(reason="Will change to project setup.")
    def test_upload_file_konfuzio_api(self):
        """Test upload of a file through API and its removal."""
        file_path = os.path.join(FOLDER_ROOT, 'test_data/pdf/1_test.pdf')
        doc = upload_file_konfuzio_api(file_path, project_id=KONFUZIO_PROJECT_ID)
        assert doc.status_code == 201
        document_id = json.loads(doc.text)['id']
        assert delete_file_konfuzio_api(document_id)

    def test_download_file_with_ocr(self):
        """Test to download the ocred version of a document."""
        document_id = 94858
        downloaded_file = download_file_konfuzio_api(document_id=document_id)
        logging.info(f'Size of file {document_id}: {sys.getsizeof(downloaded_file)}')

    def test_download_file_without_ocr(self):
        """Test to download the original version of a document."""
        document_id = 94858
        downloaded_file = download_file_konfuzio_api(document_id=document_id, ocr=False)
        logging.info(f'Size of file {document_id}: {sys.getsizeof(downloaded_file)}')

    def test_download_file_not_available(self):
        """Test to download the original version of a document."""
        document_id = 15631000000000000000000000000
        with pytest.raises(FileNotFoundError):
            download_file_konfuzio_api(document_id=document_id)

    def test_load_annotations_and_text_from_api(self):
        """Download Annotations and the Text from API for a Document and check their offset alignment."""
        text = get_document_text(TEST_DOCUMENT_ID)
        annotations = get_document_annotations(TEST_DOCUMENT_ID)
        assert len(annotations) == 17
        # check the text to be in line with the annotations offsets
        for i in range(0, len(annotations)):
            assert (
                text[annotations[1]['start_offset'] : annotations[1]['end_offset']] == annotations[1]['offset_string']
            )

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
            start_offset=24,
            end_offset=1000,
            accuracy=None,
            label_id=label_id,
            label_set_id=label_set_id,
            revised=False,
            is_correct=True,
            bboxes=bboxes,
        )

        assert response.status_code == 201
        annotation = json.loads(response.text)
        assert delete_document_annotation(TEST_DOCUMENT_ID, annotation['id'])

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
            start_offset=24,
            end_offset=1000,
            accuracy=None,
            label_id=label_id,
            label_set_id=label_set_id,
            revised=False,
            is_correct=True,
            bboxes=bboxes,
        )

        assert response.status_code == 201
        annotation = json.loads(response.text)
        assert delete_document_annotation(TEST_DOCUMENT_ID, annotation['id'])

    def test_post_document_annotation(self):
        """Create an Annotation via API."""
        document_id = TEST_DOCUMENT_ID
        start_offset = 86
        end_offset = 88
        accuracy = 0.0001
        label_id = 863  # Refers to Label Betrag (863)
        label_set_id = 64  # Refers to LabelSet Brutto-Bezug (allows multisections)
        # create a revised annotation, so we can verify its existence via get_document_annotations
        response = post_document_annotation(
            document_id=document_id,
            start_offset=start_offset,
            end_offset=end_offset,
            accuracy=accuracy,
            label_id=label_id,
            label_set_id=label_set_id,
            revised=True,
        )
        annotation = json.loads(response.text)
        annotation_ids = [annot['id'] for annot in get_document_annotations(document_id, include_extractions=True)]
        assert annotation['id'] in annotation_ids
        assert delete_document_annotation(document_id, annotation['id'])

    def test_get_project_labels(self):
        """Download Labels from API for a Project."""
        label_ids = [label["id"] for label in get_project_labels()]
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

    def test_get_results_from_segmentation(self):
        """Download segmentation results."""
        prj = Project()
        result = get_results_from_segmentation(doc_id=prj.documents[0].id, project_id=prj.id)
        assert len(result[0]) == 5  # on the first page 5 elements can be found
        assert set([box["label"] for box in result[0]]) == {"text", "figure", "table", "title"}

    def test_update_file_konfuzio_api(self):
        """Update the name of a document."""
        prj = Project()
        timestamp = str(datetime.datetime.now())
        doc = prj.documents[-1]
        # keep the dataset status
        result = update_file_konfuzio_api(document_id=doc.id, file_name=timestamp, dataset_status=doc.dataset_status)
        assert result['data_file_name'] == timestamp

    def test_get_document_hocr(self):
        """Get the HOCR of a file."""
        prj = Project()
        result = get_document_hocr(document_id=prj.documents[-1].id)
        assert result is None

    @unittest.skip(reason="Skip to test to create label, as there is no option to delete it again.")
    def test_create_label(self):
        """Create a label."""
        pass

    @unittest.skip(reason="Skip to iterate version of meta information of files, unclear when it paginates.")
    def test_meta_file_pagniation(self):
        """Iterate over Urls with a next page and for empty projects without document."""
        pass

    @unittest.skip(reason="Skip to test to create project, as there is no option to delete it again.")
    def create_new_project(self):
        """Test to create new project."""
        pass

    def test_get_project_list(self):
        """Test to get the projects of the user."""
        result = get_project_list(KONFUZIO_TOKEN, KONFUZIO_HOST)
        prj = Project()
        assert prj.id in [prj["id"] for prj in result]
