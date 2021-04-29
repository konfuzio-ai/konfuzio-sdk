"""Validate urls functions."""
import unittest
from konfuzio_sdk.urls import (
    get_documents_meta_url,
    get_upload_document_url,
    get_document_ocr_file_url,
    get_document_original_file_url,
    get_document_api_details_url,
    get_project_url,
    post_project_api_document_annotations_url,
    delete_project_api_document_annotations_url,
    get_document_result_v1,
    get_document_segmentation_details_url,
    get_create_label_url,
)


from konfuzio_sdk import KONFUZIO_HOST, KONFUZIO_PROJECT_ID

DOCUMENT_ID = 3346
ANNOTATION_ID = 56


class TestUrls(unittest.TestCase):
    """Testing endpoints of the Konfuzio Host."""

    def test_get_documents_meta_url(self):
        """Test function used to generate url to load meta information about documents."""
        document_meta_url = f"{KONFUZIO_HOST}/api/projects/{KONFUZIO_PROJECT_ID}/docs/"
        self.assertEqual(get_documents_meta_url(), document_meta_url)

    def test_get_upload_document_url(self):
        """Test function used to generate url to upload document."""
        upload_document_url = f"{KONFUZIO_HOST}/api/v2/docs/"
        self.assertEqual(get_upload_document_url(), upload_document_url)

    def test_get_create_label_url(self):
        """Test function used to generate url to create a label."""
        create_label_url = f"{KONFUZIO_HOST}/api/v2/labels/"
        self.assertEqual(get_create_label_url(), create_label_url)

    def test_get_document_ocr_file_url(self):
        """Test function used to generate url to access OCR version of document."""
        document_ocr_file_url = f'{KONFUZIO_HOST}/doc/show/{DOCUMENT_ID}/'
        self.assertEqual(get_document_ocr_file_url(DOCUMENT_ID), document_ocr_file_url)

    def test_get_document_original_file_url(self):
        """Test function used to generate url to access OCR version of document."""
        document_original_file_url = f'{KONFUZIO_HOST}/doc/show-original/{DOCUMENT_ID}/'
        self.assertEqual(get_document_original_file_url(DOCUMENT_ID), document_original_file_url)

    def test_get_document_api_details_url(self):
        """Test function used to generate url to access document details of one document in a project."""
        document_api_details_url = (
            f'{KONFUZIO_HOST}/api/projects/{KONFUZIO_PROJECT_ID}/docs/{DOCUMENT_ID}/'
            f'?include_extractions={False}&extra_fields={"bbox"}'
        )
        self.assertEqual(get_document_api_details_url(DOCUMENT_ID), document_api_details_url)

    def test_get_project_url(self):
        """Test function used to generate url to update project details."""
        project_url = f'{KONFUZIO_HOST}/api/projects/{KONFUZIO_PROJECT_ID}/'
        self.assertEqual(get_project_url(), project_url)

    def test_post_project_api_document_annotations_url(self):
        """Test function used to generate url which adds annotations of a document."""
        project_api_document_annotations_url = (
            f'{KONFUZIO_HOST}/api/projects/{KONFUZIO_PROJECT_ID}/docs/' f'{DOCUMENT_ID}/annotations/'
        )
        self.assertEqual(post_project_api_document_annotations_url(DOCUMENT_ID), project_api_document_annotations_url)

    def test_delete_project_api_document_annotations_url(self):
        """Test function used to generate url to delete annotations of a document."""
        project_api_document_annotations_url = (
            f'{KONFUZIO_HOST}/api/projects/{KONFUZIO_PROJECT_ID}'
            f'/docs/{DOCUMENT_ID}/'
            f'annotations/{ANNOTATION_ID}/'
        )
        self.assertEqual(
            delete_project_api_document_annotations_url(DOCUMENT_ID, ANNOTATION_ID),
            project_api_document_annotations_url,
        )

    def test_get_document_result_v1(self):
        """Test function used to generate url to access web interface for labeling of this project."""
        document_result_v1 = f'{KONFUZIO_HOST}/api/v1/docs/{DOCUMENT_ID}/'
        self.assertEqual(get_document_result_v1(DOCUMENT_ID), document_result_v1)

    def test_get_document_segmentation_details_url(self):
        """Test function used to generate url to access document details of one document in a project."""
        document_segmentation_details_url = (
            f'{KONFUZIO_HOST}/api/projects/{KONFUZIO_PROJECT_ID}/docs/' f'{DOCUMENT_ID}/segmentation/'
        )
        self.assertEqual(
            get_document_segmentation_details_url(DOCUMENT_ID, KONFUZIO_PROJECT_ID), document_segmentation_details_url
        )
