"""Validate urls functions."""
import unittest
from konfuzio_sdk.urls import (
    get_auth_token_url,
    get_projects_list_url,
    get_documents_meta_url,
    get_upload_document_url,
    get_document_ocr_file_url,
    get_document_original_file_url,
    get_document_api_details_url,
    get_project_url,
    get_document_segmentation_details_url,
    get_document_url,
    get_label_url,
    get_labels_url,
    get_annotation_url,
    get_document_annotations_url,
)


from konfuzio_sdk import KONFUZIO_HOST, KONFUZIO_PROJECT_ID

DOCUMENT_ID = 3346
ANNOTATION_ID = 56
LABEL_ID = 12


class TestUrls(unittest.TestCase):
    """Testing endpoints of the Konfuzio Host."""

    def test_get_auth_token_url(self):
        """Test function used to generate url to create an authentication token for the user."""
        auth_token_url_url = f"{KONFUZIO_HOST}/api/token-auth/"
        self.assertEqual(get_auth_token_url(host=KONFUZIO_HOST), auth_token_url_url)

    def test_get_projects_list_url(self):
        """Test function used to generate url to list all the projects available for the user."""
        projects_list_url = f"{KONFUZIO_HOST}/api/projects/"
        self.assertEqual(get_projects_list_url(), projects_list_url)

    def test_get_project_url(self):
        """Test function used to generate url to access project details."""
        project_url = f'{KONFUZIO_HOST}/api/projects/{KONFUZIO_PROJECT_ID}/'
        self.assertEqual(get_project_url(), project_url)

    def test_get_documents_meta_url(self):
        """Test function used to generate url to load meta information about documents."""
        document_meta_url = f"{KONFUZIO_HOST}/api/projects/{KONFUZIO_PROJECT_ID}/docs/"
        self.assertEqual(get_documents_meta_url(), document_meta_url)

    def test_get_document_annotations_url(self):
        """Test function used to generate url to access annotations from a document."""
        document_annotations_url = f"{KONFUZIO_HOST}/api/projects/{KONFUZIO_PROJECT_ID}/docs/{DOCUMENT_ID}/annotations/"
        self.assertEqual(get_document_annotations_url(document_id=DOCUMENT_ID), document_annotations_url)

    def test_get_document_segmentation_details_url(self):
        """Test function used to generate url to get the segmentation details of one document in a project."""
        document_segmentation_details_url = (
            f'{KONFUZIO_HOST}/api/projects/{KONFUZIO_PROJECT_ID}/docs/{DOCUMENT_ID}/segmentation/'
        )
        self.assertEqual(
            get_document_segmentation_details_url(DOCUMENT_ID, KONFUZIO_PROJECT_ID), document_segmentation_details_url
        )

    def test_get_upload_document_url(self):
        """Test function used to generate url to upload a document."""
        upload_document_url = f"{KONFUZIO_HOST}/api/v2/docs/"
        self.assertEqual(get_upload_document_url(), upload_document_url)

    def test_get_document_url(self):
        """Test function used to generate url to access a document."""
        document_url = f"{KONFUZIO_HOST}/api/v2/docs/{DOCUMENT_ID}/"
        self.assertEqual(get_document_url(document_id=DOCUMENT_ID), document_url)

    def test_get_document_ocr_file_url(self):
        """Test function used to generate url to access the OCR version of the document."""
        document_ocr_file_url = f'{KONFUZIO_HOST}/doc/show/{DOCUMENT_ID}/'
        self.assertEqual(get_document_ocr_file_url(DOCUMENT_ID), document_ocr_file_url)

    def test_get_document_original_file_url(self):
        """Test function used to generate url to access the original version of the document."""
        document_original_file_url = f'{KONFUZIO_HOST}/doc/show-original/{DOCUMENT_ID}/'
        self.assertEqual(get_document_original_file_url(DOCUMENT_ID), document_original_file_url)

    def test_get_document_api_details_url(self):
        """Test function used to generate url to access document details of one document in a project."""
        document_api_details_url = (
            f'{KONFUZIO_HOST}/api/projects/{KONFUZIO_PROJECT_ID}/docs/{DOCUMENT_ID}/'
            f'?include_extractions={False}&extra_fields={"bbox"}'
        )
        self.assertEqual(get_document_api_details_url(DOCUMENT_ID), document_api_details_url)

    def test_get_labels_url(self):
        """Test function used to generate url to list all labels."""
        labels_url = f"{KONFUZIO_HOST}/api/v2/labels/"
        self.assertEqual(get_labels_url(), labels_url)

    def test_get_label_url(self):
        """Test function used to generate url to access a label."""
        label_url = f"{KONFUZIO_HOST}/api/v2/labels/{LABEL_ID}/"
        self.assertEqual(get_label_url(label_id=LABEL_ID), label_url)

    def test_get_annotation_url(self):
        """Test function used to generate url to access an annotation of a document."""
        annotation_url = (
            f'{KONFUZIO_HOST}/api/projects/{KONFUZIO_PROJECT_ID}'
            f'/docs/{DOCUMENT_ID}/'
            f'annotations/{ANNOTATION_ID}/'
        )
        self.assertEqual(get_annotation_url(DOCUMENT_ID, ANNOTATION_ID), annotation_url)
