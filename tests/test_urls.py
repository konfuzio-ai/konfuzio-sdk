"""Validate urls functions."""
import unittest

from konfuzio_sdk import KONFUZIO_HOST
from konfuzio_sdk.urls import (
    get_ai_model_download_url,
    get_ai_model_url,
    get_annotation_url,
    get_annotation_view_url,
    get_auth_token_url,
    get_create_ai_model_url,
    get_document_annotations_url,
    get_document_api_details_url,
    get_document_ocr_file_url,
    get_document_original_file_url,
    get_document_segmentation_details_url,
    get_document_url,
    get_documents_meta_url,
    get_label_url,
    get_labels_url,
    get_page_image_url,
    get_project_url,
    get_projects_list_url,
    get_update_ai_model_url,
    get_upload_document_url,
)
from tests.variables import TEST_PROJECT_ID

DOCUMENT_ID = 3346
ANNOTATION_ID = 56
LABEL_ID = 12
PAGE_ID = 1
AI_ID = 1


class TestUrls(unittest.TestCase):
    """Testing endpoints of the Konfuzio Host."""

    def test_get_auth_token_url(self):
        """Test function used to generate url to create an authentication token for the user."""
        auth_token_url_url = f'{KONFUZIO_HOST}/api/token-auth/'
        self.assertEqual(get_auth_token_url(host=KONFUZIO_HOST), auth_token_url_url)

    def test_get_projects_list_url(self):
        """Test function used to generate url to list all the Projects available for the user."""
        projects_list_url = f'{KONFUZIO_HOST}/api/projects/'
        self.assertEqual(get_projects_list_url(), projects_list_url)

    def test_get_project_url(self):
        """Test function used to generate url to access Project details."""
        project_url = f'{KONFUZIO_HOST}/api/projects/{TEST_PROJECT_ID}/'
        self.assertEqual(get_project_url(TEST_PROJECT_ID), project_url)

    def test_get_documents_meta_url(self):
        """Test function used to generate url to load meta information about Documents."""
        document_meta_url = f'{KONFUZIO_HOST}/api/projects/{TEST_PROJECT_ID}/docs/?limit=10'
        self.assertEqual(document_meta_url, get_documents_meta_url(project_id=TEST_PROJECT_ID))

    def test_get_document_annotations_url(self):
        """Test function used to generate url to access Annotations from a document."""
        document_annotations_url = f'{KONFUZIO_HOST}/api/projects/{TEST_PROJECT_ID}/docs/{DOCUMENT_ID}/annotations/'
        self.assertEqual(
            get_document_annotations_url(document_id=DOCUMENT_ID, project_id=TEST_PROJECT_ID), document_annotations_url
        )

    def test_get_document_segmentation_details_url(self):
        """Test function used to generate url to get the segmentation details of one Document in a Project."""
        document_segmentation_details_url = (
            f'{KONFUZIO_HOST}/api/projects/{TEST_PROJECT_ID}/docs/{DOCUMENT_ID}/segmentation/'
        )
        self.assertEqual(
            get_document_segmentation_details_url(DOCUMENT_ID, TEST_PROJECT_ID), document_segmentation_details_url
        )

    def test_get_upload_document_url(self):
        """Test function used to generate url to upload a document."""
        upload_document_url = f'{KONFUZIO_HOST}/api/v2/docs/'
        self.assertEqual(get_upload_document_url(), upload_document_url)

    def test_get_document_url(self):
        """Test function used to generate url to access a document."""
        document_url = f'{KONFUZIO_HOST}/api/v2/docs/{DOCUMENT_ID}/'
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
        """Test function used to generate url to access Document details of one Document in a Project."""
        document_api_details_url = (
            f'{KONFUZIO_HOST}/api/projects/{TEST_PROJECT_ID}/docs/{DOCUMENT_ID}/' f'?extra_fields=bbox'
        )
        self.assertEqual(
            get_document_api_details_url(DOCUMENT_ID, project_id=TEST_PROJECT_ID), document_api_details_url
        )

    def test_get_labels_url(self):
        """Test function used to generate url to list all Labels."""
        labels_url = f'{KONFUZIO_HOST}/api/v2/labels/'
        self.assertEqual(get_labels_url(), labels_url)

    def test_get_label_url(self):
        """Test function used to generate url to access a label."""
        label_url = f'{KONFUZIO_HOST}/api/v2/labels/{LABEL_ID}/'
        self.assertEqual(get_label_url(label_id=LABEL_ID), label_url)

    def test_get_annotation_url(self):
        """Test function used to generate url to access an Annotation of a document."""
        annotation_url = (
            f'{KONFUZIO_HOST}/api/projects/{TEST_PROJECT_ID}' f'/docs/{DOCUMENT_ID}/' f'annotations/{ANNOTATION_ID}/'
        )
        self.assertEqual(get_annotation_url(DOCUMENT_ID, ANNOTATION_ID, TEST_PROJECT_ID), annotation_url)

    def test_annotation_view_url(self):
        """Test to access Annotation online."""
        self.assertEqual(get_annotation_view_url(ANNOTATION_ID), f'{KONFUZIO_HOST}/a/{ANNOTATION_ID}')

    def test_page_image_url(self):
        """Test to access Page online."""
        self.assertEqual(get_page_image_url(PAGE_ID), f'{KONFUZIO_HOST}/page/show-image/{PAGE_ID}/')

    def test_upload_ai_url(self):
        """Test to get URL to upload new AI file."""
        self.assertEqual(
            get_create_ai_model_url(ai_type='extraction'), f'{KONFUZIO_HOST}/api/v3/extraction-ais/upload/'
        )
        self.assertEqual(
            get_create_ai_model_url(ai_type='categorization'), f'{KONFUZIO_HOST}/api/v3/category-ais/upload/'
        )
        self.assertEqual(
            get_create_ai_model_url(ai_type='filesplitting'), f'{KONFUZIO_HOST}/api/v3/splitting-ais/upload/'
        )

    def test_get_ai_model_url(self):
        """Test to get URL to upload new AI file."""
        ai_model_id = 100
        self.assertEqual(
            get_ai_model_url(ai_type='extraction', ai_model_id=ai_model_id),
            f'{KONFUZIO_HOST}/api/v3/extraction-ais/{ai_model_id}/',
        )
        self.assertEqual(
            get_ai_model_url(ai_type='categorization', ai_model_id=ai_model_id),
            f'{KONFUZIO_HOST}/api/v3/category-ais/{ai_model_id}/',
        )
        self.assertEqual(
            get_ai_model_url(ai_type='filesplitting', ai_model_id=ai_model_id),
            f'{KONFUZIO_HOST}/api/v3/splitting-ais/{ai_model_id}/',
        )

    def test_download_ai_url(self):
        """Test to get URL to download AI file."""
        ai_model_id = 100
        self.assertEqual(
            get_ai_model_download_url(ai_model_id=ai_model_id, host=KONFUZIO_HOST),
            f'{KONFUZIO_HOST}/aimodel/file/{ai_model_id}/',
        )

    def test_change_ai_url(self):
        """Test to get URL to change AI."""
        self.assertEqual(get_update_ai_model_url(AI_ID), f'{KONFUZIO_HOST}/api/aimodels/{AI_ID}/')
