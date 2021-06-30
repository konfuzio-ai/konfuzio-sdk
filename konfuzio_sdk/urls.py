"""Endpoints of the Konfuzio Host."""

import logging

from konfuzio_sdk import KONFUZIO_HOST, KONFUZIO_PROJECT_ID

logger = logging.getLogger(__name__)


def get_auth_token_url() -> str:
    """
    Generate URL that creates an authentication token for the user.

    :return: URL to generate the token.
    """
    return f"{KONFUZIO_HOST}/api/token-auth/"


def get_project_list_url() -> str:
    """
    Generate URL to load all the projects available for the user.

    :return: URL to get all the projects for the user.
    """
    return f"{KONFUZIO_HOST}/api/projects/"


def create_new_project_url() -> str:
    """
    Generate URL to create a new project.

    :return: URL to create a new project.
    """
    return f"{KONFUZIO_HOST}/api/projects/"


def get_documents_meta_url() -> str:
    """
    Generate URL to load meta information about documents.

    :return: URL to get all the documents details.
    """
    return f"{KONFUZIO_HOST}/api/projects/{KONFUZIO_PROJECT_ID}/docs/"


def get_upload_document_url() -> str:
    """
    Generate URL to upload a document.

    :return: URL to upload a document
    """
    return f"{KONFUZIO_HOST}/api/v2/docs/"


def update_document_url(document_id: int) -> str:
    """
    Generate URL to update a document.

    :return: URL to update a document
    """
    return f"{KONFUZIO_HOST}/api/v2/docs/{document_id}/"


def get_create_label_url() -> str:
    """
    Generate URL to create a label.

    :return: URL to create a label.
    """
    return f"{KONFUZIO_HOST}/api/v2/labels/"


def get_document_ocr_file_url(document_id: int) -> str:
    """
    Generate URL to access OCR version of document.

    :param document_id: ID of the document as integer
    :return: URL to get OCR document file.
    """
    return f'{KONFUZIO_HOST}/doc/show/{document_id}/'


def get_document_original_file_url(document_id: int) -> str:
    """
    Generate URL to access original version of the document.

    :param document_id: ID of the document as integer
    :return: URL to get the original document
    """
    return f'{KONFUZIO_HOST}/doc/show-original/{document_id}/'


def get_document_api_details_url(document_id: int, include_extractions: bool = False, extra_fields='bbox') -> str:
    """
    Generate URL to access document details of one document in a project.

    :param document_id: ID of the document as integer
    :param include_extractions: Bool to include extractions
    :param extra_fields: Extra information to include in the response
    :return: URL to get document details
    """
    return (
        f'{KONFUZIO_HOST}/api/projects/{KONFUZIO_PROJECT_ID}/docs/{document_id}/'
        f'?include_extractions={include_extractions}&extra_fields={extra_fields}'
    )


def get_project_url(project_id=None) -> str:
    """
    Generate URL to get project details.

    :param project_id: ID of the project
    :return: URL to get project details.
    """
    project_id = project_id if project_id else KONFUZIO_PROJECT_ID
    return f'{KONFUZIO_HOST}/api/projects/{project_id}/'


def post_project_api_document_annotations_url(document_id: int) -> str:
    """
    Add new annotations to a document.

    :param document_id: ID of the document as integer
    :return: URL for adding annotations to a document
    """
    return f'{KONFUZIO_HOST}/api/projects/{KONFUZIO_PROJECT_ID}/docs/{document_id}/annotations/'


def delete_project_api_document_annotations_url(document_id: int, annotation_id: int) -> str:
    """
    Delete the annotation of a document.

    :param document_id: ID of the document as integer
    :param annotation_id: ID of the annotation as integer
    :return: URL to delete annotation of a document
    """
    return f'{KONFUZIO_HOST}/api/projects/{KONFUZIO_PROJECT_ID}/docs/{document_id}/' f'annotations/{annotation_id}/'


def get_document_result_v1(document_id: int) -> str:
    """
    Generate URL to access web interface for labeling of this project.

    :param document_id: ID of the document as integer
    :return: URL for labeling of the project.
    """
    return f'{KONFUZIO_HOST}/api/v1/docs/{document_id}/'


def get_document_segmentation_details_url(document_id: int, project_id, action='segmentation') -> str:
    """
    Generate URL to get the segmentation results of a document.

    :param document_id: ID of the document as integer
    :param project_id: ID of the project
    :param action: Action from where to get the results
    :return: URL to access the segmentation results of a document
    """
    return f'{KONFUZIO_HOST}/api/projects/{project_id}/docs/{document_id}/{action}/'
