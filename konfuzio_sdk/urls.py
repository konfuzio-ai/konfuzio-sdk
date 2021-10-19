"""Endpoints of the Konfuzio Host."""

import logging

from konfuzio_sdk import KONFUZIO_HOST, KONFUZIO_PROJECT_ID
from typing import Union

logger = logging.getLogger(__name__)


def get_auth_token_url(host: str) -> str:
    """
    Generate URL that creates an authentication token for the user.

    :return: URL to generate the token.
    """
    return f"{host}/api/token-auth/"


def get_project_list_url(host: str = KONFUZIO_HOST) -> str:
    """
    Generate URL to load all the projects available for the user.

    :return: URL to get all the projects for the user.
    """
    return f"{host}/api/projects/"


def create_new_project_url(host: str = KONFUZIO_HOST) -> str:
    """
    Generate URL to create a new project.

    :param host: Konfuzio host
    :return: URL to create a new project.
    """
    return f"{host}/api/projects/"


def get_documents_meta_url(host: str = KONFUZIO_HOST, project_id: int = KONFUZIO_PROJECT_ID) -> str:
    """
    Generate URL to load meta information about documents.

    :param host: Konfuzio host
    :param project_id: ID of the project
    :return: URL to get all the documents details.
    """
    return f"{host}/api/projects/{project_id}/docs/"


def get_upload_document_url(host: str = KONFUZIO_HOST) -> str:
    """
    Generate URL to upload a document.

    :param host: Konfuzio host
    :return: URL to upload a document
    """
    return f"{host}/api/v2/docs/"


def update_document_url(document_id: int, host: str = KONFUZIO_HOST) -> str:
    """
    Generate URL to update a document.

    :param document_id: ID of the document as integer
    :param host: Konfuzio host
    :return: URL to update a document
    """
    return f"{host}/api/v2/docs/{document_id}/"


def get_create_label_url(host: str = KONFUZIO_HOST) -> str:
    """
    Generate URL to create a label.

    :param host: Konfuzio host
    :return: URL to create a label.
    """
    return f"{host}/api/v2/labels/"


def get_document_ocr_file_url(document_id: int, host: str = KONFUZIO_HOST) -> str:
    """
    Generate URL to access OCR version of document.

    :param document_id: ID of the document as integer
    :param host: Konfuzio host
    :return: URL to get OCR document file.
    """
    return f'{host}/doc/show/{document_id}/'


def get_document_original_file_url(document_id: int, host: str = KONFUZIO_HOST) -> str:
    """
    Generate URL to access original version of the document.

    :param document_id: ID of the document as integer
    :param host: Konfuzio host
    :return: URL to get the original document
    """
    return f'{host}/doc/show-original/{document_id}/'


def get_document_api_details_url(
    document_id: int,
    host: str = KONFUZIO_HOST,
    project_id: int = KONFUZIO_PROJECT_ID,
    include_extractions: bool = False,
    extra_fields='bbox',
) -> str:
    """
    Generate URL to access document details of one document in a project.

    :param document_id: ID of the document as integer
    :param host: Konfuzio host
    :param project_id: ID of the project
    :param include_extractions: Bool to include extractions
    :param extra_fields: Extra information to include in the response
    :return: URL to get document details
    """
    return (
        f'{host}/api/projects/{project_id}/docs/{document_id}/'
        f'?include_extractions={include_extractions}&extra_fields={extra_fields}'
    )


def get_project_url(project_id: Union[int, None] = None, host: str = KONFUZIO_HOST) -> str:
    """
    Generate URL to get project details.

    :param project_id: ID of the project
    :param host: Konfuzio host
    :return: URL to get project details.
    """
    project_id = project_id if project_id else KONFUZIO_PROJECT_ID
    return f'{host}/api/projects/{project_id}/'


def post_project_api_document_annotations_url(
    document_id: int, host: str = KONFUZIO_HOST, project_id: int = KONFUZIO_PROJECT_ID
) -> str:
    """
    Add new annotations to a document.

    :param document_id: ID of the document as integer
    :param host: Konfuzio host
    :param project_id: ID of the project
    :return: URL for adding annotations to a document
    """
    return f'{host}/api/projects/{project_id}/docs/{document_id}/annotations/'


def delete_project_api_document_annotations_url(
    document_id: int, annotation_id: int, host: str = KONFUZIO_HOST, project_id: int = KONFUZIO_PROJECT_ID
) -> str:
    """
    Delete the annotation of a document.

    :param document_id: ID of the document as integer
    :param annotation_id: ID of the annotation as integer
    :param host: Konfuzio host
    :param project_id: ID of the project
    :return: URL to delete annotation of a document
    """
    return f'{host}/api/projects/{project_id}/docs/{document_id}/' f'annotations/{annotation_id}/'


def get_document_result_v1(document_id: int, host: str = KONFUZIO_HOST) -> str:
    """
    Generate URL to access web interface for labeling of this project.

    :param document_id: ID of the document as integer
    :param host: Konfuzio host
    :return: URL for labeling of the project.
    """
    return f'{host}/api/v1/docs/{document_id}/'


def get_document_segmentation_details_url(
    document_id: int, project_id: int = KONFUZIO_PROJECT_ID, host: str = KONFUZIO_HOST, action='segmentation'
) -> str:
    """
    Generate URL to get the segmentation results of a document.

    :param document_id: ID of the document as integer
    :param project_id: ID of the project
    :param host: Konfuzio host
    :param action: Action from where to get the results
    :return: URL to access the segmentation results of a document
    """
    return f'{host}/api/projects/{project_id}/docs/{document_id}/{action}/'
