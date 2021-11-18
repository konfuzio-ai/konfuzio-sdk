"""URLs to the endpoints of Konfuzio Host."""

import logging

from konfuzio_sdk import KONFUZIO_HOST, KONFUZIO_PROJECT_ID
from typing import Union

logger = logging.getLogger(__name__)


# TOKEN-AUTH


def get_auth_token_url(host: str) -> str:
    """
    Generate URL that creates an authentication token for the user.

    :return: URL to generate the token.
    """
    return f"{host}/api/token-auth/"


# PROJECTS


def get_projects_list_url(host: str = KONFUZIO_HOST) -> str:
    """
    Generate URL to list all the projects available for the user.

    :return: URL to get all the projects for the user.
    """
    return f"{host}/api/projects/"


def get_project_url(project_id: Union[int, None] = None, host: str = KONFUZIO_HOST) -> str:
    """
    Generate URL to access the project details.

    :param project_id: ID of the project
    :param host: Konfuzio host
    :return: URL to access the project details.
    """
    project_id = project_id if project_id else KONFUZIO_PROJECT_ID
    return f'{host}/api/projects/{project_id}/'


def get_documents_meta_url(host: str = KONFUZIO_HOST, project_id: int = KONFUZIO_PROJECT_ID) -> str:
    """
    Generate URL to load meta information about the documents in the project.

    :param host: Konfuzio host
    :param project_id: ID of the project
    :return: URL to get all the documents details.
    """
    return f"{host}/api/projects/{project_id}/docs/"


def get_document_annotations_url(
    document_id: int, host: str = KONFUZIO_HOST, project_id: int = KONFUZIO_PROJECT_ID
) -> str:
    """
    Access annotations of a document.

    :param document_id: ID of the document as integer
    :param host: Konfuzio host
    :param project_id: ID of the project
    :return: URL to access the annotations of a document
    """
    return f'{host}/api/projects/{project_id}/docs/{document_id}/annotations/'


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


# DOCUMENTS


def get_upload_document_url(host: str = KONFUZIO_HOST) -> str:
    """
    Generate URL to upload a document.

    :param host: Konfuzio host
    :return: URL to upload a document
    """
    return f"{host}/api/v2/docs/"


def get_document_url(document_id: int, host: str = KONFUZIO_HOST) -> str:
    """
    Generate URL to access a document.

    :param document_id: ID of the document as integer
    :param host: Konfuzio host
    :return: URL to access a document
    """
    return f"{host}/api/v2/docs/{document_id}/"


def get_document_ocr_file_url(document_id: int, host: str = KONFUZIO_HOST) -> str:
    """
    Generate URL to get the OCR version of the document.

    :param document_id: ID of the document as integer
    :param host: Konfuzio host
    :return: URL to get OCR document file.
    """
    return f'{host}/doc/show/{document_id}/'


def get_document_original_file_url(document_id: int, host: str = KONFUZIO_HOST) -> str:
    """
    Generate URL to get the original version of the document.

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
    Generate URL to access the details of a document in a project.

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


# LABELS


def get_labels_url(host: str = KONFUZIO_HOST) -> str:
    """
    Generate URL to list all labels for the user.

    :param host: Konfuzio host
    :return: URL to list all labels for the user.
    """
    return f"{host}/api/v2/labels/"


def get_label_url(label_id: int, host: str = KONFUZIO_HOST) -> str:
    """
    Generate URL to access a label.

    :param label_id: ID of the label as integer
    :param host: Konfuzio host
    :return: URL to access a label
    """
    return f"{host}/api/v2/labels/{label_id}/"


# ANNOTATIONS


def get_annotation_url(
    document_id: int, annotation_id: int, host: str = KONFUZIO_HOST, project_id: int = KONFUZIO_PROJECT_ID
) -> str:
    """
    Generate URL to access an annotation.

    :param document_id: ID of the document as integer
    :param annotation_id: ID of the annotation as integer
    :param host: Konfuzio host
    :param project_id: ID of the project
    :return: URL to access an annotation of a document
    """
    return f'{host}/api/projects/{project_id}/docs/{document_id}/annotations/{annotation_id}/'
