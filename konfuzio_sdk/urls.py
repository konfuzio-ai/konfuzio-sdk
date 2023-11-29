"""URLs to the endpoints of Konfuzio Host."""

import logging

from konfuzio_sdk import KONFUZIO_HOST
from typing import Union

logger = logging.getLogger(__name__)


# TOKEN-AUTH


def get_auth_token_url(host: str = None) -> str:
    """
    Generate URL that creates an authentication token for the user.

    :param host: Konfuzio host
    :return: URL to generate the token.
    """
    if host is None:
        host = KONFUZIO_HOST
    return f"{host}/api/token-auth/"


# PROJECTS


def get_projects_list_url(host: str = None) -> str:
    """
    Generate URL to list all the Projects available for the user.

    :param host: Konfuzio host
    :return: URL to get all the Projects for the user.
    """
    if host is None:
        host = KONFUZIO_HOST
    return f"{host}/api/projects/"


def get_project_url(project_id: Union[int, None], host: str = None) -> str:
    """
    Generate URL to access the Project details.

    :param host: Konfuzio host
    :param project_id: ID of the Project
    :return: URL to access the Project details.
    """
    if host is None:
        host = KONFUZIO_HOST
    return f'{host}/api/projects/{project_id}/'


def get_documents_meta_url(project_id: int, limit: int = 10, host: str = None) -> str:
    """
    Generate URL to load meta information about the Documents in the Project.

    :param project_id: ID of the Project
    :param host: Konfuzio host
    :return: URL to get all the Documents details.
    """
    if host is None:
        host = KONFUZIO_HOST
    return f"{host}/api/projects/{project_id}/docs/?limit={limit}"


def get_document_segmentation_details_url(
        document_id: int, project_id: int, host: str = None, action='segmentation'
) -> str:
    """
    Generate URL to get the segmentation results of a  Document.

    :param document_id: ID of the Document as integer
    :param project_id: ID of the Project
    :param host: Konfuzio host
    :param action: Action from where to get the results
    :return: URL to access the segmentation results of a  Document
    """
    if host is None:
        host = KONFUZIO_HOST
    return f'{host}/api/projects/{project_id}/docs/{document_id}/{action}/'


def get_extraction_ais_list_url(project_id: int, host: str = None) -> str:
    """
    Generate URL to get a list of Extraction AIs for a specific project.

    :param project_id: ID of the Project
    :param host: Konfuzio host
    :return: URL to get all Extraction AIs for a specific project
    """
    if host is None:
        host = KONFUZIO_HOST
    return f"{host}/api/v3/extraction-ais/?limit=1000&project_id={project_id}"


def get_splitting_ais_list_url(project_id: int, host: str = None) -> str:
    """
    Generate URL to get a list of Splitting AIs for a specific project.

    :param project_id: ID of the Project
    :param host: Konfuzio host
    :return: URL to get all Splitting AIs for a specific project
    """
    if host is None:
        host = KONFUZIO_HOST
    return f"{host}/api/v3/splitting-ais/?limit=1000&project_id={project_id}"


def get_categorization_ais_list_url(project_id: int, host: str = None) -> str:
    """
    Generate URL to get a list of Categorization AIs for a specific project.

    :param project_id: ID of the Project
    :param host: Konfuzio host
    :return: URL to get all Categorization AIs for a specific project
    """
    if host is None:
        host = KONFUZIO_HOST
    return f"{host}/api/v3/category-ais/?limit=1000&project_id={project_id}"


# DOCUMENTS


def get_upload_document_url(host: str = None) -> str:
    """
    Generate URL to upload a  Document.

    :param host: Konfuzio host
    :return: URL to upload a  Document
    """
    if host is None:
        host = KONFUZIO_HOST
    return f"{host}/api/v2/docs/"


def get_document_url(document_id: int, host: str = None) -> str:
    """
    Generate URL to access a  Document.

    :param document_id: ID of the Document as integer
    :param host: Konfuzio host
    :return: URL to access a  Document
    """
    if host is None:
        host = KONFUZIO_HOST
    return f"{host}/api/v2/docs/{document_id}/"


def get_document_ocr_file_url(document_id: int, host: str = None) -> str:
    """
    Generate URL to get the OCR version of the document.

    :param document_id: ID of the Document as integer
    :param host: Konfuzio host
    :return: URL to get OCR Document file.
    """
    if host is None:
        host = KONFUZIO_HOST
    return f'{host}/doc/show/{document_id}/'


def get_document_original_file_url(document_id: int, host: str = None) -> str:
    """
    Generate URL to get the original version of the document.

    :param document_id: ID of the Document as integer
    :param host: Konfuzio host
    :return: URL to get the original document
    """
    if host is None:
        host = KONFUZIO_HOST
    return f'{host}/doc/show-original/{document_id}/'


def get_page_image_url(page_id: int, host: str = None) -> str:
    """
    Generate URL to get Page as Image.

    :param page_id: ID of the Page
    :return: URL to get Page as PNG
    """
    if host is None:
        host = KONFUZIO_HOST
    return f'{host}/page/show-image/{page_id}/'


def get_document_api_details_url(
    document_id: int, project_id: int, host: str = None, extra_fields='bbox'
) -> str:
    """
    Generate URL to access the details of a Document in a Project.

    :param document_id: ID of the Document as integer
    :param project_id: ID of the Project
    :param host: Konfuzio host
    :param extra_fields: Extra information to include in the response
    :return: URL to get Document details
    """
    if host is None:
        host = KONFUZIO_HOST
    return f'{host}/api/projects/{project_id}/docs/{document_id}/?extra_fields={extra_fields}'


def get_annotation_view_url(annotation_id: int, host: str = None) -> str:
    """
    Generate URL to see Annotation in the SmartView.

    :param annotation_id: ID of the Annotation
    :return: URL to get visually access Annotation online.
    """
    if host is None:
        host = KONFUZIO_HOST
    return f'{host}/a/{annotation_id}'


# LABELS


def get_labels_url(host: str = None) -> str:
    """
    Generate URL to list all Labels for the user.

    :param host: Konfuzio host
    :return: URL to list all Labels for the user.
    """
    if host is None:
        host = KONFUZIO_HOST
    return f"{host}/api/v2/labels/"


def get_label_url(label_id: int, host: str = None) -> str:
    """
    Generate URL to access a Label.

    :param label_id: ID of the Label as integer
    :param host: Konfuzio host
    :return: URL to access a Label
    """
    if host is None:
        host = KONFUZIO_HOST
    return f"{host}/api/v2/labels/{label_id}/"


# ANNOTATIONS


def get_document_annotations_url(document_id: int, project_id: int, host: str = None) -> str:
    """
    Access Annotations of a document.

    :param document_id: ID of the Document as integer
    :param project_id: ID of the project
    :param host: Konfuzio host
    :return: URL to access the Annotations of a document
    """
    if host is None:
        host = KONFUZIO_HOST
    return f'{host}/api/projects/{project_id}/docs/{document_id}/annotations/'


def get_annotation_url(document_id: int, annotation_id: int, project_id: int, host: str = None) -> str:
    """
    Generate URL to access an annotation.

    :param document_id: ID of the Document as integer
    :param annotation_id: ID of the Annotation as integer
    :param project_id: ID of the project
    :param host: Konfuzio host
    :return: URL to access an Annotation of a document
    """
    if host is None:
        host = KONFUZIO_HOST
    return f'{host}/api/projects/{project_id}/docs/{document_id}/annotations/{annotation_id}/'


def get_create_ai_model_url(ai_type: str, host: str = None) -> str:
    """
    Get url to create new AiModel.

    :return: URL
    """
    if host is None:
        host = KONFUZIO_HOST
    if ai_type == 'extraction':
        return f'{host}/api/v3/extraction-ais/upload/'
    elif ai_type == 'categorization':
        return f'{host}/api/v3/category-ais/upload/'
    elif ai_type == 'filesplitting':
        return f'{host}/api/v3/splitting-ais/upload/'


def get_update_ai_model_url(ai_model_id, host: str = None) -> str:
    """
    Get url to update an AiModel.

    :return: URL
    """
    if host is None:
        host = KONFUZIO_HOST
    return f'{host}/api/aimodels/{ai_model_id}/'


def get_ai_model_url(ai_model_id: int, ai_type: str, host: str = None) -> str:
    """
    Get url to modify or delete an AI model.

    :return: a dictionary of potential URLs
    """
    if host is None:
        host = KONFUZIO_HOST
    if ai_type == 'extraction':
        return f'{host}/api/v3/extraction-ais/{ai_model_id}/'
    elif ai_type == 'categorization':
        return f'{host}/api/v3/category-ais/{ai_model_id}/'
    elif ai_type == 'filesplitting':
        return f'{host}/api/v3/splitting-ais/{ai_model_id}/'
    else:
        raise ValueError


def get_ai_model_download_url(ai_model_id: int, host: str = None) -> str:
    """
    Get url to download an AI model.

    @param ai_model_id:  ID of the AI model
    @param host: Konfuzio host
    @return: URL
    """
    if host is None:
        host = KONFUZIO_HOST
    return f"{host}/aimodel/file/{ai_model_id}/"
