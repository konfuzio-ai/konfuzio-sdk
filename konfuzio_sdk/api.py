"""Connect to the Konfuzio Server to receive or send data."""

import io
import json
import logging
import os
import time
from operator import itemgetter
from typing import List, Union
from urllib.parse import urlparse

import requests
from konfuzio_sdk import KONFUZIO_HOST, KONFUZIO_TOKEN
from konfuzio_sdk.urls import (
    get_auth_token_url,
    get_projects_list_url,
    get_document_api_details_url,
    get_project_url,
    get_document_ocr_file_url,
    get_document_original_file_url,
    get_documents_meta_url,
    get_document_annotations_url,
    get_annotation_url,
    get_upload_document_url,
    get_document_url,
    get_document_segmentation_details_url,
    get_labels_url,
)
from konfuzio_sdk.utils import is_file, load_image

logger = logging.getLogger(__name__)


def get_auth_token(username, password, host):
    """
    Generate the authentication token for the user.

    :return: The new generated token.
    """
    url = get_auth_token_url(host)
    user_credentials = {"username": username, "password": password}
    r = requests.post(url, json=user_credentials)
    return r


def get_project_list(token, host):
    """
    Get the list of all projects for the user.

    :return: Response object
    """
    session = requests.Session()
    session.headers.update({'Authorization': f'Token {token}'})
    url = get_projects_list_url(host)
    r = session.get(url=url)
    return r


def create_new_project(project_name, token=None, host=None):
    """
    Create a new project for the user.

    :return: Response object
    """
    session = konfuzio_session(token)
    url = get_projects_list_url(host)
    new_project_data = {"name": project_name}
    r = session.post(url=url, json=new_project_data)
    return r


def get_project_name_from_id(project_id: int, host: str) -> str:
    """
    Get the project name given the project_id.

    :param project_id: ID of the project
    :return: Name of the project in JSON format.
    """
    session = konfuzio_session()
    url = get_project_url(project_id, host)
    r = session.get(url=url)
    return r.json()['name']


def retry_get(session, url):
    """
    Workaround to avoid exceptions in case the server does not respond.

    :param session: Working session
    :param url: Url of the endpoint
    :return: Response.
    """
    # Retry if server is down.
    retry_count = 0

    while True:
        try:
            r = session.get(url=url, timeout=10.0)
        except requests.RequestException:
            logger.warning(f'Retry to get url >>{url}<<')
            retry_count += 1
            if retry_count >= 10:
                raise
            time.sleep(15)
            continue

        try:
            r.raise_for_status()
            break
        except requests.exceptions.HTTPError:
            if 401 <= r.status_code <= 403:
                raise ConnectionError(f'Problem with credentials: {json.loads(r.text)["detail"]}')

            elif r.status_code == 404:
                if not is_file(os.getcwd() + '/.env', raise_exception=False):
                    raise ConnectionError('.env file does not exist! Run "konfuzio_sdk init" to create it.')
                else:
                    raise ConnectionError(f'Unknown issue: {json.loads(r.text)["detail"]}')

            elif r.status_code == 500:
                raise TimeoutError(f'Problem with server: {json.loads(r.text)["detail"]} even after 10 retries')

            else:
                logger.warning(f'Retry to get url >>{url}<<')
                retry_count += 1
                if retry_count >= 10:
                    raise TimeoutError(f'Unknown issue even after 10 retries {json.loads(r.text)["detail"]}')
                time.sleep(15)
    return r


def get_csrf(session):
    """
    Get new CSRF from the host.

    :param session: Working session
    :return: New CSRF token.
    """
    login = session.get(KONFUZIO_HOST)
    csrf_token = login.cookies['csrftoken']
    return csrf_token


def konfuzio_session(token=None):
    """
    Create a session incl. base auth to the KONFUZIO_HOST.

    :return: Request session.
    """
    if not token:
        token = KONFUZIO_TOKEN
    session = requests.Session()
    session.headers.update({'Authorization': f'Token {token}'})
    return session


def get_document_details(document_id, session=konfuzio_session(), extra_fields: str = 'bbox,hocr'):
    """
    Use the text-extraction server to retrieve the data from a document.

    :param document_id: ID of the document
    :param session: Session to connect to the server
    :return: Data of the document.
    """
    url = get_document_api_details_url(document_id, include_extractions=False, extra_fields=extra_fields)
    r = retry_get(session, url)
    data = json.loads(r.text)
    text = data["text"]
    annotations = data["annotations"]
    annotations_sets = data["sections"]
    if text is None:
        logger.warning(f'Document with ID {document_id} does not contain any text, check OCR status.')
    else:
        logger.info(
            f'Document with ID {document_id} contains {len(text)} characters '
            f'and {len(annotations)} annotations in {len(annotations_sets)} annotation sets.'
        )

    return data


def get_document_text(document_id, session=konfuzio_session()):
    """
    Use the text-extraction server to retrieve the text found in the document.

    :param document_id: ID of the file
    :param session: Session to connect to the server
    :return: Document text.
    """
    url = get_document_api_details_url(document_id)
    r = retry_get(session, url)
    text = r.json()['text']
    if text is None:
        logger.warning(f'Document with ID {document_id} does not contain any text, check OCR status.')
    else:
        logger.info(f'Document with ID {document_id} contains {len(text)} characters.')

    return text


def get_document_hocr(document_id, session=konfuzio_session()):
    """
    Use the text-extraction server to retrieve the hOCR data.

    :param document_id: ID of the file
    :param session: Session to connect to the server
    :return: hOCR data of the document.
    """
    url = get_document_api_details_url(document_id, extra_fields='bbox,hocr')
    r = retry_get(session, url)
    hocr = r.json()['hocr']
    if hocr is None:
        logger.warning(f'Document with ID {document_id} does not contain hocr.')
    else:
        logger.info(f'Document with ID {document_id} contains {len(hocr)} characters.')

    return hocr


def get_document_annotations(document_id, include_extractions=False, session=konfuzio_session()):
    """
    Use the text-extraction server to retrieve human revised annotations.

    :param document_id: ID of the file
    :param include_extractions: Bool to include extractions
    :param session: Session to connect to the server
    :return: Sorted annotations.
    """
    url = get_document_api_details_url(document_id, include_extractions=include_extractions)
    r = retry_get(session, url)
    annotations = r.json()['annotations']
    not_custom_annotations = annotations
    revised_annotations_and_extractions = [
        x for x in not_custom_annotations if x['revised'] or x['is_correct'] or not x['id']
    ]
    sorted_annotations = sorted(
        revised_annotations_and_extractions, key=lambda x: (x.get('start_offset') is None, x.get('start_offset'))
    )
    logger.info(f'Document with ID {document_id} contains {len(sorted_annotations)} annotations.')

    return sorted_annotations


def post_document_bulk_annotation(document_id: int, annotation_list, session=konfuzio_session()):
    """
    Add a list of annotations to an existing document.

    :param document_id: ID of the file
    :param annotation_list: List of annotations
    :param session: Session to connect to the server
    :return: Response status.
    """
    url = get_document_annotations_url(document_id)
    r = session.post(url, json=annotation_list)
    r.raise_for_status()
    return r


def post_document_annotation(
    document_id: int,
    start_offset: int,
    end_offset: int,
    label_id: int,
    label_set_id: int,
    accuracy: float,
    revised: bool = False,
    is_correct: bool = False,
    annotation_set=None,
    define_annotation_set=True,
    session=konfuzio_session(),
    **kwargs,
):
    """
    Add an annotation to an existing document.

    For the annotation set definition, we can:
    - define the annotation set id where the annotation should belong
    (annotation_set=x (int), define_annotation_set=True)
    - pass it as None and a new annotation set will be created
    (annotation_set=None, define_annotation_set=True)
    - do not pass the annotation set field and a new annotation set will be created if does not exist any or the
    annotation will be added to the previous annotation set created (define_annotation_set=False)

    :param document_id: ID of the file
    :param start_offset: Start offset of the annotation
    :param end_offset: End offset of the annotation
    :param label_id: ID of the label.
    :param label_set_id: ID of the label set where the annotation belongs
    :param accuracy: Accuracy of the annotation
    :param revised: If the annotation is revised or not (bool)
    :param is_correct: If the annotation is corrected or not (bool)
    :param annotation_set: Annotation set to connect to the server
    :param define_annotation_set: If to define the annotation set (bool)
    :return: Response status.
    """
    url = get_document_annotations_url(document_id)

    bbox = kwargs.get('bbox', None)
    custom_bboxes = kwargs.get('bboxes', None)
    selection_bbox = kwargs.get('selection_bbox', None)
    page_number = kwargs.get('page_number', None)
    offset_string = kwargs.get('offset_string', None)

    data = {
        'start_offset': start_offset,
        'end_offset': end_offset,
        'label': label_id,
        'revised': revised,
        'section_label_id': label_set_id,
        'accuracy': accuracy,
        'is_correct': is_correct,
    }

    if define_annotation_set:
        data['section'] = annotation_set

    if page_number is not None:
        data['page_number'] = page_number

    if offset_string is not None:
        data['offset_string'] = offset_string

    if bbox is not None:
        data['bbox'] = bbox

    if custom_bboxes is not None:
        data['custom_bboxes'] = custom_bboxes

    if selection_bbox is not None:
        data['selection_bbox'] = selection_bbox

    r = session.post(url, json=data)
    return r


def delete_document_annotation(document_id: int, annotation_id: int, session=konfuzio_session()):
    """
    Delete a given annotation of the given document.

    :param document_id: ID of the document
    :param annotation_id: ID of the annotation
    :param session: Session to connect to the server.
    :return: Response status.
    """
    url = get_annotation_url(document_id=document_id, annotation_id=annotation_id)
    r = session.delete(url)
    return r


def get_meta_of_files(session=konfuzio_session()) -> List[dict]:
    """
    Get dictionary of previously uploaded document names to Konfuzio API.

    Dataset_status:
    NONE = 0
    PREPARATION = 1
    TRAINING = 2
    TEST = 3
    LOW_OCR_QUALITY = 4

    :param session: Session to connect to the server
    :return: Sorted documents names in the format {id: 'pdf_name'}.
    """
    url = get_documents_meta_url()
    result = []

    while True:
        r = retry_get(session, url)
        data = r.json()
        if isinstance(data, dict) and 'results' in data.keys():
            result += data['results']
            if 'next' in data.keys() and data['next']:
                url = data['next']
            else:
                break
        else:
            result = data
            break

    sorted_documents = sorted(result, key=itemgetter('id'))
    return sorted_documents


def get_project_labels(session=konfuzio_session()) -> List[dict]:
    """
    Get Labels available in project.

    :param session: Session to connect to the server
    :return: Sorted labels.
    """
    url = get_project_url()
    r = retry_get(session, url)
    sorted_labels = sorted(r.json()['labels'], key=itemgetter('id'))
    return sorted_labels


def create_label(
    project_id: int, label_name: str, label_sets: list, session=konfuzio_session(), **kwargs
) -> List[dict]:
    """
    Create a Label and associate it with labels sets.

    :param project_id: Project ID where to create the label
    :param label_name: Name for the label
    :param label_sets: Label sets that use the label
    :param session: Session to connect to the server
    :return: Label ID in the Konfuzio Server.
    """
    url = get_labels_url()
    label_sets_ids = [label_set.id for label_set in label_sets]

    description = kwargs.get('description', None)
    has_multiple_top_candidates = kwargs.get('has_multiple_top_candidates', False)
    data_type = kwargs.get('data_type', 'Text')

    data = {
        "project": project_id,
        "text": label_name,
        "description": description,
        "has_multiple_top_candidates": has_multiple_top_candidates,
        "get_data_type_display": data_type,
        "templates": label_sets_ids,
    }

    r = session.post(url=url, json=data)

    assert r.status_code == requests.codes.created, f'Status of request: {r}'
    label_id = r.json()['id']
    return label_id


def get_project_label_sets(session=konfuzio_session()) -> List[dict]:
    """
    Get Label Sets available in project.

    :param session: Session to connect to the server
    :return: Sorted Label Sets.
    """
    url = get_project_url()
    r = session.get(url=url)
    r.raise_for_status()
    sorted_label_sets = sorted(r.json()['section_labels'], key=itemgetter('id'))
    return sorted_label_sets


def upload_file_konfuzio_api(
    filepath: str,
    project_id: int,
    session=konfuzio_session(),
    dataset_status: int = 0,
    category_template_id: Union[None, int] = None,
):
    """
    Upload file to Konfuzio API.

    :param filepath: Path to file to be uploaded
    :param session: Session to connect to the server
    :param project_id: Project ID where to upload the document
    :return: Response status.
    """
    url = get_upload_document_url()
    is_file(filepath)

    with open(filepath, "rb") as f:
        file_data = f.read()

    files = {"data_file": (os.path.basename(filepath), file_data, "multipart/form-data")}
    data = {"project": project_id, "dataset_status": dataset_status, "category_template": category_template_id}

    r = session.post(url=url, files=files, data=data)
    return r


def delete_file_konfuzio_api(document_id: int, session=konfuzio_session()):
    """
    Delete Document by ID via Konfuzio API.

    :param document_id: ID of the document
    :param session: Session to connect to the server
    :return: File id in Konfuzio Server.
    """
    url = get_document_url(document_id)
    data = {'id': document_id}

    r = session.delete(url=url, json=data)
    assert r.status_code == 204
    return True


def update_file_konfuzio_api(
    document_id: int, file_name: str, dataset_status: int = 0, session=konfuzio_session(), **kwargs
):
    """
    Update the dataset status of an existing document via Konfuzio API.

    :param document_id: ID of the document
    :param dataset_status: New dataset status
    :param session: Session to connect to the server
    :return: Response status.
    """
    url = get_document_url(document_id)

    category_template_id = kwargs.get('category_template_id', None)

    data = {"data_file_name": file_name, "dataset_status": dataset_status, "category_template": category_template_id}

    r = session.patch(url=url, json=data)
    return r


def download_file_konfuzio_api(document_id: int, ocr: bool = True, session=konfuzio_session()):
    """
    Download file from the Konfuzio server using the document id.

    Django authentication is form-based, whereas DRF uses BasicAuth.

    :param document_id: ID of the document
    :param ocr: Bool to get the ocr version of the document
    :param session: Session to connect to the server
    :return: The downloaded file.
    """
    if ocr:
        url = get_document_ocr_file_url(document_id)
    else:
        url = get_document_original_file_url(document_id)

    r = session.get(url)

    try:
        r.raise_for_status()
    except Exception:
        if r.status_code != 200:
            logger.exception("Requests error")
            raise FileNotFoundError(json.loads(r.text)["detail"])

    content_type = r.headers.get('content-type')
    if content_type not in ['application/pdf', 'image/jpeg', 'image/png', 'image/jpg']:
        raise FileNotFoundError(f'CONTENT TYP of {document_id} is {content_type} and no PDF or image.')

    logger.info(f'Downloaded file {document_id} from {KONFUZIO_HOST}.')
    return r.content


def is_url_image(image_url):
    """
    Check if the URL will return an image.

    :param image_url: URL of the image
    :return: If the URL returns an image
    """
    image_formats = ("image/png", "image/jpeg", "image/jpg")
    r = requests.head(image_url)
    logger.info(f'{image_url} has content type {r.headers["content-type"]}')
    if r.headers["content-type"] in image_formats:
        return True
    return False


def download_images(urls: List[str] = None):
    """
    Download images by a list of urls.

    :param urls: URLs of the images
    :return: Downloaded images.
    """
    are_images = [is_url_image(url) for url in urls]
    if not are_images[: sum(are_images)]:
        raise NotImplementedError('Only images are supported')
    downloads = [requests.get(url) for url in urls]
    images = [load_image(io.BytesIO(download.content)) for download in downloads]
    return images


def is_url(url: str) -> bool:
    """
    Return true if the string is a valid URL.

    :param url: String URL
    :return: True if is a valid URL.
    """
    logger.info(url)
    result = urlparse(url)
    return all([result.scheme, result.netloc])


def get_results_from_segmentation(doc_id: int, project_id: int, host: Union[str, None] = None) -> List[dict]:
    """Get bbox results from segmentation endpoint.

    :param doc_id: ID of the document
    :param project_id: ID of the project.
    """
    session = konfuzio_session()

    segmentation_url = get_document_segmentation_details_url(doc_id, project_id, host=host, action='segmentation')
    response = retry_get(session, segmentation_url)
    segmentation_result = response.json()

    return segmentation_result
