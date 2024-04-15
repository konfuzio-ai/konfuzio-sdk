"""Connect to the Konfuzio Server to receive or send data."""

import json
import logging
import os
from json import JSONDecodeError
from operator import itemgetter
from typing import Dict, List, Optional, Union

import requests
from requests import HTTPError
from requests.adapters import HTTPAdapter
from urllib3 import Retry

from konfuzio_sdk import KONFUZIO_HOST, KONFUZIO_TOKEN
from konfuzio_sdk.urls import (
    create_annotation_url,
    get_ai_model_download_url,
    get_ai_model_url,
    get_annotation_url,
    get_auth_token_url,
    get_categorization_ais_list_url,
    get_create_ai_model_url,
    get_document_annotations_url,
    get_document_bbox_url,
    get_document_details_url,
    get_document_ocr_file_url,
    get_document_original_file_url,
    get_document_segmentation_details_url,
    get_document_url,
    get_documents_meta_url,
    get_extraction_ais_list_url,
    get_labels_url,
    get_page_url,
    get_project_categories_url,
    get_project_label_sets_url,
    get_project_labels_url,
    get_project_url,
    get_projects_list_url,
    get_splitting_ais_list_url,
    get_upload_document_url,
)
from konfuzio_sdk.utils import is_file

logger = logging.getLogger(__name__)

AI_TYPES = ['filesplitting', 'extraction', 'categorization']


def _get_auth_token(username, password, host=KONFUZIO_HOST) -> str:
    """
    Generate the authentication token for the user.

    :param username: Login username.
    :type username: str
    :param password: Login password.
    :type password: str
    :param host: Host URL.
    :type host: str
    :return: The new generated token.
    """
    url = get_auth_token_url(host)
    user_credentials = {'username': username, 'password': password}
    r = requests.post(url, json=user_credentials)
    if r.status_code == 200:
        token = r.json()['token']
    elif r.status_code in [403, 400]:
        raise PermissionError(
            '[ERROR] Your credentials are not correct! Please run init again and provide the correct credentials.'
        )
    else:
        raise ConnectionError(f'HTTP Status {r.status_code}: {r.text}')
    return token


def init_env(
    user: str, password: str, host: str = KONFUZIO_HOST, working_directory=os.getcwd(), file_ending: str = '.env'
):
    """
    Add the .env file to the working directory.

    :param user: Username to log in to the host
    :param password: Password to log in to the host
    :param host: URL of host.
    :param working_directory: Directory where file should be added
    :param file_ending: Ending of file.
    """
    token = _get_auth_token(user, password, host)

    with open(os.path.join(working_directory, file_ending), 'w') as f:
        f.write(f'KONFUZIO_HOST = {host}\n')
        f.write(f'KONFUZIO_USER = {user}\n')
        f.write(f'KONFUZIO_TOKEN = {token}\n')

    print('[SUCCESS] SDK initialized!')

    return True


class TimeoutHTTPAdapter(HTTPAdapter):
    """Combine a retry strategy with a timeout strategy.

    Documentation
    =============
        * `Urllib3 <https://urllib3.readthedocs.io/en/latest/reference/urllib3.util.html#urllib3.util.Retry>`_
        * TimeoutHTTPAdapter idea used from the following
            `Blogpost <https://findwork.dev/blog/advanced-usage-python-requests-timeouts-retries-hooks/>`_
    """

    timeout = None  # see https://stackoverflow.com/a/29649638

    def __init__(self, timeout, *args, **kwargs):
        """Force to init with timeout policy."""
        self.timeout = timeout
        super().__init__(*args, **kwargs)

    def send(self, request, *args, **kwargs):
        """Use timeout policy if not otherwise declared."""
        if request.headers['Authorization'] == 'Token None':
            raise PermissionError(f'Your Token to connect to {KONFUZIO_HOST} is missing, e.g. use "konfuzio_sdk init"')
        logger.debug(f'Sending request: {request.url}, {request.method}, {request.headers}')
        timeout = kwargs.get('timeout')
        if timeout is None:
            kwargs['timeout'] = self.timeout
        return super().send(request, *args, **kwargs)

    def build_response(self, req, resp):
        """Throw error for any HTTPError that is not part of the retry strategy."""
        response = super().build_response(req, resp)
        logger.debug(f'Received response: {response.status_code}, {response.reason}, {response.url}')
        # handle status code one by one as some status codes will cause a retry, see konfuzio_session
        if response.status_code in [403, 404]:
            # Fallback to None if there's no detail, for whatever reason.
            try:
                detail = response.json().get('detail')
            except JSONDecodeError:
                detail = 'No JSON provided by Host.'
            raise HTTPError(f'{response.status_code} {response.reason}: {detail} via {response.url}')

        try:
            response.raise_for_status()
        except HTTPError as e:
            raise HTTPError(response.text) from e

        return response


def konfuzio_session(
    token: str = None, timeout: Optional[int] = None, num_retries: Optional[int] = None, host: str = None
):
    """
    Create a session incl. Token to the KONFUZIO_HOST.

    :param token: Konfuzio Token to connect to the host.
    :param timeout: Timeout in seconds.
    :param num_retries: Number of retries if the request fails.
    :param host: Host to connect to.
    :return: Request session.
    """
    if token is None:
        token = KONFUZIO_TOKEN
    if timeout is None:
        timeout = 120
    if num_retries is None:
        num_retries = 5

    retry_strategy = Retry(
        total=num_retries,
        status_forcelist=[429, 500, 502, 503, 504],
        backoff_factor=2,
    )
    session = requests.Session()
    session.mount('https://', adapter=TimeoutHTTPAdapter(max_retries=retry_strategy, timeout=timeout))
    session.headers.update({'Authorization': f'Token {token}'})
    session.host = host
    return session


def get_project_list(session=None):
    """
    Get the list of all Projects for the user.

    :param session: Konfuzio session with Retry and Timeout policy
    :return: Response object
    """
    if session is None:
        session = konfuzio_session()
    url = get_projects_list_url()
    r = session.get(url=url)
    return r.json()


def get_project_details(project_id: int, session=None) -> dict:
    """
    Get Project's metadata.

    :param project_id: ID of the Project
    :param session: Konfuzio session with Retry and Timeout policy
    :return: Project metadata
    """
    if session is None:
        session = konfuzio_session()
    if hasattr(session, 'host'):
        host = session.host
    else:
        host = None

    url = get_project_url(project_id=project_id, host=host)
    r = session.get(url=url)

    return r.json()


def get_project_labels(project_id: int, session=None) -> dict:
    """
    Get Project's Labels.

    :param project_id: An ID of a Project to get Labels from.
    :param session: Konfuzio session with Retry and Timeout policy
    """
    if session is None:
        session = konfuzio_session()
    if hasattr(session, 'host'):
        host = session.host
    else:
        host = None

    url = get_project_labels_url(project_id=project_id, host=host)
    r = session.get(url=url)

    return r.json()


def get_project_label_sets(project_id: int, session=None) -> dict:
    """
    Get Project's Label Sets.

    :param project_id: An ID of a Project to get Label Sets from.
    :param session: Konfuzio session with Retry and Timeout policy
    """
    if session is None:
        session = konfuzio_session()
    if hasattr(session, 'host'):
        host = session.host
    else:
        host = None

    url = get_project_label_sets_url(project_id=project_id, host=host)
    r = session.get(url=url)

    return r.json()


def create_new_project(project_name, session=None):
    """
    Create a new Project for the user.

    :param project_name: name of the project you want to create
    :param session: Konfuzio session with Retry and Timeout policy
    :return: Response object
    """
    if session is None:
        session = konfuzio_session()
    url = get_projects_list_url()
    new_project_data = {'name': project_name}
    r = session.post(url=url, json=new_project_data)

    if r.status_code == 201:
        project_id = r.json()['id']
        logger.info(f'Project {project_name} (ID {project_id}) was created successfully!')
        return project_id
    else:
        raise PermissionError(
            f'HTTP Status {r.status_code}: The project {project_name} was not created, please check'
            f' your permissions. Error {r.json()}'
        )


def get_document_details(document_id: int, session=None):
    """
    Use the text-extraction server to retrieve the data from a document.

    :param document_id: ID of the document
    :param session: Konfuzio session with Retry and Timeout policy
    :return: Data of the document.
    """
    if session is None:
        session = konfuzio_session()
    if hasattr(session, 'host'):
        host = session.host
    else:
        host = None
    url = get_document_details_url(document_id=document_id, host=host)
    r = session.get(url)
    return r.json()


def get_document_annotations(document_id: int, session=None):
    """
    Get Annotations of a Document.

    :param document_id: ID of the Document.
    :param session: Konfuzio session with Retry and Timeout policy
    :return: List of the Annotations of the Document.
    """
    if session is None:
        session = konfuzio_session()
    if hasattr(session, 'host'):
        host = session.host
    else:
        host = None
    url = get_document_annotations_url(document_id=document_id, host=host)
    r = session.get(url)
    return r.json()


def get_document_bbox(document_id: int, session=None):
    """
    Get Bboxes for a Document.

    :param document_id: ID of the Document.
    :param session: Konfuzio session with Retry and Timeout policy
    :return: List of Bboxes of characters in the Document
    """
    if session is None:
        session = konfuzio_session()
    if hasattr(session, 'host'):
        host = session.host
    else:
        host = None
    url = get_document_bbox_url(document_id=document_id, host=host)
    r = session.get(url)
    return r.json()


def get_page_image(document_id: int, page_number: int, session=None, thumbnail: bool = False):
    """
    Load image of a Page as Bytes.

    :param page_number: Number of the Page
    :param thumbnail: Download Page image as thumbnail
    :param session: Konfuzio session with Retry and Timeout policy
    :return: Bytes of the Image.
    """
    if session is None:
        session = konfuzio_session()
    if thumbnail:
        raise NotImplementedError
    else:
        if hasattr(session, 'host'):
            host = session.host
        else:
            host = None
    url = get_page_url(document_id=document_id, page_number=page_number, host=host)

    r = session.get(url)
    image_url = f"{host or KONFUZIO_HOST}{r.json()['image_url']}"
    r = session.get(image_url)

    content_type = r.headers.get('content-type')
    if content_type != 'image/png':
        raise TypeError(f'CONTENT TYPE of Image {page_number} is {content_type} and no PNG.')

    return r.content


# def post_document_bulk_annotation(document_id: int, project_id: int, annotation_list, session=konfuzio_session()):
#     """
#     Add a list of Annotations to an existing document.
#
#     :param document_id: ID of the file
#     :param project_id: ID of the project
#     :param annotation_list: List of Annotations
#     :param session: Konfuzio session with Retry and Timeout policy
#     :return: Response status.
#     """
#     url = get_document_annotations_url(document_id, project_id=project_id)
#     r = session.post(url, json=annotation_list)
#     r.raise_for_status()
#     return r


def post_document_annotation(
    document_id: int,
    spans: List,
    label_id: int,
    confidence: Union[float, None] = None,
    revised: bool = False,
    is_correct: bool = False,
    session=None,
    **kwargs,
):
    """
    Add an Annotation to an existing document.

    You must specify either annotation_set_id or label_set_id.

    Use annotation_set_id if an Annotation Set already exists. You can find the list of existing Annotation Sets by
    using the GET endpoint of the Document.

    Using label_set_id will create a new Annotation Set associated with that Label Set. You can only do this if the
    Label Set has has_multiple_sections set to True.

    :param document_id: ID of the file
    :param spans: Spans that constitute the Annotation
    :param label_id: ID of the Label
    :param confidence: Confidence of the Annotation still called Accuracy by text-annotation
    :param revised: If the Annotation is revised or not (bool)
    :param is_correct: If the Annotation is corrected or not (bool)
    :param session: Konfuzio session with Retry and Timeout policy
    :return: Response status.
    """
    if session is None:
        session = konfuzio_session()

    url = create_annotation_url()

    custom_bboxes = kwargs.get('selection_bbox', None)
    annotation_set_id = kwargs.get('annotation_set_id', None)
    label_set_id = kwargs.get('label_set_id', None)
    if not all(isinstance(span, dict) for span in spans):
        spans = [
            {
                'x0': span.bbox().x0,
                'x1': span.bbox().x1,
                'y0': span.bbox().y0,
                'y1': span.bbox().y1,
                'page_index': span.page.index,
                'offset_string': span.offset_string,
            }
            for span in spans
        ]

    data = {
        'document': document_id,
        'label': label_id,
        'revised': revised,
        'confidence': confidence,
        'is_correct': is_correct,
        'origin': 'api.v3',
        'span': spans,
    }

    if annotation_set_id is not None:
        data['annotation_set'] = annotation_set_id
    elif label_set_id is not None:
        data['label_set'] = label_set_id

    if custom_bboxes is not None:
        data['selection_bbox'] = custom_bboxes

    r = session.post(url, json=data)
    if r.status_code != 201:
        logger.error(f'Received response status code {r.status_code}.')
    assert r.status_code == 201
    return r


def change_document_annotation(annotation_id: int, session=None, **kwargs):
    """
    Change something about an Annotation.

    :param annotation_id: ID of an Annotation to be changed
    :param session: Konfuzio session with Retry and Timeout policy
    :return: Response status.
    """
    if session is None:
        session = konfuzio_session()

    url = get_annotation_url(annotation_id=annotation_id)
    label = kwargs.get('label', None)
    is_correct = kwargs.get('is_correct', None)
    revised = kwargs.get('revised', None)
    span = kwargs.get('span', None)
    label_set = kwargs.get('label_set', None)
    annotation_set = kwargs.get('annotation_set', None)
    selection_bbox = kwargs.get('selection_bbox', None)
    data = {}
    if label is not None:
        data['label'] = label
    if is_correct is not None:
        data['is_correct'] = is_correct
    if revised is not None:
        data['revised'] = revised
    if span is not None:
        data['span'] = span
    if label_set is not None:
        data['label_set'] = label_set
    if annotation_set is not None:
        data['annotation_set'] = annotation_set
    if selection_bbox is not None:
        data['selection_bbox'] = selection_bbox
    r = session.patch(url, json=data)
    if r.status_code != 200:
        logger.error(f'Received response status code {r.status_code}.')
    assert r.status_code == 200
    return r


def delete_document_annotation(annotation_id: int, session=None, delete_from_database: bool = False, **kwargs):
    """
    Delete a given Annotation of the given document.

    For AI training purposes, we recommend setting `delete_from_database` to False if you don't want to remove
    Annotation permanently. This creates a negative feedback Annotation and does not remove it from the database.

    :param annotation_id: ID of the annotation
    :param session: Konfuzio session with Retry and Timeout policy
    :return: Response status.
    """

    data = {'annotation_id': annotation_id}

    if not delete_from_database:
        data['is_correct'] = False
        data['revised'] = True

    if session is None:
        session = konfuzio_session()
    url = get_annotation_url(annotation_id=annotation_id)
    if not delete_from_database:
        r = session.patch(url=url, json=data)
    else:
        r = session.delete(url=url)
    if r.status_code == 200:
        # the text Annotation received negative feedback and copied the Annotation and created a new one
        return json.loads(r.text)['id']
    elif r.status_code == 204:
        return r
    else:
        raise ConnectionError(f'Error{r.status_code}: {r.content} {r.url}')


def get_meta_of_files(project_id: int, pagination_limit: int = 100, limit: int = None, session=None) -> List[dict]:
    """
    Get meta information of Documents in a Project.

    :param project_id: ID of the Project
    :param pagination_limit: Number of Documents returned in a single paginated response
    :param limit: Number of Documents returned in general
    :param session: Konfuzio session with Retry and Timeout policy
    :return: Sorted Documents names in the format {id_: 'pdf_name'}.
    """
    if session is None:
        session = konfuzio_session()
    if hasattr(session, 'host'):
        host = session.host
    else:
        host = None
    if limit:
        url = get_documents_meta_url(project_id=project_id, offset=0, limit=limit)
    else:
        url = get_documents_meta_url(project_id=project_id, limit=pagination_limit, host=host)
    result = []
    r = session.get(url)
    data = r.json()
    result += data['results']

    if not limit:
        while 'next' in data.keys() and data['next']:
            logger.info(f'Iterate on paginated {url}.')
            url = data['next']
            r = session.get(url)
            data = r.json()
            result += data['results']

    sorted_documents = sorted(result, key=itemgetter('id'))
    return sorted_documents


def create_label(
    project_id: int,
    label_name: str,
    label_sets: list,
    session=None,
    description=None,
    has_multiple_top_candidates=None,
    data_type=None,
) -> List[dict]:
    """
    Create a Label and associate it with Labels sets.

    :param project_id: Project ID where to create the label
    :param label_name: Name for the label
    :param label_sets: Label sets that use the label
    :param session: Konfuzio session with Retry and Timeout policy
    :param description: Test to describe the label
    :param has_multiple_top_candidates: If multiple Annotations can be correct in a single Annotation Set
    :param data_type: Expected data type of any Span of Annotations related to this Label.
    :return: Label ID in the Konfuzio Server.
    """
    if session is None:
        session = konfuzio_session()
    url = get_labels_url()
    label_sets_ids = [label_set.id_ for label_set in label_sets]

    data = {
        'project': project_id,
        'text': label_name,
        'description': description,
        'has_multiple_top_candidates': has_multiple_top_candidates,
        'get_data_type_display': data_type,
        'templates': label_sets_ids,
    }

    r = session.post(url=url, json=data)

    assert r.status_code == 201
    label_id = r.json()['id']
    return label_id


def upload_file_konfuzio_api(
    filepath: str,
    project_id: int,
    dataset_status: int = 0,
    session=None,
    category_id: Union[None, int] = None,
    callback_url: str = '',
    callback_status_code: int = None,
    sync: bool = False,
    assignee: str = '',
):
    """
    Upload Document to Konfuzio API.

    :param filepath: Path to file to be uploaded
    :param project_id: ID of the project
    :param session: Konfuzio session with Retry and Timeout policy
    :param dataset_status: Set data set status of the document.
    :param category_id: Define a Category the Document belongs to
    :param callback_url: Callback URL receiving POST call once extraction is done
    :param callback_status_code: The HTTP response code of the callback server (in case a callback URL is set)
    :param sync: If True, will run synchronously and only return once the online database is updated
    :param assignee: The user who is currently assigned to the Document.
    :return: Response status.
    """
    if session is None:
        session = konfuzio_session()
    url = get_upload_document_url()
    is_file(filepath)

    with open(filepath, 'rb') as f:
        file_data = f.read()

    files = {'data_file': (os.path.basename(filepath), file_data, 'multipart/form-data')}
    data = {
        'project': project_id,
        'dataset_status': dataset_status,
        'category': category_id,
        'sync': sync,
        'callback_url': callback_url,
        'callback_status_code': callback_status_code,
    }
    if assignee:
        data['assignee'] = assignee

    r = session.post(url=url, files=files, data=data)

    return r


def delete_file_konfuzio_api(document_id: int, session=None):
    """
    Delete Document by ID via Konfuzio API.

    :param document_id: ID of the document
    :param session: Konfuzio session with Retry and Timeout policy
    :return: File id_ in Konfuzio Server.
    """
    if session is None:
        session = konfuzio_session()
    url = get_document_url(document_id)

    session.delete(url=url)

    return True


def update_document_konfuzio_api(document_id: int, session=None, **kwargs):
    """
    Update an existing Document via Konfuzio API.

    :param document_id: ID of the document
    :param session: Konfuzio session with Retry and Timeout policy
    :return: Response status.
    """
    if session is None:
        session = konfuzio_session()
    url = get_document_url(document_id)

    data = {}
    file_name = kwargs.get('file_name', None)
    dataset_status = kwargs.get('dataset_status', None)
    category_id = kwargs.get('category_id', None)
    assignee = kwargs.get('assignee', None)

    if file_name is not None:
        data.update({'data_file_name': file_name})

    if dataset_status is not None:
        data.update({'dataset_status': dataset_status})

    if category_id is not None:
        data.update({'category': category_id})

    if assignee is not None:
        data.update({'assignee': assignee})

    r = session.patch(url=url, json=data)

    return json.loads(r.text)


def download_file_konfuzio_api(document_id: int, ocr: bool = True, session=None):
    """
    Download file from the Konfuzio server using the Document id_.

    Django authentication is form-based, whereas DRF uses BasicAuth.

    :param document_id: ID of the document
    :param ocr: Bool to get the ocr version of the document
    :param session: Konfuzio session with Retry and Timeout policy
    :return: The downloaded file.
    """
    if session is None:
        session = konfuzio_session()
    if hasattr(session, 'host'):
        host = session.host
    else:
        host = None
    if ocr:
        url = get_document_ocr_file_url(document_id, host=host)
    else:
        url = get_document_original_file_url(document_id, host=host)

    r = session.get(url)

    content_type = r.headers.get('content-type')
    if content_type not in ['application/pdf', 'image/jpeg', 'image/png', 'image/jpg']:
        logger.info(f'CONTENT TYP of {document_id} is {content_type} and no PDF or image.')

    logger.info(f'Downloaded file {document_id} from {url}.')
    return r.content


def get_results_from_segmentation(doc_id: int, project_id: int, session=None) -> List[List[dict]]:
    """Get bbox results from segmentation endpoint.

    :param doc_id: ID of the document
    :param project_id: ID of the Project.
    :param session: Konfuzio session with Retry and Timeout policy
    """
    if session is None:
        session = konfuzio_session()
    segmentation_url = get_document_segmentation_details_url(doc_id, project_id)
    response = session.get(segmentation_url)

    segmentation_result = response.json()
    return segmentation_result


def get_project_categories(project_id: int = None, session=None) -> List[Dict]:
    """
    Get a list of Categories of a Project.

    :param project_id: ID of the Project.
    :param session: Konfuzio session with Retry and Timeout policy
    """
    if session is None:
        session = konfuzio_session()
    if hasattr(session, 'host'):
        host = session.host
    else:
        host = None
    url = get_project_categories_url(project_id=project_id, host=host)
    r = session.get(url=url)
    return r.json()['results']


def upload_ai_model(ai_model_path: str, project_id: int = None, category_id: int = None, session=None):
    """
    Upload an ai_model to the text-annotation server.

    :param ai_model_path: Path to the ai_model
    :param project_id: An ID of a Project to which the AI is uploaded. Needed for the File Splitting and Categorization
    AIs because they function on a Project level.
    :param category_id: An ID of a Category on which the AI is trained. Needed for the Extraction AI because it
    functions on a Category level and requires a single Category.
    :param session: session to connect to server
    :raises: ValueError when neither project_id nor category_id is specified.
    :raises: HTTPError when a request is unsuccessful.
    :return:
    """
    if session is None:
        session = konfuzio_session()
    if (not project_id) and (not category_id):
        raise ValueError('Project ID or Category ID has to be specified; both values cannot be empty.')
    for cur_ai_type in AI_TYPES:
        if cur_ai_type in ai_model_path:
            ai_type = cur_ai_type
            break
    else:
        raise ValueError(
            "Cannot define AI type by the file name. Pass an AI model that is named according to the \
                          SDK's naming conventions."
        )
    url = get_create_ai_model_url(ai_type)
    if is_file(ai_model_path):
        model_name = os.path.basename(ai_model_path)
        with open(ai_model_path, 'rb') as f:
            multipart_form_data = {'file': (model_name, f)}
            headers = {'Prefer': 'respond-async'}
            if not ai_type == 'extraction':
                data = {'project': str(project_id)}
            else:
                data = {'category': str(category_id)}
            r = session.post(url, files=multipart_form_data, data=data, headers=headers)
            try:
                r.raise_for_status()
            except HTTPError as e:
                raise HTTPError(r.text) from e
    data = r.json()
    ai_model_id = data['id']

    logger.info(f'New AI Model {ai_model_id} uploaded to {url}')
    return ai_model_id


def delete_ai_model(ai_model_id: int, ai_type: str, session=None):
    """
    Delete an AI model from the server.

    :param ai_model_id: an ID of the model to be deleted.
    :param ai_type: Should be one of the following: 'filesplitting', 'extraction', 'categorization'.
    :param session: session to connect to the server.
    :raises: ValueError if ai_type is not correctly specified.
    :raises: ConnectionError when a request is unsuccessful.
    """
    if session is None:
        session = konfuzio_session()
    if ai_type not in AI_TYPES:
        raise ValueError(f"ai_type should be one of the following: {', '.join(AI_TYPES)}")
    url = get_ai_model_url(ai_model_id, ai_type)
    r = session.delete(url)
    if r.status_code == 200:
        return json.loads(r.text)['id']
    elif r.status_code == 204:
        return r
    else:
        raise ConnectionError(f'Error{r.status_code}: {r.content} {r.url}')


def update_ai_model(ai_model_id: int, ai_type: str, patch: bool = True, session=None, **kwargs):
    """
    Update an AI model from the server.

    :param ai_model_id: an ID of the model to be updated.
    :param ai_type: Should be one of the following: 'filesplitting', 'extraction', 'categorization'.
    :param patch: If true, adds info instead of replacing it.
    :param session: session to connect to the server.
    :raises: ValueError if ai_type is not correctly specified.
    :raises: HTTPError when a request is unsuccessful.
    """
    if session is None:
        session = konfuzio_session()
    if ai_type not in AI_TYPES:
        raise ValueError(f"ai_type should be one of the following: {', '.join(AI_TYPES)}")
    url = get_ai_model_url(ai_model_id, ai_type)

    data = {}
    description = kwargs.get('description', None)

    if description is not None:
        data.update({'description': description})
    if patch:
        r = session.patch(url=url, json=data)
    else:
        r = session.put(url=url, json=data)
    try:
        r.raise_for_status()
    except HTTPError as e:
        raise HTTPError(r.text) from e

    return json.loads(r.text)


def get_all_project_ais(project_id: int, session=None) -> dict:
    """
    Fetch all types of AIs for a specific project.

    :param project_id: ID of the Project
    :param session: Konfuzio session with Retry and Timeout policy
    :param host: Konfuzio host
    :return: Dictionary with lists of all AIs for a specific project
    """
    if session is None:
        session = konfuzio_session()
    if hasattr(session, 'host'):
        host = session.host
    else:
        host = None

    urls = {
        'extraction': get_extraction_ais_list_url(project_id, host),
        'filesplitting': get_splitting_ais_list_url(project_id, host),
        'categorization': get_categorization_ais_list_url(project_id, host),
    }

    all_ais = {}

    for ai_type, url in urls.items():
        try:
            response = session.get(url=url)
            response.raise_for_status()

            if response.status_code == 200:
                all_ais[ai_type] = json.loads(response.text)
        except HTTPError as e:
            all_ais[ai_type] = {'error': e}
            print(f'[ERROR] while fetching {ai_type} AIs: {e}')

    return all_ais


def export_ai_models(project, session=None) -> int:
    """
    Export all AI Model files for a specific Project.

    :param: project: Konfuzio Project
    :return: Number of exported AIs
    """
    ai_types = set()  # Using a set to store unique AI types
    exported_ais = {}  # Keeping track of the AIs that have been exported

    project_ai_models = project.ai_models
    for model_type, details in project_ai_models.items():
        count = details.get('count')
        if count and count > 0:
            # Only AI types with at least one model will be exported
            ai_types.add(model_type)

    if session is None:
        session = konfuzio_session()
    if hasattr(session, 'host'):
        host = session.host
    else:
        host = None

    for ai_type in ai_types:
        variant = ai_type
        folder = os.path.join(project.project_folder, 'models', variant + '_ais')

        ai_models = project_ai_models.get(variant, {}).get('results', [])

        for index, ai_model in enumerate(ai_models):
            # Only export fully trained AIs which are set as active
            if not ai_model.get('status') == 'done' or not ai_model.get('active'):
                logger.error(f'Skip {ai_model} in export.')
                continue
            ai_model_id = ai_model.get('id')
            ai_model_version = ai_model.get('id')

            if not ai_model_id or not ai_model_version:
                continue

            model_url = get_ai_model_download_url(ai_model_id=ai_model_id, host=host)

            try:
                response = project.session.get(model_url)
                response.raise_for_status()
            except HTTPError:
                logger.error(
                    f'Skip {ai_model} in export because this AI is corrupted (i.e. it does not have a file associated).'
                )
                continue

            if response.status_code == 200:
                alternative_name = f'{variant}_ai_{ai_model_id}_version_{ai_model_version}'

                # Current implementation automatically downloads the AI file through the Content-Disposition header

                content_disposition = response.headers.get('Content-Disposition', alternative_name)
                if 'filename=' in content_disposition:
                    # Split the string by 'filename=' and get the second part
                    file_name = content_disposition.split('filename=')[1].strip()

                    # Remove double quotes from the beginning and end if present
                    file_name = file_name.strip('"')
                else:
                    file_name = alternative_name

                # Create the model directory
                models_dir = os.path.join(folder)
                if not os.path.exists(models_dir):
                    os.makedirs(models_dir)

                local_model_path = os.path.join(models_dir, file_name)

                with open(local_model_path, 'wb') as f:
                    f.write(response.content)

                exported_ais[variant + '_' + str(index)] = local_model_path
                print(f'[SUCCESS] Exported {variant} AI Model to {file_name}')

    return exported_ais.keys().__len__()
