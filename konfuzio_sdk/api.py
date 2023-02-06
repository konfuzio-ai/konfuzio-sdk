"""Connect to the Konfuzio Server to receive or send data."""

import json
import logging
import os
from json import JSONDecodeError
from operator import itemgetter
from typing import List, Union

import requests
from requests import HTTPError
from requests.adapters import HTTPAdapter
from urllib3 import Retry

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
    get_update_ai_model_url,
    get_create_ai_model_url,
    get_page_image_url,
)
from konfuzio_sdk.utils import is_file

logger = logging.getLogger(__name__)


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
    user_credentials = {"username": username, "password": password}
    r = requests.post(url, json=user_credentials)
    if r.status_code == 200:
        token = r.json()['token']
    elif r.status_code in [403, 400]:
        raise PermissionError(
            "[ERROR] Your credentials are not correct! Please run init again and provide the correct credentials."
        )
    else:
        raise ConnectionError(f'HTTP Status {r.status_code}: {r.text}')
    return token


def init_env(
    user: str, password: str, host: str = KONFUZIO_HOST, working_directory=os.getcwd(), file_ending: str = ".env"
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

    with open(os.path.join(working_directory, file_ending), "w") as f:
        f.write(f"KONFUZIO_HOST = {host}\n")
        f.write(f"KONFUZIO_USER = {user}\n")
        f.write(f"KONFUZIO_TOKEN = {token}\n")

    print("[SUCCESS] SDK initialized!")

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
        timeout = kwargs.get("timeout")
        if timeout is None:
            kwargs["timeout"] = self.timeout
        return super().send(request, *args, **kwargs)

    def build_response(self, req, resp):
        """Throw error for any HTTPError that is not part of the retry strategy."""
        response = super().build_response(req, resp)
        # handle status code one by one as some status codes will cause a retry, see konfuzio_session
        if response.status_code in [403, 404]:
            # Fallback to None if there's no detail, for whatever reason.
            try:
                detail = response.json().get('detail')
            except JSONDecodeError:
                detail = 'No JSON provided by Host.'
            raise HTTPError(f'{response.status_code} {response.reason}: {detail} via {response.url}')

        return response


def konfuzio_session(token: str = KONFUZIO_TOKEN):
    """
    Create a session incl. Token to the KONFUZIO_HOST.

    :return: Request session.
    """
    retry_strategy = Retry(
        total=5,
        status_forcelist=[429, 500, 502, 503, 504],
        backoff_factor=2,
        # allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE"],  # POST excluded
    )
    session = requests.Session()
    session.mount('https://', adapter=TimeoutHTTPAdapter(max_retries=retry_strategy, timeout=120))
    session.headers.update({'Authorization': f'Token {token}'})
    return session


def get_project_list(session=konfuzio_session()):
    """
    Get the list of all Projects for the user.

    :param session: Konfuzio session with Retry and Timeout policy
    :return: Response object
    """
    url = get_projects_list_url()
    r = session.get(url=url)
    return r.json()


def get_project_details(project_id: int, session=konfuzio_session()) -> dict:
    """
    Get Label Sets available in Project.

    :param project_id: ID of the Project
    :param session: Konfuzio session with Retry and Timeout policy
    :return: Sorted Label Sets.
    """
    url = get_project_url(project_id=project_id)
    r = session.get(url=url)
    try:
        r.raise_for_status()
    except HTTPError as e:
        raise HTTPError(r.text) from e
    return r.json()


def create_new_project(project_name, session=konfuzio_session()):
    """
    Create a new Project for the user.

    :param project_name: name of the project you want to create
    :param session: Konfuzio session with Retry and Timeout policy
    :return: Response object
    """
    url = get_projects_list_url()
    new_project_data = {"name": project_name}
    r = session.post(url=url, json=new_project_data)

    if r.status_code == 201:
        project_id = r.json()["id"]
        print(f"Project {project_name} (ID {project_id}) was created successfully!")
        return project_id
    else:
        raise PermissionError(
            f'HTTP Status {r.status_code}: The project {project_name} was not created, please check'
            f' your permissions. Error {r.json()}'
        )


def get_document_details(document_id: int, project_id: int, session=konfuzio_session(), extra_fields: str = ''):
    """
    Use the text-extraction server to retrieve the data from a document.

    :param document_id: ID of the document
    :param project_id: ID of the Project
    :param session: Konfuzio session with Retry and Timeout policy
    :param extra_fields: Retrieve bounding boxes and HOCR from document, too. Can be "bbox,hocr", it's a hotfix
    :return: Data of the document.
    """
    url = get_document_api_details_url(document_id=document_id, project_id=project_id, extra_fields=extra_fields)
    r = session.get(url)
    return r.json()


def get_page_image(page_id: int, session=konfuzio_session(), thumbnail: bool = False):
    """
    Load image of a Page as Bytes.

    :param page_id: ID of the Page
    :param thumbnail: Download Page image as thumbnail
    :param session: Konfuzio session with Retry and Timeout policy
    :return: Bytes of the Image.
    """
    if thumbnail:
        raise NotImplementedError
    else:
        url = get_page_image_url(page_id=page_id)

    r = session.get(url)

    content_type = r.headers.get('content-type')
    if content_type != 'image/png':
        raise TypeError(f'CONTENT TYP of Image {page_id} is {content_type} and no PNG.')

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
    project_id: int,
    label_id: int,
    label_set_id: int,
    confidence: Union[float, None] = None,
    revised: bool = False,
    is_correct: bool = False,
    annotation_set=None,
    session=konfuzio_session(),
    **kwargs,
):
    """
    Add an Annotation to an existing document.

    For the Annotation Set definition, we can:
    - define the Annotation Set id_ where the Annotation should belong
    (annotation_set=x (int), define_annotation_set=True)
    - pass it as None and a new Annotation Set will be created
    (annotation_set=None, define_annotation_set=True)
    - do not pass the Annotation Set field and a new Annotation Set will be created if does not exist any or the
    Annotation will be added to the previous Annotation Set created (define_annotation_set=False)

    :param document_id: ID of the file
    :param project_id: ID of the project
    :param label_id: ID of the Label
    :param label_set_id: ID of the Label Set where the Annotation belongs
    :param confidence: Confidence of the Annotation still called Accuracy by text-annotation
    :param revised: If the Annotation is revised or not (bool)
    :param is_correct: If the Annotation is corrected or not (bool)
    :param annotation_set: Annotation Set to connect to the server
    :param session: Konfuzio session with Retry and Timeout policy
    :return: Response status.
    """
    url = get_document_annotations_url(document_id, project_id=project_id)

    # bbox = kwargs.get('bbox', None)
    custom_bboxes = kwargs.get('bboxes', None)
    # selection_bbox = kwargs.get('selection_bbox', None)
    # page_number = kwargs.get('page_number', None)
    # offset_string = kwargs.get('offset_string', None)
    start_offset = kwargs.get('start_offset', None)
    end_offset = kwargs.get('end_offset', None)

    data = {
        'start_offset': start_offset,
        'end_offset': end_offset,
        'label': label_id,
        'revised': revised,
        'section_label_id': label_set_id,
        'accuracy': confidence,
        'is_correct': is_correct,
    }

    if end_offset:
        data['end_offset'] = end_offset

    if start_offset is not None:
        data['start_offset'] = start_offset

    data['section'] = annotation_set

    # if page_number is not None:
    #     data['page_number'] = page_number
    #
    # if offset_string is not None:
    #     data['offset_string'] = offset_string
    #
    # if bbox is not None:
    #     data['bbox'] = bbox
    #
    # if selection_bbox is not None:
    #     data['selection_bbox'] = selection_bbox

    if custom_bboxes is not None:
        data['custom_bboxes'] = custom_bboxes

    r = session.post(url, json=data)
    if r.status_code != 201:
        logger.error(f"Received response status code {r.status_code}.")
    assert r.status_code == 201
    return r


def delete_document_annotation(document_id: int, annotation_id: int, project_id: int, session=konfuzio_session()):
    """
    Delete a given Annotation of the given document.

    :param document_id: ID of the document
    :param annotation_id: ID of the annotation
    :param project_id: ID of the project
    :param session: Konfuzio session with Retry and Timeout policy
    :return: Response status.
    """
    url = get_annotation_url(document_id=document_id, annotation_id=annotation_id, project_id=project_id)
    r = session.delete(url)
    if r.status_code == 200:
        # the text Annotation received negative feedback and copied the Annotation and created a new one
        return json.loads(r.text)['id']
    elif r.status_code == 204:
        return r
    else:
        raise ConnectionError(f'Error{r.status_code}: {r.content} {r.url}')


def get_meta_of_files(project_id: int, limit: int = 1000, session=konfuzio_session()) -> List[dict]:
    """
    Get meta information of Documents in a Project.

    :param project_id: ID of the Project
    :param limit: Number of Documents per Page
    :param session: Konfuzio session with Retry and Timeout policy
    :return: Sorted Documents names in the format {id_: 'pdf_name'}.
    """
    url = get_documents_meta_url(project_id=project_id, limit=limit)
    result = []
    r = session.get(url)
    data = r.json()
    result += data['results']

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
    session=konfuzio_session(),
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
    url = get_labels_url()
    label_sets_ids = [label_set.id_ for label_set in label_sets]

    data = {
        "project": project_id,
        "text": label_name,
        "description": description,
        "has_multiple_top_candidates": has_multiple_top_candidates,
        "get_data_type_display": data_type,
        "templates": label_sets_ids,
    }

    r = session.post(url=url, json=data)

    assert r.status_code == 201
    label_id = r.json()['id']
    return label_id


def upload_file_konfuzio_api(
    filepath: str,
    project_id: int,
    dataset_status: int = 0,
    session=konfuzio_session(),
    category_id: Union[None, int] = None,
    callback_url: str = '',
    sync: bool = False,
):
    """
    Upload Document to Konfuzio API.

    :param filepath: Path to file to be uploaded
    :param project_id: ID of the project
    :param session: Konfuzio session with Retry and Timeout policy
    :param dataset_status: Set data set status of the document.
    :param category_id: Define a Category the Document belongs to
    :param callback_url: Callback URL receiving POST call once extraction is done
    :param sync: If True, will run synchronously and only return once the online database is updated
    :return: Response status.
    """
    url = get_upload_document_url()
    is_file(filepath)

    with open(filepath, "rb") as f:
        file_data = f.read()

    files = {"data_file": (os.path.basename(filepath), file_data, "multipart/form-data")}
    data = {
        "project": project_id,
        "dataset_status": dataset_status,
        "category_template": category_id,
        "sync": sync,
        "callback_url": callback_url,
    }

    r = session.post(url=url, files=files, data=data)

    try:
        r.raise_for_status()
    except HTTPError as e:
        raise HTTPError(r.text) from e
    return r


def delete_file_konfuzio_api(document_id: int, session=konfuzio_session()):
    """
    Delete Document by ID via Konfuzio API.

    :param document_id: ID of the document
    :param session: Konfuzio session with Retry and Timeout policy
    :return: File id_ in Konfuzio Server.
    """
    url = get_document_url(document_id)
    data = {'id': document_id}

    r = session.delete(url=url, json=data)

    try:
        r.raise_for_status()
    except HTTPError as e:
        raise HTTPError(r.text) from e

    return True


def update_document_konfuzio_api(document_id: int, session=konfuzio_session(), **kwargs):
    """
    Update an existing Document via Konfuzio API.

    :param document_id: ID of the document
    :param session: Konfuzio session with Retry and Timeout policy
    :return: Response status.
    """
    url = get_document_url(document_id)

    data = {}
    file_name = kwargs.get('file_name', None)
    dataset_status = kwargs.get('dataset_status', None)
    category_id = kwargs.get('category_template_id', None)
    assignee = kwargs.get('assignee', None)

    if file_name is not None:
        data.update({"data_file_name": file_name})

    if dataset_status is not None:
        data.update({"dataset_status": dataset_status})

    if category_id is not None:
        data.update({"category_template": category_id})

    if assignee is not None:
        data.update({"assignee": assignee})

    r = session.patch(url=url, json=data)

    try:
        r.raise_for_status()
    except HTTPError as e:
        raise HTTPError(r.text) from e

    return json.loads(r.text)


def download_file_konfuzio_api(document_id: int, ocr: bool = True, session=konfuzio_session()):
    """
    Download file from the Konfuzio server using the Document id_.

    Django authentication is form-based, whereas DRF uses BasicAuth.

    :param document_id: ID of the document
    :param ocr: Bool to get the ocr version of the document
    :param session: Konfuzio session with Retry and Timeout policy
    :return: The downloaded file.
    """
    if ocr:
        url = get_document_ocr_file_url(document_id)
    else:
        url = get_document_original_file_url(document_id)

    r = session.get(url)

    content_type = r.headers.get('content-type')
    if content_type not in ['application/pdf', 'image/jpeg', 'image/png', 'image/jpg']:
        logger.info(f'CONTENT TYP of {document_id} is {content_type} and no PDF or image.')

    logger.info(f'Downloaded file {document_id} from {KONFUZIO_HOST}.')
    return r.content


def get_results_from_segmentation(doc_id: int, project_id: int, session=konfuzio_session()) -> List[List[dict]]:
    """Get bbox results from segmentation endpoint.

    :param doc_id: ID of the document
    :param project_id: ID of the Project.
    :param session: Konfuzio session with Retry and Timeout policy
    """
    segmentation_url = get_document_segmentation_details_url(doc_id, project_id)
    response = session.get(segmentation_url)
    segmentation_result = response.json()

    return segmentation_result


def upload_ai_model(ai_model_path: str, category_ids: List[int] = None, session=konfuzio_session()):  # noqa: F821
    """
    Upload an ai_model to the text-annotation server.

    :param ai_model_path: Path to the ai_model
    :param category_ids: define ids of Categories the model should become available after upload.
    :param session: session to connect to server
    :return:
    """
    url = get_create_ai_model_url()
    if is_file(ai_model_path):
        model_name = os.path.basename(ai_model_path)
        with open(ai_model_path, 'rb') as f:
            multipart_form_data = {'ai_model': (model_name, f)}
            headers = {"Prefer": "respond-async"}
            r = session.post(url, files=multipart_form_data, headers=headers)
            r.raise_for_status()
    data = r.json()
    ai_model_id = data['id']
    ai_model = data['ai_model']

    if category_ids:
        url = get_update_ai_model_url(ai_model_id)
        data = {'templates': category_ids}
        headers = {'content-type': 'application/json'}
        response = session.patch(url, data=json.dumps(data), headers=headers)
        response.raise_for_status()

    logger.info(f'New AI Model uploaded {ai_model} to {url}')
    return ai_model
