"""Utility functions for adapting Konfuzio concepts to be used with Pydantic models for categorization."""
from pydantic import BaseModel

from konfuzio_sdk.data import Document, Page, Project

from .schemas import CategorizeResponse20240729

NOT_IMPLEMENTED_ERROR_MESSAGE = (
    'The request does not adhere to any schema version. Please modify the request to fit one of the schemas from '
    'bento/categorization/schemas.py.'
)


def prepare_request(request: BaseModel, project: Project) -> Document:
    """
    Receive a request and prepare it for the categorization runner.

    :param request: Unprocessed request.
    :param project: A Project instance.
    :returns: An instance of a Document class.
    """
    if request.__class__.__name__ == 'CategorizeRequest20240729':
        bboxes = {
            str(bbox_id): {
                'x0': bbox.x0,
                'x1': bbox.x1,
                'y0': bbox.y0,
                'y1': bbox.y1,
                'page_number': bbox.page_number,
                'text': bbox.text,
            }
            for bbox_id, bbox in request.bboxes.items()
        }
        document = Document(
            text=request.text,
            bbox=bboxes,
            project=project,
        )
        for page in request.pages:
            p = Page(id_=page.number, document=document, number=page.number, original_size=page.original_size)
            if page.segmentation:
                p._segmentation = page.segmentation
            if page.image:
                p.image_bytes = page.image
    else:
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR_MESSAGE)
    return document


def process_response(result, schema: BaseModel = CategorizeResponse20240729) -> BaseModel:
    """
    Process a raw response from the runner to contain only selected fields.

    :param result: A raw response to be processed.
    :param schema: A schema of the response.
    :returns: A list of dictionaries with Pages and their Categories.
    """
    # pages_result = []
    if schema.__name__ == 'CategorizeResponse20240729':
        for page in result.pages:
            _ = {'number': page.number, 'categories': []}
