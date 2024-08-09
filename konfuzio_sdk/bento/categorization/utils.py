"""Utility functions for adapting Konfuzio concepts to be used with Pydantic models for categorization."""
from typing import Optional

from pydantic import BaseModel

from konfuzio_sdk.data import CategoryAnnotation, Document, Page, Project

from .schemas import CategorizeRequest20240729, CategorizeRequest20240729Page, CategorizeResponse20240729

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
    pages_result = []
    if schema.__name__ == 'CategorizeResponse20240729':
        for page in result.pages:
            current_page = {'number': page.number, 'categories': []}
            for category_annotation in page.category_annotations:
                current_page['categories'] = {
                    'category_id': category_annotation.category.id_,
                    'confidence': category_annotation.confidence,
                }
            pages_result.append(current_page)
    else:
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR_MESSAGE)
    return schema(pages=pages_result)


def convert_document_to_request(document: Document, schema: BaseModel = CategorizeRequest20240729) -> BaseModel:
    """
    Receive a Document and convert it into a request in accordance to a passed schema.

    :param document: A Document to be converted.
    :param schema: A schema to which the request should adhere.
    :returns: A Document converted in accordance with the schema.
    """
    if schema.__name__ == 'CategorizeRequest20240729':
        pages = [
            CategorizeRequest20240729Page(
                number=page.number,
                image=page.image_bytes,
                original_size=page._original_size,
                segmentation=page._segmentation,
            )
            for page in document.pages()
        ]
        converted = schema(
            text=document.text,
            bboxes={
                k: {
                    'x0': v.x0,
                    'x1': v.x1,
                    'y0': v.y0,
                    'y1': v.y1,
                    'page_number': v.page.number,
                    'top': v.top,
                    'bottom': v.bottom,
                    'text': document.text[k],
                }
                for k, v in document.bboxes.items()
            },
            pages=pages,
        )
    else:
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR_MESSAGE)
    return converted


def convert_response_to_categorized_pages(
    response: BaseModel, document: Document, mappings: Optional[dict] = None
) -> Document:
    """
    Receive a CategorizeResponse and convert it into a list of categorized Pages to be added to the Document.

    :param response: A CategorizeResponse to be converted.
    :param document: A Document to which the categorized Pages should be added.
    :param mappings: A dict with "categories" keys containing mappings from old to new IDs. Original
        IDs are used if no mapping is provided or if the mapping is not found.
    :returns: The original Document with added categorized Pages.
    """
    if mappings is None:
        mappings = {}

    # Mappings might be from JSON, so we need to convert keys to integers.
    category_mappings = {int(k): v for k, v in mappings.get('categories', {}).items()}

    if response.__class__.__name__ == 'CategorizeResponse20240729':
        for page in response.pages:
            document_page = document.get_page_by_index(page['number'] - 1)
            for category in page.categories:
                category_id = category_mappings.get(category.id_, category.id_)
                confidence = category.confidence
                document_page.add_category_annotation(
                    category_annotation=CategoryAnnotation(
                        category=document.project.get_category_by_id(category_id), confidence=confidence
                    )
                )
        return document
    else:
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR_MESSAGE)
