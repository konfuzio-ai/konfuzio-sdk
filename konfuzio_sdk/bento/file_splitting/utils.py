"""Utility functions for adapting Konfuzio concepts to be used with Pydantic models for file splitting."""
from typing import Optional

from pydantic import BaseModel

from konfuzio_sdk.data import Document, Page, Project

from .schemas import SplitRequest20240930, SplitRequest20240930Page, SplitResponse20240930

NOT_IMPLEMENTED_ERROR_MESSAGE = (
    'The request does not adhere to any schema version. Please modify the request to fit one of the schemas from '
    'bento/file_splitting/schemas.py.'
)


def prepare_request(request: BaseModel, project: Project, konfuzio_sdk_version: Optional[str] = None) -> Document:
    """
    Receive a request and prepare it for the file splitting runner.
    :param request: Unprocessed request.
    :param project: A Project instance.
    :returns: An instance of a Document class.
    """
    document_id = max((doc.id_ for doc in project._documents if doc.id_), default=0) + 1

    if request.__class__.__name__ == 'SplitRequest20240930':
        bboxes = {}
        for bbox_id, bbox in request.bboxes.items():
            bboxes[str(bbox_id)] = {
                'x0': bbox.x0,
                'x1': bbox.x1,
                'y0': bbox.y0,
                'y1': bbox.y1,
                'page_number': bbox.page_number,
                'text': bbox.text,
            }
            # Backwards compatibility with Konfuzio SDK versions < 0.3.
            # In newer versions, the top and bottom values are not needed.
            if konfuzio_sdk_version:
                if konfuzio_sdk_version < '0.3':
                    page = next(page for page in request.pages if page.number == bbox.page_number)
                    bboxes[str(bbox_id)]['top'] = round(page.original_size[1] - bbox.y0, 4)
                    bboxes[str(bbox_id)]['bottom'] = round(page.original_size[1] - bbox.y1, 4)
        document = Document(
            id_=document_id,
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


def process_response(result, schema: BaseModel = SplitResponse20240930) -> BaseModel:
    """
    Process a raw response from the runner to contain only selected fields.
    :param result: A raw response to be processed.
    :param schema: A schema of the response.
    :returns: A list of dictionaries with Pages and their Categories.
    """
    results = []
    if schema.__name__ == 'SplitResponse20240930':
        for document in result:
            category_annotations = []
            for category_annotation in document.category_annotations:
                category_annotations.append(
                    schema.SplittingResult.CategoryAnnotation(
                        category_id=category_annotation.category.id_,
                        confidence=category_annotation.confidence,
                        category_name=category_annotation.category.name,
                    )
                )
            results.append(
                schema.SplittingResult(
                    page_ids=[page.id_ for page in document.pages()],
                    category=document.category.id_,
                    categories=category_annotations,
                )
            )
    else:
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR_MESSAGE)
    return schema(splitting_results=results)


def convert_document_to_request(document: Document, schema: BaseModel = SplitRequest20240930) -> BaseModel:
    """
    Receive a Document and convert it into a request in accordance to a passed schema.
    :param document: A Document to be converted.
    :param schema: A schema to which the request should adhere.
    :returns: A Document converted in accordance with the schema.
    """
    if schema.__name__ == 'SplitRequest20240930':
        pages = [
            SplitRequest20240930Page(
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
