"""Utility functions for adapting Konfuzio concepts to be used with Pydantic models."""
from pydantic import BaseModel

from konfuzio_sdk.data import Category, Document, Page, Project

from .schemas import ExtractRequest20240117Page, ExtractResponse20240117


def prepare_request(request: BaseModel) -> Document:
    """
    Receive a request and prepare it for the extraction runner.

    :param request: Unprocessed request.
    :returns: An instance of a Document class.
    """

    project = Project(id_=None)
    project.set_offline()
    category = Category(project=project)
    if 'ExtractRequest20240117' in str(request):
        bboxes = {
            bbox_id: {
                'x0': bbox.x0,
                'x1': bbox.x1,
                'y0': bbox.y0,
                'y1': bbox.y1,
                'page_number': bbox.page.number,
                'text': bbox.text,
            }
            for bbox_id, bbox in request.bboxes.items()
        }
        document = Document(
            text=request.text,
            bbox=bboxes,
            project=project,
            category=category,
        )
        for page in request.pages:
            Page(id_=page.number, document=document, number=page.number, original_size=page.original_size)
    else:
        raise NotImplementedError(
            'The request does not adhere to any version of schema. Please, modify the request to '
            'fit one of the schemas from bento/extraction/schemas.py.'
        )
    return document


def convert_document_to_request(document: Document, schema: BaseModel):
    """
    Receive a Document and convert it into a request in accordance to a passed schema.

    :param document: A Document to be converted.
    :param schema: A schema to which the request should adhere.
    :returns: A Document converted in accordance with the schema.
    """
    pages = [
        ExtractRequest20240117Page(number=page.number, image=page.image, original_size=page._original_size)
        for page in document.pages()
    ]
    bboxes = {
        str(bbox_id): schema.Bbox(
            x0=bbox.x0,
            x1=bbox.x1,
            y0=bbox.y0,
            y1=bbox.y1,
            page=ExtractRequest20240117Page(
                number=bbox.page.number, image=bbox.page.image, original_size=bbox.page._original_size
            ),
        )
        for bbox_id, bbox in document.bboxes.items()
    }
    if 'ExtractRequest20240117' in str(schema):
        converted = schema(text=document.text, bboxes=bboxes, pages=pages)
    else:
        raise NotImplementedError(
            'The request does not adhere to any version of schema. Please, modify the request to '
            'fit one of the schemas from bento/extraction/schemas.py.'
        )
    return converted


def process_response(result, schema=ExtractResponse20240117):
    """
    Process a raw response from the runner to contain only selected fields.

    :param result: A raw response to be processed.
    :param schema: A schema of the response.
    :returns: A list of dictionaries with Label Set IDs and Annotation data.
    """
    annotations_result = []
    if schema.__class__.__name__ == 'ExtractResponse20240117':
        for annotation_set in result.annotation_sets():
            current_annotation_set = {'label_set_id': annotation_set.label_set.id_, 'annotations': []}
            for annotation in annotation_set.annotations(use_correct=False, ignore_below_threshold=True):
                spans_list_of_dicts = [
                    schema.AnnotationSet.Annotation.Span(
                        x0=span.bbox().x0,
                        x1=span.bbox().x1,
                        y0=span.bbox().y0,
                        y1=span.bbox().y1,
                        page_index=span.page.index,
                        start_offset=span.start_offset,
                        end_offset=span.end_offset,
                        offset_string=span.offset_string,
                        offset_string_original=span.offset_string,
                    )
                    for span in annotation.spans()
                ]
                current_annotation_set['annotations'].append(
                    schema.AnnotationSet.Annotation(
                        offset_string=annotation.offset_string,
                        translated_string=annotation.translated_string,
                        normalized=annotation.normalized,
                        label=schema.AnnotationSet.Annotation.Label(
                            id=annotation.label.id_,
                            name=annotation.label.name,
                            has_multiple_top_candidates=annotation.label.has_multiple_top_candidates,
                            data_type=annotation.label.data_type,
                            threshold=annotation.label.threshold,
                        ),
                        confidence=annotation.confidence,
                        annotation_set=annotation.annotation_set.id_,
                        span=spans_list_of_dicts,
                        selection_bbox=annotation.selection_bbox,
                    )
                )
            annotations_result.append(current_annotation_set)
    else:
        raise NotImplementedError(
            'The request does not adhere to any version of schema. Please, modify the request to '
            'fit one of the schemas from bento/extraction/schemas.py.'
        )
    return schema(annotation_sets=annotations_result)
