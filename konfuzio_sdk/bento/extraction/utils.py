"""Utility functions for adapting Konfuzio concepts to be used with Pydantic models."""
from pydantic import BaseModel

from konfuzio_sdk.data import Annotation, AnnotationSet, Document, Page, Project, Span

from .schemas import ExtractRequest20240117, ExtractRequest20240117Page, ExtractResponse20240117

NOT_IMPLEMENTED_ERROR_MESSAGE = (
    'The request does not adhere to any schema version. Please modify the request to fit one of the schemas from '
    'bento/extraction/schemas.py.'
)


def prepare_request(request: BaseModel, project: Project) -> Document:
    """
    Receive a request and prepare it for the extraction runner.

    :param request: Unprocessed request.
    :param project: A Project instance.
    :returns: An instance of a Document class.
    """
    # Extraction AIs include only one Category per Project.
    category = project.categories[0]
    if request.__class__.__name__ == 'ExtractRequest20240117':
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
            category=category,
        )
        for page in request.pages:
            p = Page(id_=page.number, document=document, number=page.number, original_size=page.original_size)
            if page.segmentation:
                p._segmentation = page.segmentation
    else:
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR_MESSAGE)
    return document


def process_response(result, schema=ExtractResponse20240117):
    """
    Process a raw response from the runner to contain only selected fields.

    :param result: A raw response to be processed.
    :param schema: A schema of the response.
    :returns: A list of dictionaries with Label Set IDs and Annotation data.
    """
    annotations_result = []
    if schema.__name__ == 'ExtractResponse20240117':
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
                    for span in annotation.spans
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
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR_MESSAGE)
    return schema(annotation_sets=annotations_result)


def convert_document_to_request(document: Document, schema: BaseModel = ExtractRequest20240117):
    """
    Receive a Document and convert it into a request in accordance to a passed schema.

    :param document: A Document to be converted.
    :param schema: A schema to which the request should adhere.
    :returns: A Document converted in accordance with the schema.
    """
    pages = [
        ExtractRequest20240117Page(
            number=page.number, image=page.image, original_size=page._original_size, segmentation=page._segmentation
        )
        for page in document.pages()
    ]
    if schema.__name__ == 'ExtractRequest20240117':
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


def convert_response_to_annotations(response: BaseModel, document: Document):
    """
    Receive an ExtractResponse and convert it into a list of Annotations to be added to the Document.

    :param annotations: An ExtractResponse to be converted.
    :param document: A Document to which the annotations should be added.
    :param schema: A schema to which the annotations should adhere.
    :returns: The original Document with added Annotations.
    """
    if response.__class__.__name__ == 'ExtractResponse20240117':
        for annotation_set in response.annotation_sets:
            sdk_annotation_set = AnnotationSet(
                document=document, label_set=document.project.get_label_set_by_id(annotation_set.label_set_id)
            )
            for annotation in annotation_set.annotations:
                Annotation(
                    document=document,
                    annotation_set=sdk_annotation_set,
                    label=document.project.get_label_by_id(annotation.label.id),
                    offset_string=annotation.offset_string,
                    translated_string=annotation.translated_string,
                    normalized=annotation.normalized,
                    confidence=annotation.confidence,
                    spans=[
                        Span(
                            start_offset=span.start_offset,
                            end_offset=span.end_offset,
                        )
                        for span in annotation.span
                    ],
                )
        return document
    else:
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR_MESSAGE)
