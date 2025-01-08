"""Utility functions for adapting Konfuzio concepts to be used with Pydantic models."""
import logging
import typing as t

from pydantic import BaseModel

from konfuzio_sdk.data import Annotation, AnnotationSet, Document, Page, Project, Span

from .schemas import ExtractRequest20240117Page, ExtractRequest20241227, ExtractResponse20240117

NOT_IMPLEMENTED_ERROR_MESSAGE = (
    'The request does not adhere to any schema version. Please modify the request to fit one of the schemas from ' 'bento/extraction/schemas.py.'
)

# Pydanctic conversion functions

logger = logging.getLogger()


def prepare_request(request: BaseModel, project: Project, konfuzio_sdk_version: t.Optional[str] = None) -> Document:
    """
    Receive a request and prepare it for the extraction runner.

    :param request: Unprocessed request.
    :param project: A Project instance.
    :param konfuzio_sdk_version: The version of the Konfuzio SDK used by the embedded AI model. Used to apply backwards
        compatibility changes for older SDK versions.
    :returns: An instance of a Document class.
    """
    # Extraction AIs include only one Category per Project.
    category = project.categories[0]
    # Calculate next available ID based on current Project documents to avoid conflicts.
    document_id = max((doc.id_ for doc in project._documents if doc.id_), default=0) + 1

    if request.__class__.__name__ == 'ExtractRequest20240117':
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
            if konfuzio_sdk_version and konfuzio_sdk_version < '0.3':
                page = next(page for page in request.pages if page.number == bbox.page_number)
                bboxes[str(bbox_id)]['top'] = round(page.original_size[1] - bbox.y0, 4)
                bboxes[str(bbox_id)]['bottom'] = round(page.original_size[1] - bbox.y1, 4)
        document = Document(
            id_=document_id,
            text=request.text,
            bbox=bboxes,
            project=project,
            category=category,
        )
        for page in request.pages:
            p = Page(id_=page.number, document=document, number=page.number, original_size=page.original_size)
            if page.segmentation:
                p._segmentation = page.segmentation
            if page.image:
                p.image_bytes = page.image
    elif request.__class__.__name__ == 'ExtractRequest20241223':
        bboxes = {}
        for bbox_id, bbox in request.bboxes.items():
            bboxes[str(bbox_id)] = {
                'x0': bbox.x0,
                'x1': bbox.x1,
                'y0': bbox.y0,
                'y1': bbox.y1,
                'page_number': bbox.page_number,
                'text': bbox.text,
                'line_index': bbox.line_index,
            }
            # Backwards compatibility with Konfuzio SDK versions < 0.3.
            # In newer versions, the top and bottom values are not needed.
            if konfuzio_sdk_version and konfuzio_sdk_version < '0.3':
                page = next(page for page in request.pages if page.number == bbox.page_number)
                bboxes[str(bbox_id)]['top'] = round(page.original_size[1] - bbox.y0, 4)
                bboxes[str(bbox_id)]['bottom'] = round(page.original_size[1] - bbox.y1, 4)
        document = Document(
            id_=document_id,
            text=request.text,
            bbox=bboxes,
            project=project,
            category=category,
        )
        for page in request.pages:
            p = Page(id_=page.number, document=document, number=page.number, original_size=page.original_size)
            if page.segmentation:
                p._segmentation = page.segmentation
            if page.image:
                p.image_bytes = page.image
    elif request.__class__.__name__ == 'ExtractRequest20241227':
        bboxes = {}
        for bbox_id, bbox in request.bboxes.items():
            bboxes[str(bbox_id)] = {
                'x0': bbox.x0,
                'x1': bbox.x1,
                'y0': bbox.y0,
                'y1': bbox.y1,
                'page_number': bbox.page_number,
                'text': bbox.text,
                'line_index': bbox.line_index,
            }
            # Backwards compatibility with Konfuzio SDK versions < 0.3.
            # In newer versions, the top and bottom values are not needed.
            if konfuzio_sdk_version and konfuzio_sdk_version < '0.3':
                page = next(page for page in request.pages if page.number == bbox.page_number)
                bboxes[str(bbox_id)]['top'] = round(page.original_size[1] - bbox.y0, 4)
                bboxes[str(bbox_id)]['bottom'] = round(page.original_size[1] - bbox.y1, 4)
        document = Document(
            id_=document_id,
            text=request.text,
            bbox=bboxes,
            project=project,
            category=category,
        )
        document.raw_ocr_response = request.raw_ocr_response
        for page in request.pages:
            p = Page(id_=page.number, document=document, number=page.number, original_size=page.original_size)
            if page.segmentation:
                p._segmentation = page.segmentation
            if page.image:
                p.image_bytes = page.image
    else:
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR_MESSAGE)
    return document


def process_response(result, schema: BaseModel = ExtractResponse20240117) -> BaseModel:
    """
    Process a raw response from the runner to contain only selected fields.

    :param result: A raw response to be processed.
    :param schema: A schema of the response.
    :returns: A list of dictionaries with Label Set IDs and Annotation data.
    """
    annotations_result = []
    if schema.__name__ == 'ExtractResponse20240117':
        for annotation_set in result.annotation_sets():
            if not annotation_set.label_set.id_:
                continue
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
        return schema(annotation_sets=annotations_result)
    elif schema.__name__ == 'ExtractResponseForLegacyTrainer20240912':
        import json

        class JSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if hasattr(obj, 'to_json'):
                    return obj.to_json()
                return json.JSONEncoder.default(self, obj)

        json_result = json.loads(json.dumps(result, cls=JSONEncoder))
        return schema(json_result)
    else:
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR_MESSAGE)


def convert_document_to_request(document: Document, schema: BaseModel = ExtractRequest20241227) -> BaseModel:
    """
    Receive a Document and convert it into a request in accordance to a passed schema.

    :param document: A Document to be converted.
    :param schema: A schema to which the request should adhere.
    :returns: A Document converted in accordance with the schema.
    """
    if schema.__name__ == 'ExtractRequest20240117':
        pages = [
            ExtractRequest20240117Page(
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
    elif schema.__name__ == 'ExtractRequest20241223':
        pages = [
            ExtractRequest20240117Page(
                number=page.number,
                image=page.image_bytes,
                original_size=page._original_size,
                segmentation=page._segmentation,
            )
            for page in document.pages()
        ]
        doc_bbox = document.get_bbox()
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
                    'line_index': doc_bbox[str(k)].get('line_index') or doc_bbox[str(k)]['line_number'] - 1,
                }
                for k, v in document.bboxes.items()
            },
            pages=pages,
        )
    elif schema.__name__ == 'ExtractRequest20241227':
        pages = [
            ExtractRequest20240117Page(
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
                    'line_index': doc_bbox[str(k)].get('line_index') or doc_bbox[str(k)]['line_number'] - 1,
                }
                for k, v in document.bboxes.items()
            },
            pages=pages,
            raw_ocr_response=document.raw_ocr_response if hasattr(document, 'raw_ocr_response') else None,
        )

    else:
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR_MESSAGE)
    return converted


def convert_response_to_annotations(response, response_schema_class: BaseModel, document: Document, mappings: t.Optional[dict] = None) -> Document:
    """
    Receive an ExtractResponse and convert it into a list of Annotations to be added to the Document.

    :param response: A Response instance to be converted.
    :param response_schema_class: The schema class.
    :param document: A Document to which the annotations should be added.
    :param mappings: A dict with "label_sets" and "labels" keys, both containing mappings from old to new IDs. Original
        IDs are used if no mapping is provided or if the mapping is not found.
    :returns: The original Document with added Annotations.
    """
    if mappings is None:
        mappings = {}

    # Mappings might be from JSON, so we need to convert keys to integers.
    label_set_mappings = {int(k): v for k, v in mappings.get('label_sets', {}).items()}
    label_mappings = {int(k): v for k, v in mappings.get('labels', {}).items()}

    if response_schema_class.__name__ == 'ExtractResponse20240117':
        response = response_schema_class(annotation_sets=response.json()['annotation_sets'])

        for annotation_set in response.annotation_sets:
            label_set_id = label_set_mappings.get(annotation_set.label_set_id, annotation_set.label_set_id)
            sdk_annotation_set = AnnotationSet(document=document, label_set=document.project.get_label_set_by_id(label_set_id))
            for annotation in annotation_set.annotations:
                label_id = label_mappings.get(annotation.label.id, annotation.label.id)
                Annotation(
                    document=document,
                    annotation_set=sdk_annotation_set,
                    label=document.project.get_label_by_id(label_id),
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
    elif response_schema_class.__name__ == 'ExtractResponseForLegacyTrainer20240912':
        """Restore Pandas Dataframe which has been converted to JSON for sending via API."""
        import json

        result = response.json()

        def my_convert(res_dict):
            import pandas as pd

            new_dict = {}
            for k, v in res_dict.items():
                if isinstance(v, dict):
                    new_dict[k] = my_convert(v)
                if isinstance(v, list):
                    new_dict[k] = [my_convert(x) for x in v]
                if isinstance(v, str):
                    new_dict[k] = pd.DataFrame.from_dict(json.loads(v))
            return new_dict

        my_converted_json = my_convert(result)
        return my_converted_json
    else:
        logger.error(f'Schema "{response.__class__.__name__}" is not implemented.')
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR_MESSAGE)
