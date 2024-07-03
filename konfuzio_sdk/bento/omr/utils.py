"""Utility functions for adapting OMR functionality to be used with Pydantic models."""

from pydantic import BaseModel

from konfuzio_sdk.data import Document

from .schemas import (
    CheckboxRequest20240523,
    CheckboxResponse20240523,
)

NOT_IMPLEMENTED_ERROR_MESSAGE = (
    'The request does not adhere to any schema version. Please modify the request to fit one of the schemas from '
    'bento/omr/schemas.py.'
)


def convert_document_to_request(document: Document, schema: BaseModel = CheckboxRequest20240523) -> BaseModel:
    """
    Receive a Document and convert it into a request in accordance to a passed schema.

    :param document: A Document to be converted.
    :param schema: A schema to which the request should adhere.
    :returns: A Document converted in accordance with the schema.
    """

    if schema.__name__ == 'CheckboxRequest20240523':
        pages = [
            {
                'page_id': page.id_,
                'width': int(page.width),
                'height': int(page.height),
                'image': open(page.image_path, 'rb').read(),
            }
            for page in document.pages()
        ]

        annotations = [
            {
                'page_id': a.page.id_,
                'annotation_id': a.id_,
                # bbox=CheckboxRequest20240523.Annotation.Box(x0=int(a.x0), x1=int(a.x1), y0=int(a.y0), y1=int(a.y1)),
                'bbox': {
                    'x0': int(a.bboxes[0]['x0']),
                    'x1': int(a.bboxes[0]['x1']),
                    'y0': int(a.bboxes[0]['y0']),
                    'y1': int(a.bboxes[0]['y1']),
                },
            }
            for a in document.annotations()
            if (
                getattr(a.label, 'is_linked_to_checkbox', True) or a.label.is_linked_to_checkbox is None
            )  # TODO: Should be changed to an explicit check if True once the checkbox service is fully integrated.
        ]
        converted = schema(pages=pages, annotations=annotations)

    else:
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR_MESSAGE)
    return converted


def convert_response_to_checkbox_annotations(
    document: Document, response: BaseModel = CheckboxResponse20240523
) -> Document:
    """
    Receive a CheckboxResponse and add the metadata to the Annotations of the Document.

    :param document: A Document where the annotations should be checkbox-annotated.
    :param response: A CheckboxResponse to be converted.
    :returns: The original Document with checkbox-annotated Annotations.
    """
    if response.__class__.__name__ == 'CheckboxResponse20240523':
        for annotation_metadata in response.metadata:
            annotation = document.get_annotation_by_id(annotation_metadata.annotation_id)
            if annotation.metadata is None:
                annotation.metadata = {}
            annotation.metadata['checkbox'] = annotation_metadata.checkbox
        return document
    else:
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR_MESSAGE)
