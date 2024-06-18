"""Define pydantic models for request and response from the Extraction AI."""
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, PlainSerializer, PlainValidator, WithJsonSchema, errors
from typing_extensions import Annotated


def hex_bytes_validator(o: Any) -> bytes:
    """
    Custom validator to be able to correctly serialize and unserialize bytes.
    See https://github.com/pydantic/pydantic/issues/3756#issuecomment-1654425270
    """
    if isinstance(o, bytes):
        return o
    elif isinstance(o, bytearray):
        return bytes(o)
    elif isinstance(o, str):
        return bytes.fromhex(o)
    raise errors.BytesError


HexBytes = Annotated[
    bytes, PlainValidator(hex_bytes_validator), PlainSerializer(lambda b: b.hex()), WithJsonSchema({'type': 'string'})
]


class ExtractRequest20240117Page(BaseModel):
    """Describe a scheme for the Page class on 17/01/2024."""

    number: int
    image: Optional[HexBytes] = None
    original_size: Tuple[float, float]
    segmentation: Optional[list] = None


class ExtractRequest20240117(BaseModel):
    """Describe a scheme for the extraction request on 17/01/2024."""

    class Bbox(BaseModel):
        """Describe a scheme for the Bbox class on 17/01/2024."""

        x0: float
        x1: float
        y0: float
        y1: float
        page_number: int
        text: Optional[str]

    text: str
    bboxes: Optional[Dict[int, Bbox]]
    pages: Optional[List[ExtractRequest20240117Page]]


class ExtractResponse20240117(BaseModel):
    """Describe a scheme for the extraction response on 17/01/2024."""

    class AnnotationSet(BaseModel):
        """Describe a scheme for the AnnotationSet class on 17/01/2024."""

        class Annotation(BaseModel):
            """Describe a scheme for the Annotation class on 17/01/2024."""

            class Label(BaseModel):
                """Describe a scheme for the Label class on 17/01/2024."""

                id: int
                name: str
                has_multiple_top_candidates: bool
                data_type: str
                threshold: float

            class Span(BaseModel):
                """Describe a scheme for the Span class on 17/01/2024."""

                x0: Union[int, float]
                x1: Union[int, float]
                y0: Union[int, float]
                y1: Union[int, float]
                page_index: int
                start_offset: int
                end_offset: int
                offset_string: Optional[str]
                offset_string_original: str

            offset_string: Optional[Union[str, List[str]]]
            translated_string: Optional[str]
            normalized: Union[str, int, None]
            label: Label
            annotation_set: int
            confidence: float
            span: List[Span]

        label_set_id: int
        annotations: List[Annotation]

    annotation_sets: List[AnnotationSet]
