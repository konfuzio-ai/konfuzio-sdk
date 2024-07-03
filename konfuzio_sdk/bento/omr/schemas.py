"""Define pydantic models for request and response from the Extraction AI."""

from typing import Any, List, Optional, Union

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


class CheckboxRequest20240523(BaseModel):
    """
    Describe a scheme for the Optical Mark Recognition (OMR) request on 15/02/2024.

    Info:
    Just Annotations with the attribute is_linked_to_checkbox set to True are allowed.
    The Page images need to be str-base64 encoded.
    """

    class Annotation(BaseModel):
        """Describe a scheme for the Annotation info needed for OMR on 15/02/2024."""

        class Box(BaseModel):
            """Describe a scheme for BBox on 15/02/2024."""

            x0: int
            x1: int
            y0: int
            y1: int

        page_id: int
        annotation_id: int
        bbox: Box

    class Page(BaseModel):
        """Describe a scheme for the Page info needed for OMR on 15/02/2024."""

        page_id: int
        width: int
        height: int
        image: HexBytes

    pages: List[Page]
    annotations: List[Annotation]
    # by default the threshold is defined by the checkbox detector bento, this argument allows to flexibly overwrite it
    detection_threshold: Optional[float] = None


class CheckboxResponse20240523(BaseModel):
    """Describe a scheme for the Optical Mark Recognition (OMR) response on 15/02/2024."""

    class MetaData(BaseModel):
        """Describe a scheme for OMR meta data on 15/02/2024."""

        class Checkbox(BaseModel):
            """Describe a scheme for OMR checkbox data on 15/02/2024."""

            class Box(BaseModel):
                """Describe a scheme for BBox on 15/02/2024."""

                x0: int
                x1: int
                y0: int
                y1: int

            is_checked: Union[bool, None]  # None if undetermined or invalid
            bbox: Box
            confidence: float

        annotation_id: int
        checkbox: Checkbox

    metadata: List[MetaData]
