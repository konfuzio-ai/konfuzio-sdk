"""Define pydantic models for request and response from the Categorization AI."""
from typing import Any, Dict, List, Optional, Tuple

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


class CategorizeRequest20240729Page(BaseModel):
    """Describe a scheme for Page class on 29/07/2024."""

    number: int
    image: Optional[HexBytes] = None
    original_size: Tuple[float, float]
    segmentation: Optional[list] = None


class CategorizeRequest20240729(BaseModel):
    """Describe a scheme for the categorization request on 29/07/2024."""

    class Bbox(BaseModel):
        """Describe a scheme for the Bbox class on 29/07/2024."""

        x0: float
        x1: float
        y0: float
        y1: float
        page_number: int
        text: Optional[str]

    text: str
    bboxes: Optional[Dict[int, Bbox]]
    pages: Optional[List[CategorizeRequest20240729Page]]


class CategorizeResponse20240729(BaseModel):
    """Describe a scheme for the categorization response on 29/07/2024."""

    class CategorizedPage(BaseModel):
        """Describe a scheme for the CategorizedPage class after categorization on 29/07/2024."""

        class PredictedCategory(BaseModel):
            """Describe a scheme for the PredictedCategory class on 29/07/2024."""

            category_id: int
            confidence: float

        number: int
        categories: List[PredictedCategory]

    pages: List[CategorizedPage]
