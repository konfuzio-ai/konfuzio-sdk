"""Define pydantic models for request and response from the Splitting AI."""
from typing import Dict, List, Optional, Tuple, Union

from pydantic import BaseModel

# Use relative or top module import based on whether this is run as an actual service or imported
try:
    from ..base.utils import HexBytes
except (ImportError, ValueError):
    from base.utils import HexBytes


class CategoryAnnotation20240930(BaseModel):
    """Describe a scheme for a CategoryAnnotation on 30/09/2024."""

    id: int
    name: str
    confidence: float


class SplitRequest20240930Page(BaseModel):
    """Describe a scheme for Page class on 30/09/2024."""

    id: int
    number: int
    image: Optional[HexBytes] = None
    original_size: Tuple[float, float]
    segmentation: Optional[list] = None
    categories: List[CategoryAnnotation20240930]


class Bbox20240930(BaseModel):
    """Describe a scheme for the Bbox class on 30/09/2024."""

    x0: float
    x1: float
    y0: float
    y1: float
    page_number: int
    text: Optional[str]


class SplitRequest20240930(BaseModel):
    """Describe a scheme for the splitting request on 30/09/2024."""

    text: str
    bboxes: Optional[Dict[int, Bbox20240930]]
    pages: Optional[List[SplitRequest20240930Page]]


class SplitRequest20241227(BaseModel):
    """Describe a scheme for the splitting request on 27/12/2024."""

    text: str
    bboxes: Optional[Dict[int, Bbox20240930]]
    pages: Optional[List[SplitRequest20240930Page]]
    raw_ocr_response: Optional[Union[Dict, List]]


class SplitResponse20240930(BaseModel):
    """Describe a scheme for the splitting response on 30/09/2024."""

    class SplittingResult(BaseModel):
        """Describe a scheme for a subdocument constructed after splitting on 30/09/2024."""

        class Page(BaseModel):
            """Describe a scheme for the page as part of a subdocument on 30/09/2024."""

            id: int

        pages: List[Page]
        category: Optional[int]
        categories: List[CategoryAnnotation20240930]

    splitting_results: List[SplittingResult]
