"""Define pydantic models for request and response from the Splitting AI."""
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel

# Use relative or top module import based on whether this is run as an actual service or imported
try:
    from ..base.utils import HexBytes
except (ImportError, ValueError):
    from base.utils import HexBytes


class SplitRequest20240930Page(BaseModel):
    """Describe a scheme for Page class on 30/09/2024."""

    number: int
    image: Optional[HexBytes] = None
    original_size: Tuple[float, float]
    segmentation: Optional[list] = None


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


class SplitResponse20240930(BaseModel):
    """Describe a scheme for the splitting response on 30/09/2024."""

    class Subdocument(BaseModel):
        """Describe a scheme for a subdocument constructed after splitting on 30/09/2024."""

        class PageId(BaseModel):
            """Describe a scheme for Page ID object on 30/09/2024."""

            page_id: int

        class CategoryAnnotation(BaseModel):
            """Describe a scheme for a CategoryAnnotation on 30/09/2024."""

            category_id: int
            confidence: float
            category_name: str

        page_ids: List[PageId]
        category: int
        categories: List[CategoryAnnotation]

    subdocuments: List[Subdocument]
