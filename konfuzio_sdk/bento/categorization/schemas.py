"""Define pydantic models for request and response from the Categorization AI."""
from typing import Dict, List, Optional, Tuple, Union

from pydantic import BaseModel

# Use relative or top module import based on whether this is run as an actual service or imported
try:
    from ..base.utils import HexBytes
except (ImportError, ValueError):
    from base.utils import HexBytes


class CategorizeRequest20240729Page(BaseModel):
    """Describe a scheme for Page class on 29/07/2024."""

    number: int
    image: Optional[HexBytes] = None
    original_size: Tuple[float, float]
    segmentation: Optional[list] = None


class CategorizeRequest20241227(BaseModel):
    """Describe a scheme for the categorization request on 27/12/2024."""

    class Bbox(BaseModel):
        """Describe a scheme for the Bbox class on 27/12/2024."""

        x0: float
        x1: float
        y0: float
        y1: float
        page_number: int
        text: Optional[str]

    text: Optional[str]
    bboxes: Optional[Dict[int, Bbox]]
    pages: Optional[List[CategorizeRequest20240729Page]]
    raw_ocr_response: Optional[Union[Dict, List]]


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

    text: Optional[str]
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
            category_name: str

        number: int
        categories: List[PredictedCategory]

    pages: List[CategorizedPage]
