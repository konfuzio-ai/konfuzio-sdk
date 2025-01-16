"""Define pydantic models for request and response from the Extraction AI."""
from typing import Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, RootModel

# Use relative or top module import based on whether this is run as an actual service or imported
try:
    from ..base.utils import HexBytes
except (ImportError, ValueError):
    from base.utils import HexBytes


class ExtractRequest20240117Page(BaseModel):
    """Describe a scheme for the Page class on 17/01/2024."""

    number: int
    image: Optional[HexBytes] = None
    original_size: Tuple[float, float]
    segmentation: Optional[list] = None


class ExtractRequest20241227(BaseModel):
    """Describe a scheme for the extraction request on 27/12/2024."""

    class Bbox(BaseModel):
        """Describe a scheme for the Bbox class on 27/12/2024."""

        x0: float
        x1: float
        y0: float
        y1: float
        page_number: int
        text: Optional[str]
        line_index: int

    text: Optional[str]
    bboxes: Optional[Dict[int, Bbox]]
    pages: Optional[List[ExtractRequest20240117Page]]
    raw_ocr_response: Optional[Union[Dict, List]]


class ExtractRequest20241223(BaseModel):
    """Describe a scheme for the extraction request on 23/12/2024."""

    class BboxExtractRequest20241223(BaseModel):
        """Describe a scheme for the Bbox class on 23/12/2024."""

        x0: float
        x1: float
        y0: float
        y1: float
        page_number: int
        text: Optional[str]
        line_index: int

    text: Optional[str]
    bboxes: Optional[Dict[int, BboxExtractRequest20241223]]
    pages: Optional[List[ExtractRequest20240117Page]]


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

    text: Optional[str]
    bboxes: Optional[Dict[int, Bbox]]
    pages: Optional[List[ExtractRequest20240117Page]]


class ExtractRequest20241223(BaseModel):
    """Describe a scheme for the extraction request on 23/12/2024."""

    class BboxExtractRequest20241223(BaseModel):
        """Describe a scheme for the Bbox class on 23/12/2024."""

        x0: float
        x1: float
        y0: float
        y1: float
        page_number: int
        text: Optional[str]
        line_index: int

    text: Optional[str]
    bboxes: Optional[Dict[int, BboxExtractRequest20241223]]
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


class ExtractResponseForLegacyTrainer20240912(RootModel):
    root: Dict
