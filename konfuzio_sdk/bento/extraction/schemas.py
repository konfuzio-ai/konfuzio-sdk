"""Define pydantic models for request and response from the Extraction AI."""
from typing import Dict, List, Optional, Tuple, Union

from pydantic import BaseModel


class ExtractRequest20240117Page(BaseModel):
    """Describe a scheme for the Page class on 17/01/2024."""

    number: int
    image: Optional[bytes] = None
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


class OMRRequest20240215(BaseModel):
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
        annotation_id: str
        bbox: Box

    class Page(BaseModel):
        """Describe a scheme for the Page info needed for OMR on 15/02/2024."""

        page_id: int
        width: int
        height: int
        image: str

    pages: List[Page]
    annotations: List[Annotation]


class OMRResponse20240215(BaseModel):
    """Describe a scheme for the Optical Mark Recognition (OMR) response on 15/02/2024."""

    class MetaData(BaseModel):
        """Describe a scheme for OMR meta data on 15/02/2024."""

        class Meta(BaseModel):
            """Describe a scheme for OMR meta data instance on 15/02/2024."""

            class OMR(BaseModel):
                """Describe a scheme for OMR checkbox data on 15/02/2024."""

                class Box(BaseModel):
                    """Describe a scheme for BBox on 15/02/2024."""

                    x0: int
                    x1: int
                    y0: int
                    y1: int

                is_checked: Union[bool, None]  # None if undetermined or invalid
                checkbox_bbox: Box
                confidence: float

            omr: OMR

        annotation_id: Meta

    metadata: MetaData
