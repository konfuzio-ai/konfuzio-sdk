from typing import List, Optional, Tuple

from pydantic import BaseModel


class ExtractRequest20240117(BaseModel):
    """Describe a scheme for the extraction request on 17/01/2024."""

    text: str
    bboxes: Optional[dict]

    class Page(BaseModel):
        """Describe a scheme for the Page class on 17/01/2024."""

        number: int
        image: Optional[bytes]
        original_size: Tuple[float, float]

    pages: Optional[List[Page]]


class ExtractResponse20240117(BaseModel):
    """Describe a scheme for the extraction response on 17/01/2024."""

    class Annotation(BaseModel):
        """Describe a scheme for the Annotation class on 17/01/2024."""

        label: int
        annotation_set: int

    annotations: List[Annotation]
