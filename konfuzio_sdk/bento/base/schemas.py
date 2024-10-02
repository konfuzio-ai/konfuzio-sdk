from pydantic import BaseModel
from typing import List, Optional, Dict, Union, Tuple
from konfuzio_sdk.data import Document, Page, Project
from utils import HexBytes

class BboxModel20240729(BaseModel):
    """Describe a scheme for the Bbox class on 29/07/2024."""

    x0: float
    x1: float
    y0: float
    y1: float
    page_number: int
    text: Optional[str]


class LabelModel20240117(BaseModel):
    """Describe a scheme for the Label class on 17/01/2024."""

    id: int
    name: str
    has_multiple_top_candidates: bool
    data_type: str
    threshold: float


class SpanModel20240117(BaseModel):
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


class AnnotationModel20240117(BaseModel):
    """Describe a scheme for the Annotation class on 17/01/2024."""

    offset_string: Optional[Union[str, List[str]]]
    translated_string: Optional[str]
    normalized: Union[str, int, None]
    label: LabelModel20240117
    annotation_set: int
    confidence: float
    span: List[SpanModel20240117]


class CategoryModel20240117(BaseModel):
    """Describe a scheme for the Category class on 29/07/2024."""

    category_id: int
    confidence: float
    category_name: str


class AnnotationSetModel20240117(BaseModel):
    """Describe a scheme for the AnnotationSet class on 17/01/2024."""

    label_set_id: int
    annotations: List[AnnotationModel20240117]


class PageModel20240117(BaseModel):
    """Describe a scheme for the Page class on 17/01/2024."""

    number: int
    image: Optional[HexBytes] = None
    original_size: Tuple[float, float]
    segmentation: Optional[list] = None


class CategorizedPageModel20240729(BaseModel):
    """Describe a scheme for the CategorizedPage class after categorization on 29/07/2024."""

    number: int
    categories: List[CategoryModel20240117]


class DocumentModel20240117(BaseModel):
    """Describe a scheme for a Document on 17/01/2024."""

    text: str
    bboxes: Optional[Dict[int, BboxModel20240729]]
    pages: Optional[List[PageModel20240117]]

    def prepare_request(self, request: BaseModel, project: Project, konfuzio_sdk_version: Optional[str] = None) -> Document:
        """
        Receive a request and prepare it for the extraction runner.

        :param request: Unprocessed request.
        :param project: A Project instance.
        :param konfuzio_sdk_version: The version of the Konfuzio SDK used by the embedded AI model. Used to apply backwards
            compatibility changes for older SDK versions.
        :returns: An instance of a Document class.
        """
        # Extraction AIs include only one Category per Project.
        category = project.categories[0]
        # Calculate next available ID based on current Project documents to avoid conflicts.
        document_id = max((doc.id_ for doc in project._documents if doc.id_), default=0) + 1

        bboxes = {}
        for bbox_id, bbox in request.bboxes.items():
            bboxes[str(bbox_id)] = {
                'x0': bbox.x0,
                'x1': bbox.x1,
                'y0': bbox.y0,
                'y1': bbox.y1,
                'page_number': bbox.page_number,
                'text': bbox.text,
            }
            # Backwards compatibility with Konfuzio SDK versions < 0.3.
            # In newer versions, the top and bottom values are not needed.
            if konfuzio_sdk_version and konfuzio_sdk_version < '0.3':
                page = next(page for page in request.pages if page.number == bbox.page_number)
                bboxes[str(bbox_id)]['top'] = round(page.original_size[1] - bbox.y0, 4)
                bboxes[str(bbox_id)]['bottom'] = round(page.original_size[1] - bbox.y1, 4)
        document = Document(
            id_=document_id,
            text=request.text,
            bbox=bboxes,
            project=project,
            category=category,
        )
        for page in request.pages:
            p = Page(id_=page.number, document=document, number=page.number, original_size=page.original_size)
            if page.segmentation:
                p._segmentation = page.segmentation
            if page.image:
                p.image_bytes = page.image

        return document

    @classmethod
    def convert_document_to_request(cls, document: Document) -> BaseModel:
        """
        Receive a Document and convert it into a request in accordance to a passed schema.

        :param document: A Document to be converted.
        :param schema: A schema to which the request should adhere.
        :returns: A Document converted in accordance with the schema.
        """
        pages = [
            CategorizedPageModel20240729(
                number=page.number,
                image=page.image_bytes,
                original_size=page._original_size,
                segmentation=page._segmentation,
            )
            for page in document.pages()
        ]
        converted = cls(
            text=document.text,
            bboxes={
                k: {
                    'x0': v.x0,
                    'x1': v.x1,
                    'y0': v.y0,
                    'y1': v.y1,
                    'page_number': v.page.number,
                    'top': v.top,
                    'bottom': v.bottom,
                    'text': document.text[k],
                }
                for k, v in document.bboxes.items()
            },
            pages=pages,
        )

        return converted
