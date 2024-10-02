"""Define pydantic models for request and response from the Categorization AI."""
from typing import List, Optional
from konfuzio_sdk.data import Document, CategoryAnnotation

from pydantic import BaseModel


# Use relative or top module import based on whether this is run as an actual service or imported
try:
    from ..base.utils import HexBytes
    from ..base.schemas import BboxModel20240729, CategoryModel20240117, PageModel20240117, CategorizedPageModel20240729, DocumentModel20240117
except (ImportError, ValueError):
    from base.utils import HexBytes
    from base.schemas import BboxModel20240729, CategoryModel20240117, PageModel20240117, CategorizedPageModel20240729, DocumentModel20240117


class CategorizeRequest20240729(DocumentModel20240117):
    """Describe a scheme for the categorization request on 29/07/2024."""

    pass

class CategorizeResponse20240729(BaseModel):
    """Describe a scheme for the categorization response on 29/07/2024."""

    pages: List[CategorizedPageModel20240729]

    @classmethod
    def process_response(cls, result) -> BaseModel:
        """
        Process a raw response from the runner to contain only selected fields.

        :param result: A raw response to be processed.
        :param schema: A schema of the response.
        :returns: A list of dictionaries with Pages and their Categories.
        """
        pages_result = []
        for page in result.pages():
            current_page = {'number': page.number, 'original_size': page._original_size, 'categories': []}
            for category_annotation in page.category_annotations:
                current_page['categories'].append(
                    CategorizedPageModel20240729.PredictedCategory(
                        category_id=category_annotation.category.id_,
                        confidence=category_annotation.confidence,
                        category_name=category_annotation.category.name,
                    )
                )
            pages_result.append(
                CategorizedPageModel20240729(
                    number=current_page['number'],
                    categories=current_page['categories'],
                )
            )
        return cls(pages=pages_result)

    @classmethod
    def convert_response_to_categorized_pages(
            cls, response: BaseModel, document: Document, mappings: Optional[dict] = None
    ) -> Document:
        """
        Receive a CategorizeResponse and convert it into a list of categorized Pages to be added to the Document.

        :param response: A CategorizeResponse to be converted.
        :param document: A Document to which the categorized Pages should be added.
        :param mappings: A dict with "label_sets" keys containing mappings from old to new IDs (categories are a subset
            of label sets for mapping purposes). Original IDs are used if no mapping is provided or if the mapping is not
            found.
        :returns: The original Document with added categorized Pages.
        """
        if mappings is None:
            mappings = {}

        # Mappings might be from JSON, so we need to convert keys to integers.
        label_set_mappings = {int(k): v for k, v in mappings.get('label_sets', {}).items()}

        for page in response.pages:
            page_to_update = document.get_page_by_index(page.number - 1)
            for category in page.categories:
                category_id = label_set_mappings.get(category.category_id, category.category_id)
                confidence = category.confidence
                page_to_update.add_category_annotation(
                    category_annotation=CategoryAnnotation(
                        category=document.project.get_category_by_id(category_id), confidence=confidence
                    )
                )
        return document
