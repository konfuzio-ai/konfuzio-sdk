"""Run extraction service for a dockerized AI."""

from typing import List, Optional, Tuple

import bentoml
from pydantic import BaseModel

from konfuzio_sdk.data import Document, Page, Project

extraction_runner = bentoml.picklable_model.get('rfextractionai:latest').to_runner(embedded=True)

svc = bentoml.Service('extraction_svc', runners=[extraction_runner])


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


@svc.api(
    input=bentoml.io.JSON(pydantic_model=ExtractRequest20240117),
    output=bentoml.io.JSON(pydantic_model=ExtractResponse20240117),
)
async def extract(request: ExtractRequest20240117) -> ExtractResponse20240117:
    """Send an asynchronous call to the Extraction AI and process the response."""
    project = Project(id_=None)
    project.set_offline()
    category = project.no_category
    document = Document(
        text=request.text,
        bbox=request.bboxes,
        project=project,
        category=category,
    )
    for page in request.pages:
        Page(document=document, number=page.number, original_size=page.original_size)

    result = await extraction_runner.extract.async_run(document)

    annotations_result = []
    for annotation_set in result.annotation_sets():
        current_annotation_set = {'label_set_id': annotation_set.label_set.id_, 'annotations': []}
        for annotation in annotation_set.annotations():
            current_annotation_set['annotations'].append(
                {
                    'id': annotation.id_,
                    'document': annotation.document.id_,
                    'offset_string': annotation.offset_string,
                    'offset_string_original': annotation.offset_string_original,
                    'translated_string': annotation.translated_string,
                    'normalized': annotation.normalized,
                    'label': {
                        'id': annotation.label.id_,
                        'name': annotation.label.name,
                        'has_multiple_top_candidates': annotation.label.has_multiple_top_candidates,
                        'data_type': annotation.label.data_type,
                        'threshold': annotation.label.threshold,
                    },
                    'confidence': annotation.confidence,
                    'is_correct': annotation.is_correct,
                    'revised': annotation.is_revised,
                    'origin': annotation.orgign,
                    'created_by': annotation.created_by,
                    'revised_by': annotation.revised_by,
                    'span': annotation.spans,
                    'selection_bbox': annotation.selection_bbox,
                    'custom_offset_string': annotation.custom_offset_string,
                }
            )
        annotations_result.append(current_annotation_set)
    return annotations_result
