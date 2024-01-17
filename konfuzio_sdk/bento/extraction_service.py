import bentoml
from pydantic import BaseModel
from typing import Optional, List, Tuple
from konfuzio_sdk.data import Document, Project, Category, Page

extraction_runner = bentoml.picklable_model.get("extraction:latest").to_runner()

svc = bentoml.Service("extraction_svc", runners=[extraction_runner])


class ExtractRequest20240117(BaseModel):
    text: str
    bboxes: Optional[dict]

    class Page(BaseModel):
        number: int
        image: Optional[bytes]
        original_size: Tuple[float, float]

    pages: Optional[List[Page]]


class ExtractResponse20240117(BaseModel):
    class Annotation(BaseModel):
        label: int
        annotation_set: int

    annotations: List[Annotation]


@svc.api(
    input=bentoml.io.JSON(pydantic_model=ExtractRequest20240117),
    output=bentoml.io.JSON(pydantic_model=ExtractResponse20240117),
)
async def extract(request: ExtractRequest20240117) -> ExtractResponse20240117:
    project = Project(id_=None)
    project.set_offline()
    category = Category(project=project)
    document = Document(
        text=request.text,
        bbox=request.bboxes,
        project=project,
        category=category,
    )
    for page in request.pages:
        Page(id_=page.number, document=document, number=page.number, original_size=page.original_size)

    result = await extraction_runner.extract.async_run(document)
    # return result.annotation_sets()
    res = []
    for annotation_set in result.annotation_sets():
        for annotation in annotation_set.annotations():
            res.append({"label": annotation.label.id, "annotation_set": annotation_set.id})
    return res
