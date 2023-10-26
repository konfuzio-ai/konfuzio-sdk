import bentoml
import jsonpickle
from pydantic import BaseModel
from typing import Optional, List, Tuple
from konfuzio_sdk.data import Document, Project, Category, Page

extraction_runner = bentoml.picklable_model.get("extraction:latest").to_runner()

svc = bentoml.Service("extraction_svc", runners=[extraction_runner])


class ExtractRequest(BaseModel):
    text: str
    bboxes: Optional[dict]

    class Page(BaseModel):
        number: int
        image: Optional[bytes]
        original_size: Tuple[float, float]

    pages: Optional[List[Page]]


@svc.api(input=bentoml.io.JSON(pydantic_model=ExtractRequest), output=bentoml.io.JSON())
async def extract(request: ExtractRequest) -> dict:
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
    return result.annotation_sets()
