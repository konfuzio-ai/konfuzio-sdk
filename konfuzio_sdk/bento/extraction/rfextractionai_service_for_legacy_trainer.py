"""Run extraction service for a dockerized AI."""

import asyncio
import json
import os
import typing as t
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional, Tuple, Union
import bentoml
from fastapi import Depends, FastAPI, HTTPException

from .schemas import ExtractRequest20240117, ExtractResponse20240117
from .utils import handle_exceptions, prepare_request, process_response
from konfuzio_sdk.data import Project, Category

# load ai model name from AI_MODEL_NAME file in parent directory
ai_model_name_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'AI_MODEL_NAME')
ai_model_name = open(ai_model_name_file).read().strip()

app = FastAPI()


@bentoml.service
@bentoml.mount_asgi_app(app, path='/v1')
class ExtractionService:
    model_ref = bentoml.models.get(ai_model_name)

    def __init__(self):
        """Load the extraction model into memory."""
        self.extraction_model = bentoml.picklable_model.load_model(self.model_ref)
        self.executor = ThreadPoolExecutor()

    @bentoml.api(input_spec=ExtractRequest20240117)
    @handle_exceptions
    async def extract(self, ctx: bentoml.Context, **request: t.Any) -> Dict:
        """Send a call to the Extraction AI and process the response."""
        # Even though the request is already validated against the pydantic schema, we need to get it back as an
        # instance of the pydantic model to be able to pass it to the prepare_request function.
        request = ExtractRequest20240117(**request)
        project = Project(None, strict_data_validation=False, credentials={})
        project.set_offline()
        Category(project=project)

        # Add credentials from the request headers to the Project object, but only if the SDK version supports this.
        # Older SDK versions do not have the credentials attribute on Project.
        if hasattr(project, 'credentials'):
            for key, value in ctx.request.headers.items():
                if key.startswith('env_'):
                    key = key.replace('env_', '', 1)
                    project.credentials[key.upper()] = value
        document = prepare_request(
            request=request,
            project=project,
            konfuzio_sdk_version=getattr(self.extraction_model, 'konfuzio_sdk_version', None),
        )

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
            page = next(page for page in request.pages if page.number == bbox.page_number)
            bboxes[str(bbox_id)]['top'] = round(page.original_size[1] - bbox.y0, 4)
            bboxes[str(bbox_id)]['bottom'] = round(page.original_size[1] - bbox.y1, 4)

        pages = []
        for _page in request.pages:
            pages.append(dict(_page))

        result = await asyncio.get_event_loop().run_in_executor(self.executor, self.extraction_model.extract, request.text, bboxes, pages)

        import json
        class JSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if hasattr(obj, 'to_json'):
                    return obj.to_json()
                return json.JSONEncoder.default(self, obj)

        json_result = json.loads(json.dumps(result, cls=JSONEncoder))
        project._documents = [d for d in project._documents if d.id_ != document.id_ and d.copy_of_id != document.id_]
        return json_result


@app.get('/project-metadata')
async def project_metadata(service=Depends(bentoml.get_current_service)):
    """Return the embedded JSON data about the project."""
    project_metadata_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'categories_and_labels_data.json5')
    try:
        with open(project_metadata_file) as f:
            project_metadata = json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail='Project metadata not found')
    return project_metadata
