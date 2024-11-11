"""Run extraction service for a dockerized AI."""

import asyncio
import json
import os
import typing as t
from concurrent.futures import ThreadPoolExecutor

import bentoml
from fastapi import Depends, FastAPI, HTTPException

from .schemas import ExtractRequest20240117, ExtractResponseForLegacyTrainer20240912
from .utils import process_response

try:
    from ..base.utils import handle_exceptions
except (ImportError, ValueError):
    from base.utils import handle_exceptions

from konfuzio_sdk.data import Category, Project

# load ai model name from AI_MODEL_NAME file in parent directory
ai_model_name_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'AI_MODEL_NAME')
ai_model_name = open(ai_model_name_file).read().strip()

app = FastAPI()


@bentoml.service(
    traffic={
        'timeout': int(os.environ.get('BENTO_SERVICE_TIMEOUT', '3600')),
        # Don't process more than 2 documents at a time. Will respond with 429 if more come.
        # Clients should implement a retry strategy for 429.
        # Servers should implement a scaling strategy and start multiple services when high load is present.
        'max_concurrency': int(os.environ.get('BENTO_SERVICE_MAX_CONCURRENCY', '2')),
    }
)
@bentoml.mount_asgi_app(app, path='/v1')
class ExtractionService:
    model_ref = bentoml.models.get(ai_model_name)

    def __init__(self):
        """Initialize the extraction service."""
        print(f'Initializing service for model {self.model_ref}')
        self.extraction_model = None
        self.executor = ThreadPoolExecutor()
        self.model_load_task = asyncio.create_task(self.load_model())

    async def load_model(self):
        """Asynchronously load the extraction model into memory using the executor."""
        print(f'Loading model {self.model_ref}')
        loop = asyncio.get_event_loop()
        self.extraction_model = await loop.run_in_executor(
            self.executor, bentoml.picklable_model.load_model, self.model_ref
        )
        print(f'Model {self.model_ref} loaded')

    async def get_model(self):
        """Ensure the model is loaded before returning it."""
        await self.model_load_task
        if self.extraction_model is None:
            raise RuntimeError('Model failed to load')
        return self.extraction_model

    @bentoml.api(input_spec=ExtractRequest20240117)
    @handle_exceptions
    async def extract(self, ctx: bentoml.Context, **request: t.Any) -> ExtractResponseForLegacyTrainer20240912:
        """Send a call to the Extraction AI and process the response."""

        # Ensure the model is loaded
        extraction_model = await self.get_model()

        # Even though the request is already validated against the pydantic schema, we need to get it back as an
        # instance of the pydantic model to be able to pass it to the prepare_request function.
        request = ExtractRequest20240117(**request)
        project = Project(None, strict_data_validation=False, credentials={})
        project.set_offline()
        Category(project=project)

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

        result = await asyncio.get_event_loop().run_in_executor(
            self.executor, extraction_model.extract, request.text, bboxes, pages
        )
        json_result = process_response(result, schema=ExtractResponseForLegacyTrainer20240912)

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
