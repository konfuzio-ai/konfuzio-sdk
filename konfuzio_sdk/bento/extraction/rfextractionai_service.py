"""Run extraction service for a dockerized AI."""

import asyncio
import json
import os
import typing as t

import bentoml
from fastapi import Depends, FastAPI, HTTPException

from .schemas import ExtractRequest20241227, ExtractResponse20240117
from .utils import prepare_request, process_response

# Use relative or top module import based on whether this is run as an actual service or imported
try:
    from ..base.base_services import PicklableModelService
    from ..base.utils import add_credentials_to_project, cleanup_project_after_document_processing, handle_exceptions
except (ImportError, ValueError):
    from base.base_services import PicklableModelService
    from base.utils import add_credentials_to_project, cleanup_project_after_document_processing, handle_exceptions

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
class ExtractionService(PicklableModelService):
    model_ref = bentoml.models.get(ai_model_name)


    @bentoml.api(input_spec=ExtractRequest20241227)
    @handle_exceptions
    async def extract(self, ctx: bentoml.Context, **request: t.Any) -> ExtractResponse20240117:
        """Send a call to the Extraction AI and process the response."""
        # Ensure the model is loaded
        extraction_model = await self.get_model()

        # The rest of the method remains the same
        request = ExtractRequest20241227(**request)
        project = extraction_model.project

        # Add credentials from the request headers to the Project object, but only if the SDK version supports this.
        add_credentials_to_project(project, ctx)
        # Older SDK versions do not have the credentials attribute on Project.
        document = prepare_request(
            request=request,
            project=project,
            konfuzio_sdk_version=getattr(extraction_model, 'konfuzio_sdk_version', None),
        )
        # Run the extraction in a separate thread, otherwise the API server will block

        result = await asyncio.get_event_loop().run_in_executor(self.executor, extraction_model.extract, document)
        annotations_result = process_response(result)

        # Remove the Document and its copies from the Project to avoid memory leaks
        cleanup_project_after_document_processing(project, document)
        return annotations_result


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
