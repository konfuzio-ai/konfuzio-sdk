"""Run a service for a containerized instance of Splitting AI."""
import asyncio
import json
import os
from typing import Any

import bentoml
from fastapi import Depends, FastAPI, HTTPException

from .schemas import SplitRequest20241227, SplitResponse20240930
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
class FileSplittingService(PicklableModelService):
    model_ref = bentoml.models.get(ai_model_name)

    @bentoml.api(input_spec=SplitRequest20241227)
    @handle_exceptions
    async def split(self, ctx: bentoml.Context, **request: Any) -> SplitResponse20240930:
        """Send a call to the Splitting AI and process the response."""
        # Ensure the model is loaded
        splitting_model = await self.get_model()
        request = SplitRequest20241227(**request)
        project = splitting_model.model.project
        add_credentials_to_project(project, ctx)
        document = prepare_request(request=request, project=project)
        result = await asyncio.get_event_loop().run_in_executor(self.executor, splitting_model.propose_split_documents, document)
        split_result = process_response(result)
        cleanup_project_after_document_processing(project, document)
        return split_result


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
