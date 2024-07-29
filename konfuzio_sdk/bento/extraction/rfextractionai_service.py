"""Run extraction service for a dockerized AI."""
import json
import os
import typing as t

import bentoml
from fastapi import Depends, FastAPI, HTTPException

from .schemas import ExtractRequest20240117, ExtractResponse20240117
from .utils import handle_exceptions, prepare_request, process_response

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

    @bentoml.api(input_spec=ExtractRequest20240117)
    @handle_exceptions
    async def extract(self, ctx: bentoml.Context, **request: t.Any) -> ExtractResponse20240117:
        """Send a call to the Extraction AI and process the response."""
        # Even though the request is already validated against the pydantic schema, we need to get it back as an
        # instance of the pydantic model to be able to pass it to the prepare_request function.
        request = ExtractRequest20240117(**request)
        project = self.extraction_model.project
        # Add credentials from the request headers to the Project object, but only if the SDK version supports this.
        # Older SDK versions do not have the credentials attribute on Project.
        if hasattr(project, 'credentials'):
            for key, value in ctx.request.headers.items():
                if key.startswith('env_'):
                    key = key.replace('env_', '', 1)
                    project.credentials[key.upper()] = value
        document = prepare_request(request=request, project=project)
        result = self.extraction_model.extract(document)
        annotations_result = process_response(result)
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
