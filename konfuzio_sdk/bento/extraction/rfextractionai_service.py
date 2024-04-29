"""Run extraction service for a dockerized AI."""
import json
import logging
import os

import bentoml
from fastapi import FastAPI, HTTPException

from .schemas import ExtractRequest20240117, ExtractResponse20240117
from .utils import prepare_request, process_response

# load ai model name from AI_MODEL_NAME file in parent directory
ai_model_name_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'AI_MODEL_NAME')
ai_model_name = open(ai_model_name_file).read().strip()

extraction_runner = bentoml.picklable_model.get(f'{ai_model_name}').to_runner(embedded=True)
app = FastAPI()
svc = bentoml.Service('extraction_svc', runners=[extraction_runner])

logger = logging.getLogger(__name__)


@svc.api(
    input=bentoml.io.JSON(pydantic_model=ExtractRequest20240117),
    output=bentoml.io.JSON(pydantic_model=ExtractResponse20240117),
)
async def extract(request: ExtractRequest20240117) -> ExtractResponse20240117:
    """Send an asynchronous call to the Extraction AI and process the response."""
    # The runner references the runnable instance, which in turn references the original pickled model.
    # From the pickled model we can retrieve the original Project instance that was used to train the model, and use it
    # to prepare the request.
    project = extraction_runner._runner_handle._runnable.model.project
    document = prepare_request(request=request, project=project)
    result = await extraction_runner.extract.async_run(document)
    annotations_result = process_response(result)
    return annotations_result


@app.get('/project-metadata')
async def project_metadata():
    """Get the project metadata."""
    project_metadata_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'categories_and_labels_data.json5')
    try:
        with open(project_metadata_file) as f:
            project_metadata = json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail='Project metadata not found')
    return project_metadata


svc.mount_asgi_app(app)
app = svc.asgi_app
