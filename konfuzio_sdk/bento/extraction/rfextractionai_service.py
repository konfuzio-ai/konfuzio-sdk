"""Run extraction service for a dockerized AI."""
import logging

import bentoml

from .schemas import ExtractRequest20240117, ExtractResponse20240117
from .utils import prepare_request, process_response

extraction_runner = bentoml.picklable_model.get('rfextractionai:latest').to_runner(embedded=True)

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
