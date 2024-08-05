"""Run a service for a containerized instance of Categorization AI."""
import os
from typing import Any

import bentoml
from fastapi import FastAPI

from .schemas import CategorizeRequest20240729, CategorizeResponse20240729
from .utils import prepare_request, process_response

# load ai model name from AI_MODEL_NAME file in parent directory
ai_model_name_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'AI_MODEL_NAME')
ai_model_name = open(ai_model_name_file).read().strip()

app = FastAPI()


@bentoml.service
@bentoml.mount_asgi_app(app, path='/v1')
class CategorizationService:
    model_ref = bentoml.models.get(ai_model_name)

    def __init__(self):
        """Load the categorization model into memory."""
        self.categorization_model = bentoml.picklable_model.load_model(self.model_ref)

    @bentoml.api(input_spec=CategorizeRequest20240729)
    async def categorize(self, **request: Any) -> CategorizeResponse20240729:
        """Send an call to the Categorization AI and process the response."""
        request = CategorizeRequest20240729(**request)
        project = self.categorization_model.project
        document = prepare_request(request=request, project=project)
        result = self.categorization_model.extract(document)
        categories_result = process_response(result)
        return categories_result
