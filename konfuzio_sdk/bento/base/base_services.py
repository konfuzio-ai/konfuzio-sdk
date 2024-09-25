"""Base services for Bento-based models."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import bentoml


class PicklableModelService:
    """A general class for any AI models that will be packaged into Bento archive and served as a containerized version."""

    def __init__(self):
        """Initialize the extraction service."""
        print(f'Initializing service for model {self.model_ref}')
        self.model = None
        self.executor = ThreadPoolExecutor()
        self.model_load_task = asyncio.create_task(self.load_model())

    async def load_model(self):
        """Asynchronously load the extraction model into memory using the executor."""
        print(f'Loading model {self.model_ref}')
        loop = asyncio.get_event_loop()
        self.model = await loop.run_in_executor(self.executor, bentoml.picklable_model.load_model, self.model_ref)
        print(f'Model {self.model_ref} loaded')

    async def get_model(self):
        """Ensure the model is loaded before returning it."""
        await self.model_load_task
        if self.model is None:
            raise RuntimeError('Model failed to load')
        return self.model
