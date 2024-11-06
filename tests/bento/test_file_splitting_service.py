"""Test interfaces created for containerization of File Splitting AIs."""
import subprocess
import time
import unittest

import pytest
import requests

from konfuzio_sdk.data import Project
from konfuzio_sdk.settings_importer import is_dependency_installed
from konfuzio_sdk.trainer.file_splitting import SplittingAI, TextualFileSplittingModel


@pytest.mark.skipif(
    not is_dependency_installed('torch'),
    reason='Required dependencies not installed.',
)
class TestFileSplittingAIBento(unittest.TestCase):
    """Test that Bento-based AI works."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create a model and its Bento instance of File Splitting AI."""

        cls.project = Project(id_=14392, update=True)
        file_splitting_model = TextualFileSplittingModel(categories=cls.project.categories)
        file_splitting_model.documents = file_splitting_model.categories[0].documents()
        file_splitting_model.test_documents = file_splitting_model.categories[0].test_documents()
        cls.test_document = cls.project.get_document_by_id(6357341)
        file_splitting_model.fit(epochs=3, eval_batch_size=1, train_batch_size=1)
        cls.splitting_ai = SplittingAI(model=file_splitting_model)
        bento, path = cls.splitting_ai.save_bento()
        cls.bento_name = bento.tag.name + ':' + bento.tag.version
        cls.bento_process = subprocess.Popen(
            ['bentoml', 'serve', cls.bento_name], stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )

        time.sleep(5)
        print('served bento')
        cls.request_url = 'http://0.0.0.0:3000/split'

    def test_run_splitting_ai_prediction(self):
        """Test Splitting AI integration with the Textual File Splitting Model in Bento service."""
        data = {
            'text': self.test_document.text,
            'bboxes': {
                str(bbox_id): {
                    'x0': bbox['x0'],
                    'x1': bbox['x1'],
                    'y0': bbox['y0'],
                    'y1': bbox['y1'],
                    'page_number': bbox['page_number'] if 'page_number' in bbox else bbox['page_index'] + 1,
                    'text': bbox['text'],
                }
                for bbox_id, bbox in self.test_document.get_bbox().items()
            },
            'pages': [
                {
                    'number': page.number,
                    'original_size': page._original_size,
                    'image': page.image_bytes,
                    'segmentation': page._segmentation,
                }
                for page in self.test_document.pages()
            ],
        }
        response = requests.post(url=self.request_url, json=data)
        # print(response.json())
        assert response.status_code == 200
        assert response.json()['splitting_results']
