"""Test interfaces created for containerization of File Splitting AIs."""
import subprocess
import time
import unittest

import pytest

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

        cls.project = Project(id_=14392)
        file_splitting_model = TextualFileSplittingModel(categories=cls.project.categories)
        file_splitting_model.documents = file_splitting_model.categories[0].documents()
        file_splitting_model.test_documents = file_splitting_model.categories[0].test_documents()
        cls.test_document = file_splitting_model.test_documents[-1]
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
        pred = self.splitting_ai.propose_split_documents(self.test_document)
        assert len(pred) == 1
        for page in pred[0].pages():
            if page.number == 1:
                assert page.is_first_page
                assert page.is_first_page_confidence > 0.5
            else:
                assert not page.is_first_page
                assert page.is_first_page_confidence
