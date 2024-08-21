"""Test interfaces created for containerization of Categorization AIs."""
import subprocess
import time
import unittest

import pytest
import requests

from konfuzio_sdk.bento.categorization.schemas import CategorizeRequest20240729, CategorizeResponse20240729
from konfuzio_sdk.bento.categorization.utils import convert_document_to_request, convert_response_to_categorized_pages
from konfuzio_sdk.data import Project
from konfuzio_sdk.settings_importer import is_dependency_installed
from konfuzio_sdk.trainer.document_categorization import BERT, CategorizationAI, PageTextCategorizationModel


@pytest.mark.skipif(
    not is_dependency_installed('torch'),
    reason='Required dependencies not installed.',
)
class TestCategorizationAIBento(unittest.TestCase):
    """Test that Bento-based AI works."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create a model and its Bento instance of Categorization AI."""
        cls.project = Project(id_=14392)
        cls.test_document = cls.project.get_document_by_id(5589057)
        cls.categorization_pipeline = CategorizationAI(cls.project.categories)
        cls.category_1 = cls.project.get_category_by_name(category_name='Employee and Family Medic')
        cls.category_2 = cls.project.get_category_by_name(category_name='Patient Registration Form')
        cls.categorization_pipeline.documents = cls.category_1.documents() + cls.category_2.documents()
        cls.categorization_pipeline.test_documents = cls.category_1.test_documents() + cls.category_2.test_documents()
        cls.categorization_pipeline.category_vocab = cls.categorization_pipeline.build_template_category_vocab()
        text_model = BERT(name='prajjwal1/bert-tiny')
        cls.categorization_pipeline.classifier = PageTextCategorizationModel(
            text_model=text_model,
            output_dim=len(cls.categorization_pipeline.category_vocab),
        )
        cls.categorization_pipeline.classifier.eval()
        cls.categorization_pipeline.build_preprocessing_pipeline(use_image=False)
        cls.categorization_pipeline.fit(n_epochs=5, optimizer={'name': 'Adam'})
        bento, path = cls.categorization_pipeline.save_bento()
        cls.bento_name = bento.tag.name + ':' + bento.tag.version
        cls.bento_process = subprocess.Popen(
            ['bentoml', 'serve', cls.bento_name], stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )

        time.sleep(5)
        print('served bento')
        cls.request_url = 'http://0.0.0.0:3000/categorize'

    def test_categorize(self):
        """Test that only a schema-adhering response is accepted and processed by categorize method of service."""
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
        # logging_from_subprocess(process=self.bento_process, breaking_point='status=')
        assert len(response.json()['pages']) == 2
        assert response.json()['pages'][0]['categories'][0]['category_id'] == 19827
        assert response.json()['pages'][0]['categories'][0]['confidence'] > 0.9

    def test_categorize_converted(self):
        """Test that a converting function creates request that can be accepted by categorize method of service."""
        document = self.test_document
        prepared = convert_document_to_request(document=document, schema=CategorizeRequest20240729)
        response = requests.post(url=self.request_url, json=prepared.dict())
        response_schema = CategorizeResponse20240729(pages=response.json()['pages'])
        document_with_categorized_pages = convert_response_to_categorized_pages(
            response=response_schema, document=document
        )
        assert len(document_with_categorized_pages.pages()) == 2
        assert document_with_categorized_pages.get_page_by_index(0).category_annotations[0].category.id_ == 19827
        assert document_with_categorized_pages.get_page_by_index(0).category_annotations[0].confidence > 0.9

    @classmethod
    def tearDownClass(cls) -> None:
        """Kill process."""
        cls.bento_process.kill()
