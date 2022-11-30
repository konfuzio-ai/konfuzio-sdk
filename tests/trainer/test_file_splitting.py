"""Test SplittingAI and the model's training, saving and prediction."""
import pathlib
import unittest

from konfuzio_sdk.data import Project
from konfuzio_sdk.samples import LocalTextProject
from konfuzio_sdk.tokenizer.regex import ConnectedTextTokenizer
from konfuzio_sdk.trainer.file_splitting import ContextAwareFileSplittingModel, SplittingAI
from tests.variables import TEST_PROJECT_ID

TEST_WITH_FULL_DATASET = True


class TestFileSplittingModel(unittest.TestCase):
    """Test filesplitting model."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the tested class."""
        cls.project = LocalTextProject()
        cls.file_splitting_model = ContextAwareFileSplittingModel()
        cls.file_splitting_model.categories = cls.project.categories
        cls.file_splitting_model.train_data = [
            document for category in cls.file_splitting_model.categories for document in category.documents()
        ]
        cls.file_splitting_model.test_data = [
            document for category in cls.file_splitting_model.categories for document in category.test_documents()
        ]
        cls.file_splitting_model.tokenizer = ConnectedTextTokenizer()
        cls.file_splitting_model.first_page_spans = None

    def test_fit_context_aware_splitting_model(self):
        """Test pseudotraining of the context-aware splitting model."""
        self.file_splitting_model.first_page_spans = self.file_splitting_model.fit()
        non_first_page_spans = {}
        for category in self.file_splitting_model.categories:
            cur_non_first_page_spans = []
            for doc in category.documents():
                for page in doc.pages():
                    if page.number > 1:
                        cur_non_first_page_spans.append({span.offset_string for span in page.spans()})
            if not cur_non_first_page_spans:
                cur_non_first_page_spans.append(set())
            true_non_first_page_spans = set.intersection(*cur_non_first_page_spans)
            non_first_page_spans[category.id_] = true_non_first_page_spans
        for category in self.file_splitting_model.categories:
            for span in self.file_splitting_model.first_page_spans[category.id_]:
                assert span not in non_first_page_spans[category.id_]

    def test_save_context_aware_splitting_model(self):
        """Test saving of the first-page Spans."""
        self.file_splitting_model.save(self.project.model_folder)
        assert pathlib.Path(self.project.model_folder + '/first_page_spans.cloudpickle').exists()
        pathlib.Path(self.project.model_folder + '/first_page_spans.cloudpickle').unlink()

    def test_predict_context_aware_splitting_model(self):
        """Test correct first Page prediction."""
        for document in self.file_splitting_model.test_data:
            for page in document.pages():
                pred = self.file_splitting_model.predict(page)
                assert hasattr(pred, 'is_first_page')


def test_split_document_splitting_ai():
    """Test the SplittingAI."""
    project = Project(id_=TEST_PROJECT_ID)
    project_receipts = Project(id_=1644)
    test_document = project.get_document_by_id(399140)
    model = ContextAwareFileSplittingModel()
    model.categories = project.categories + [project_receipts.get_category_by_id(5196)]
    if TEST_WITH_FULL_DATASET:
        model.train_data = [document for category in model.categories for document in category.documents()]
    else:
        model.train_data = [document for category in model.categories for document in category.documents()[:10]]
    model.test_data = [document for category in model.categories for document in category.test_documents()]
    model.tokenizer = ConnectedTextTokenizer()
    model.first_page_spans = model.fit()
    model.save(project.model_folder)
    splitting_ai = SplittingAI(project_id=TEST_PROJECT_ID)
    suggested_splits = splitting_ai.propose_split_documents(test_document)
    assert len(suggested_splits) == 5
    assert [len(doc.pages()) for doc in suggested_splits] == [2, 1, 1, 1, 1]
    pathlib.Path(project.model_folder + '/first_page_spans.cloudpickle').unlink()
