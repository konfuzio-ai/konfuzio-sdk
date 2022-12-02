"""Test SplittingAI and the model's training, saving and prediction."""
import pathlib
import unittest

from copy import deepcopy

from konfuzio_sdk.samples import LocalTextProject
from konfuzio_sdk.tokenizer.regex import ConnectedTextTokenizer
from konfuzio_sdk.trainer.file_splitting import ContextAwareFileSplittingModel, SplittingAI

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

    def test_predict_context_aware_splitting_model(self):
        """Test correct first Page prediction."""
        test_document = self.file_splitting_model.tokenizer.tokenize(
            deepcopy(self.project.get_category_by_id(3).test_documents()[0])
        )
        for page in test_document.pages():
            intersections = []
            for category in self.file_splitting_model.categories:
                intersection = {span.offset_string for span in page.spans()}.intersection(
                    self.file_splitting_model.first_page_spans[category.id_]
                )
                if intersection:
                    intersections.append(intersection)
            if category.id_ == 3:
                if page.number == 1:
                    assert intersections == [{'I like bread.'}]
                if page.number in (2, 4):
                    assert intersections == []
                if page.number in (3, 5):
                    assert intersections == [{'Morning,'}]

    def test_splitting_ai_predict(self):
        """Test SplittingAI's predict method."""
        splitting_ai = SplittingAI(self.project)
        test_document = self.project.get_category_by_id(3).test_documents()[0]
        pred = splitting_ai.propose_split_documents(test_document)
        assert len(pred) == 3
        pathlib.Path(self.project.model_folder + '/first_page_spans.cloudpickle').unlink()
