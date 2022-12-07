"""Test SplittingAI and the model's training, saving and prediction."""
import glob
import unittest

from copy import deepcopy

from konfuzio_sdk.evaluate import FileSplittingEvaluation
from konfuzio_sdk.samples import LocalTextProject
from konfuzio_sdk.tokenizer.regex import ConnectedTextTokenizer
from konfuzio_sdk.trainer.file_splitting import ContextAwareFileSplittingModel, SplittingAI


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
        cls.file_splitting_model.test_data = cls.project.get_category_by_id(3).test_documents()
        cls.file_splitting_model.tokenizer = ConnectedTextTokenizer()
        cls.file_splitting_model.first_page_spans = None
        cls.test_document = cls.project.get_category_by_id(3).test_documents()[0]

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
        assert glob.glob(self.project.model_folder + '/*.pkl')

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
                self.file_splitting_model.predict(page)
                if page.number == 1:
                    assert intersections == [{'I like bread.'}]
                    assert page.is_first_page
                if page.number in (2, 4):
                    assert intersections == []
                if page.number in (3, 5):
                    assert intersections == [{'Morning,'}]
                    assert page.is_first_page

    def test_splitting_ai_predict(self):
        """Test SplittingAI's Document-splitting method."""
        splitting_ai = SplittingAI(self.file_splitting_model)
        pred = splitting_ai.propose_split_documents(self.test_document)
        assert len(pred) == 3

    def test_suggest_first_pages(self):
        """Test SplittingAI's suggesting first Pages."""
        splitting_ai = SplittingAI(self.file_splitting_model)
        pred = splitting_ai.propose_split_documents(self.test_document, return_pages=True)
        for page in pred.pages():
            if page.number in (1, 3, 5):
                assert page.is_first_page
            else:
                assert not page.is_first_page

    def test_metrics_calculation(self):
        """Test Evaluation class for ContextAwareFileSplitting."""
        splitting_ai = SplittingAI(self.file_splitting_model)
        ground_truth = self.test_document
        for page in ground_truth.pages():
            if page.number in (1, 3, 5):
                page.is_first_page = True
        pred = splitting_ai.propose_split_documents(self.test_document, return_pages=True)
        documents = [[ground_truth, pred]]
        evaluation = FileSplittingEvaluation(documents)
        assert evaluation.tp() == 3
        assert evaluation.fp() == 0
        assert evaluation.fn() == 0
        assert evaluation.precision() == 1.0
        assert evaluation.recall() == 1.0
        assert evaluation.f1() == 1.0

    def test_metrics_calculation_by_category(self):
        """Test Evaluation by Category."""
        splitting_ai = SplittingAI(self.file_splitting_model)
        ground_truth = self.test_document
        for page in ground_truth.pages():
            if page.number in (1, 3, 5):
                page.is_first_page = True
        pred = splitting_ai.propose_split_documents(self.test_document, return_pages=True)
        documents = [[ground_truth, pred]]
        evaluation = FileSplittingEvaluation(documents)
        assert evaluation.tp(search=ground_truth.category) == 3
        assert evaluation.fp(search=ground_truth.category) == 0
        assert evaluation.fn(search=ground_truth.category) == 0
        assert evaluation.precision(search=ground_truth.category) == 1.0
        assert evaluation.recall(search=ground_truth.category) == 1.0
        assert evaluation.f1(search=ground_truth.category) == 1.0

    def test_splitting_ai_evaluation(self):
        """Test evaluate_full method of SplittingAI."""
        splitting_ai = SplittingAI(self.file_splitting_model)
        splitting_ai.evaluate_full()
        assert splitting_ai.full_evaluation.evaluation_results['tp'] == 4
        assert splitting_ai.full_evaluation.evaluation_results['fp'] == 0
        assert splitting_ai.full_evaluation.evaluation_results['fn'] == 0
        assert splitting_ai.full_evaluation.evaluation_results['precision'] == 1.0
        assert splitting_ai.full_evaluation.evaluation_results['recall'] == 1.0
        assert splitting_ai.full_evaluation.evaluation_results['f1'] == 1.0
