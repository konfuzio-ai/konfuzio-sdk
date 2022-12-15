"""Test SplittingAI and the model's training, saving and prediction."""
import os
import unittest

from copy import deepcopy

from konfuzio_sdk.data import Project
from konfuzio_sdk.samples import LocalTextProject
from konfuzio_sdk.tokenizer.regex import ConnectedTextTokenizer
from konfuzio_sdk.trainer.file_splitting import ContextAwareFileSplittingModel, SplittingAI, FusionModel


class TestContextAwareFileSplittingModel(unittest.TestCase):
    """Test context-aware filesplitting model."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the tested class."""
        cls.project = LocalTextProject()
        cls.file_splitting_model = ContextAwareFileSplittingModel()
        cls.file_splitting_model.categories = [cls.project.get_category_by_id(3), cls.project.get_category_by_id(4)]
        cls.file_splitting_model.train_data = [
            document for document in cls.project.documents if document.category in cls.file_splitting_model.categories
        ]
        cls.file_splitting_model.test_data = [
            document
            for document in cls.project.test_documents
            if document.category in cls.file_splitting_model.categories
        ]
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
        splitting_ai = SplittingAI(self.file_splitting_model, use_fallback_logic=True)
        pred = splitting_ai.propose_split_documents(self.test_document)
        assert len(pred) == 3

    def test_suggest_first_pages(self):
        """Test SplittingAI's suggesting first Pages."""
        splitting_ai = SplittingAI(self.file_splitting_model, use_fallback_logic=True)
        pred = splitting_ai.propose_split_documents(self.test_document, return_pages=True)
        for page in pred.pages():
            if page.number in (1, 3, 5):
                assert page.is_first_page
            else:
                assert not page.is_first_page
        for item in self.project.model_folder:
            if item.endswith('.pkl'):
                os.remove(os.path.join(self.project.model_folder, item))

    def test_splitting_ai_evaluate_full_on_training(self):
        """Test SplittingAI's evaluate_full on training Documents."""
        splitting_ai = SplittingAI(self.file_splitting_model)
        splitting_ai.evaluate_full(use_training_docs=True)
        assert splitting_ai.full_evaluation.tp() == 3
        assert splitting_ai.full_evaluation.fp() == 0
        assert splitting_ai.full_evaluation.fn() == 0
        assert splitting_ai.full_evaluation.tn() == 3
        assert splitting_ai.full_evaluation.precision() == 1.0
        assert splitting_ai.full_evaluation.recall() == 1.0
        assert splitting_ai.full_evaluation.f1() == 1.0

    def test_splitting_ai_evaluate_full_on_testing(self):
        """Test SplittingAI's evaluate_full on testing Documents."""
        splitting_ai = SplittingAI(self.file_splitting_model)
        splitting_ai.evaluate_full()
        print(splitting_ai.full_evaluation.evaluation_results)
        assert splitting_ai.full_evaluation.tp() == 9
        assert splitting_ai.full_evaluation.fp() == 0
        assert splitting_ai.full_evaluation.fn() == 0
        assert splitting_ai.full_evaluation.tn() == 7
        assert splitting_ai.full_evaluation.precision() == 1.0
        assert splitting_ai.full_evaluation.recall() == 1.0
        assert splitting_ai.full_evaluation.f1() == 1.0


TEST_WITH_FULL_DATASET = False


class TestFusionFileSplittingModel(unittest.TestCase):
    """Test fusion file-splitting model."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the tested class."""
        cls.project = Project(id_=46)
        cls.file_splitting_model = FusionModel()
        cls.file_splitting_model.categories = cls.project.categories
        if TEST_WITH_FULL_DATASET:
            cls.file_splitting_model.train_data = cls.project.documents
        else:
            cls.file_splitting_model.train_data = cls.project.documents[:10]
        cls.file_splitting_model.test_data = cls.project.test_documents

    def test_model_training(self):
        """Test model's fit() method."""
        self.file_splitting_model.model = self.file_splitting_model.fit()
        assert self.file_splitting_model.model

    def test_run_page_prediction(self):
        """Test model's prediction."""
        for doc in self.file_splitting_model.test_data:
            for page in doc.pages():
                page.is_first_page = None
                page = self.file_splitting_model.predict(page)
                assert page.is_first_page

    def test_run_splitting_ai_prediction(self):
        """Test SplittingAI integration with the Fusion model."""
