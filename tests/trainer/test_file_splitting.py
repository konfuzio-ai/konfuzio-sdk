"""Test SplittingAI and the model's training, saving and prediction."""
import os
import pathlib
import unittest

from copy import deepcopy

from konfuzio_sdk.samples import LocalTextProject
from konfuzio_sdk.tokenizer.regex import ConnectedTextTokenizer
from konfuzio_sdk.trainer.file_splitting import ContextAwareFileSplittingModel, SplittingAI
from konfuzio_sdk.trainer.information_extraction import load_model


class TestFileSplittingModel(unittest.TestCase):
    """Test filesplitting model."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the tested class."""
        cls.project = LocalTextProject()
        cls.file_splitting_model = ContextAwareFileSplittingModel()
        cls.file_splitting_model.categories = [cls.project.get_category_by_id(3), cls.project.get_category_by_id(4)]
        cls.file_splitting_model.documents = [
            document for document in cls.project.documents if document.category in cls.file_splitting_model.categories
        ]
        cls.file_splitting_model.test_documents = [
            document
            for document in cls.project.test_documents
            if document.category in cls.file_splitting_model.categories
        ][:-1]
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
        # deepcopying because we do not want changes in an original test Document.
        # typically this happens in one of the private methods, but since here we pass a Document Page by Page, we
        # need to tokenize it explicitly (compared to when we pass a full Document to the SplittingAI).
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

    def test_json_model_save_and_load(self):
        """Test saving and loading first_page_spans as JSON."""
        self.file_splitting_model.output_dir = self.project.model_folder
        self.file_splitting_model.path = self.file_splitting_model.save(save_json=True)
        assert os.path.isfile(self.file_splitting_model.path)
        self.file_splitting_model.load_json(self.file_splitting_model.path)
        assert 3 in self.file_splitting_model.first_page_spans
        assert 4 in self.file_splitting_model.first_page_spans
        assert "Morning," in self.file_splitting_model.first_page_spans[3]
        assert "I like bread." in self.file_splitting_model.first_page_spans[3]
        assert "Evening," in self.file_splitting_model.first_page_spans[4]
        assert "I like fish." in self.file_splitting_model.first_page_spans[4]
        assert len(self.file_splitting_model.first_page_spans) == 2
        assert len(self.file_splitting_model.first_page_spans[3]) == 2
        assert len(self.file_splitting_model.first_page_spans[4]) == 2
        pathlib.Path(self.file_splitting_model.path).unlink()

    def test_pickle_model_save_load(self):
        """Test saving ContextAwareFileSplittingModel to pickle."""
        self.file_splitting_model.output_dir = self.project.model_folder
        self.file_splitting_model.path = self.file_splitting_model.save(save_json=False)
        assert os.path.isfile(self.file_splitting_model.path)
        model = load_model(self.file_splitting_model.path)
        assert model.first_page_spans == self.file_splitting_model.first_page_spans
        pathlib.Path(self.file_splitting_model.path).unlink()

    def test_splitting_ai_predict(self):
        """Test SplittingAI's Document-splitting method."""
        splitting_ai = SplittingAI(self.file_splitting_model)
        pred = splitting_ai.propose_split_documents(self.test_document)
        assert len(pred) == 3

    def test_suggest_first_pages(self):
        """Test SplittingAI's suggesting first Pages."""
        splitting_ai = SplittingAI(self.file_splitting_model)
        test_document = self.file_splitting_model.tokenizer.tokenize(
            deepcopy(self.project.get_category_by_id(3).test_documents()[0])
        )
        pred = splitting_ai.propose_split_documents(test_document, return_pages=True)
        for page in pred.pages():
            if page.number in (1, 3, 5):
                assert page.is_first_page
            else:
                assert not page.is_first_page

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

    def test_splitting_with_inplace(self):
        """Test ContextAwareFileSplittingModel's predict method with inplace=True."""
        splitting_ai = SplittingAI(self.file_splitting_model)
        test_document = self.file_splitting_model.tokenizer.tokenize(
            self.project.get_category_by_id(3).test_documents()[0]
        )
        pred = splitting_ai.propose_split_documents(test_document, return_pages=True, inplace=True)
        for page in pred.pages():
            if page.number in (1, 3, 5):
                assert page.is_first_page
            else:
                assert not page.is_first_page
        assert pred == test_document
