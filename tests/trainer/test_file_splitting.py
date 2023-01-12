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
        cls.file_splitting_model = ContextAwareFileSplittingModel(
            categories=[cls.project.get_category_by_id(3), cls.project.get_category_by_id(4)]
        )
        cls.file_splitting_model.test_documents = cls.file_splitting_model.test_documents[:-2]
        cls.file_splitting_model.tokenizer = ConnectedTextTokenizer()
        cls.test_document = cls.project.get_category_by_id(3).test_documents()[0]

    def test_fit_context_aware_splitting_model(self):
        """Test pseudotraining of the context-aware splitting model."""
        self.file_splitting_model.fit()
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
            for span in category.exclusive_first_page_strings:
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
                    category.exclusive_first_page_strings
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

    def test_pickle_model_save_load(self):
        """Test saving ContextAwareFileSplittingModel to pickle."""
        self.file_splitting_model.output_dir = self.project.model_folder
        self.file_splitting_model.path = self.file_splitting_model.save(
            keep_documents=True, max_ram='5MB', include_konfuzio=False
        )
        assert os.path.isfile(self.file_splitting_model.path)
        model = load_model(self.file_splitting_model.path)
        for category_gt, category_load in zip(self.file_splitting_model.categories, model.categories):
            assert category_gt.exclusive_first_page_strings == category_load.exclusive_first_page_strings
        pathlib.Path(self.file_splitting_model.path).unlink()

    def test_pickle_model_save_lose_weight(self):
        """Test saving ContextAwareFileSplittingModel with reduce_weight."""
        self.file_splitting_model.output_dir = self.project.model_folder
        self.file_splitting_model.path = self.file_splitting_model.save(
            reduce_weight=True, keep_documents=True, max_ram='5MB', include_konfuzio=False
        )
        assert os.path.isfile(self.file_splitting_model.path)
        model = load_model(self.file_splitting_model.path)
        for category_gt, category_load in zip(self.file_splitting_model.categories, model.categories):
            assert category_gt.exclusive_first_page_strings == category_load.exclusive_first_page_strings
        pathlib.Path(self.file_splitting_model.path).unlink()

    def test_splitting_ai_predict(self):
        """Test SplittingAI's Document-splitting method."""
        splitting_ai = SplittingAI(self.file_splitting_model)
        pred = splitting_ai.propose_split_documents(self.test_document)
        assert len(pred) == 3

    def test_splitting_ai_predict_one_file_document(self):
        """Test SplittingAI's Document-splitting method on a single-file Document."""
        splitting_ai = SplittingAI(self.file_splitting_model)
        test_document = self.project.get_category_by_id(4).test_documents()[-1]
        pred = splitting_ai.propose_split_documents(test_document)
        assert len(pred) == 1
        assert len(pred[0].pages()) == 2

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
        self.file_splitting_model.test_documents = self.file_splitting_model.test_documents[:-2]
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
        pred = splitting_ai.propose_split_documents(test_document, return_pages=True, inplace=True)[0]
        for page in pred.pages():
            if page.number in (1, 3, 5):
                assert page.is_first_page
            else:
                assert not page.is_first_page
        assert pred == test_document

    def test_suggest_first_pages(self):
        """Test SplittingAI's suggesting first Pages."""
        splitting_ai = SplittingAI(self.file_splitting_model)
        test_document = self.file_splitting_model.tokenizer.tokenize(
            deepcopy(self.project.get_category_by_id(3).test_documents()[0])
        )
        pred = splitting_ai.propose_split_documents(test_document, return_pages=True)[0]
        for page in pred.pages():
            if page.number in (1, 3, 5):
                assert page.is_first_page
            else:
                assert not page.is_first_page
