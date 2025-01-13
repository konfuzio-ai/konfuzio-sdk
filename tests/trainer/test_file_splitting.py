"""Test Splitting AI and the models' training, saving and prediction."""
import bz2
import os
import pathlib
import shutil
import sys
import unittest
from copy import deepcopy

import cloudpickle
import numpy
import parameterized
import pytest
from requests import HTTPError

from konfuzio_sdk.api import (
    delete_ai_model,
    konfuzio_session,
    update_ai_model,
    upload_ai_model,
)
from konfuzio_sdk.data import Category, Document, Project
from konfuzio_sdk.samples import LocalTextProject
from konfuzio_sdk.settings_importer import is_dependency_installed
from konfuzio_sdk.tokenizer.regex import ConnectedTextTokenizer
from konfuzio_sdk.trainer.file_splitting import (
    ContextAwareFileSplittingModel,
    MultimodalFileSplittingModel,
    SplittingAI,
    TextualFileSplittingModel,
)
from konfuzio_sdk.urls import get_create_ai_model_url
from tests.variables import OFFLINE_PROJECT

TEST_WITH_FULL_DATASET = False
DEVICE = 'cpu'


@pytest.mark.skipif(
    not is_dependency_installed('torch') and not is_dependency_installed('transformers') and not is_dependency_installed('tensorflow'),
    reason='Required dependencies not installed.',
)
class TestContextAwareFileSplittingModel(unittest.TestCase):
    """Test Context Aware File Splitting Model."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the tested class."""
        cls.project = LocalTextProject()
        cls.file_splitting_model = ContextAwareFileSplittingModel(
            categories=[cls.project.get_category_by_id(3), cls.project.get_category_by_id(4)],
            tokenizer=ConnectedTextTokenizer(),
        )
        cls.file_splitting_model.test_documents = cls.file_splitting_model.test_documents[:-2]
        cls.test_document = cls.project.get_document_by_id(9)

    def test_fit_context_aware_splitting_model(self):
        """Test pseudotraining of the Context Aware File Splitting Model."""
        self.file_splitting_model.fit(allow_empty_categories=True)
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
            cur_exclusive_first_page_strings = category.exclusive_first_page_strings(tokenizer=ConnectedTextTokenizer())
            for span in cur_exclusive_first_page_strings:
                assert span not in non_first_page_spans[category.id_]

    def test_init_file_splitting_model_empty_list(self):
        """Test running Context Aware File Splitting Model with an empty Categories list."""
        with pytest.raises(ValueError, match='an empty list'):
            ContextAwareFileSplittingModel(categories=[], tokenizer=ConnectedTextTokenizer())

    def test_init_file_splitting_model_not_a_category(self):
        """Test passing a list with an element that is not a Category as an input."""
        with pytest.raises(ValueError, match='have to be Categories'):
            ContextAwareFileSplittingModel(categories=[self.project.get_category_by_id(3), ''], tokenizer=ConnectedTextTokenizer())

    def test_init_file_splitting_model_category_no_documents(self):
        """Test passing a Category that does not have Documents."""
        _ = Category(project=self.project, id_=5, name='CategoryName 5')
        with pytest.raises(ValueError, match='At least one Category'):
            ContextAwareFileSplittingModel(categories=[_], tokenizer=ConnectedTextTokenizer())

    def test_init_file_splitting_model_category_no_test_documents(self):
        """Test passing a Category that does not have test Documents."""
        _ = Category(project=self.project, id_=6, name='CategoryName 6')
        Document(project=self.project, category=_, text='Hi all, I like fish.', dataset_status=2)
        initialized_ai = ContextAwareFileSplittingModel(categories=[_], tokenizer=ConnectedTextTokenizer())
        assert _ in initialized_ai.categories

    def test_load_incompatible_model(self):
        """Test initializing a model that does not pass has_compatible_interface check."""
        wrong_class = ConnectedTextTokenizer()
        assert not self.file_splitting_model.has_compatible_interface(wrong_class)

    def test_load_model_from_different_class(self):
        """Test Splitting AI with a model that doesn't inherit from AbstractFileSplittingModel class."""
        wrong_class = ConnectedTextTokenizer()
        with pytest.raises(ValueError, match='model is not inheriting from AbstractFileSplittingModel'):
            SplittingAI(model=wrong_class)

    def test_predict_context_aware_splitting_model(self):
        """Test correct first Page prediction."""
        test_document = self.file_splitting_model.tokenizer.tokenize(deepcopy(self.project.get_category_by_id(3).test_documents()[0]))
        # deep copying because we do not want changes in an original test Document.
        # typically this happens in one of the private methods, but since here we pass a Document Page by Page, we
        # need to tokenize it explicitly (compared to when we pass a full Document to the Splitting AI).
        for page in test_document.pages():
            page.is_first_page = False
            for category in self.file_splitting_model.categories:
                cur_first_page_strings = category.exclusive_first_page_strings(tokenizer=self.file_splitting_model.tokenizer)
                intersection = {span.offset_string.strip('\f').strip('\n') for span in page.spans()}.intersection(cur_first_page_strings)
                if len(intersection) > 0:
                    page.is_first_page = True
                    break
            if page.number == 1:
                assert intersection == {'I like bread.'}
                assert page.is_first_page
            if page.number in (2, 4):
                assert intersection == set()
            if page.number in (3, 5):
                assert intersection == {'Morning,'}
                assert page.is_first_page

    def test_predict_with_empty_categories(self):
        """Test predicting with all empty Categories."""
        model = ContextAwareFileSplittingModel(categories=[self.project.get_category_by_id(2)], tokenizer=self.file_splitting_model.tokenizer)
        model.fit(allow_empty_categories=True)
        with pytest.raises(ValueError, match='Cannot run prediction as none of the Categories in'):
            model.predict(self.test_document.pages()[0])

    def test_pickle_model_save_load(self):
        """Test saving Context Aware File Splitting Model to a pickle."""
        self.file_splitting_model.path = self.file_splitting_model.save(keep_documents=True, max_ram='5MB')
        assert os.path.isfile(self.file_splitting_model.path)
        model = ContextAwareFileSplittingModel.load_model(self.file_splitting_model.path)
        for category_gt, category_load in zip(self.file_splitting_model.categories, model.categories):
            gt_exclusive_first_page_strings = category_gt.exclusive_first_page_strings(tokenizer=ConnectedTextTokenizer())
            load_exclusive_first_page_strings = category_load.exclusive_first_page_strings(tokenizer=ConnectedTextTokenizer())
            assert gt_exclusive_first_page_strings == load_exclusive_first_page_strings

    def test_pickle_model_save_lose_weight(self):
        """Test saving Context Aware File Splitting Model with reduce_weight."""
        self.file_splitting_model.path = self.file_splitting_model.save(reduce_weight=True, keep_documents=True, max_ram='5MB')
        assert os.path.isfile(self.file_splitting_model.path)
        model = ContextAwareFileSplittingModel.load_model(self.file_splitting_model.path)
        for category_gt, category_load in zip(self.file_splitting_model.categories, model.categories):
            gt_exclusive_first_page_strings = category_gt.exclusive_first_page_strings(tokenizer=ConnectedTextTokenizer())
            load_exclusive_first_page_strings = category_load.exclusive_first_page_strings(tokenizer=ConnectedTextTokenizer())
            assert gt_exclusive_first_page_strings == load_exclusive_first_page_strings

    @unittest.skipIf(sys.version_info[:2] != (3, 8), reason='This AI can only be loaded on Python 3.8.')
    def test_pickle_model_upload_modify_delete(self):
        """Upload the model."""
        assert os.path.isfile(self.file_splitting_model.path)
        try:
            model_id = upload_ai_model(ai_model_path=self.file_splitting_model.path, project_id=46)
            assert isinstance(model_id, int)
            updated = update_ai_model(model_id, ai_type='filesplitting', description='test_description')
            assert updated['description'] == 'test_description'
            updated = update_ai_model(model_id, ai_type='filesplitting', patch=False, description='test_description')
            assert updated['description'] == 'test_description'
            delete_ai_model(model_id, ai_type='filesplitting')
            url = get_create_ai_model_url(ai_type='filesplitting')
            session = konfuzio_session()
            not_found = session.get(url)
            assert not_found.status_code == 204
        except HTTPError as e:
            assert ('403' in str(e)) or ('404' in str(e))

    def test_old_pickle_model_save_load(self):
        """Test saving and loading a model using the older bz2-compression-including save and load methods."""
        temp_pkl_file_path = self.project.project_folder + '/models/temp.pkl'
        pkl_file_path = self.project.project_folder + '/models/model.pkl'
        pathlib.Path(self.project.project_folder + '/models').mkdir(parents=True, exist_ok=True)
        self.file_splitting_model.ensure_model_memory_usage_within_limit(max_ram=None)
        with open(temp_pkl_file_path, 'wb') as f:
            cloudpickle.dump(self.file_splitting_model, f)
        with open(temp_pkl_file_path, 'rb') as input_f:
            with bz2.open(pkl_file_path, 'wb') as output_f:
                shutil.copyfileobj(input_f, output_f)
        os.remove(temp_pkl_file_path)
        with bz2.open(pkl_file_path, 'rb') as file:
            loaded = cloudpickle.load(file)
        for category_gt, category_load in zip(self.file_splitting_model.categories, loaded.categories):
            gt_exclusive_first_page_strings = category_gt.exclusive_first_page_strings(tokenizer=ConnectedTextTokenizer())
            load_exclusive_first_page_strings = category_load.exclusive_first_page_strings(tokenizer=ConnectedTextTokenizer())
            assert gt_exclusive_first_page_strings == load_exclusive_first_page_strings

    def test_splitting_ai_predict(self):
        """Test Splitting AI's Document-splitting method."""
        splitting_ai = SplittingAI(self.file_splitting_model)
        pred = splitting_ai.propose_split_documents(self.test_document, return_pages=False)
        assert len(pred) == 3

    def test_splitting_ai_predict_one_file_document(self):
        """Test Splitting AI's Document-splitting method on a single-file Document."""
        splitting_ai = SplittingAI(self.file_splitting_model)
        test_document = self.project.get_document_by_id(17)
        pred = splitting_ai.propose_split_documents(test_document)
        assert len(pred) == 1
        assert len(pred[0].pages()) == 2
        assert test_document.pages()[0].category == pred[0].pages()[0].category

    def test_splitting_ai_evaluate_full_on_training(self):
        """Test Splitting AI's evaluate_full on training Documents."""
        splitting_ai = SplittingAI(self.file_splitting_model)
        splitting_ai.evaluate_full(use_training_docs=True)
        assert splitting_ai.full_evaluation.tp() >= 0
        assert splitting_ai.full_evaluation.fp() >= 0
        assert splitting_ai.full_evaluation.fn() >= 0
        assert splitting_ai.full_evaluation.tn() >= 0
        assert splitting_ai.full_evaluation.precision() == 1.0
        assert splitting_ai.full_evaluation.recall() == 1.0
        assert splitting_ai.full_evaluation.f1() == 1.0

    def test_splitting_ai_evaluate_full_on_testing(self):
        """Test Splitting AI's evaluate_full on testing Documents."""
        splitting_ai = SplittingAI(self.file_splitting_model)
        splitting_ai.evaluate_full()
        assert splitting_ai.full_evaluation.tp() >= 0
        assert splitting_ai.full_evaluation.fp() >= 0
        assert splitting_ai.full_evaluation.fn() >= 0
        assert splitting_ai.full_evaluation.tn() >= 0
        assert splitting_ai.full_evaluation.precision() == 1.0
        assert splitting_ai.full_evaluation.recall() == 1.0
        assert splitting_ai.full_evaluation.f1() == 1.0

    def test_splitting_no_category_document(self):
        """Test running Splitting AI on a Document with Category == NO_CATEGORY."""
        splitting_ai = SplittingAI(self.file_splitting_model)
        test_document = self.project.get_document_by_id(19)
        assert test_document._category == self.project.no_category
        pred = splitting_ai.propose_split_documents(test_document, return_pages=False)
        assert len(pred) == 1
        assert pred[0].text == test_document.text

    def test_splitting_with_inplace(self):
        """Test Context Aware File Splitting Model's predict method with inplace=True."""
        splitting_ai = SplittingAI(self.file_splitting_model)
        pred = splitting_ai.propose_split_documents(self.test_document, return_pages=True, inplace=True)[0]
        for page in pred.pages():
            if page.number in (1, 3, 5):
                assert page.is_first_page
                assert page.is_first_page_confidence == 1
            else:
                assert not page.is_first_page
                assert page.is_first_page_confidence == 0
        assert pred == self.test_document

    def test_suggest_first_pages(self):
        """Test AI's suggesting potential split points of a deepcopy of a Document without the actual splitting."""
        splitting_ai = SplittingAI(self.file_splitting_model)
        predictions = splitting_ai.propose_split_documents(self.test_document, return_pages=True)
        assert len(predictions) == 1
        prediction = predictions[0]
        for page in prediction.pages():
            if page.number in (1, 3, 5):
                assert page.is_first_page
                assert page.is_first_page_confidence == 1
            else:
                assert not page.is_first_page
                assert page.is_first_page_confidence == 0
        for page_original, page_predicted in zip(self.test_document.pages(), prediction.pages()):
            assert page_original.category == page_predicted.category
        pathlib.Path(self.file_splitting_model.path).unlink()

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove Project None folder."""
        cls.project.delete()  # created when saving the AI model


@pytest.mark.skipif(
    not is_dependency_installed('torch') and not is_dependency_installed('transformers') and not is_dependency_installed('tensorflow'),
    reason='Required dependencies not installed.',
)
@parameterized.parameterized_class(
    ('model_name'),
    [(None, 'prajjwal1/bert-tiny')],
)
class TestTextualFileSplittingModel(unittest.TestCase):
    """Test Textual File Splitting Model."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the tested class."""
        # setting up project documents and model
        cls.project = Project(id_=14392)
        if cls.model_name:
            cls.file_splitting_model = TextualFileSplittingModel(categories=cls.project.categories, model_name=cls.model_name)
        else:
            cls.file_splitting_model = TextualFileSplittingModel(categories=cls.project.categories)
        if not TEST_WITH_FULL_DATASET:
            cls.file_splitting_model.documents = cls.file_splitting_model.categories[0].documents()
            cls.file_splitting_model.test_documents = cls.file_splitting_model.categories[0].test_documents()
        cls.training_args = {
            'epochs': 3,
            'train_batch_size': 1,
            'eval_batch_size': 1,
            'learning_rate': 1e-5,
            'weight_decay': 2e-5,
            'lr_scheduler_type': 'cosine',
            'device': DEVICE,
        }
        cls.test_document = cls.file_splitting_model.test_documents[-1]
        # starting MLflow server
        cls.mlflow_process, cls.mlflow_host, cls.mlflow_port, cls.mlflow_url = cls.start_mlflow_server()

    @classmethod
    def start_mlflow_server(cls):
        import logging
        import random
        import subprocess
        import sys
        import time

        logger = logging.getLogger(__name__)

        mlflow_host = '127.0.0.1'
        mlflow_port = random.randint(5000, 9000)
        mlflow_url = f'http://{mlflow_host}:{mlflow_port}'
        logger.info(f'Starting MLflow server under {mlflow_url}...')
        mlflow_process = subprocess.Popen(
            [sys.executable, '-m', 'mlflow', 'server', '--host', mlflow_host, '--port', str(mlflow_port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(5)
        return mlflow_process, mlflow_host, mlflow_port, mlflow_url

    def test_model_training_without_mlflow(self):
        """Test model's fit() method without Mlflow arguments."""
        # check the name of the model being trained
        if self.model_name:
            assert self.file_splitting_model.model_name == self.model_name
        else:
            assert self.file_splitting_model.model_name == 'distilbert-base-uncased'
        self.file_splitting_model.fit(**self.training_args)
        # checking if training arguments are correctly set
        assert self.file_splitting_model.training_args.num_train_epochs == self.training_args['epochs']
        assert self.file_splitting_model.training_args.per_device_train_batch_size == self.training_args['train_batch_size']
        assert self.file_splitting_model.training_args.per_device_eval_batch_size == self.training_args['eval_batch_size']
        assert self.file_splitting_model.training_args.learning_rate == self.training_args['learning_rate']
        assert self.file_splitting_model.training_args.weight_decay == self.training_args['weight_decay']
        assert self.file_splitting_model.training_args.lr_scheduler_type == self.training_args['lr_scheduler_type']
        # checking if trained model has been saved
        assert self.file_splitting_model.model is not None
        # checking if mlflow was used
        assert self.file_splitting_model.use_mlflow is False

    def test_model_training_with_mlflow(self):
        """Test model's fit() method with Mlflow arguments."""
        from mlflow import MlflowClient

        # creating mlflow client
        mlflow_client = MlflowClient(self.mlflow_url)
        # if experiment with experiment_name does not exist, create it
        if not mlflow_client.get_experiment_by_name('test_experiment'):
            _ = mlflow_client.create_experiment('test_experiment')
        # check the name of the model being trained
        if self.model_name:
            assert self.file_splitting_model.model_name == self.model_name
        else:
            assert self.file_splitting_model.model_name == 'distilbert-base-uncased'
        # training the model
        self.file_splitting_model.fit(experiment_name='test_experiment', tracking_uri=self.mlflow_url, **self.training_args)
        # checking if training arguments are correctly set
        assert self.file_splitting_model.training_args.num_train_epochs == self.training_args['epochs']
        assert self.file_splitting_model.training_args.per_device_train_batch_size == self.training_args['train_batch_size']
        assert self.file_splitting_model.training_args.per_device_eval_batch_size == self.training_args['eval_batch_size']
        assert self.file_splitting_model.training_args.learning_rate == self.training_args['learning_rate']
        assert self.file_splitting_model.training_args.weight_decay == self.training_args['weight_decay']
        assert self.file_splitting_model.training_args.lr_scheduler_type == self.training_args['lr_scheduler_type']
        # checking if trained model has been saved
        assert self.file_splitting_model.model is not None
        # checking if mlflow was used
        assert self.file_splitting_model.use_mlflow is True
        # checking if the experiment was created
        try:
            experiment = mlflow_client.get_experiment_by_name('test_experiment')
            assert experiment.name == 'test_experiment'
        except Exception as e:
            assert False, f'Experiment not created: {e}'
        # checking if the run was created
        try:
            expected_run_id = self.file_splitting_model.mlflow_run_id
            run = mlflow_client.get_run(expected_run_id)
            assert run.info.run_id == expected_run_id
            assert 'splitting_run' in run.info.run_name
        except Exception as e:
            assert False, f'Run not created: {e}'

    def test_run_page_prediction(self):
        """Test model's prediction."""
        for doc in self.file_splitting_model.test_documents:
            for page in doc.pages():
                ground_truth_is_first_page = page.is_first_page
                page.is_first_page = None
                page = self.file_splitting_model.predict(page, device=DEVICE)
                predicted_is_first_page = page.is_first_page
                assert ground_truth_is_first_page == predicted_is_first_page
                assert isinstance(page.is_first_page_confidence, float)

    def test_run_splitting_ai_prediction(self):
        """Test Splitting AI integration with the Textual File Splitting Model."""
        splitting_ai = SplittingAI(self.file_splitting_model)
        pred = splitting_ai.propose_split_documents(self.test_document)
        assert len(pred) == 1
        for page in pred[0].pages():
            if page.number == 1:
                assert page.is_first_page
                assert page.is_first_page_confidence > 0.5
            else:
                assert not page.is_first_page
                assert page.is_first_page_confidence

    def test_save_load_model(self):
        """Test saving and loading pickle file of the model."""
        path = self.file_splitting_model.save()
        loaded = TextualFileSplittingModel.load_model(path)
        splitting_ai = SplittingAI(model=loaded)
        assert isinstance(splitting_ai.model, TextualFileSplittingModel)
        pathlib.Path(path).unlink()

    def test_splitting_ai_evaluate_full_on_training(self):
        """Test Splitting AI's evaluate_full on training Documents."""
        splitting_ai = SplittingAI(self.file_splitting_model)
        splitting_ai.evaluate_full(use_training_docs=True)
        assert splitting_ai.full_evaluation.tp() >= 0.0
        assert splitting_ai.full_evaluation.fp() == 0.0
        assert splitting_ai.full_evaluation.fn() == 0.0
        assert splitting_ai.full_evaluation.tn() >= 0.0
        assert splitting_ai.full_evaluation.precision() == 1.0
        assert splitting_ai.full_evaluation.recall() == 1.0
        assert splitting_ai.full_evaluation.f1() == 1.0

    def test_splitting_ai_evaluate_full_on_testing(self):
        """Test Splitting AI's evaluate_full on testing Documents."""
        splitting_ai = SplittingAI(self.file_splitting_model)
        splitting_ai.evaluate_full()
        assert splitting_ai.full_evaluation.tp() >= 0.0
        assert splitting_ai.full_evaluation.fp() == 0.0
        assert splitting_ai.full_evaluation.fn() == 0.0
        assert splitting_ai.full_evaluation.tn() >= 0.0
        assert splitting_ai.full_evaluation.precision() == 1.0
        assert splitting_ai.full_evaluation.recall() == 1.0
        assert splitting_ai.full_evaluation.f1() == 1.0

    @classmethod
    def tearDownClass(cls) -> None:
        """Terminate MLflow server."""
        import psutil

        # get the parent process of the MLflow server and it's children processes
        parent = psutil.Process(cls.mlflow_process.pid)
        children = parent.children(recursive=True)
        # terminate the children processes
        for child in children:
            child.terminate()
        # wait for the children processes to terminate
        _, __ = psutil.wait_procs(children, timeout=5)
        # terminate the parent process if it's still alive
        if parent.is_running():
            parent.terminate()
        # wait for the parent process to terminate
        parent.wait(5)


@pytest.mark.skipif(
    not is_dependency_installed('torch') and not is_dependency_installed('transformers') and not is_dependency_installed('tensorflow'),
    reason='Required dependencies not installed.',
)
class TestMultimodalFileSplittingModel(unittest.TestCase):
    """Test Multimodal File Splitting Model."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the tested class."""
        cls.project = Project(id_=None, project_folder=OFFLINE_PROJECT)
        cls.file_splitting_model = MultimodalFileSplittingModel(categories=cls.project.categories)
        if not TEST_WITH_FULL_DATASET:
            cls.file_splitting_model.documents = cls.file_splitting_model.categories[0].documents()
            cls.file_splitting_model.test_documents = cls.file_splitting_model.categories[0].test_documents()
        cls.test_document = cls.file_splitting_model.test_documents[-1]

    def test_model_training(self):
        """Test model's fit() method."""
        self.file_splitting_model.fit(epochs=10, train_batch_size=2)
        assert self.file_splitting_model.model

    def test_run_page_prediction(self):
        """Test model's prediction."""
        for doc in self.file_splitting_model.test_documents:
            for page in doc.pages():
                ground_truth_is_first_page = page.is_first_page
                page.is_first_page = None
                page = self.file_splitting_model.predict(page)
                predicted_is_first_page = page.is_first_page
                assert ground_truth_is_first_page == predicted_is_first_page
                assert type(page.is_first_page_confidence) is numpy.float32

    def test_run_splitting_ai_prediction(self):
        """Test Splitting AI integration with the Multimodal File Splitting Model."""
        splitting_ai = SplittingAI(self.file_splitting_model)
        pred = splitting_ai.propose_split_documents(self.test_document)
        assert len(pred) == 1
        for page in pred[0].pages():
            ground_truth_is_first_page = page.is_first_page
            page.is_first_page = None
            page = self.file_splitting_model.predict(page)
            predicted_is_first_page = page.is_first_page
            assert ground_truth_is_first_page == predicted_is_first_page
            assert type(page.is_first_page_confidence) is numpy.float32

    def test_save_load_model(self):
        """Test saving and loading pickle file of the model."""
        path = self.file_splitting_model.save()
        loaded = MultimodalFileSplittingModel.load_model(path)
        splitting_ai = SplittingAI(model=loaded)
        assert isinstance(splitting_ai.model, MultimodalFileSplittingModel)
        pathlib.Path(path).unlink()

    def test_splitting_ai_evaluate_full_on_training(self):
        """Test Splitting AI's evaluate_full on training Documents."""
        splitting_ai = SplittingAI(self.file_splitting_model)
        splitting_ai.evaluate_full(use_training_docs=True)
        assert splitting_ai.full_evaluation.tp() >= 0.0
        assert splitting_ai.full_evaluation.fp() == 0.0
        assert splitting_ai.full_evaluation.fn() == 0.0
        assert splitting_ai.full_evaluation.tn() >= 0.0
        assert splitting_ai.full_evaluation.precision() >= 0.0
        assert splitting_ai.full_evaluation.recall() >= 0.0
        assert splitting_ai.full_evaluation.f1() == 1.0

    def test_splitting_ai_evaluate_full_on_testing(self):
        """Test Splitting AI's evaluate_full on testing Documents."""
        splitting_ai = SplittingAI(self.file_splitting_model)
        splitting_ai.evaluate_full()
        assert splitting_ai.full_evaluation.tp() >= 0.0
        assert splitting_ai.full_evaluation.fp() == 0.0
        assert splitting_ai.full_evaluation.fn() == 0.0
        assert splitting_ai.full_evaluation.tn() >= 0.0
        assert splitting_ai.full_evaluation.precision() >= 0.0
        assert splitting_ai.full_evaluation.recall() >= 0.0
        assert splitting_ai.full_evaluation.f1() == 1.0
