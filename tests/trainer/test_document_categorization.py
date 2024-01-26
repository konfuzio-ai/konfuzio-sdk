# -*- coding: utf-8 -*-
"""Test to train a Categorization AI."""
import logging
import os
import sys
import unittest
from copy import deepcopy
from typing import Dict, List

import parameterized
import pytest
from requests import HTTPError, ReadTimeout

from konfuzio_sdk.api import delete_ai_model, konfuzio_session, update_ai_model, upload_ai_model
from konfuzio_sdk.data import Document, Page, Project
from konfuzio_sdk.extras import FloatTensor, torch
from konfuzio_sdk.settings_importer import is_dependency_installed
from konfuzio_sdk.tokenizer.regex import ConnectedTextTokenizer, WhitespaceTokenizer
from konfuzio_sdk.trainer.document_categorization import (
    BERT,
    LSTM,
    NBOW,
    VGG,
    AbstractCategorizationModel,
    AbstractTextCategorizationModel,
    CategorizationAI,
    EfficientNet,
    ImageModel,
    MultimodalConcatenate,
    NameBasedCategorizationAI,
    NBOWSelfAttention,
    PageImageCategorizationModel,
    PageMultimodalCategorizationModel,
    PageTextCategorizationModel,
    TextModel,
    build_categorization_ai_pipeline,
)
from konfuzio_sdk.urls import get_create_ai_model_url
from tests.variables import (
    OFFLINE_PROJECT,
    TEST_CATEGORIZATION_DOCUMENT_ID,
    TEST_DOCUMENT_ID,
    TEST_PAYSLIPS_CATEGORY_ID,
    TEST_RECEIPTS_CATEGORY_ID,
)

logger = logging.getLogger(__name__)


@pytest.mark.skipif(
    not is_dependency_installed('timm')
    and not is_dependency_installed('torch')
    and not is_dependency_installed('transformers')
    and not is_dependency_installed('torchvision'),
    reason='Required dependencies not installed.',
)
class TestNameBasedCategorizationAI(unittest.TestCase):
    """Test the fallback logic for predicting the Category of a Document."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the Data and Categorization Pipeline."""
        cls.project = Project(id_=None, project_folder=OFFLINE_PROJECT)
        cls.categorization_pipeline = NameBasedCategorizationAI(cls.project.categories)
        cls.payslips_category = cls.project.get_category_by_id(TEST_PAYSLIPS_CATEGORY_ID)
        cls.receipts_category = cls.project.get_category_by_id(TEST_RECEIPTS_CATEGORY_ID)

    def test_1_configure_pipeline(self) -> None:
        """
        No pipeline to configure for the fallback logic.

        Documents can be specified to calculate Evaluation metrics.
        """
        assert self.categorization_pipeline.categories is not None

        payslips_training_documents = self.project.get_category_by_id(TEST_PAYSLIPS_CATEGORY_ID).documents()
        receipts_training_documents = self.project.get_category_by_id(TEST_RECEIPTS_CATEGORY_ID).documents()
        self.categorization_pipeline.documents = payslips_training_documents + receipts_training_documents
        assert all(doc.category is not None for doc in self.categorization_pipeline.documents)

        payslips_test_documents = self.project.get_category_by_id(TEST_PAYSLIPS_CATEGORY_ID).test_documents()
        receipts_test_documents = self.project.get_category_by_id(TEST_RECEIPTS_CATEGORY_ID).test_documents()
        self.categorization_pipeline.test_documents = payslips_test_documents + receipts_test_documents
        assert all(doc.category is not None for doc in self.categorization_pipeline.test_documents)

    def test_2_fit(self) -> None:
        """Start to train the Model."""
        # since we are using the fallback logic, this should not require training anything
        with self.assertRaises(NotImplementedError):
            self.categorization_pipeline.fit()

    def test_3_save_model(self):
        """Save the model."""
        # since we are using the fallback logic, this should not save any model to disk
        with self.assertRaises(NotImplementedError):
            self.categorization_pipeline.pipeline_path = self.categorization_pipeline.save(
                output_dir=self.project.model_folder
            )

    def test_4_evaluate(self):
        """Evaluate NameBasedCategorizationAI."""
        categorization_evaluation = self.categorization_pipeline.evaluate()
        # can't categorize any of the 3 payslips docs since they don't contain the word "lohnabrechnung"
        assert categorization_evaluation.f1(self.categorization_pipeline.categories[0]) == 1.0
        # can categorize 1 out of 2 receipts docs since one contains the word "quittung"
        # therefore recall == 1/2 and precision == 1.0, implying f1 == 2/3
        assert categorization_evaluation.f1(self.categorization_pipeline.categories[1]) == 1.0
        # global f1 score
        assert categorization_evaluation.f1(None) == 1.0

    def test_5a_categorize_test_document(self):
        """Test extract Category for a selected Test Document with the Category name contained within its text."""
        test_receipt_document = deepcopy(self.project.get_document_by_id(TEST_CATEGORIZATION_DOCUMENT_ID))
        # reset each Page.category attribute to test that it can be categorized successfully
        test_receipt_document.set_category(self.project.no_category)
        result = self.categorization_pipeline.categorize(document=test_receipt_document)
        assert isinstance(result, Document)
        assert result.category == self.receipts_category
        for page in result.pages():
            assert page.category == self.receipts_category

    def test_5b_categorize_test_document_check_category_annotation(self):
        """Test extract Category for a selected Test Document and ensure that maximum_confidence_category is set."""
        test_receipt_document = deepcopy(self.project.get_document_by_id(TEST_CATEGORIZATION_DOCUMENT_ID))
        test_receipt_document.set_category(self.project.no_category)
        result = self.categorization_pipeline.categorize(document=test_receipt_document, recategorize=True)
        assert isinstance(result, Document)
        assert result.maximum_confidence_category == self.receipts_category
        assert result.category == result.maximum_confidence_category

    def test_6a_categorize_document_with_no_pages(self):
        """Test extract Category for a Document without a Page."""
        document_with_no_pages = Document(project=self.project, text='hello')
        result = self.categorization_pipeline.categorize(document=document_with_no_pages)
        assert isinstance(result, Document)
        assert result.category == result.project.no_category
        assert result.pages() == []

    def test_7_already_existing_categorization(self):
        """Test that the existing Category attribute for a Test Document will be reused as the fallback result."""
        test_payslip_document = self.project.get_document_by_id(TEST_DOCUMENT_ID)
        result = self.categorization_pipeline.categorize(document=test_payslip_document)
        assert isinstance(result, Document)
        assert result.category == test_payslip_document.category
        for page in result.pages():
            assert page.category == test_payslip_document.category

    def test_8_cannot_categorize_test_documents_with_category_name_not_contained_in_text(self):
        """Test cannot extract Category for two Test Document if their texts don't contain the Category name."""
        test_receipt_document = self.project.get_category_by_id(TEST_RECEIPTS_CATEGORY_ID).test_documents()[0]
        # reset each Page.category attribute to test that it can't be categorized successfully
        test_receipt_document.set_category(self.project.no_category)
        result = self.categorization_pipeline.categorize(document=test_receipt_document)
        assert isinstance(result, Document)
        assert result.category == self.project.no_category
        for page in result.pages():
            assert page.category == self.project.no_category

        test_payslip_document = self.project.get_document_by_id(TEST_DOCUMENT_ID)
        # reset each Page.category attribute to test that it can't be categorized successfully
        test_payslip_document.set_category(self.project.no_category)
        result = self.categorization_pipeline.categorize(document=test_payslip_document)
        assert isinstance(result, Document)
        assert result.category == self.project.no_category
        for page in result.pages():
            assert page.category == self.project.no_category

    def test_9_force_categorization(self):
        """Test re-extract Category for two selected Test Documents that already contain a Category attribute."""
        # This Document can be recategorized successfully because its text contains the word "quittung" (receipt)
        # in it.
        # Recall that the check is case-insensitive.
        test_receipt_document = self.project.get_document_by_id(TEST_CATEGORIZATION_DOCUMENT_ID)
        result = self.categorization_pipeline.categorize(document=test_receipt_document, recategorize=True)
        assert isinstance(result, Document)
        assert result.category == test_receipt_document.category
        for page in result.pages():
            assert page.category == test_receipt_document.category

        # This Document is originally categorized as "Lohnabrechnung".
        # It cannot be recategorized successfully because its text does not contain
        # the word "lohnabrechnung" (payslip) in it.
        # Recall that the check is case-insensitive.
        test_payslip_document = self.project.get_document_by_id(TEST_DOCUMENT_ID)
        result = self.categorization_pipeline.categorize(document=test_payslip_document, recategorize=True)
        assert isinstance(result, Document)
        assert result.category == result.project.no_category
        for page in result.pages():
            assert page.category == self.project.no_category

    def test_9a_categorize_in_place(self):
        """Test inplace re-extract Category for a selected Test Document that contains a Category attribute."""
        # This Document can be recategorized successfully because its text contains the word "quittung" (receipt)
        # in it.
        # Recall that the check is case-insensitive.
        test_receipt_document = self.project.get_document_by_id(TEST_CATEGORIZATION_DOCUMENT_ID)
        test_receipt_document.set_category(self.project.no_category)
        self.categorization_pipeline.categorize(document=test_receipt_document, inplace=True)
        assert test_receipt_document.category == self.receipts_category
        for page in test_receipt_document.pages():
            assert page.category == self.receipts_category

    def test_9b_categorize_in_place_document_with_no_pages(self):
        """Test extract Category in place for a Document without a Page."""
        document_with_no_pages = Document(project=self.project, text='hello')
        result = self.categorization_pipeline.categorize(document=document_with_no_pages, inplace=True)
        assert isinstance(result, Document)
        assert result.category == self.project.no_category
        assert result.pages() == []

    def test_9c_categorize_defaults_not_in_place(self):
        """Test cannot re-extract Category for a selected Test Document that contain a Category attribute."""
        # This Document can be recategorized successfully because its text contains the word "quittung" (receipt)
        # in it.
        # Recall that the check is case-insensitive.
        test_receipt_document = self.project.get_document_by_id(TEST_CATEGORIZATION_DOCUMENT_ID)
        test_receipt_document.set_category(self.project.no_category)
        self.categorization_pipeline.categorize(document=test_receipt_document)
        assert test_receipt_document.category == self.project.no_category
        for page in test_receipt_document.pages():
            assert page.category == self.project.no_category

    def test_10_run_model_incompatible_interface(self):
        """Test initializing a model that does not pass has_compatible_interface check."""
        wrong_class = ConnectedTextTokenizer()
        assert not self.categorization_pipeline.has_compatible_interface(wrong_class)


@pytest.mark.skipif(
    not is_dependency_installed('timm')
    and not is_dependency_installed('torch')
    and not is_dependency_installed('transformers')
    and not is_dependency_installed('torchvision'),
    reason='Required dependencies not installed.',
)
class TestAbstractCategorizationModel(unittest.TestCase):
    """Test general functionality that uses nn.Module classes for classification."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the classifier and test setup."""

        # DummyClassifier definition
        class DummyClassifier(AbstractCategorizationModel):
            def _valid(self) -> None:
                """Validate architecture sizes."""
                pass

            def _load_architecture(self) -> None:
                """Load NN architecture."""
                pass

            def _define_features(self) -> None:
                """Define number of features as self.n_features: int."""
                self.n_features = 0

        cls.classifier = DummyClassifier()

    def test_create_instance(self):
        """Test create instance of the AbstractCategorizationModel."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class AbstractCategorizationModel"):
            _ = AbstractCategorizationModel()


@pytest.mark.skipif(
    not is_dependency_installed('timm')
    and not is_dependency_installed('torch')
    and not is_dependency_installed('transformers')
    and not is_dependency_installed('torchvision'),
    reason='Required dependencies not installed.',
)
class TestAbstractTextCategorizationModel(unittest.TestCase):
    """Test general functionality that uses nn.Module classes for text classification."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the classifier and test setup."""

        # DummyClassifier definition
        class DummyTextClassifier(AbstractTextCategorizationModel):
            def _valid(self) -> None:
                """Validate architecture sizes."""
                pass

            def _load_architecture(self) -> None:
                """Load NN architecture."""
                pass

            def _define_features(self) -> None:
                """Define number of features as self.n_features: int."""
                self.n_features = self.emb_dim

            def _output(self, text: torch.Tensor) -> List[FloatTensor]:
                """Define number of features as self.n_features: int."""
                features = torch.ones([1, self.input_dim, self.emb_dim], dtype=torch.int64)
                if self.uses_attention:
                    attention = torch.ones([1, self.input_dim, self.input_dim], dtype=torch.int64)
                    return [features, attention]
                else:
                    return [features]

        cls.classifier = DummyTextClassifier(input_dim=100, emb_dim=64)

    def test_create_instance(self):
        """Test create instance of the AbstractTextCategorizationModel."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class AbstractTextCategorizationModel"):
            _ = AbstractTextCategorizationModel()

    def test_n_features(self):
        """Test number of features."""
        assert self.classifier.n_features == 64

    @pytest.mark.parametrize('uses_attention', [True, False])
    def test_output(self):
        """Test output shape of the text classifier."""
        text = torch.ones([1, self.classifier.input_dim], dtype=torch.int64)
        result: List[FloatTensor] = self.classifier._output(text=text)
        if self.classifier.uses_attention:
            # In trainer/document_categorization.py, only NBOWSelfAttention and BERT use attention
            # (see docstring of the classes for details)
            assert len(result) == 2
            assert result[1].shape == (1, self.classifier.input_dim, self.classifier.input_dim)
        else:
            assert len(result) == 1
            assert result[0].shape == (1, self.classifier.input_dim, self.classifier.n_features)


@pytest.mark.skipif(
    not is_dependency_installed('timm')
    and not is_dependency_installed('torch')
    and not is_dependency_installed('transformers')
    and not is_dependency_installed('torchvision'),
    reason='Required dependencies not installed.',
)
@parameterized.parameterized_class(
    ('text_class', 'input_dim', 'emb_dim', 'n_heads', 'test_name'),
    [
        (NBOW, 100, 64, None, 'nbow'),
        (NBOWSelfAttention, 100, 64, 8, 'nbowselfattention'),
        (NBOWSelfAttention, 100, 64, 8, 'nbowselfattention-invalid'),
        (LSTM, 100, 64, None, 'lstm'),
    ],
)
class TestTextCategorizationModels(unittest.TestCase):
    """
    Test the currently four text modules available (NBOW, NBOWSelfAttention, LSTM, BERT).

    Each module takes a sequence of tokens as input and outputs a sequence of “hidden states”, i.e. one vector per
    input token. The size of each of the hidden states can be found with the module’s `n_features` parameter.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the Text Classifier."""
        cls.text_model = cls.text_class(
            input_dim=cls.input_dim, emb_dim=cls.emb_dim, n_heads=cls.n_heads, name=cls.test_name
        )

    @classmethod
    def tearDownClass(cls) -> None:
        """Delete the Text Classifier."""
        del cls.text_model

    def test_valid(self) -> None:
        """Test _valid method."""
        if self.test_name == 'nbowselfattention-invalid':
            with pytest.raises(ValueError, match='must be a multiple of n_heads'):
                _ = NBOWSelfAttention(input_dim=self.input_dim, emb_dim=self.emb_dim, n_heads=self.n_heads + 1)

    def test_n_features(self) -> None:
        """Test n_features."""
        if 'bert' in self.test_name:
            # The transformers library stores the number of features in the config dict so no need
            # to check the value
            assert self.text_model._feature_size in self.text_model.bert.config.to_dict()
            return
        bidirectional = self.text_model.bidirectional
        n_features = self.text_model.n_features
        emb_dim = self.text_model.emb_dim
        if bidirectional is None:
            # Only trainer/document_categorization.py::LSTM has a bidirectional option
            # (see docstring of the classes for details)
            assert n_features == emb_dim
        else:
            hid_dim = self.text_model.hid_dim
            assert n_features == hid_dim * 2 if bidirectional else hid_dim

    def test_output(self) -> None:
        """Test collect output of NN architecture."""
        text = torch.ones([1, self.text_model.input_dim], dtype=torch.int64)
        result: List[FloatTensor] = self.text_model._output(text=text)
        if self.text_model.uses_attention:
            # In trainer/document_categorization.py, only NBOWSelfAttention and BERT use attention
            # (see docstring of the classes for details)
            assert len(result) == 2
            assert result[1].shape == (1, self.text_model.input_dim, self.text_model.input_dim)
        else:
            assert len(result) == 1
            assert result[0].shape == (1, self.text_model.input_dim, self.text_model.n_features)

    def test_forward(self) -> None:
        """Test the computation performed at every call."""
        text = torch.ones([1, self.text_model.input_dim], dtype=torch.int64)
        _input = {'text': text}
        res: Dict[str, FloatTensor] = self.text_model(input=_input)
        assert 'features' in res
        assert res['features'].shape == (1, self.text_model.input_dim, self.text_model.n_features)
        if self.text_model.uses_attention:
            # In trainer/document_categorization.py, only NBOWSelfAttention and BERT use attention
            # (see docstring of the classes for details)
            assert 'attention' in res
            assert res['attention'].shape == (1, self.text_model.input_dim, self.text_model.input_dim)


@parameterized.parameterized_class(
    ('bert_name', 'n_epochs'),
    [
        ('prajjwal1/bert-tiny', 5),
        # ('german-nlp-group/electra-base-german-uncased',5),
        # ('bert-base-german-dbmdz-cased',3),  # commented out for passing on push in github actions (RAM limitations).
        # feel free to uncomment when testing locally
        # ('distilbert-base-german-cased',3),
        # ('albert-base-v2',3),
        # ('bert-base-chinese',3),
        # ('bert-base-german-cased',3),
        # ('google/mobilebert-uncased',3),
        # ('vinai/phobert-base',3)
    ],
)
@pytest.mark.skipif(
    not is_dependency_installed('timm')
    and not is_dependency_installed('torch')
    and not is_dependency_installed('transformers')
    and not is_dependency_installed('torchvision'),
    reason='Required dependencies not installed.',
)
class TestBertCategorizationModels(unittest.TestCase):
    """Test BERT models with different names."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the Data and Categorization Pipeline."""
        cls.training_prj = Project(id_=14392)
        cls.categorization_pipeline = CategorizationAI(cls.training_prj.categories)
        cls.category_1 = cls.training_prj.get_category_by_id(19827)
        cls.category_2 = cls.training_prj.get_category_by_id(19828)

    @classmethod
    def tearDownClass(cls) -> None:
        """Delete the Data and Categorization Pipeline."""
        os.remove(cls.categorization_pipeline.pipeline_path)
        del cls.training_prj
        del cls.categorization_pipeline
        del cls.category_1
        del cls.category_2

    def test_1_configure_dataset(self):
        """Test configuring the dataset for the training and testing of the model."""
        category_1_documents = self.category_1.documents()
        category_2_documents = self.category_2.documents()
        self.categorization_pipeline.documents = category_1_documents + category_2_documents
        assert all(doc.category is not None for doc in self.categorization_pipeline.documents)

        category_1_test_documents = self.category_1.test_documents()
        category_2_test_documents = self.category_2.test_documents()
        self.categorization_pipeline.test_documents = category_1_test_documents + category_2_test_documents
        assert all(doc.category is not None for doc in self.categorization_pipeline.documents)

    def test_2_configure_pipeline(self):
        """Test configuring the training pipeline of the model."""
        self.categorization_pipeline.category_vocab = self.categorization_pipeline.build_template_category_vocab()
        text_model = BERT(name=self.bert_name)
        bert_config = text_model.bert.config
        assert hasattr(bert_config, '_name_or_path')
        assert bert_config._name_or_path == self.bert_name
        self.categorization_pipeline.classifier = PageTextCategorizationModel(
            text_model=text_model,
            output_dim=len(self.categorization_pipeline.category_vocab),
        )
        self.categorization_pipeline.classifier.eval()

    def test_3_fit(self):
        """Test training the model."""
        self.categorization_pipeline.build_preprocessing_pipeline(use_image=False)
        self.categorization_pipeline.fit(n_epochs=self.n_epochs, optimizer={'name': 'Adam'})

    def test_4_categorize_document(self):
        """Test categorizing the Document."""
        test_document = self.training_prj.get_document_by_id(5589058)
        ground_truth_category = test_document.category
        test_document.set_category(self.training_prj.no_category)
        result = self.categorization_pipeline.categorize(document=test_document)
        assert isinstance(result, Document)
        assert result.category == ground_truth_category
        for page in result.pages():
            assert page.category == ground_truth_category
            assert page.maximum_confidence_category_annotation.confidence > 0.9
        # restore category attribute to not interfere with next tests
        test_document.set_category(result.category)

    def test_5_categorize_page(self):
        """Test categorizing the Document's Page."""
        test_document = self.training_prj.get_document_by_id(5589057)
        ground_truth_category = test_document.category
        test_page = WhitespaceTokenizer().tokenize(deepcopy(test_document)).pages()[0]
        # reset the category attribute to test that it can be categorized successfully
        test_page.set_category(self.training_prj.no_category)
        result = self.categorization_pipeline._categorize_page(test_page)
        assert isinstance(result, Page)
        assert result.category == ground_truth_category
        assert result.maximum_confidence_category_annotation.confidence > 0.9

    def test_6_save(self):
        """Test saving the model."""
        self.categorization_pipeline.pipeline_path = self.categorization_pipeline.save(reduce_weight=False)
        assert os.path.isfile(self.categorization_pipeline.pipeline_path)

    def test_7_load_ai_model(self):
        """Test loading a previously saved model."""
        loaded = CategorizationAI.load_model(self.categorization_pipeline.pipeline_path)
        test_document = self.training_prj.get_document_by_id(5589058)
        ground_truth_category = test_document.category
        test_document.set_category(self.training_prj.no_category)
        result = loaded.categorize(document=test_document)
        assert isinstance(result, Document)
        assert result.category == ground_truth_category
        for page in result.pages():
            assert page.category == ground_truth_category
            assert page.maximum_confidence_category_annotation.confidence > 0.9
        # restore category attribute to not interfere with next tests
        test_document.set_category(result.category)

    @unittest.skipIf(sys.version_info[:2] != (3, 8), reason='This AI can only be loaded on Python 3.8.')
    def test_8_upload_ai_model(self) -> None:
        """Upload the model."""
        assert os.path.isfile(self.categorization_pipeline.pipeline_path)
        try:
            model_id = upload_ai_model(
                ai_model_path=self.categorization_pipeline.pipeline_path, project_id=self.training_prj.id_
            )
            assert isinstance(model_id, int)
            updated = update_ai_model(model_id, ai_type='categorization', description='test_description')
            assert updated['description'] == 'test_description'
            updated = update_ai_model(model_id, ai_type='categorization', patch=False, description='test_description')
            assert updated['description'] == 'test_description'
            delete_ai_model(model_id, ai_type='categorization')
            url = get_create_ai_model_url(ai_type='categorization')
            session = konfuzio_session()
            not_found = session.get(url)
            assert not_found.status_code == 204
        except (HTTPError, ReadTimeout) as e:
            assert ('403' in str(e)) or ('500' in str(e)) or ('ReadTimeout' in str(e))


@parameterized.parameterized_class(
    ('text_class', 'tokenizer', 'image_class', 'image_class_version', 'n_epochs', 'test_quality'),
    [
        (NBOW, WhitespaceTokenizer, None, None, 20, True),
        (NBOWSelfAttention, WhitespaceTokenizer, None, None, 20, True),
        (LSTM, WhitespaceTokenizer, None, None, 1, False),
        (NBOW, ConnectedTextTokenizer, EfficientNet, 'efficientnet_b0', 1, False),
        # (None, None, EfficientNet, "efficientnet_b3", 5),  # commented out because of length of execution
        # (NBOWSelfAttention, ConnectedTextTokenizer, VGG, "vgg11", 2),
        # (LSTM, ConnectedTextTokenizer, VGG, "vgg13", 3),
        # (None, None, VGG, "vgg19", 2),
    ],
)
@pytest.mark.skipif(
    not is_dependency_installed('timm')
    and not is_dependency_installed('torch')
    and not is_dependency_installed('transformers')
    and not is_dependency_installed('torchvision'),
    reason='Required dependencies not installed.',
)
class TestCategorizationConfigurations(unittest.TestCase):
    """Test configurations that use both image and text model for Categorization."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the Data and Categorization Pipeline."""
        cls.training_prj = Project(id_=14392)
        cls.categorization_pipeline = CategorizationAI(cls.training_prj.categories)
        cls.category_1 = cls.training_prj.get_category_by_id(19827)
        cls.category_2 = cls.training_prj.get_category_by_id(19828)

    @classmethod
    def tearDownClass(cls) -> None:
        """Delete the Data and Categorization Pipeline."""
        os.remove(cls.categorization_pipeline.pipeline_path)
        del cls.training_prj
        del cls.categorization_pipeline
        del cls.category_1
        del cls.category_2

    def test_1_configure_dataset(self):
        """Test configuring the dataset for the training and testing of the model."""
        category_1_documents = self.category_1.documents()
        category_2_documents = self.category_2.documents()
        self.categorization_pipeline.documents = category_1_documents + category_2_documents
        assert all(doc.category is not None for doc in self.categorization_pipeline.documents)

        category_1_test_documents = self.category_1.test_documents()
        category_2_test_documents = self.category_2.test_documents()
        self.categorization_pipeline.test_documents = category_1_test_documents + category_2_test_documents
        assert all(doc.category is not None for doc in self.categorization_pipeline.documents)

    def test_2_configure_pipeline(self):
        """Test configuring the training pipeline of the model."""
        if self.text_class:
            self.categorization_pipeline.tokenizer = self.tokenizer()
            self.categorization_pipeline.text_vocab = self.categorization_pipeline.build_text_vocab()
        self.categorization_pipeline.category_vocab = self.categorization_pipeline.build_template_category_vocab()

        image_model = None
        text_model = None

        if self.image_class:
            image_model = self.image_class(name=self.image_class_version)
        if self.text_class:
            text_model = self.text_class(input_dim=len(self.categorization_pipeline.text_vocab))
        if not self.image_class:
            self.categorization_pipeline.classifier = PageTextCategorizationModel(
                text_model=text_model,
                output_dim=len(self.categorization_pipeline.category_vocab),
            )
        elif not self.text_class:
            self.categorization_pipeline.classifier = PageImageCategorizationModel(
                image_model=image_model,
                output_dim=len(self.categorization_pipeline.category_vocab),
            )
        else:
            multimodal_model = MultimodalConcatenate(
                n_image_features=image_model.n_features,
                n_text_features=text_model.n_features,
            )
            self.categorization_pipeline.classifier = PageMultimodalCategorizationModel(
                image_model=image_model,
                text_model=text_model,
                multimodal_model=multimodal_model,
                output_dim=len(self.categorization_pipeline.category_vocab),
            )

        self.categorization_pipeline.classifier.eval()

    def test_3_fit(self):
        """Test training the model."""
        if self.image_class:
            self.categorization_pipeline.build_preprocessing_pipeline(use_image=True)
        else:
            self.categorization_pipeline.build_preprocessing_pipeline(use_image=False)
        self.categorization_pipeline.fit(n_epochs=self.n_epochs, optimizer={'name': 'Adam'})

    def test_4_categorize_document(self):
        """Test categorizing the Document."""
        test_document = self.training_prj.get_document_by_id(5589058)
        ground_truth_category = test_document.category
        test_document.set_category(self.training_prj.no_category)
        result = self.categorization_pipeline.categorize(document=test_document)
        assert isinstance(result, Document)
        if self.test_quality:
            assert result.category == ground_truth_category
            for page in result.pages():
                assert page.category == ground_truth_category
        # restore category attribute to not interfere with next tests
        test_document.set_category(result.category)

    def test_5_categorize_page(self):
        """Test categorizing the Document's Page."""
        test_document = self.training_prj.get_document_by_id(5589057)
        ground_truth_category = test_document.category
        test_page = WhitespaceTokenizer().tokenize(deepcopy(test_document)).pages()[0]
        # reset the category attribute to test that it can be categorized successfully
        test_page.set_category(self.training_prj.no_category)
        result = self.categorization_pipeline._categorize_page(test_page)
        assert isinstance(result, Page)
        if self.test_quality:
            assert result.category == ground_truth_category

    def test_6_save(self):
        """Test saving the model."""
        self.categorization_pipeline.pipeline_path = self.categorization_pipeline.save(reduce_weight=False)
        assert os.path.isfile(self.categorization_pipeline.pipeline_path)

    def test_7_load_ai_model(self):
        """Test loading a previously saved model."""
        loaded = CategorizationAI.load_model(self.categorization_pipeline.pipeline_path)
        test_document = self.training_prj.get_document_by_id(5589058)
        ground_truth_category = test_document.category
        test_document.set_category(self.training_prj.no_category)
        result = loaded.categorize(document=test_document)
        assert isinstance(result, Document)
        if self.test_quality:
            assert result.category == ground_truth_category
            for page in result.pages():
                assert page.category == ground_truth_category
        # restore category attribute to not interfere with next tests
        test_document.set_category(result.category)

    @unittest.skipIf(sys.version_info[:2] != (3, 8), reason='This AI can only be loaded on Python 3.8.')
    def test_8_upload_ai_model(self) -> None:
        """Upload the model."""
        assert os.path.isfile(self.categorization_pipeline.pipeline_path)
        try:
            model_id = upload_ai_model(
                ai_model_path=self.categorization_pipeline.pipeline_path, project_id=self.training_prj.id_
            )
            assert isinstance(model_id, int)
            updated = update_ai_model(model_id, ai_type='categorization', description='test_description')
            assert updated['description'] == 'test_description'
            updated = update_ai_model(model_id, ai_type='categorization', patch=False, description='test_description')
            assert updated['description'] == 'test_description'
            delete_ai_model(model_id, ai_type='categorization')
            url = get_create_ai_model_url(ai_type='categorization')
            session = konfuzio_session()
            not_found = session.get(url)
            assert not_found.status_code == 204
        except (HTTPError, ReadTimeout) as e:
            assert ('403' in str(e)) or ('500' in str(e)) or ('Read timed out' in str(e)) or ('404' in str(e))


@pytest.mark.skipif(
    not is_dependency_installed('timm')
    and not is_dependency_installed('torch')
    and not is_dependency_installed('transformers')
    and not is_dependency_installed('torchvision'),
    reason='Required dependencies not installed.',
)
@unittest.skipIf(sys.version_info[:2] != (3, 8), reason='This AI can only be loaded on Python 3.8.')
def test_bert_in_multimodal_categorization_ai():
    """Test compatibility of BERT-based model and image-processing model for Categorization."""
    training_prj = Project(id_=14392)
    categorization_pipeline = CategorizationAI(training_prj.categories)
    category_1 = training_prj.get_category_by_id(19827)
    category_2 = training_prj.get_category_by_id(19828)
    category_1_documents = category_1.documents()
    category_2_documents = category_2.documents()
    categorization_pipeline.documents = category_1_documents + category_2_documents
    assert all(doc.category is not None for doc in categorization_pipeline.documents)
    category_1_test_documents = category_1.test_documents()
    category_2_test_documents = category_2.test_documents()
    categorization_pipeline.test_documents = category_1_test_documents + category_2_test_documents
    assert all(doc.category is not None for doc in categorization_pipeline.documents)
    categorization_pipeline.category_vocab = categorization_pipeline.build_template_category_vocab()
    text_model = BERT(name='prajjwal1/bert-tiny')
    bert_config = text_model.bert.config
    assert hasattr(bert_config, '_name_or_path')
    assert bert_config._name_or_path == 'prajjwal1/bert-tiny'
    image_model = VGG(name='vgg11')
    multimodal_model = MultimodalConcatenate(
        n_image_features=image_model.n_features,
        n_text_features=text_model.n_features,
    )
    categorization_pipeline.classifier = PageMultimodalCategorizationModel(
        image_model=image_model,
        text_model=text_model,
        multimodal_model=multimodal_model,
        output_dim=len(categorization_pipeline.category_vocab),
    )
    categorization_pipeline.classifier.eval()
    categorization_pipeline.build_preprocessing_pipeline(use_image=True)
    categorization_pipeline.fit(n_epochs=3, optimizer={'name': 'Adam'})
    test_document = training_prj.get_document_by_id(5589058)
    ground_truth_category = test_document.category
    test_document.set_category(training_prj.no_category)
    result = categorization_pipeline.categorize(document=test_document)
    assert isinstance(result, Document)
    assert result.category == ground_truth_category
    for page in result.pages():
        assert page.category == ground_truth_category
    test_document.set_category(result.category)
    categorization_pipeline.pipeline_path = categorization_pipeline.save(reduce_weight=False)
    assert os.path.isfile(categorization_pipeline.pipeline_path)
    loaded = CategorizationAI.load_model(categorization_pipeline.pipeline_path)
    test_document.set_category(training_prj.no_category)
    result = loaded.categorize(document=test_document)
    assert isinstance(result, Document)
    assert result.category == ground_truth_category
    for page in result.pages():
        assert page.category == ground_truth_category
        assert page.maximum_confidence_category_annotation.confidence > 0.9
    test_document.set_category(result.category)
    os.remove(categorization_pipeline.pipeline_path)


@pytest.mark.skipif(
    not is_dependency_installed('timm')
    and not is_dependency_installed('torch')
    and not is_dependency_installed('transformers')
    and not is_dependency_installed('torchvision'),
    reason='Required dependencies not installed.',
)
def test_build_categorization_ai() -> None:
    """Test building a Categorization AI by choosing an ImageModel and a TextModel."""
    project = Project(id_=None, project_folder=OFFLINE_PROJECT)
    categorization_pipeline = build_categorization_ai_pipeline(
        categories=project.categories,
        documents=[project.documents[0]],
        test_documents=[project.test_documents[1]],
        image_model_name=ImageModel.EfficientNetB0,
        text_model_name=TextModel.NBOWSelfAttention,
    )
    pipeline_path = categorization_pipeline.save(output_dir='custom_output_dir')
    with pytest.raises(ValueError, match='output_dir'):
        categorization_pipeline.save(path='path')
    CategorizationAI.load_model(pipeline_path)
    os.remove(pipeline_path)


@pytest.mark.skipif(
    not is_dependency_installed('timm')
    and not is_dependency_installed('torch')
    and not is_dependency_installed('transformers')
    and not is_dependency_installed('torchvision'),
    reason='Required dependencies not installed.',
)
def test_categorize_no_category_document():
    """Test categorization in case a NO_CATEGORY is predicted."""
    project = Project(id_=14392)
    test_document = project.documents[0]
    test_document.set_category(None)
    categorization_pipeline = build_categorization_ai_pipeline(
        categories=[test_document.category],
        documents=[project.documents[0]],
        test_documents=[project.test_documents[1]],
        image_model_name=ImageModel.EfficientNetB0,
        text_model_name=TextModel.NBOWSelfAttention,
    )
    categorization_pipeline.categorize(document=test_document, recategorize=True)
    assert test_document.category == project.no_category
