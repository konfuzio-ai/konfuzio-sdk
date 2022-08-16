# -*- coding: utf-8 -*-
"""Test to train an Extraction AI."""
import linecache
import logging
import math
import tracemalloc
import unittest

import pytest
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from konfuzio_sdk.data import Project, Document, AnnotationSet
from konfuzio_sdk.trainer.information_extraction import (
    DocumentAnnotationMultiClassModel,
    num_count,
    date_count,
    digit_count,
    space_count,
    special_count,
    vowel_count,
    upper_count,
    duplicate_count,
    substring_count,
    unique_char_count,
    strip_accents,
    count_string_differences,
    year_month_day_count,
    add_extractions_as_annotations,
    extraction_result_to_document,
    SeparateLabelsEntityMultiClassModel,
    DocumentEntityMulticlassModel,
    SeparateLabelsAnnotationMultiClassModel,
)
from konfuzio_sdk.api import upload_ai_model
from konfuzio_sdk.tokenizer.regex import RegexTokenizer, WhitespaceTokenizer
from konfuzio_sdk.tokenizer.base import ListTokenizer
from tests.variables import OFFLINE_PROJECT, TEST_DOCUMENT_ID
from konfuzio_sdk.samples import LocalTextProject

from konfuzio_sdk.evaluate import Evaluation

logger = logging.getLogger(__name__)

FEATURE_COUNT = 49


def display_top(snapshot, key_type='lineno', limit=30):
    """Trace memory allocations, see https://docs.python.org/3/library/tracemalloc.html."""
    snapshot = snapshot.filter_traces(
        (
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
            tracemalloc.Filter(False, "<logging>"),
            tracemalloc.Filter(False, "<tracemalloc>"),
        )
    )
    top_stats = snapshot.statistics(key_type)

    logger.info("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        logger.info("#%s: %s:%s: %.1f KiB" % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            logger.info('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        logger.info("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    logger.info("Total allocated size: %.1f KiB" % (total / 1024))


class TestNewSDKInformationExtraction(unittest.TestCase):
    """Test New SDK Information Extraction."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the Data and Pipeline."""
        cls.project = Project(id_=46, update=True)
        cls.pipeline = DocumentEntityMulticlassModel()
        cls.pipeline.category = cls.project.categories[0]
        documents = cls.project.documents
        cls.pipeline.test_documents = cls.pipeline.category.test_documents()
        documents = [doc for doc in documents if doc.category]
        assert len(documents) == 25
        cls.pipeline.tokenizer = WhitespaceTokenizer()
        for doc in documents:
            cls.pipeline.tokenizer.tokenize(doc)

        cls.pipeline.df_train, cls.pipeline.label_feature_list = cls.pipeline.feature_function(documents=documents)
        cls.pipeline.df_test, cls.pipeline.test_label_feature_list = cls.pipeline.feature_function(
            documents=cls.pipeline.test_documents
        )

        cls.pipeline.fit()
        # pipeline_path = cls.pipeline.save(output_dir='.')

    @unittest.skip(reason='Test run offline.')
    def test_sdk_vs_server_diff_44855(self):
        """Test to find differerences between SDK and server with test doc 44855 from project 46."""
        app_doc44855 = self.project.get_document_by_id(311644)
        for ann in app_doc44855.annotations(use_correct=False):
            ann.is_correct = True

        result = self.pipeline.extract(app_doc44855)
        virt_doc = extraction_result_to_document(app_doc44855, result)

        # comp_res = compare(app_doc44855, virt_doc)
        evaluation = Evaluation([(app_doc44855, virt_doc)], strict=True)
        evaluation.data.to_csv('test_eval_44855_app_sdk_1.csv')
        assert evaluation.f1(None) == 1.0

    @unittest.skip(reason='Test run offline.')
    def test_sdk_vs_server_diff_44855_tokenizer(self):
        """Test to find differerences between SDK tokenizer and server with test doc 44855 from project 46."""
        app_doc44855 = self.project.get_document_by_id(311644)
        for ann in app_doc44855.annotations(use_correct=False):
            ann.is_correct = True

        # result = self.pipeline.extract(app_doc44855)
        tokenized_doc = app_doc44855.__deepcopy__(None)
        self.pipeline.tokenizer.tokenize(tokenized_doc)
        # virt_doc = extraction_result_to_document(app_doc44855, result)

        # comp_res = compare(app_doc44855, virt_doc)
        evaluation = Evaluation([(app_doc44855, tokenized_doc)], strict=True)
        evaluation.data.to_csv('test_eval_44855_app_sdk_tokenizer_1.csv')
        # assert evaluation.f1(None) == 1.0

    @unittest.skip(reason='Test run offline.')
    def test_eval_44865_sdk(self):
        """Test sdk with reupploaded 1st test doc in project 46 (id=44855)."""
        app_doc44865_test_doc = self.project.get_document_by_id(314250)

        extraction_result = self.pipeline.extract(document=app_doc44865_test_doc)
        predicted_doc = extraction_result_to_document(app_doc44865_test_doc, extraction_result)
        eval_list = [(app_doc44865_test_doc, predicted_doc)]

        evaluation = Evaluation(eval_list, strict=True)

        evaluation.data.to_csv('test_eval_44865_sdk_1.csv')

    @unittest.skip(reason='Test run offline.')
    def test_eval_44866_sdk(self):
        """Test sdk with reupploaded 2nd test doc in project 46 (id=44856)."""
        app_doc44866_test_doc = self.project.get_document_by_id(314074)

        extraction_result = self.pipeline.extract(document=app_doc44866_test_doc)
        predicted_doc = extraction_result_to_document(app_doc44866_test_doc, extraction_result)
        eval_list = [(app_doc44866_test_doc, predicted_doc)]

        evaluation = Evaluation(eval_list, strict=True)

        evaluation.data.to_csv('test_eval_44866_sdk_1.csv')

    @unittest.skip(reason='Test run offline.')
    def test_eval_44867_sdk(self):
        """Test sdk with reupploaded 3rd test doc in project 46 (id=44857)."""
        app_doc44867_test_doc = self.project.get_document_by_id(314249)

        extraction_result = self.pipeline.extract(document=app_doc44867_test_doc)
        predicted_doc = extraction_result_to_document(app_doc44867_test_doc, extraction_result)
        eval_list = [(app_doc44867_test_doc, predicted_doc)]

        evaluation = Evaluation(eval_list, strict=True)

        evaluation.data.to_csv('test_eval_44867_sdk_1.csv')

    # @unittest.skip(reason='Test run offline.')
    # def test_eval_sdk(self):

    #     app_doc44865_eval = self.project.get_document_by_id(314250)
    #     app_doc44866_eval = self.project.get_document_by_id(314074)
    #     app_doc44867_eval = self.project.get_document_by_id(314249)
    #     app_docs = [app_doc44865_eval, app_doc44866_eval, app_doc44867_eval]

    #     eval_list = []
    #     for i, document in enumerate(app_docs):
    #         extraction_result = self.pipeline.extract(document=document)
    #         predicted_doc = extraction_result_to_document(document, extraction_result)
    #         eval_list.append((document, predicted_doc))

    #     evaluation = Evaluation(eval_list, strict=True)

    #     assert evaluation.f1(None) == 1.0 # 0.8546255506607929 # 0.8660714285714286
    #     # return self.evaluation

    # @unittest.skip(reason='Test run offline.')
    # def test_eval_app(self):

    #     app_doc44865_eval = self.project.get_document_by_id(314250)
    #     app_doc44866_eval = self.project.get_document_by_id(314074)
    #     app_doc44867_eval = self.project.get_document_by_id(314249)
    #     app_docs_eval = [app_doc44865_eval, app_doc44866_eval, app_doc44867_eval]
    #     app_doc44865 = self.project.get_document_by_id(314273)
    #     app_doc44866 = self.project.get_document_by_id(314272)
    #     app_doc44867 = self.project.get_document_by_id(314274)
    #     app_docs = [app_doc44865, app_doc44866, app_doc44867]
    #     eval_list = []
    #     for i, document in enumerate(app_docs_eval):
    #         eval_list.append((document, app_docs[i]))

    #     evaluation = Evaluation(eval_list, strict=True)
    #     # F1 0.8725868725868726
    #     # TP 113
    #     # FP 33
    #     # FN 0
    #     assert evaluation.f1(None) == 1.0


class TestSequenceInformationExtraction(unittest.TestCase):
    """Test to train an extraction Model for Documents."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the Data and Pipeline."""
        tracemalloc.start()
        cls.project = Project(id_=None, project_folder=OFFLINE_PROJECT)
        cls.pipeline = DocumentAnnotationMultiClassModel()
        display_top(tracemalloc.take_snapshot())

    def tearDown(self) -> None:
        """Print a Snapshot after running a test."""
        display_top(tracemalloc.take_snapshot())

    def test_1_configure_pipeline(self):
        """Make sure the Data and Pipeline is configured."""
        self.pipeline.tokenizer = WhitespaceTokenizer()
        self.pipeline.category = self.project.get_category_by_id(id_=63)
        self.pipeline.documents = self.pipeline.category.documents()[:5]
        self.pipeline.test_documents = self.pipeline.category.test_documents()[:1]

    def test_2_make_features(self):
        """Make sure the Data and Pipeline is configured."""
        self.pipeline.df_train, self.pipeline.label_feature_list = self.pipeline.feature_function(
            documents=self.pipeline.documents
        )
        self.pipeline.df_test, self.pipeline.test_label_feature_list = self.pipeline.feature_function(
            documents=self.pipeline.test_documents
        )

    def test_3_fit(self) -> None:
        """Start to train the Model."""
        self.pipeline.fit()

    def test_4_save_model(self):
        """Save the model."""
        self.pipeline_path = self.pipeline.save(output_dir=self.project.model_folder)

    def test_5_evaluate_model(self):
        """Evaluate the model."""
        self.pipeline.evaluate()

    def test_6_extract_test_document(self):
        """Extract a randomly selected Test Document."""
        test_document = self.project.get_document_by_id(TEST_DOCUMENT_ID)
        result = self.pipeline.extract(document=test_document)
        assert len(result['Brutto-Bezug']) > 0  # todo add more test for inference on data level

    @unittest.skip(reason='Test run offline.')
    def test_7_upload_ai_model(self):
        """Upload the model."""
        upload_ai_model(ai_model_path=self.pipeline_path, category_ids=[self.pipeline.category.id_])


class TestSequenceInformationSeparateLabelsExtraction(unittest.TestCase):
    """Test to train an extraction Model for Documents."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the Data and Pipeline."""
        cls.project = Project(id_=None, project_folder=OFFLINE_PROJECT)
        cls.pipeline = SeparateLabelsEntityMultiClassModel()

    def test_1_configure_pipeline(self):
        """Make sure the Data and Pipeline is configured."""
        self.pipeline.tokenizer = WhitespaceTokenizer()
        self.pipeline.category = self.project.get_category_by_id(id_=63)
        self.pipeline.documents = self.pipeline.category.documents()[:5]
        self.pipeline.test_documents = self.pipeline.category.test_documents()[:1]
        # todo have a separate test case for calculating features of offline documents
        for doc in self.pipeline.documents + self.pipeline.test_documents:
            doc.set_offline()

    def test_2_make_features(self):
        """Make sure the Data and Pipeline is configured."""
        self.pipeline.df_train, self.pipeline.label_feature_list = self.pipeline.feature_function(
            documents=self.pipeline.documents
        )
        self.pipeline.df_test, self.pipeline.test_label_feature_list = self.pipeline.feature_function(
            documents=self.pipeline.test_documents
        )

    def test_3_fit(self) -> None:
        """Start to train the Model."""
        self.pipeline.fit()

    def test_4_save_model(self):
        """Save the model."""
        self.pipeline_path = self.pipeline.save(output_dir=self.project.model_folder)

    def test_5_evaluate_model(self):
        """Evaluate the model."""
        self.pipeline.evaluate()

    def test_6_extract_test_document(self):
        """Extract a randomly selected Test Document."""
        test_document = self.project.get_document_by_id(TEST_DOCUMENT_ID)
        result = self.pipeline.extract(document=test_document)
        assert len(result['Brutto-Bezug']) > 0  # todo add more test for inference on data level

    @unittest.skip(reason='Test run offline.')
    def test_7_upload_ai_model(self):
        """Upload the model."""
        upload_ai_model(ai_model_path=self.pipeline_path, category_ids=[self.pipeline.category.id_])


class TestSequenceDocumentEntityMulticlassModelExtraction(unittest.TestCase):
    """Test to train an extraction Model for Documents."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the Data and Pipeline."""
        cls.project = Project(id_=None, project_folder=OFFLINE_PROJECT)
        cls.pipeline = DocumentEntityMulticlassModel()

    def test_1_configure_pipeline(self):
        """Make sure the Data and Pipeline is configured."""
        self.pipeline.tokenizer = WhitespaceTokenizer()
        self.pipeline.category = self.project.get_category_by_id(id_=63)
        self.pipeline.documents = self.pipeline.category.documents()[:5]
        self.pipeline.test_documents = self.pipeline.category.test_documents()[:1]
        # todo have a separate test case for calculating features of offline documents
        for doc in self.pipeline.documents + self.pipeline.test_documents:
            doc.set_offline()

    def test_2_make_features(self):
        """Make sure the Data and Pipeline is configured."""
        self.pipeline.df_train, self.pipeline.label_feature_list = self.pipeline.feature_function(
            documents=self.pipeline.documents
        )
        self.pipeline.df_test, self.pipeline.test_label_feature_list = self.pipeline.feature_function(
            documents=self.pipeline.test_documents
        )

    def test_3_fit(self) -> None:
        """Start to train the Model."""
        self.pipeline.fit()

    def test_4_save_model(self):
        """Save the model."""
        self.pipeline_path = self.pipeline.save(output_dir=self.project.model_folder)

    def test_5_evaluate_model(self):
        """Evaluate the model."""
        self.pipeline.evaluate()

    def test_6_extract_test_document(self):
        """Extract a randomly selected Test Document."""
        test_document = self.project.get_document_by_id(TEST_DOCUMENT_ID)
        result = self.pipeline.extract(document=test_document)
        assert len(result['Brutto-Bezug']) > 0  # todo add more test for inference on data level

    @unittest.skip(reason='Test run offline.')
    def test_7_upload_ai_model(self):
        """Upload the model."""
        upload_ai_model(ai_model_path=self.pipeline_path, category_ids=[self.pipeline.category.id_])


class TestSequenceSeparateLabelsAnnotationMultiClassModelExtraction(unittest.TestCase):
    """Test to train an extraction Model for Documents."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the Data and Pipeline."""
        cls.project = Project(id_=None, project_folder=OFFLINE_PROJECT)
        cls.pipeline = SeparateLabelsAnnotationMultiClassModel()

    def test_1_configure_pipeline(self):
        """Make sure the Data and Pipeline is configured."""
        self.pipeline.tokenizer = WhitespaceTokenizer()
        self.pipeline.category = self.project.get_category_by_id(id_=63)
        self.pipeline.documents = self.pipeline.category.documents()[:5]
        self.pipeline.test_documents = self.pipeline.category.test_documents()[:1]

    def test_2_make_features(self):
        """Make sure the Data and Pipeline is configured."""
        self.pipeline.df_train, self.pipeline.label_feature_list = self.pipeline.feature_function(
            documents=self.pipeline.documents
        )
        self.pipeline.df_test, self.pipeline.test_label_feature_list = self.pipeline.feature_function(
            documents=self.pipeline.test_documents
        )

    def test_3_fit(self) -> None:
        """Start to train the Model."""
        self.pipeline.fit()

    def test_4_save_model(self):
        """Save the model."""
        self.pipeline_path = self.pipeline.save(output_dir=self.project.model_folder)

    def test_5_evaluate_model(self):
        """Evaluate the model."""
        self.pipeline.evaluate()

    def test_6_extract_test_document(self):
        """Extract a randomly selected Test Document."""
        test_document = self.project.get_document_by_id(TEST_DOCUMENT_ID)
        result = self.pipeline.extract(document=test_document)
        # todo: this extract method should use a Document
        assert len(result['Brutto-Bezug']) > 0  # todo add more test for inference on data level

    @unittest.skip(reason='Test run offline.')
    def test_7_upload_ai_model(self):
        """Upload the model."""
        upload_ai_model(ai_model_path=self.pipeline_path, category_ids=[self.pipeline.category.id_])


class TestFindRegexSeparateLabelsAnnotationMultiClassModelExtraction(unittest.TestCase):
    """Test to train an extraction Model for Documents."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the Data and Pipeline."""
        cls.project = Project(id_=None, project_folder=OFFLINE_PROJECT)
        cls.pipeline = SeparateLabelsAnnotationMultiClassModel()

    def test_1_configure_pipeline(self):
        """Make sure the Data and Pipeline is configured."""
        self.pipeline.tokenizer = ListTokenizer(tokenizers=[])
        self.pipeline.category = self.project.get_category_by_id(id_=63)
        self.pipeline.documents = self.pipeline.category.documents()[:5]
        self.pipeline.test_documents = self.pipeline.category.test_documents()[:1]

    def test_2_find_regex(self):
        """Fit the tokenizer."""
        for label in self.pipeline.category.labels:
            for regex in label.find_regex(category=self.pipeline.category):
                self.pipeline.tokenizer.tokenizers.append(RegexTokenizer(regex=regex))

    @unittest.skip(reason='We do not achieve this at the moment.')
    def test_3_perfect_tokenizer_coverage(self):
        """Check 100% tokenizer coverage."""
        tokenizer_eval = self.pipeline.tokenizer.evaluate_dataset(self.pipeline.test_documents)
        for document in self.pipeline.test_documents:
            assert tokenizer_eval.tokenizer(search=document) == len(document.spans)  # currently 34==35

    def test_4_make_features(self):
        """Make sure the Data and Pipeline is configured."""
        self.pipeline.df_train, self.pipeline.label_feature_list = self.pipeline.feature_function(
            documents=self.pipeline.documents
        )
        self.pipeline.df_test, self.pipeline.test_label_feature_list = self.pipeline.feature_function(
            documents=self.pipeline.test_documents
        )

    def test_5_fit(self) -> None:
        """Start to train the Model."""
        self.pipeline.fit()

    def test_6_save_model(self):
        """Save the model."""
        self.pipeline_path = self.pipeline.save(output_dir=self.project.model_folder)

    def test_7_evaluate_model(self):
        """Evaluate the model."""
        self.pipeline.evaluate_full()

    @unittest.skip(reason='We do not achieve this at the moment.')
    def test_8_perfect_evaluation_f1(self):
        """Check 100% strict evaluation score."""
        for document in self.pipeline.test_documents:
            assert self.pipeline.evaluation.f1(search=document) == 1

    def test_9_extract_test_document(self):
        """Extract a randomly selected Test Document."""
        test_document = self.project.get_document_by_id(TEST_DOCUMENT_ID)
        result = self.pipeline.extract(document=test_document)
        # todo: this extract method should use a Document
        assert len(result['Brutto-Bezug']) > 0  # todo add more test for inference on data level

    @unittest.skip(reason='Test run offline.')
    def test_10_upload_ai_model(self):
        """Upload the model."""
        upload_ai_model(ai_model_path=self.pipeline_path, category_ids=[self.pipeline.category.id_])


class TestInformationExtraction(unittest.TestCase):
    """Test to train an extraction Model for Documents."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the Data and Pipeline."""
        cls.project = Project(id_=None, project_folder=OFFLINE_PROJECT)

    def test_extraction_without_tokenizer(self):
        """Test extraction on a Document."""
        pipeline = DocumentAnnotationMultiClassModel()
        document = self.project.get_document_by_id(TEST_DOCUMENT_ID)
        with pytest.raises(AttributeError) as einfo:
            pipeline.extract(document)
        assert 'missing Tokenizer' in str(einfo.value)

    def test_extraction_without_clf(self):
        """Test extraction without classifier."""
        document = self.project.get_document_by_id(TEST_DOCUMENT_ID)
        pipeline = DocumentAnnotationMultiClassModel()
        pipeline.tokenizer = WhitespaceTokenizer()
        with pytest.raises(AttributeError) as einfo:
            pipeline.extract(document)
        assert 'does not provide a Label Classifier' in str(einfo.value)

    def test_feature_function(self):
        """Test to generate features."""
        document = self.project.get_document_by_id(TEST_DOCUMENT_ID)
        pipeline = DocumentAnnotationMultiClassModel()
        pipeline.tokenizer = WhitespaceTokenizer()
        features, feature_names, errors = pipeline.features(document)
        assert len(feature_names) == 270  # todo investigate if all features are calculated correctly, see #9289

    def test_extract_with_unfitted_clf(self):
        """Test to extract a Document."""
        document = self.project.get_document_by_id(TEST_DOCUMENT_ID)
        pipeline = DocumentAnnotationMultiClassModel()
        pipeline.tokenizer = WhitespaceTokenizer()
        pipeline.clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
        with pytest.raises(AttributeError) as einfo:
            _, _ = pipeline.extract(document)
        assert 'instance is not fitted yet' in str(einfo.value)

    def test_extract_with_fitted_clf(self):
        """Test to extract a Document."""
        document = self.project.get_document_by_id(TEST_DOCUMENT_ID)
        pipeline = DocumentAnnotationMultiClassModel()
        pipeline.tokenizer = WhitespaceTokenizer()
        pipeline.clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
        X, y = make_classification(
            n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False
        )
        pipeline.clf.fit(X, y)
        with pytest.raises(KeyError) as einfo:
            pipeline.extract(document)
        assert 'do not match the features of the pipeline' in str(einfo.value)

    def test_extract_with_correctly_fitted_clf(self):
        """Test to extract a Document."""
        document = self.project.get_document_by_id(TEST_DOCUMENT_ID)
        pipeline = DocumentAnnotationMultiClassModel()
        pipeline.tokenizer = WhitespaceTokenizer()
        pipeline.clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
        X, y = make_classification(
            n_samples=1000, n_features=2, n_informative=2, n_redundant=0, random_state=0, shuffle=False
        )
        pipeline.clf.fit(X, y)
        pipeline.label_feature_list = ['start_offset', 'end_offset']
        pipeline.category = document.category
        pipeline.extract(document)
        # todo
        # virtual_doc = extraction_result_to_document(document, extraction_result)
        # assert len(virtual_doc.annotations(use_correct=False)) > 0
        # assert len(virtual_doc.annotation_sets()) > 0

    def test_feature_function_with_label_limit(self):
        """Test to generate features with many spatial features.."""
        document = self.project.get_document_by_id(TEST_DOCUMENT_ID)
        pipeline = DocumentAnnotationMultiClassModel()
        pipeline.no_label_limit = 0.5
        pipeline.tokenizer = WhitespaceTokenizer()
        pipeline.n_nearest = 10
        features, feature_names, errors = pipeline.features(document)
        assert len(feature_names) == 1102  # todo investigate if all features are calculated correctly, see #9289
        assert features['is_correct'].sum() == 19
        assert features['revised'].sum() == 2


class TestAddExtractionAsAnnotation(unittest.TestCase):
    """Test add an Extraction result as Annotation to a Document."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set LocalTextProject with example prediction."""
        cls.project = LocalTextProject()
        cls.category = cls.project.get_category_by_id(1)
        cls.label_set = cls.project.get_label_set_by_id(3)
        cls.label = cls.project.get_label_by_id(4)
        cls.sample_document = cls.project.local_none_document
        # example of an extraction
        cls.extraction = {
            'Start': 15,
            'End': 20,
            'Accuracy': 0.2,
            'page_index': 0,
            'x0': 10,
            'x1': 20,
            'y0': 10,
            'y1': 20,
            'top': 200,
            'bottom': 210,
        }
        cls.extraction_df = pd.DataFrame(data=[cls.extraction])

    def test_1_add_extraction_to_sample_document(self):
        """Test add extraction to the sample document."""
        annotation_set = AnnotationSet(id_=99, document=self.sample_document, label_set=self.label_set)

        add_extractions_as_annotations(
            extractions=self.extraction_df,
            document=self.sample_document,
            label=self.label,
            label_set=self.label_set,
            annotation_set=annotation_set,
        )

        assert len(self.sample_document.annotations(use_correct=False)) == 1

    def test_2_status_of_annotation_created(self):
        """Test status of te annotation created in the sample document."""
        annotation = self.sample_document.annotations(use_correct=False)[0]
        assert not annotation.is_correct
        assert not annotation.revised

    def test_3_number_of_spans_of_annotation_created(self):
        """Test number of Spans in the annotation created in the sample document."""
        annotation = self.sample_document.annotations(use_correct=False)[0]
        assert len(annotation.spans) == 1

    def test_4_span_attributes_of_annotation_created(self):
        """Test attributes of the span in the annotation created in the sample document."""
        annotation = self.sample_document.annotations(use_correct=False)[0]
        assert annotation.spans[0].start_offset == self.extraction_df.loc[0, 'Start']
        assert annotation.spans[0].end_offset == self.extraction_df.loc[0, 'End']
        # The document used does not have bounding boxes, so we cannot have the coordinates
        assert annotation.spans[0].offset_string == 'pizza'
        assert annotation.spans[0].x0 is None

    def test_add_empty_extraction_to_empty_document(self):
        """Test add empty extraction to an empty document - no text."""
        document = Document(text='', project=self.project, category=self.category)
        annotation_set_1 = AnnotationSet(id_=97, document=document, label_set=self.label_set)
        extraction_df = pd.DataFrame()

        add_extractions_as_annotations(
            extractions=extraction_df,
            document=document,
            label=self.label,
            label_set=self.label_set,
            annotation_set=annotation_set_1,
        )
        assert document.annotations(use_correct=False) == []

    def test_add_empty_extraction_to_document(self):
        """Test add empty extraction to a document."""
        document = Document(text='Hello', project=self.project, category=self.category)
        annotation_set_1 = AnnotationSet(id_=98, document=document, label_set=self.label_set)
        extraction_df = pd.DataFrame()

        add_extractions_as_annotations(
            extractions=extraction_df,
            document=document,
            label=self.label,
            label_set=self.label_set,
            annotation_set=annotation_set_1,
        )
        assert document.annotations(use_correct=False) == []

    def test_add_extraction_to_empty_document(self):
        """Test add extraction to an empty document - no text."""
        document = Document(text='', project=self.project, category=self.category)
        annotation_set_1 = AnnotationSet(id_=1, document=document, label_set=self.label_set)

        add_extractions_as_annotations(
            extractions=self.extraction_df,
            document=document,
            label=self.label,
            label_set=self.label_set,
            annotation_set=annotation_set_1,
        )

        annotation = document.annotations(use_correct=False)[0]
        # The document used is an empty document, therefore it does not have text or bounding boxes,
        # so we cannot have the offset string or the coordinates
        assert annotation.spans[0].offset_string is None
        assert annotation.spans[0].x0 is None
        assert annotation.spans[0].y0 is None
        assert annotation.spans[0].x1 is None
        assert annotation.spans[0].y1 is None

    def test_add_invalid_extraction(self):
        """Test add an invalid extraction - missing fields."""
        document = Document(project=self.project, category=self.category, text='From 14.12.2021 to 1.1.2022.')
        annotation_set_1 = AnnotationSet(id_=1, document=document, label_set=self.label_set)
        extraction = {'start_offset': 5, 'end_offset': 10}

        extraction_df = pd.DataFrame(data=[extraction])

        with self.assertRaises(ValueError) as context:
            add_extractions_as_annotations(
                extractions=extraction_df,
                document=document,
                label=self.label,
                label_set=self.label_set,
                annotation_set=annotation_set_1,
            )
            assert 'Extraction do not contain all required fields' in context.exception


class TestExtractionToDocument(unittest.TestCase):
    """Test the conversion of the Extraction results from the AI to a Document."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set LocalTextProject with example predictions."""
        cls.project = LocalTextProject()
        cls.category = cls.project.get_category_by_id(1)
        cls.label_set_0 = cls.project.get_label_set_by_id(2)
        # cls.label_set_1 = cls.project.get_label_set_by_id(3)
        cls.label_0 = cls.project.get_label_by_id(4)
        cls.label_1 = cls.project.get_label_by_id(5)
        cls.sample_document = cls.project.local_none_document
        # label_set_1 = LabelSet(id_=10, name='label set name', project=project, categories=[category])

        # example 1 of an extraction
        cls.extraction_1 = {
            'Start': 5,
            'End': 10,
            'Accuracy': 0.2,
            'page_index': 0,
            'x0': 10,
            'x1': 20,
            'y0': 10,
            'y1': 20,
            'top': 200,
            'bottom': 210,
        }

        # example 2 of an extraction
        cls.extraction_2 = {
            'Start': 15,
            'End': 20,
            'Accuracy': 0.2,
            'page_index': 0,
            'x0': 20,
            'x1': 30,
            'y0': 20,
            'y1': 30,
            'top': 200,
            'bottom': 210,
        }

    def test_empty_extraction_result_to_document(self):
        """Test conversion of an empty AI output to a Document."""
        virtual_doc = extraction_result_to_document(self.sample_document, extraction_result={})
        assert virtual_doc.annotations(use_correct=False) == []

    def test_empty_extraction_result_to_empty_document(self):
        """Test conversion of an empty AI output to an empty Document."""
        document = Document(text='', project=self.project, category=self.category)
        virtual_doc = extraction_result_to_document(document, extraction_result={})
        assert virtual_doc.annotations(use_correct=False) == []

    def test_extraction_result_with_empty_dataframe_to_document(self):
        """Test conversion of an AI output with an empty dataframe to a Document."""
        document = Document(project=self.project, category=self.category, text='From 14.12.2021 to 1.1.2022.')
        virtual_doc = extraction_result_to_document(
            document, extraction_result={'label in category label set': pd.DataFrame()}
        )
        assert virtual_doc.annotations(use_correct=False) == []

    def test_extraction_result_with_empty_dictionary_to_document(self):
        """Test conversion of an AI output with an empty dictionary to a Document."""
        virtual_doc = extraction_result_to_document(self.sample_document, extraction_result={'LabelSetName': {}})
        assert virtual_doc.annotations(use_correct=False) == []

    def test_extraction_result_with_empty_list_to_document(self):
        """Test conversion of an AI output with an empty list to a Document."""
        virtual_doc = extraction_result_to_document(self.sample_document, extraction_result={'LabelSetName': []})
        assert virtual_doc.annotations(use_correct=False) == []

    def test_extraction_result_with_empty_list_to_empty_document(self):
        """Test conversion of an AI output with an empty list to an empty Document."""
        virtual_doc = extraction_result_to_document(self.sample_document, extraction_result={'LabelSetName': []})
        assert virtual_doc.annotations(use_correct=False) == []

    def test_extraction_result_for_category_label_set(self):
        """Test conversion of an AI output with an extraction for a label in the Category Label Set."""
        extraction_result = {'DefaultLabelName': pd.DataFrame(data=[self.extraction_1])}
        virtual_doc = extraction_result_to_document(self.sample_document, extraction_result=extraction_result)
        assert len(virtual_doc.annotations(use_correct=False)) == 1
        annotation = virtual_doc.annotations(use_correct=False)[0]
        assert annotation.label.name == 'DefaultLabelName'
        assert annotation.label_set == self.project.get_label_set_by_name('CategoryName')

    def test_extraction_result_for_label_set_with_single_annotation_set(self):
        """Test conversion of an AI output with multiple extractions for a label in a Label Set - 1 Annotation Set."""
        extraction_result = {'LabelSetName': {'LabelName': pd.DataFrame(data=[self.extraction_1, self.extraction_2])}}
        virtual_doc = extraction_result_to_document(self.sample_document, extraction_result=extraction_result)
        assert len(virtual_doc.annotations(use_correct=False)) == 2
        annotation_1 = virtual_doc.annotations(use_correct=False)[0]
        annotation_2 = virtual_doc.annotations(use_correct=False)[1]
        assert annotation_1.label.name == annotation_2.label.name == 'LabelName'
        assert annotation_1.label_set == annotation_2.label_set == self.project.get_label_set_by_name('LabelSetName')
        assert annotation_1.annotation_set.id_ == annotation_2.annotation_set.id_

    def test_extraction_result_for_label_set_with_multiple_annotation_sets(self):
        """Test conversion of an AI output with extractions for a label in a Label Set for different Annotation Sets."""
        extraction_result = {
            'LabelSetName': [
                {'LabelName': pd.DataFrame(data=[self.extraction_1])},
                {'LabelName': pd.DataFrame(data=[self.extraction_2])},
            ]
        }
        virtual_doc = extraction_result_to_document(self.sample_document, extraction_result=extraction_result)
        assert len(virtual_doc.annotations(use_correct=False)) == 2
        annotation_1 = virtual_doc.annotations(use_correct=False)[0]
        annotation_2 = virtual_doc.annotations(use_correct=False)[1]
        assert annotation_1.label.name == annotation_2.label.name == 'LabelName'
        assert annotation_1.label_set == annotation_2.label_set == self.project.get_label_set_by_name('LabelSetName')
        assert annotation_1.annotation_set.id_ != annotation_2.annotation_set.id_


def test_feat_num_count():
    """Test string conversion."""
    # Debug code for df: df[df[self.label_feature_list].isin([np.nan, np.inf, -np.inf]).any(1)]
    error_string_1 = '10042020200917e747'
    res = num_count(error_string_1)
    assert not math.isinf(res)

    error_string_2 = '26042020081513e749'
    res = num_count(error_string_2)
    assert not math.isinf(res)


def test_date_count():
    """Test string conversion."""
    result = date_count("01.01.2010")
    assert result == 1


def test_date_count_right_format_wrong_date():
    """Test string conversion."""
    date_count("aa.dd.dhsfkbhsdf")


def test_date_count_index_error():
    """Test string conversion."""
    date_count("ad")


def test_digit_count():
    """Test string conversion."""
    result = digit_count("123456789ABC")
    assert result == 9


def test_num_count_wrong_format():
    """Test string conversion."""
    num_count("word")


def test_space_count():
    """Test string conversion."""
    result = space_count("1 2 3 4 5 ")
    assert result == 5


def test_space_count_with_tabs():
    """Test string conversion."""
    result = space_count("\t")
    assert result == 4


def test_special_count():
    """Test string conversion."""
    result = special_count("!_:ThreeSpecialChars")
    assert result == 3


def test_vowel_count():
    """Test string conversion."""
    result = vowel_count("vowel")
    assert result == 2


def test_upper_count():
    """Test string conversion."""
    result = upper_count("UPPERlower!")
    assert result == 5


def test_num_count():
    """Test string conversion."""
    result = num_count("1.500,34")
    assert result == 1500.34


def test_duplicate_count():
    """Test string conversion."""
    result = duplicate_count("AAABBCCDDE")
    assert result == 9


def test_substring_count():
    """Test string conversion."""
    result = substring_count(["Apple", "Annaconda"], "a")
    assert result == [1, 3]


def test_unique_char_count():
    """Test string conversion."""
    result = unique_char_count("12345678987654321")
    assert result == 9


def test_accented_char_strip_and_count():
    """Test string conversion."""
    l_test = ['Hallà', 'àèìòùé', 'Nothing']

    l_stripped = [strip_accents(s) for s in l_test]
    assert l_stripped[0] == 'Halla'
    assert l_stripped[1] == 'aeioue'
    assert l_stripped[2] == 'Nothing'

    l_diff = [count_string_differences(s1, s2) for s1, s2 in zip(l_test, l_stripped)]
    assert l_diff[0] == 1
    assert l_diff[1] == 6
    assert l_diff[2] == 0


test_data_year_month_day_count = [
    (['1. November 2019'], ([2019], [11], [1]), 51453),
    (['1.Oktober2019 '], ([2019], [10], [1]), 51452),
    (['1. September 2019'], ([2019], [9], [1]), 51451),
    (['1.August2019'], ([2019], [8], [1]), 51450),
    (['23.0919'], ([2019], [9], [23]), 51449),
    (['011019'], ([2019], [10], [1]), 51449),
    (['0210.19'], ([2019], [10], [2]), 51449),
    (['1. Mai 2019'], ([2019], [5], [1]), 51448),
    (['16.122019'], ([2019], [12], [16]), 50954),
    (['07092012'], ([2012], [9], [7]), 0),
    (['14132020'], ([0], [0], [0]), 0),
    (['250785'], ([1985], [7], [25]), 0),
    (['1704.2020'], ([2020], [4], [17]), 0),
    (['/04.12.'], ([0], [12], [4]), 47776),
    (['04.12./'], ([0], [12], [4]), 47776),
    (['02-05-2019'], ([2019], [5], [2]), 54858),
    (['1. Oktober2019'], ([2019], [10], [1]), 0),
    (['13 Mar 2020'], ([2020], [3], [13]), 37527),
    (['30, Juni'], ([0], [6], [30]), 53921),
    (['2019-06-01'], ([2019], [6], [1]), 38217),
    (['30 Sep 2019'], ([2019], [9], [30]), 39970),
    (['July 1, 2019'], ([2019], [7], [1]), 38208),
    (['(29.03.2018)'], ([2018], [3], [29]), 51432),
    (['03,12.'], ([0], [12], [3]), 51439),
    (['23,01.'], ([0], [1], [23]), 51430),
    (['03,07,'], ([0], [0], [0]), 51435),
    (['05.09;'], ([0], [9], [5]), 51436),
    (['24,01.'], ([0], [1], [24]), 51430),
    (['15.02.‚2019'], ([2019], [2], [15]), 54970),
]


@pytest.mark.parametrize("test_input, expected, document_id", test_data_year_month_day_count)
def test_dates(test_input, expected, document_id):
    """Test string conversion."""
    res = year_month_day_count(test_input)
    assert res[0][0] == expected[0][0]
    assert res[1][0] == expected[1][0]
    assert res[2][0] == expected[2][0]


test_data_num = [
    ('3,444, 40+', 3444.4, 51438),
    ('5.473,04S', -5473.04, 51443),
    (' 362,85H', 362.85, 51443),
    ('3,288,50', 3288.50, 45551),
    ('1,635,74', 1635.74, 51426),
    ('0,00', 0, 514449),
    ('331.500', 331500, 57398),
    ('4.361.163', 4361163, 57268),
    ('4.361.163-', -4361163, 0),
    ('aghdabh', 0, 0),
    ('2019-20-12', 20192012.0, 0),
]


@pytest.mark.parametrize("test_input, expected, document_id", test_data_num)
def test_num(test_input, expected, document_id):
    """Test string conversion."""
    assert num_count(test_input) == expected


#
# """Test models in models_labels_multiclass."""
# import logging
# import unittest
#
# import pandas
# import pandas as pd
# import pytest
# from konfuzio_sdk.data import Project, Category, Document, Label, LabelSet, AnnotationSet, Annotation, Span
# from pympler import asizeof
#
# from konfuzio.wrapper import get_bboxes, is_valid_extraction_dataframe
#
# from konfuzio_sdk.pipelines.extraction_ai import DocumentAnnotationMultiClassModel
#
# logger = logging.getLogger(__name__)
#
#
# # 1) dataset creation/cleaning (458).
# # 2) Searching existing testcases and writing the module/name to this file
# # 3) Complete testcases, what is missing.
#
# # Tests should inside the functions and test should indicate which variables are changed.
# # Tests should verify that overall for TestProject (46, Payslip / Trainticket) is correct (e.g. 100% Quality)
#
#
# @unittest.skip("Empty document crashes n-nearest.")
# def test_build_with_labels_only_associated_with_category():
#     """Test training with a document where the labels belong to the Category Label Set - no other Label Sets."""
#     project = Project(id_=46)
#     category = project.get_category_by_id(63)
#     category.label_sets = [project.get_label_set_by_id(63)]
#     label = category.label_sets[0].labels[0]
#
#     project._documents = [x for x in project._documents if x.id_ == 44834]
#
#     # Training documents only with Annotations for Label "label"
#     for document in category.documents():
#         document._annotations = document.annotations(use_correct=True, label=label)
#
#     extraction_ai = DocumentAnnotationMultiClassModel(category=category)
#     extraction_ai.build()
#     result = extraction_ai.extract(text=document.text, bbox=document.get_bbox(), pages=document.pages)
#
#     assert set(extraction_ai.df_train.label_text.unique()) == set([label.name, extraction_ai.no_label.name])
#     assert len(result.keys()) == 1
#     assert isinstance(result[label.name], pd.DataFrame)
#
#
# def test_build_with_label_set_without_multiple_annotation_sets():
#     """Test training with a document where the labels belong to a Label Set without multiple Annotation Sets."""
#     project = Project(id_=46)
#     category = project.get_category_by_id(63)
#     label_set = project.get_label_set_by_id(3707)
#     category.label_sets = [label_set]
#     label = project.get_label_by_id(12503)
#     assert label in category.label_sets[0].labels
#
#     project._documents = [x for x in project._documents if x.id_ == 44834]
#
#     # Training documents only with Annotations for Label "label"
#     for document in category.documents():
#         document._annotations = document.annotations(use_correct=True, label=label)
#
#     extraction_ai = DocumentAnnotationMultiClassModel(category=category)
#     extraction_ai.build()
#     result = extraction_ai.extract(text=document.text, bbox=document.get_bbox(), pages=document.pages)
#
#     assert len(result.keys()) == 1
#     assert isinstance(result[label_set.name], dict)
#     assert label.name in result[label_set.name].keys()
#     assert isinstance(result[label_set.name][label.name], pd.DataFrame)
#
#
# def test_build_label_set_with_multiple_annotation_sets():
#     """Test training with a document where the labels belong to a Label Set with multiple Annotation Sets."""
#     project = Project(id_=46)
#     category = project.get_category_by_id(63)
#     label_set = project.get_label_set_by_id(64)
#     category.label_sets = [label_set]
#     label = project.get_label_by_id(861)
#     assert label in category.label_sets[0].labels
#
#     project._documents = [x for x in project._documents if x.id_ == 44834]
#
#     # Training documents only with Annotations for Label "label"
#     for document in category.documents():
#         document._annotations = document.annotations(use_correct=True, label=label)
#
#     extraction_ai = DocumentAnnotationMultiClassModel(category=category)
#     extraction_ai.build()
#     result = extraction_ai.extract(text=document.text, bbox=document.get_bbox(), pages=document.pages)
#
#     assert len(result.keys()) == 1
#     assert isinstance(result[label_set.name], list)
#     assert isinstance(result[label_set.name][0], dict)
#     assert isinstance(result[label_set.name][0][label.name], pd.DataFrame)
#
#
# @unittest.skip("Needs revision and implementation.")
# def test_build_label_sets_with_shared_labels():
#     """
#     Test training with a document where a Label is shared between 2 Label Sets.
#
#     The first Label Set (first ID) does not have the option for multiple Annotation Sets and the second has the option
#     for multiple Annotation Sets.
#     The order is important.
#
#     Both Label Sets should have results.
#     TODO: atm the first Label Set without the option for multiple Annotation Sets,
#      takes all the results for the shared
#         Label.
#     """
#     project = Project(id_=46)
#     category = project.get_category_by_id(63)
#     label_set_1 = project.get_label_set_by_id(64)
#     label_set_1.has_multiple_annotation_sets = False
#     label_set_2 = project.get_label_set_by_id(3706)
#     category.label_sets = [label_set_1, label_set_2]
#     label = project.get_label_by_id(861)
#     assert label in category.label_sets[0].labels
#
#     # Training documents only with Annotations for Label "label"
#     for document in category.documents():
#         document._annotations = document.annotations(use_correct=True, label=label)
#
#     extraction_ai = DocumentAnnotationMultiClassModel(category=category)
#     extraction_ai.build()
#     result = extraction_ai.extract(text=document.text, bbox=document.get_bbox(), pages=document.pages)
#
#     assert len(result.keys()) == 2
#     assert isinstance(result[label_set_1.name], dict)
#     assert isinstance(result[label_set_2.name], list)
#     assert not result[label_set_1.name][label.name].empty
#     assert len(result[label_set_2.name]) > 0
#     assert set(extraction_ai.label_set_clf.classes_.tolist()) == set([label_set_1.name, label_set_2.name, 'No'])
#
#
# @unittest.skip("Separate labels not supported")
# class SeparateLabelsTestDocumentEntityMulticlassModel(unittest.TestCase):
#     """
#     Test separate_labels function.
#
#     Existing test cases:
#
#     test_data.TestAPIDataSetup.test_separate_labels
#     => Tests training and the number of labels after separation using project 46
#
#     Missing test:
#     - number of annotations per label after separation
#     - Test label IDs or names instead of length of label list
#     - Test annotation_set label IDs or names instead of length of annotation_set label list
#     - Test new instances created by separations have negative ids
#
#     test_models_labels_multiclass.TestSeparateLabelsEntityMultiClassModel.test_training
#     => Tests training and the number of labels after separation using project 458
#
#     Missing test:
#     - content of extract() result
#     - No leakage of annotations from shared labels across categories and within categories (Nachname)
#     """
#
#     @classmethod
#     def setUpClass(cls) -> None:
#         """Set the test data (Project 1100)."""
#         cls.prj = Project(id_=1199)
#
#     def test_separate_labels(self):
#         # res = separate_labels(project=self.prj)
#         # assert....
#         pass
#
#
# class TestDocumentModelInitialization(unittest.TestCase):
#     @classmethod
#     def setUpClass(cls) -> None:
#         """Set the test data (Project 1199)."""
#         # TODO: use emtpy project
#         cls.prj = Project(id_=46)
#         cls.category = cls.prj.categories[0]
#         cls.document_id = 196137
#         cls.document = cls.prj.get_document_by_id(cls.document_id)
#         cls.extraction_ai = DocumentAnnotationMultiClassModel(category=cls.category)
#
#     def test_category(self):
#         """Test Extraction AI has Category."""
#         assert self.extraction_ai.category == self.category
#
#     def test_label_sets_with_ids(self):
#         """Test Extraction AI has Label Sets from category."""
#         assert set(filter(lambda x: x.id_, self.extraction_ai.label_sets)) == set(
#             filter(lambda x: x.id_, self.category.label_sets)
#         )
#
#     def test_labels_with_ids(self):
#         """Test Extraction AI has Labels from category."""
#         assert set(filter(lambda x: x.id_, self.extraction_ai.labels)) == set(
#             filter(lambda x: x.id_ and set(x.label_sets).issubset(self.category.label_sets), self.prj.labels)
#         )
#
#     @unittest.skip('Tokenizer creates new NO_Label')
#     def test_labels(self):
#         # TODO: NO LABEL is not added to the extraction AI but is added to the project
#         assert self.extraction_ai.labels == self.prj.labels
#
#     @unittest.skip('Tokenizer creates new NO_Label')
#     def test_label_sets(self):
#         # TODO: Label Set for NO LABEL is added to extraction AI but not to the project
#         assert self.extraction_ai.label_sets == self.prj.label_sets
#
#
# class TestLabelSetClfBasicProject(unittest.TestCase):
#     """Test fit of the Label Set classifier."""
#
#     # WIP: tests need to be completed
#
#     @classmethod
#     def setUpClass(cls) -> None:
#         """Set the test data."""
#         cls.prj = Project(id_=None)
#         cls.category = Category(project=cls.prj, id_=1)
#         cls.prj.add_category(cls.category)
#
#         label_set = LabelSet(
#             project=cls.prj, name='LabelSet1', categories=[cls.category], id_=2, has_multiple_annotation_sets=True
#         )
#         label = Label(project=cls.prj, label_sets=[label_set], text='FirstName', id_=3)
#
#         cls.document = Document(project=cls.prj, category=cls.category, text='A\nB')
#
#         # TODO: we need to add annotation even if we pass df_train
#         span = Span(start_offset=0, end_offset=1)
#         annotation_set = AnnotationSet(document=cls.document, label_set=label_set)
#         _ = Annotation(
#             label=label, annotation_set=annotation_set, label_set=label_set, document=cls.document, spans=[span]
#         )
#
#         cls.extraction_ai = DocumentAnnotationMultiClassModel(category=cls.category)
#
#     def test_fit_document_not_belonging_to_category(self):
#         """Test fit() the Label Set classifier with an invalid document - not belonging to the category."""
#         self.extraction_ai.label_feature_list = ['dummy_feat_1']
#         self.extraction_ai.df_train = pandas.DataFrame([{'document_id': 1, 'label_text': 'test', 'dummy_feat_1': 1}])
#         self.extraction_ai.df_valid = pandas.DataFrame()
#         with self.assertRaises(IndexError):
#             self.extraction_ai.fit_label_set_clf()
#
#     def test_fit_document_without_id(self):
#         """Test fit() the Label Set classifier with an invalid document - without ID."""
#         self.extraction_ai.label_feature_list = ['dummy_feat_1']
#         self.extraction_ai.df_train = \
#         pandas.DataFrame([{'document_id': None, 'label_text': 'test', 'dummy_feat_1': 1}])
#         self.extraction_ai.df_valid = pandas.DataFrame()
#         with self.assertRaises(ValueError) as context:
#             self.extraction_ai.fit_label_set_clf()
#             assert 'No objects to concatenate' in context
#
#     @unittest.skip(reason='Suggestion for change.')
#     def test_fit_document_without_label_classifier(self):
#         """Test fit() the Label Set classifier without having fitted the Label classifier."""
#         self.extraction_ai.label_feature_list = ['dummy_feat_1']
#         self.extraction_ai.df_train = pandas.DataFrame([{'document_id': 4, 'label_text': 'test', 'dummy_feat_1': 1}])
#         self.extraction_ai.df_valid = pandas.DataFrame()
#         # TODO: fit should e possible without label clf (8857)
#         self.extraction_ai.fit_label_set_clf()
#         # TODO: add assert
#
#     @unittest.skip(reason='Suggestion for change.')
#     def test_fit_document_without_label_features(self):
#         """Test fit() the Label Set classifier without Label features."""
#         self.extraction_ai.label_feature_list = []
#         self.extraction_ai.df_train = pandas.DataFrame([{'document_id': 4, 'label_text': 'test'}])
#         self.extraction_ai.df_valid = pandas.DataFrame()
#         self.extraction_ai.fit()
#         # TODO: fit should e possible without label features (8857)
#         self.extraction_ai.fit_label_set_clf()
#         # TODO: add assert
#
#     @unittest.skip(reason='Suggestion for change.')
#     def test_fit_document_without_label_features_list(self):
#         """Test fit() the Label Set classifier without the Label features list."""
#         self.extraction_ai.label_feature_list = []
#         self.extraction_ai.df_train = pandas.DataFrame([{'document_id': 4, 'label_text': 'test', 'dummy_feat_1': 1}])
#         self.extraction_ai.df_valid = pandas.DataFrame()
#         self.extraction_ai.fit()
#         # TODO: fit should be possible without label features (8857)
#         self.extraction_ai.fit_label_set_clf()
#         # TODO: add assert
#
#     @unittest.skip(reason='Suggestion for change.')
#     def test_fit_document_complete_features(self):
#         """Test fit() the Label Set classifier with complete features."""
#         self.extraction_ai.label_feature_list = ['dummy_feat_1']
#         span_dict = {
#             'document_id': 4,
#             'label_text': 'FirstName',
#             'dummy_feat_1': 1,
#             'label_id': 3,
#             'start_offset': 0,
#             'end_offset': 1,
#             'line_index': 0,
#         }
#         self.extraction_ai.df_train = pandas.DataFrame([span_dict])
#         self.extraction_ai.df_valid = pandas.DataFrame()
#         self.extraction_ai.fit()
#         # TODO: fit should be possible with offline document if we have df_train (already with span info)
#         self.extraction_ai.fit_label_set_clf()
#         # TODO: add assert
#
#     @unittest.skip(reason='Suggestion for change.')
#     def test_fit_document_without_offsets_features(self):
#         """Test fit() the Label Set classifier without offset featuers."""
#         self.extraction_ai.label_feature_list = ['dummy_feat_1']
#         span_dict = {'document_id': 4, 'label_text': 'FirstName', 'dummy_feat_1': 1, 'label_id': 3, 'line_index': 0}
#         self.extraction_ai.df_train = pandas.DataFrame([span_dict])
#         self.extraction_ai.df_valid = pandas.DataFrame()
#         self.extraction_ai.fit()
#         # TODO: fit should be possible with offline document if we have df_train (already with span info) (8856)
#         self.extraction_ai.fit_label_set_clf()
#         # TODO: add assert
#
#
# class TestLabelSetClfExtractDefaultOnly(unittest.TestCase):
#     """Test Label Set classifier extract method when the only Label Set is the Category."""
#
#     @classmethod
#     def setUpClass(cls) -> None:
#         """Set the test data."""
#         cls.project = Project(id_=46)
#         category = cls.project.get_category_by_id(63)
#         cls.label_set = cls.project.get_label_set_by_id(63)  # default
#         category.label_sets = [cls.label_set]
#         cls.test_document = category.documents()[0]
#
#         for document in category.documents():
#             filtered_annotations = [
#                 annot for annot in document.annotations(use_correct=True) if annot.label_set == cls.label_set
#             ]
#             document._annotations = filtered_annotations
#
#         cls.extraction_ai = DocumentAnnotationMultiClassModel(category=category)
#
#     def test_1_build_project_with_default_label_set_only(self):
#         """Test training with a Document where there are no Label Sets other than the default one."""
#         self.extraction_ai.build()
#         assert self.extraction_ai.label_set_clf is None
#         result = self.extraction_ai.extract(
#             text=self.test_document.text, bbox=self.test_document.get_bbox(), pages=self.test_document.pages
#         )
#
#         assert len(result.keys()) == 9
#         for label_name in result.keys():
#             assert label_name in [label.name for label in self.label_set.labels]
#             assert isinstance(result[label_name], pd.DataFrame)
#
#     def test_2_extract_label_set_with_clf(self):
#         """Test result of extract of Label Set clf with no Label Sets other than the default one."""
#         res_dict = {
#             # label from default Label Set
#             'Vorname': pd.DataFrame(
#                 [
#                     {'label_text': 'Vorname', 'confidence': 0.4, 'line_index': 1},
#                     {'label_text': 'Vorname', 'confidence': 0.6, 'line_index': 2},
#                 ]
#             )
#         }
#
#         self.assertIsNone(self.extraction_ai.label_set_clf)
#         extract_result = self.extraction_ai.extract_label_set_with_clf(self.test_document, pd.DataFrame(), res_dict)
#         assert extract_result == res_dict
#
#
# class TestLabelSetClfExtractMultipleFalseOnly(unittest.TestCase):
#     """Test Label Set classifier extract method when the only Label Set has no option for multiple Annotation Sets."""
#
#     @classmethod
#     def setUpClass(cls) -> None:
#         """Project with 1 Label Set with multiple=False."""
#         cls.project = Project(id_=46)
#         category = cls.project.get_category_by_id(63)
#         cls.label_set1 = cls.project.get_label_set_by_id(63)  # default
#         cls.label_set2 = cls.project.get_label_set_by_id(3707)  # multiple false
#         category.label_sets = [cls.label_set1, cls.label_set2]
#         cls.test_document = category.documents()[0]
#
#         for document in category.documents():
#             filtered_annotations = [
#                 annot
#                 for annot in document.annotations(use_correct=True)
#                 if annot.label_set in [cls.label_set1, cls.label_set2]
#             ]
#             document._annotations = filtered_annotations
#
#         cls.extraction_ai = DocumentAnnotationMultiClassModel(category=category)
#
#     def test_1_extract_method_default_label_set_prediction(self):
#         """Test extract with default predictions from the Label Set Classifier."""
#         res_dict = {
#             # label from default Label Set
#             'Vorname': pd.DataFrame(
#                 [
#                     {'label_text': 'Vorname', 'confidence': 0.4, 'line_index': 1},
#                     {'label_text': 'Vorname', 'confidence': 0.6, 'line_index': 2},
#                 ]
#             ),
#             # Label from Label Set with multiple False
#             'Steuer-Brutto': pd.DataFrame(
#                 [
#                     {'label_text': 'Steuer-Brutto', 'confidence': 0.2, 'line_index': 3},
#                     {'label_text': 'Steuer-Brutto', 'confidence': 0.4, 'line_index': 4},
#                 ]
#             ),
#         }
#
#         res_label_sets = pd.DataFrame(['No', 'No', 'No', 'No'])
#
#         result = self.extraction_ai.extract_from_label_set_output(res_dict, res_label_sets)
#         assert all(result['Vorname'] == res_dict['Vorname'])
#         assert all(result['Verdiensibescheinigung']['Steuer-Brutto'] == res_dict['Steuer-Brutto'])
#
#     def test_2_extract_method_correct_label_cls_correct_label_set_clf(self):
#         """Test extract with correct predictions from the Label Classifier and Label Set Classifier."""
#         res_dict = {
#             # label from default Label Set
#             'Vorname': pd.DataFrame(
#                 [
#                     {'label_text': 'Vorname', 'confidence': 0.4, 'line_index': 1},
#                     {'label_text': 'Vorname', 'confidence': 0.6, 'line_index': 2},
#                 ]
#             ),
#             # Label from Label Set with multiple False
#             'Steuer-Brutto': pd.DataFrame(
#                 [
#                     {'label_text': 'Steuer-Brutto', 'confidence': 0.2, 'line_index': 3},
#                     {'label_text': 'Steuer-Brutto', 'confidence': 0.4, 'line_index': 4},
#                 ]
#             ),
#         }
#         res_label_sets = pd.DataFrame(['Lohnabrechnung', 'No', 'Verdiensibescheinigung', 'No'])
#
#         result = self.extraction_ai.extract_from_label_set_output(res_dict, res_label_sets)
#         assert all(result['Vorname'] == res_dict['Vorname'])
#         assert all(result['Verdiensibescheinigung']['Steuer-Brutto'] == res_dict['Steuer-Brutto'])
#
#     @unittest.skip(reason="Not currently using the choose_top option, which also needs revision.")
#     def test_3_extract_method_correct_label_cls_correct_label_set_clf_choose_top(self):
#         """Test extract with correct predictions from the Label Classifier and Label Set Classifier and choose top."""
#         res_dict = {
#             # label from default Label Set
#             'Vorname': pd.DataFrame(
#                 [
#                     {'label_text': 'Vorname', 'confidence': 0.4, 'line_index': 1},
#                     {'label_text': 'Vorname', 'confidence': 0.6, 'line_index': 2},
#                 ]
#             ),
#             # Label from Label Set with multiple False
#             'Steuer-Brutto': pd.DataFrame(
#                 [
#                     {'label_text': 'Steuer-Brutto', 'confidence': 0.2, 'line_index': 3},
#                     {'label_text': 'Steuer-Brutto', 'confidence': 0.4, 'line_index': 4},
#                 ]
#             ),
#         }
#         res_label_sets = pd.DataFrame(['Lohnabrechnung', 'No', 'Verdiensibescheinigung', 'No'])
#
#         result = self.extraction_ai.extract_from_label_set_output(res_dict, res_label_sets, choose_top=True)
#         assert (
#             result['Vorname']
#             .reset_index(drop=True)
#             .equals(pd.DataFrame([{'label_text': 'Vorname', 'confidence': 0.6, 'line_index': 2}]))
#         )
#         assert (
#             result['Verdiensibescheinigung']['Steuer-Brutto']
#             .reset_index(drop=True)
#             .equals(pd.DataFrame([{'label_text': 'Steuer-Brutto', 'confidence': 0.4, 'line_index': 4}]))
#         )
#
#     def test_4_extract_method_incorrect_label_cls_correct_label_set_clf(self):
#         """Test extract with missing Label predictions and correct Label Set predictions."""
#         res_dict = {
#             # label from default Label Set
#             'Vorname': pd.DataFrame(
#                 [
#                     {'label_text': 'Vorname', 'confidence': 0.4, 'line_index': 1},
#                     {'label_text': 'Vorname', 'confidence': 0.6, 'line_index': 2},
#                 ]
#             ),
#         }
#         res_label_sets = pd.DataFrame(['No', 'No'])
#         result = self.extraction_ai.extract_from_label_set_output(res_dict.copy(), res_label_sets)
#         assert all(result['Vorname'] == res_dict['Vorname'])
#
#     def test_5_extract_method_incorrect_label_cls_incorrect_label_set_clf(self):
#         """Test extract with missing Label predictions and incorrect Label Set predictions."""
#         res_dict = {
#             # label from default Label Set
#             'Vorname': pd.DataFrame(
#                 [
#                     {'label_text': 'Vorname', 'confidence': 0.4, 'line_index': 1},
#                     {'label_text': 'Vorname', 'confidence': 0.6, 'line_index': 2},
#                 ]
#             ),
#         }
#         res_label_sets = pd.DataFrame(['No', 'Verdiensibescheinigung'])
#         result = self.extraction_ai.extract_from_label_set_output(res_dict, res_label_sets)
#         assert all(result['Vorname'] == res_dict['Vorname'])
#
#     def test_6_extract_method_correct_label_cls_incorrect_label_set_clf(self):
#         """Test extract with correct Label predictions and incorrect Label Set predictions."""
#         res_dict = {
#             # label from default Label Set
#             'Vorname': pd.DataFrame(
#                 [
#                     {'label_text': 'Vorname', 'confidence': 0.4, 'line_index': 1},
#                     {'label_text': 'Vorname', 'confidence': 0.6, 'line_index': 2},
#                 ]
#             ),
#             # Label from Label Set with multiple False
#             'Steuer-Brutto': pd.DataFrame(
#                 [
#                     {'label_text': 'Steuer-Brutto', 'confidence': 0.2, 'line_index': 3},
#                     {'label_text': 'Steuer-Brutto', 'confidence': 0.4, 'line_index': 4},
#                 ]
#             ),
#         }
#         res_label_sets = pd.DataFrame(['No', 'No', 'No', 'No'])
#         result = self.extraction_ai.extract_from_label_set_output(res_dict.copy(), res_label_sets)
#         assert all(result['Vorname'] == res_dict['Vorname'])
#         assert all(result['Verdiensibescheinigung']['Steuer-Brutto'] == res_dict['Steuer-Brutto'])
#
#     def test_7_build_project_with_multiple_false_label_sets_only(self):
#         """Test training with a Document where there are no Label Sets other than multiple=False ones."""
#         self.extraction_ai.build()
#         assert self.extraction_ai.label_set_clf is not None
#         result = self.extraction_ai.extract(
#             text=self.test_document.text, bbox=self.test_document.get_bbox(), pages=self.test_document.pages
#         )
#
#         labels_names_1 = [label.name for label in self.label_set1.labels]
#
#         for key in result.keys():
#             assert key in labels_names_1 + [self.label_set2.name]
#
#
# class TestLabelSetClfExtractMultipleTrueOnly(unittest.TestCase):
#     """Test Label Set classifier extract method when the only Label Set has option for multiple Annotation Sets."""
#
#     @classmethod
#     def setUpClass(cls) -> None:
#         """Project with 1 Label Set with multiple=False."""
#         cls.project = Project(id_=46)
#         category = cls.project.get_category_by_id(63)
#         cls.label_set1 = cls.project.get_label_set_by_id(63)  # default
#         cls.label_set2 = cls.project.get_label_set_by_id(64)  # multiple true
#         category.label_sets = [cls.label_set1, cls.label_set2]
#         cls.test_document = category.documents()[0]
#
#         for document in category.documents():
#             filtered_annotations = [
#                 annot
#                 for annot in document.annotations(use_correct=True)
#                 if annot.label_set in [cls.label_set1, cls.label_set2]
#             ]
#             document._annotations = filtered_annotations
#
#         cls.extraction_ai = DocumentAnnotationMultiClassModel(category=category)
#
#     def test_1_extract_method_default_label_set_prediction(self):
#         """Test extract with default predictions from the Label Set Classifier."""
#         res_dict = {
#             # label from default Label Set
#             'Vorname': pd.DataFrame([{'label_text': 'Vorname', 'confidence': 0.4, 'line_index': 1}]),
#             # Label from Label Set with multiple True
#             'Betrag': pd.DataFrame(
#                 [
#                     {'label_text': 'Betrag', 'confidence': 0.2, 'line_index': 3},
#                     {'label_text': 'Betrag', 'confidence': 0.4, 'line_index': 4},
#                     {'label_text': 'Betrag', 'confidence': 0.3, 'line_index': 5},
#                 ]
#             ),
#         }
#
#         res_label_sets = pd.DataFrame(['No', 'No', 'No', 'No', 'No'])
#
#         result = self.extraction_ai.extract_from_label_set_output(res_dict, res_label_sets)
#         assert all(result['Vorname'] == res_dict['Vorname'])
#         assert len(result['Brutto-Bezug']) == 1
#         assert all(result['Brutto-Bezug'][0]['Betrag'] == res_dict['Betrag'])
#
#     def test_2_extract_method_correct_label_cls_correct_label_set_clf(self):
#         """Test extract with correct predictions from the Label Classifier and Label Set Classifier."""
#         res_dict = {
#             # label from default Label Set
#             'Vorname': pd.DataFrame([{'label_text': 'Vorname', 'confidence': 0.4, 'line_index': 1}]),
#             # Label from Label Set with multiple True
#             'Betrag': pd.DataFrame(
#                 [
#                     {'label_text': 'Betrag', 'confidence': 0.2, 'line_index': 3},
#                     {'label_text': 'Betrag', 'confidence': 0.4, 'line_index': 4},
#                     {'label_text': 'Betrag', 'confidence': 0.3, 'line_index': 5},
#                 ]
#             ),
#         }
#         res_label_sets = pd.DataFrame(['Lohnabrechnung', 'No', 'Brutto-Bezug', 'Brutto-Bezug', 'Brutto-Bezug'])
#
#         result = self.extraction_ai.extract_from_label_set_output(res_dict, res_label_sets)
#         assert all(result['Vorname'] == res_dict['Vorname'])
#         assert len(result['Brutto-Bezug']) == 3
#         assert all(result['Brutto-Bezug'][0]['Betrag'] == res_dict['Betrag'].loc[0, :])
#         assert all(result['Brutto-Bezug'][1]['Betrag'] == res_dict['Betrag'].loc[1, :])
#         assert all(result['Brutto-Bezug'][2]['Betrag'] == res_dict['Betrag'].loc[2, :])
#
#     @unittest.skip(reason="Not currently using the choose_top option, which also needs revision.")
#     def test_3_extract_method_correct_label_cls_correct_label_set_clf_choose_top(self):
#         """Test extract with correct predictions from the Label Classifier and Label Set Classifier and choose top."""
#         res_dict = {
#             # label from default Label Set
#             'Vorname': pd.DataFrame([{'label_text': 'Vorname', 'confidence': 0.4, 'line_index': 1}]),
#             # Label from Label Set with multiple True
#             'Betrag': pd.DataFrame(
#                 [
#                     {'label_text': 'Betrag', 'confidence': 0.2, 'line_index': 3},
#                     {'label_text': 'Betrag', 'confidence': 0.4, 'line_index': 4},
#                     {'label_text': 'Betrag', 'confidence': 0.3, 'line_index': 5},
#                 ]
#             ),
#         }
#         res_label_sets = pd.DataFrame(['Lohnabrechnung', 'No', 'Brutto-Bezug', 'Brutto-Bezug', 'Brutto-Bezug'])
#
#         result = self.extraction_ai.extract_from_label_set_output(res_dict, res_label_sets, choose_top=True)
#         assert all(result['Vorname'] == res_dict['Vorname'])
#         assert len(result['Brutto-Bezug']) == 3
#         assert all(result['Brutto-Bezug'][0]['Betrag'] == res_dict['Betrag'].loc[0, :])
#         assert all(result['Brutto-Bezug'][1]['Betrag'] == res_dict['Betrag'].loc[1, :])
#         assert all(result['Brutto-Bezug'][2]['Betrag'] == res_dict['Betrag'].loc[2, :])
#
#     def test_4_extract_method_incorrect_label_cls_correct_label_set_clf(self):
#         """Test extract with missing Label predictions and correct Label Set predictions."""
#         res_dict = {
#             # label from default Label Set
#             'Vorname': pd.DataFrame([{'label_text': 'Vorname', 'confidence': 0.4, 'line_index': 1}]),
#             # Label from Label Set with multiple True
#             'Betrag': pd.DataFrame([{'label_text': 'Betrag', 'confidence': 0.3, 'line_index': 5}]),
#         }
#         res_label_sets = pd.DataFrame(['No', 'No', 'No', 'No', 'Brutto-Bezug'])
#         result = self.extraction_ai.extract_from_label_set_output(res_dict.copy(), res_label_sets)
#
#         assert all(result['Vorname'] == res_dict['Vorname'])
#         assert len(result['Brutto-Bezug']) == 1
#         assert all(result['Brutto-Bezug'][0]['Betrag'] == res_dict['Betrag'])
#
#     def test_5_extract_method_incorrect_label_cls_incorrect_label_set_clf(self):
#         """Test extract with missing Label predictions and incorrect Label Set predictions."""
#         res_dict = {
#             # label from default Label Set
#             'Vorname': pd.DataFrame([{'label_text': 'Vorname', 'confidence': 0.4, 'line_index': 1}]),
#             # Label from Label Set with multiple True
#             'Betrag': pd.DataFrame([{'label_text': 'Betrag', 'confidence': 0.3, 'line_index': 5}]),
#         }
#         res_label_sets = pd.DataFrame(['Brutto-Bezug', 'No', 'No', 'No', 'No'])
#         result = self.extraction_ai.extract_from_label_set_output(res_dict, res_label_sets)
#         assert all(result['Vorname'] == pd.DataFrame([{'label_text': 'Vorname', 'confidence': 0.4, 'line_index': 1}]))
#         assert len(result['Brutto-Bezug']) == 1
#         assert all(result['Brutto-Bezug'][0]['Betrag'] == res_dict['Betrag'])
#
#     def test_6_extract_method_correct_label_cls_incorrect_label_set_clf(self):
#         """Test extract with correct Label predictions and incorrect Label Set predictions."""
#         res_dict = {
#             # label from default Label Set
#             'Vorname': pd.DataFrame([{'label_text': 'Vorname', 'confidence': 0.4, 'line_index': 1}]),
#             # Label from Label Set with multiple True
#             'Betrag': pd.DataFrame(
#                 [
#                     {'label_text': 'Betrag', 'confidence': 0.2, 'line_index': 3},
#                     {'label_text': 'Betrag', 'confidence': 0.4, 'line_index': 4},
#                     {'label_text': 'Betrag', 'confidence': 0.3, 'line_index': 5},
#                 ]
#             ),
#         }
#         res_label_sets = pd.DataFrame(['Lohnabrechnung', 'No', 'No', 'No', 'No'])
#         result = self.extraction_ai.extract_from_label_set_output(res_dict.copy(), res_label_sets)
#         assert all(result['Vorname'] == res_dict['Vorname'])
#         assert len(result['Brutto-Bezug']) == 1
#         assert all(result['Brutto-Bezug'][0]['Betrag'] == res_dict['Betrag'])
#
#     def test_7_build_project_with_multiple_true_label_sets_only(self):
#         """Test training with a Document where there are no Label Sets other than multiple=True ones."""
#         self.extraction_ai.build()
#         assert self.extraction_ai.label_set_clf is not None
#         result = self.extraction_ai.extract(
#             text=self.test_document.text, bbox=self.test_document.get_bbox(), pages=self.test_document.pages
#         )
#
#         labels_names_1 = [label.name for label in self.label_set1.labels]
#
#         for key in result.keys():
#             assert key in labels_names_1 + [self.label_set2.name]
#
#
# class TestLabelSetClfExtractMultipleTrueAndMultipleFalse(unittest.TestCase):
#     """Test Label Set classifier extract method when the only Label Set is the Category."""
#
#     @classmethod
#     def setUpClass(cls) -> None:
#         """Set the test data."""
#         cls.project = Project(id_=46)
#         category = cls.project.get_category_by_id(63)
#         label_set0 = cls.project.get_label_set_by_id(63)  # default
#         label_set1 = cls.project.get_label_set_by_id(64)  # multiple true
#         label_set2 = cls.project.get_label_set_by_id(3707)  # multiple false
#         category.label_sets = [label_set0, label_set1, label_set2]
#         cls.test_document = category.documents()[0]
#
#         for document in category.documents():
#             filtered_annotations = [
#                 annot
#                 for annot in document.annotations(use_correct=True)
#                 if annot.label_set in [label_set0, label_set1, label_set2]
#             ]
#             document._annotations = filtered_annotations
#
#         cls.extraction_ai = DocumentAnnotationMultiClassModel(category=category)
#
#     def test_1_extract_method_default_label_set_prediction(self):
#         """Test extract with default predictions from the Label Set Classifier."""
#         res_dict = {
#             # label from default Label Set
#             'Vorname': pd.DataFrame([{'label_text': 'Vorname', 'confidence': 0.4, 'line_index': 1}]),
#             # Label from Label Set with multiple True
#             'Betrag': pd.DataFrame(
#                 [
#                     {'label_text': 'Betrag', 'confidence': 0.2, 'line_index': 3},
#                     {'label_text': 'Betrag', 'confidence': 0.4, 'line_index': 4},
#                     {'label_text': 'Betrag', 'confidence': 0.3, 'line_index': 5},
#                     {'label_text': 'Betrag', 'confidence': 0.3, 'line_index': 10},
#                 ]
#             ),
#             # Label from Label Set with multiple False
#             'Steuer-Brutto': pd.DataFrame(
#                 [
#                     {'label_text': 'Steuer-Brutto', 'confidence': 0.2, 'line_index': 7},
#                     {'label_text': 'Steuer-Brutto', 'confidence': 0.4, 'line_index': 8},
#                 ]
#             ),
#         }
#
#         res_label_sets = pd.DataFrame(['No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No'])
#
#         result = self.extraction_ai.extract_from_label_set_output(res_dict, res_label_sets)
#         assert all(result['Vorname'] == res_dict['Vorname'])
#         assert len(result['Brutto-Bezug']) == 1
#         assert all(result['Brutto-Bezug'][0]['Betrag'] == res_dict['Betrag'])
#         assert all(result['Verdiensibescheinigung']['Steuer-Brutto'] == res_dict['Steuer-Brutto'])
#
#     def test_2_extract_method_correct_label_cls_correct_label_set_clf(self):
#         """Test extract with correct predictions from the Label Classifier and Label Set Classifier."""
#         res_dict = {
#             # label from default Label Set
#             'Vorname': pd.DataFrame([{'label_text': 'Vorname', 'confidence': 0.4, 'line_index': 1}]),
#             # Label from Label Set with multiple True
#             'Betrag': pd.DataFrame(
#                 [
#                     {'label_text': 'Betrag', 'confidence': 0.2, 'line_index': 3},
#                     {'label_text': 'Betrag', 'confidence': 0.4, 'line_index': 4},
#                     {'label_text': 'Betrag', 'confidence': 0.3, 'line_index': 5},
#                     {'label_text': 'Betrag', 'confidence': 0.3, 'line_index': 10},
#                 ]
#             ),
#             # Label from Label Set with multiple False
#             'Steuer-Brutto': pd.DataFrame(
#                 [
#                     {'label_text': 'Steuer-Brutto', 'confidence': 0.2, 'line_index': 7},
#                     {'label_text': 'Steuer-Brutto', 'confidence': 0.4, 'line_index': 8},
#                 ]
#             ),
#         }
#         res_label_sets = pd.DataFrame(
#             [
#                 'Lohnabrechnung',
#                 'No',
#                 'Brutto-Bezug',
#                 'Brutto-Bezug',
#                 'Brutto-Bezug',
#                 'No',
#                 'Verdiensibescheinigung',
#                 'No',
#                 'No',
#                 'Brutto-Bezug',
#             ]
#         )
#
#         result = self.extraction_ai.extract_from_label_set_output(res_dict, res_label_sets)
#         assert all(result['Vorname'] == res_dict['Vorname'])
#         assert len(result['Brutto-Bezug']) == 4
#         assert all(result['Brutto-Bezug'][0]['Betrag'] == res_dict['Betrag'].loc[0, :])
#         assert all(result['Brutto-Bezug'][1]['Betrag'] == res_dict['Betrag'].loc[1, :])
#         assert all(result['Brutto-Bezug'][2]['Betrag'] == res_dict['Betrag'].loc[2, :])
#         assert all(result['Brutto-Bezug'][3]['Betrag'] == res_dict['Betrag'].loc[3, :])
#         assert all(result['Verdiensibescheinigung']['Steuer-Brutto'] == res_dict['Steuer-Brutto'])
#
#     @unittest.skip(reason="Not currently using the choose_top option, which also needs revision.")
#     def test_3_extract_method_correct_label_cls_correct_label_set_clf_choose_top(self):
#         """Test extract with correct predictions from the Label Classifier and Label Set Classifier and choose top."""
#         res_dict = {
#             # label from default Label Set
#             'Vorname': pd.DataFrame([{'label_text': 'Vorname', 'confidence': 0.4, 'line_index': 1}]),
#             # Label from Label Set with multiple True
#             'Betrag': pd.DataFrame(
#                 [
#                     {'label_text': 'Betrag', 'confidence': 0.2, 'line_index': 3},
#                     {'label_text': 'Betrag', 'confidence': 0.4, 'line_index': 4},
#                     {'label_text': 'Betrag', 'confidence': 0.3, 'line_index': 5},
#                 ]
#             ),
#         }
#         res_label_sets = pd.DataFrame(['Lohnabrechnung', 'Brutto-Bezug', 'Brutto-Bezug', 'Brutto-Bezug'])
#
#         result = self.extraction_ai.extract_from_label_set_output(res_dict, res_label_sets, choose_top=True)
#         assert all(result['Vorname'] == res_dict['Vorname'])
#         assert len(result['Brutto-Bezug']) == 3
#         assert all(result['Brutto-Bezug'][0]['Betrag'] == res_dict['Betrag'].loc[0, :])
#         assert all(result['Brutto-Bezug'][1]['Betrag'] == res_dict['Betrag'].loc[1, :])
#         assert all(result['Brutto-Bezug'][2]['Betrag'] == res_dict['Betrag'].loc[2, :])
#
#     def test_4_extract_method_incorrect_label_cls_correct_label_set_clf(self):
#         """Test extract with missing Label predictions and correct Label Set predictions."""
#         res_dict = {
#             # label from default Label Set
#             'Vorname': pd.DataFrame([{'label_text': 'Vorname', 'confidence': 0.4, 'line_index': 1}]),
#             # Label from Label Set with multiple True
#             'Betrag': pd.DataFrame(
#                 [
#                     {'label_text': 'Betrag', 'confidence': 0.3, 'line_index': 5},
#                     {'label_text': 'Betrag', 'confidence': 0.3, 'line_index': 10},
#                 ]
#             ),
#             # Label from Label Set with multiple False
#             'Steuer-Brutto': pd.DataFrame(
#                 [
#                     {'label_text': 'Steuer-Brutto', 'confidence': 0.2, 'line_index': 7},
#                 ]
#             ),
#         }
#         res_label_sets = pd.DataFrame(
#             [
#                 'Lohnabrechnung',
#                 'No',
#                 'No',
#                 'No',
#                 'Brutto-Bezug',
#                 'No',
#                 'Verdiensibescheinigung',
#                 'No',
#                 'No',
#                 'Brutto-Bezug',
#             ]
#         )
#
#         result = self.extraction_ai.extract_from_label_set_output(res_dict.copy(), res_label_sets)
#         assert all(result['Vorname'] == res_dict['Vorname'])
#         assert len(result['Brutto-Bezug']) == 2
#         assert all(result['Brutto-Bezug'][0]['Betrag'] == res_dict['Betrag'].loc[0, :])
#         assert all(result['Brutto-Bezug'][1]['Betrag'] == res_dict['Betrag'].loc[1, :])
#         assert all(result['Verdiensibescheinigung']['Steuer-Brutto'] == res_dict['Steuer-Brutto'])
#
#     def test_5_extract_method_incorrect_label_cls_incorrect_label_set_clf(self):
#         """Test extract with missing Label predictions and incorrect Label Set predictions."""
#         res_dict = {
#             # label from default Label Set
#             'Vorname': pd.DataFrame([{'label_text': 'Vorname', 'confidence': 0.4, 'line_index': 1}]),
#             # Label from Label Set with multiple True
#             'Betrag': pd.DataFrame(
#                 [
#                     {'label_text': 'Betrag', 'confidence': 0.3, 'line_index': 5},
#                     {'label_text': 'Betrag', 'confidence': 0.3, 'line_index': 10},
#                 ]
#             ),
#             # Label from Label Set with multiple False
#             'Steuer-Brutto': pd.DataFrame(
#                 [
#                     {'label_text': 'Steuer-Brutto', 'confidence': 0.2, 'line_index': 7},
#                 ]
#             ),
#         }
#         res_label_sets = pd.DataFrame(
#             ['Brutto-Bezug', 'No', 'No', 'No', 'Verdiensibescheinigung', 'No', 'Lohnabrechnung', 'No', 'No', 'No']
#         )
#
#         result = self.extraction_ai.extract_from_label_set_output(res_dict, res_label_sets)
#         assert all(result['Vorname'] == pd.DataFrame([{'label_text': 'Vorname', 'confidence': 0.4, 'line_index': 1}]))
#         assert len(result['Brutto-Bezug']) == 1
#         assert all(result['Brutto-Bezug'][0]['Betrag'] == res_dict['Betrag'])
#         assert all(result['Verdiensibescheinigung']['Steuer-Brutto'] == res_dict['Steuer-Brutto'])
#
#     def test_6_build_project_with_all_label_set_types(self):
#         """Test training with a document where there are all label set types."""
#         self.extraction_ai.build()
#         assert self.extraction_ai.label_set_clf is not None
#         result = self.extraction_ai.extract(
#             text=self.test_document.text, bbox=self.test_document.get_bbox(), pages=self.test_document.pages
#         )
#         assert len(result.keys()) == 10
#
#
# class TestLabelSetClf(unittest.TestCase):
#     # TODO: use empty project? (TestLabelSetClfBasicProject)
#
#     @classmethod
#     def setUpClass(cls) -> None:
#         """Set the test data (Project 1199)."""
#         cls.prj = Project(id_=46, update=True)
#         cls.category = cls.prj.categories[0]
#         cls.document_id = 196137
#         cls.document = cls.prj.get_document_by_id(cls.document_id)
#         cls.extraction_ai = DocumentAnnotationMultiClassModel(category=cls.category)
#
#     def test_1_fit_invalid_document(self):
#         """Test fit() the Label Set classifier with an invalid document - not belonging to the category."""
#         self.extraction_ai.label_feature_list = ['dummy_feat_1']
#         self.extraction_ai.df_train = pandas.DataFrame([{'document_id': 1, 'label_text': 'test', 'dummy_feat_1': 1}])
#         self.extraction_ai.df_valid = pandas.DataFrame()
#         with self.assertRaises(IndexError):
#             self.extraction_ai.fit()
#             self.extraction_ai.fit_label_set_clf()
#
#     def test_2_fit_label_clf(self):
#         """Minimal setup to do the fitting."""
#         self.extraction_ai.label_feature_list = ['dummy_feat_1']
#         document_id = self.category.documents()[0].id_
#         self.extraction_ai.df_train = pandas.DataFrame(
#             [
#                 {
#                     'document_id': document_id,
#                     'start_offset': 0,
#                     'end_offset': 1,
#                     'line_index': 0,
#                     'label_text': 'Vorname',
#                     'dummy_feat_1': 1,
#                 },
#                 {
#                     'document_id': document_id,
#                     'start_offset': 0,
#                     'end_offset': 1,
#                     'line_index': 0,
#                     'label_text': 'Nachname',
#                     'dummy_feat_1': 1,
#                 },
#             ]
#         )
#         self.extraction_ai.df_valid = pandas.DataFrame()
#         self.extraction_ai.fit()
#         self.extraction_ai.fit_label_set_clf()
#         assert len(self.extraction_ai.label_set_feature_list) == len(self.extraction_ai.labels)
#
#     def test_3_fit_label_extract(self):
#         """Minimal setup to do the fitting."""
#         document = self.category.documents()[0]
#         self.extraction_ai.build()
#         assert sorted(self.extraction_ai.label_set_feature_list) == sorted(
#             [
#                 'EMPTY_LABEL',
#                 'Vorname',
#                 'Betrag',
#                 'Faktor',
#                 'Menge',
#                 'Netto-Verdienst',
#                 'Steuerrechtliche Abzüge',
#                 'Nachname',
#                 'Steuer-Brutto',
#                 'Gesamt-Brutto',
#                 'Auszahlungsbetrag',
#                 'Austellungsdatum',
#                 'Bezeichnung',
#                 'Lohnart',
#                 'Bank inkl. IBAN',
#                 'Personalausweis',
#                 'Steuerklasse',
#                 'Sozialversicherung',
#             ]
#         )
#         self.extraction_ai.extract(text=document.text, bbox=document.get_bbox(), pages=document.pages)
#
#     def test_8_fit_label_set_clf_is_skipped(self):
#         """Test if the fit of the Label Set Classifier is skipped if there are only default Label Sets."""
#         for label_set in self.category.label_sets:
#             label_set.is_default = True
#         extraction_ai = DocumentAnnotationMultiClassModel(category=self.category)
#         extraction_ai.build()
#         # restore project status
#         for label_set in self.category.label_sets:
#             if label_set.id_ != 63:
#                 label_set.is_default = False
#         assert extraction_ai.label_set_clf is None
#
#     def test_7_fit_label_set_clf_is_skipped_wrong_behaviour(self):
#         """Test if the fit of the Label Set Classifier happens if there are only Label Sets with multiple False."""
#         for label_set in self.category.label_sets:
#             label_set.has_multiple_annotation_sets = False
#         extraction_ai = DocumentAnnotationMultiClassModel(category=self.category)
#         extraction_ai.build()
#         # restore project status
#         for label_set in self.category.label_sets:
#             if label_set.id_ in [64, 3706, 3606]:
#                 label_set.has_multiple_annotation_sets = True
#         assert extraction_ai.label_set_clf is not None
#
#     def test_6_extract_method_both_predictions_empty(self):
#         """Test extract with empty predictions from Label Classifier and Label Set Classifier."""
#         with self.assertRaises(ValueError) as context:
#             _ = self.extraction_ai.extract_from_label_set_output({}, pd.DataFrame())
#             assert (
#                 'Label Set Classifier result is empty and it should have the default value "No".' in context.exception
#             )
#
#     def test_4_extract_method_empty_label_prediction(self):
#         """Test extract with empty predictions from the Label Classifier."""
#         res_dict = {}
#         res_label_sets = pd.DataFrame(['No', 'No', 'Verdiensibescheinigung', 'No'])
#         result = self.extraction_ai.extract_from_label_set_output(res_dict, res_label_sets)
#         assert result == {}
#
#     def test_5_extract_method_empty_label_set_prediction(self):
#         """Test extract with empty predictions from the Label Set Classifier."""
#         res_dict = {
#             # label from default Label Set
#             'Vorname': pd.DataFrame(
#                 [
#                     {'label_text': 'Vorname', 'confidence': 0.4, 'line_index': 1},
#                     {'label_text': 'Vorname', 'confidence': 0.6, 'line_index': 2},
#                 ]
#             ),
#         }
#         res_label_sets = pd.DataFrame()
#         with self.assertRaises(ValueError) as context:
#             _ = self.extraction_ai.extract_from_label_set_output(res_dict, res_label_sets)
#
#             assert (
#                 'Label Set Classifier result is empty and it should have the default value "No".' in context.exception
#             )
#
#
# class CreateCandidatesDatasetTestDocumentAnnotationMultiClassModel(unittest.TestCase):
#     """Test for create_candidates_dataset."""
#
#     @classmethod
#     def setUpClass(cls) -> None:
#         """Set the test data (Project 46)."""
#         # TODO: use an empty Project
#         cls.prj = Project(id_=46, update=True)
#
#         cls.document_id = 44863  # 196137
#         # cls.document = cls.prj.get_document_by_id(cls.document_id)
#         cls.document = next(x for x in cls.prj.documents if x.id_ == cls.document_id)
#         # cls.document = next(x for x in cls.prj.no_status_documents if x.id_ == cls.document_id)
#         cls.category = cls.prj.categories[0]
#         extraction_ai = DocumentAnnotationMultiClassModel(category=cls.category)
#         cls.extraction_ai = extraction_ai.create_candidates_dataset()
#         cls.df = cls.extraction_ai.df_train
#
#     def test_length_df_train(self):
#         df = self.df.copy()
#         assert sum(len(x.spans) for x in self.document.annotations()) == \
#         df[df.document_id == self.document_id].shape[0]
#
#     def test_filter_feedback_required(self):
#         """All existing feedback required annotations (with ID) need to be filtered out."""
#         df = self.df.copy()
#         assert df[(~df['annotation_id'].isnull()) & (~df['revised']) & (~df['is_correct'])].shape[0] == 0
#
#     def test_filter_declined(self):
#         """All existing declined annotations (with ID) need to be filtered out."""
#         df = self.df.copy()
#         assert df[~df['id_'].isnull() & df['revised'] & ~df['is_correct']].shape[0] == 0
#
#     def test_document_id(self):
#         """All annotations must be of the category documents."""
#         df = self.df.copy()
#         assert list(df['document_id'].unique()) == list(set([doc.id_ for doc in self.category.documents()]))
#
#     def test_multiline_positional_attributes(self):
#         """Test splitting of annotations provide valid results."""
#         df = self.df.copy()
#
#         label = next(x for x in self.prj.labels if x.id_ == 12470)
#         multiline_annotations = [x for x in self.document.annotations() if x.label and x.label.id_ == label.id_]
#
#         for annotation in multiline_annotations:
#             # assert annotation.id
#             assert annotation.annotation_set
#             assert annotation.annotation_set.label_set.name
#             assert annotation.bboxes
#
#             for span in annotation._spans:
#                 box = get_bboxes(self.document.get_bbox(), span.start_offset, span.end_offset)[0]
#                 filter_df = df[(df['start_offset'] == span.start_offset) & (df['end_offset'] == span.end_offset)]
#
#                 assert filter_df['offset_string'].iat[0] == span.offset_string
#                 assert filter_df['l_dist0'].iat[0] >= 0
#                 assert filter_df['l_dist1'].iat[0] >= 0
#                 assert filter_df['r_dist0'].iat[0] >= 0
#                 assert filter_df['r_dist1'].iat[0] >= 0
#
#             for attribute in ['page_index', 'x0', 'x1', 'y0', 'y1']:
#                 logger.info(f'Check {attribute} for annotation {annotation}.')
#                 assert abs(box[attribute] - filter_df[attribute].iat[0]) < 0.1
#
#     @unittest.skip(reason="Needs revision/ implementation. - we are getting a negative 'l_dist0'")
#     def test_positional_attributes(self):
#         df = self.df.copy()
#         df = df[df['document_id'] == self.document_id]
#
#         label = next(x for x in self.prj.labels if x.id_ == 12470)
#         not_multiline_annotations = [x for x in self.document.annotations() if x.label and x.label.id_ != label.id_]
#         for annotation in not_multiline_annotations:
#             for span in annotation._spans:
#                 assert annotation.id_
#                 assert annotation.annotation_set
#                 assert annotation.annotation_set.label_set.name
#                 assert annotation.bboxes
#
#                 box = get_bboxes(annotation.document.get_bbox(), annotation.start_offset, annotation.end_offset)[0]
#                 filter_df = df[
#                     (df['start_offset'] == annotation.start_offset) & (df['end_offset'] == annotation.end_offset)
#                 ]
#
#                 assert filter_df['offset_string'].iat[0] == span.offset_string
#                 assert filter_df['l_dist0'].iat[0] >= 0
#                 assert filter_df['l_dist1'].iat[0] >= 0
#                 assert filter_df['r_dist0'].iat[0] >= 0
#                 assert filter_df['r_dist1'].iat[0] >= 0
#
#                 for attribute in ['page_index', 'x0', 'x1', 'y0', 'y1']:
#                     logger.info(f'Check {attribute} for annotation {annotation}.')
#                     assert abs(box[attribute] - filter_df[attribute].iat[0]) < 0.1
#
#     def test_feature_function_number_of_features(self):
#         """
#         Test feature function.
#
#         feature_list consists of:
#         - string_feature_column_order
#             Generated by convert_to_feat() in multiclass_clf.py, generates 51 features.
#             builds on:
#             - strip_accents()
#             - vowel_count()
#             - special_count()
#             - space_count()
#             - digit_count()
#             - upper_count()
#             - date_count()
#             - num_count()
#             - normalize_to_python_float()
#             - unique_char_count()
#             - duplicate_count()
#             - count_string_differences()
#             - year_month_day_count()
#             - substring_count()
#             - starts_with_substring()
#             - ends_with_substring()
#         - abs_pos_feature_list
#             Uses ["x0", "y0", "x1", "y1", "page_index", "area"] which are direct annotation attributes
#         - l_keys
#             Defined by n_left_nearest:
#             l_dist_n distance to nth left neighbour
#             if n_nearest_across_lines is more keys are present: why?
#         - r_keys
#             Defined by n_right_nearest:
#             r_dist_n distance to nth right neighbour
#             if n_nearest_across_lines is more keys are present: why?
#         - relative_string_feature_list
#             for each left and right neighbour take the full 51 features.
#         - relative_pos_feature_list
#             "relative_position_in_page" page index as percentage of page length.
#         - word_on_page_feature_name_list
#         - first_word_features (if first_word)
#             uses 4 features ['first_word_x0', 'first_word_y0', 'first_word_x1', 'first_word_y1']
#             51 string features are generated but no used
#
#         """
#         # We use
#         string_features = [
#             'accented_char_count',
#             # 'feat_as_float', Deactivated as we already have normalize_to_float.
#             'feat_day_count',
#             'feat_digit_len',
#             'feat_duplicate_count',
#             'feat_ends_with_minus',
#             'feat_ends_with_plus',
#             'feat_len',
#             'feat_month_count',
#             'feat_num_count',
#             'feat_space_len',
#             'feat_special_len',
#             'feat_starts_with_minus',
#             'feat_starts_with_plus',
#             'feat_substring_count_a',
#             'feat_substring_count_ae',
#             'feat_substring_count_c',
#             'feat_substring_count_ch',
#             'feat_substring_count_comma',
#             'feat_substring_count_e',
#             'feat_substring_count_ei',
#             'feat_substring_count_en',
#             'feat_substring_count_er',
#             'feat_substring_count_f',
#             'feat_substring_count_g',
#             'feat_substring_count_h',
#             'feat_substring_count_i',
#             'feat_substring_count_j',
#             'feat_substring_count_k',
#             'feat_substring_count_m',
#             'feat_substring_count_minus',
#             'feat_substring_count_n',
#             'feat_substring_count_oe',
#             'feat_substring_count_ohn',
#             'feat_substring_count_on',
#             'feat_substring_count_percent',
#             'feat_substring_count_period',
#             'feat_substring_count_plus',
#             'feat_substring_count_r',
#             'feat_substring_count_s',
#             'feat_substring_count_sch',
#             'feat_substring_count_slash',
#             'feat_substring_count_str',
#             'feat_substring_count_u',
#             'feat_substring_count_ue',
#             'feat_substring_count_y',
#             'feat_unique_char_count',
#             'feat_upper_len',
#             'feat_vowel_len',
#             'feat_year_count',
#         ]
#
#         abs_pos_feature = [
#             "x0",
#             "y0",
#             "x1",
#             "y1",
#             "x0_relative",
#             "y0_relative",
#             "x1_relative",
#             "y1_relative",
#             "page_index",
#             "page_index_relative"
#             # "area"
#         ]
#         # relative_position_on_page = ["relative_position_in_page"]
#
#         neighbours_distances = 4
#         neighbours_features = 4 * len(string_features)
#         first_page_features = 0  # Deactivated
#         word_on_page_features = 0  # Deactivated
#
#         assert set(string_features).issubset(set(self.extraction_ai.label_feature_list))
#         assert set(abs_pos_feature).issubset(set(self.extraction_ai.label_feature_list))
#         # assert set(relative_position_on_page).issubset(set(self.extraction_ai.label_feature_list))
#
#         expected_feature_count = (
#             len(string_features)
#             + len(abs_pos_feature)
#             + neighbours_distances
#             + neighbours_features
#             + first_page_features
#             + word_on_page_features
#         )
#         assert len(self.extraction_ai.label_feature_list) == expected_feature_count
#
#     def test_non_feature_columns(self):
#         expected_unused_dataframe_columns = {
#             'label_text',
#             'start_offset',
#             'end_offset',
#             'annotation_id',
#             'document_id',
#             'line_index',
#             'offset_string',
#             'r_offset_string0',
#             'r_offset_string1',
#             'l_offset_string0',
#             'l_offset_string1',
#             'confidence',
#             'normalized',
#             'is_correct',
#             'revised',
#             # 'top',
#             # 'bottom',
#             'id_',
#         }
#         unused_dataframe_column = set(list(self.df)) - set(self.extraction_ai.label_feature_list)
#         assert expected_unused_dataframe_columns == unused_dataframe_column
#
#
# class FitTestDocumentAnnotationMultiClassModel(unittest.TestCase):
#     """Test fit() method."""
#
#     @classmethod
#     def setUpClass(cls) -> None:
#         """Set the test data (Project 1100)."""
#         cls.prj = Project(id_=46)
#         cls.category = cls.prj.get_category_by_id(63)
#         cls.extraction_ai = DocumentAnnotationMultiClassModel(category=cls.category)
#
#     def test_fit(self):
#         """Minimal setup to do the fitting."""
#         self.extraction_ai.label_feature_list = ['dummy_feat_1']
#         document_id = self.category.documents()[0].id_
#         self.extraction_ai.df_train = pandas.DataFrame(
#             [{'document_id': document_id, 'label_text': 'test', 'dummy_feat_1': 1}]
#         )
#         self.extraction_ai.df_valid = pandas.DataFrame()
#         self.extraction_ai.fit()
#
#
# class TestExtractDocumentAnnotationMultiClassModel(unittest.TestCase):
#     """Test extract() method."""
#
#     @classmethod
#     def setUpClass(cls) -> None:
#         """Set the test data (Project 46)."""
#         cls.prj = Project(id_=46)
#         cls.category = cls.prj.get_category_by_id(63)
#         cls.documents = cls.category.documents()
#         cls.extraction_ai = DocumentAnnotationMultiClassModel(category=cls.category)
#         cls.extraction_ai = cls.extraction_ai.build()  # TODO why do we need build() here this makes test slow.
#
#     def test_extract_output_format(self):
#         """Test output format of the extract method."""
#         document = self.documents[0]
#         ai_result = self.extraction_ai.extract(text=document.text, bbox=document.get_bbox(), pages=document.pages)
#         assert isinstance(ai_result, dict)
#         for key, value in ai_result.items():
#             assert isinstance(value, pd.DataFrame) or isinstance(value, dict) or isinstance(value, list)
#
#     def test_extract_labels_and_label_sets(self):
#         """Test labels and label sets in the result keys."""
#         document = self.documents[0]
#         ai_result = self.extraction_ai.extract(text=document.text, bbox=document.get_bbox(), pages=document.pages)
#         category_labels_names = [
#             label.name for label in self.prj.labels if self.extraction_ai.category in label.label_sets
#         ]
#         category_label_sets_names = [label_set.name for label_set in self.extraction_ai.category.label_sets]
#
#         assert len(ai_result.keys()) > 0
#         for key, value in ai_result.items():
#             # key needs to be either a label or label set from the category
#             assert key in category_labels_names or key in category_label_sets_names
#
#     def test_extract_on_empty_document(self):
#         """Test extract() on an empty Document - no text."""
#         document = Document(text='', project=self.prj, category=self.extraction_ai.category)
#         ai_result = self.extraction_ai.extract(text=document.text, bbox=document.get_bbox(), pages=document.pages)
#         assert ai_result == {}
#
#     def test_extract_result_is_valid(self):
#         document = Document(text='', project=self.prj, category=self.extraction_ai.category)
#         ai_result = self.extraction_ai.extract(text=document.text, bbox=document.get_bbox(), pages=document.pages)
#
#         for _, value in ai_result.items():
#             if isinstance(value, pd.DataFrame):
#                 assert is_valid_extraction_dataframe(ai_result, n_features_columns=260)
#
#             elif isinstance(value, list) or isinstance(value, dict):
#                 if not isinstance(value, list):
#                     value = [value]
#
#                 for entry in value:
#                     for _, extraction in entry.items():
#                         assert is_valid_extraction_dataframe(extraction, n_features_columns=260)
#
#
# class EvaluateTestDocumentAnnotationMultiClassModel(unittest.TestCase):
#     """Test evaluate() method."""
#
#     @classmethod
#     def setUpClass(cls) -> None:
#         """Set the test data (Project 1100)."""
#         # TODO: use empty Project
#         cls.prj = Project(id_=46)
#         cls.category = cls.prj.get_category_by_id(63)
#         cls.extraction_ai = DocumentAnnotationMultiClassModel(category=cls.category)
#
#     def test_evaluate_empty_test_df(self):
#         self.extraction_ai.df_test = None
#         with self.assertRaises(AttributeError):
#             self.extraction_ai.evaluate()
#
#     def test_evaluate(self):
#         # Do minimal training
#         self.extraction_ai.label_feature_list = ['dummy_feat_1']
#         self.extraction_ai.df_train = pandas.DataFrame([{'label_text': 'test', 'dummy_feat_1': 1}])
#         self.extraction_ai.df_test = pandas.DataFrame(
#             [{'confidence': 0.1, 'is_correct': True, 'label_text': 'test', 'dummy_feat_1': 1}]
#         )
#         self.extraction_ai.fit()
#
#         self.extraction_ai.evaluate()
#         df = self.extraction_ai.df_prob.iloc[:, 1]
#         assert len(df[df[0] != df.isnull()]) == 1
#
#
# class TestLoseWeight(unittest.TestCase):
#     """Test lose_weight() method."""
#
#     def test_lose_weight_without_documents(self):
#         """Test lose_weight without loading the documents in the project."""
#         prj = Project(id_=46, update=True)
#         category = prj.get_category_by_id(63)
#         extraction_ai = DocumentAnnotationMultiClassModel(category=category)
#         extraction_ai.lose_weight()
#         size_in_mb = asizeof.asizeof(extraction_ai) / 1_000_000  # Convert to MB
#         self.assertTrue(size_in_mb < 0.25)  # 2.46 MB
#
#     def test_lose_weight(self):
#         """Test lose_weight after loading the documents in the project (with build())."""
#         prj = Project(id_=46, update=True)
#         category = prj.get_category_by_id(63)
#         extraction_ai = DocumentAnnotationMultiClassModel(category=category)
#         extraction_ai.build()
#         extraction_ai.lose_weight()
#         size_in_mb = asizeof.asizeof(extraction_ai) / 1_000_000  # Convert to MB
#         self.assertTrue(size_in_mb < 0.5)
#
#     def test_lose_weight_changes_in_category_documents(self):
#         """
#         Lose weight removes Documents in the Category.
#
#         It's necessary for running multiple training iterations (e.g. parameters search)
#         """
#         prj = Project(id_=46, update=True)
#         category = prj.get_category_by_id(63)
#         extraction_ai = DocumentAnnotationMultiClassModel(category=category)
#         extraction_ai.build()
#         extraction_ai.lose_weight()
#         assert category.documents() == []
#         assert category.test_documents() == []
#
#
# class TestSave(unittest.TestCase):
#     def test_save(self):
#         """The name of the saved file should contain the name of the Category."""
#         prj = Project(id_=46)
#         category = prj.get_category_by_id(63)
#         extraction_ai = DocumentAnnotationMultiClassModel(category=category)
#         file_name = extraction_ai.save()
#         self.assertTrue(category.name.lower() in file_name)
#
#     def test_saved_model_without_documents(self):
#         """Saved model does not include Documents."""
#         prj = Project(id_=46)
#         category = prj.get_category_by_id(63)
#         extraction_ai = DocumentAnnotationMultiClassModel(category=category)
#         extraction_ai.build()
#         file_name = extraction_ai.save()
#         model = load_pickle(file_name)
#         assert len(category.documents()) == 25
#         assert model.documents == []
#         assert model.test_documents == []
#
#     def test_not_possible_to_get_documents_from_category_of_saved_model(self):
#         """Saved model does not keep the Documents of the Category."""
#         prj = Project(id_=46)
#         category = prj.get_category_by_id(63)
#         extraction_ai = DocumentAnnotationMultiClassModel(category=category)
#         extraction_ai.build()
#         file_name = extraction_ai.save()
#         model = load_pickle(file_name)
#         assert len(category.documents()) == 25
#         assert model.category.project.documents == []
#         assert model.category.documents() == []
#
#     def test_model_size_after_save(self):
#         """Save should create a file with less than 0.5 MB."""
#         prj = Project(id_=46)
#         category = prj.get_category_by_id(63)
#         extraction_ai = DocumentAnnotationMultiClassModel(category=category)
#         extraction_ai.build()
#         file_name = extraction_ai.save()
#         model = load_pickle(file_name)
#         size_in_mb = asizeof.asizeof(model) / 1_000_000  # Convert to MB
#         self.assertTrue(size_in_mb < 0.5)
#
#
# def test_get_n_nearest_features_empty():
#     """Test calling get_n_nearest_features with empty annoation document."""
#
#     project = Project(id_=None)
#     category = Category(project=project, id_=1)
#     project.add_category(category)
#     document = Document(project=project, category=category)
#     assert len(project.virtual_documents) == 1
#
#     extraction_ai = DocumentAnnotationMultiClassModel(category=category)
#     extraction_ai.get_n_nearest_features(document=document, annotations=[])
#
#
# class TestNnearestFeatures(unittest.TestCase):
#     def setUp(self):
#         self.project = Project(id_=None)
#         self.category = Category(project=self.project, id_=1)
#         self.project.add_category(self.category)
#
#         self.label_set = LabelSet(id_=33, project=self.project, categories=[self.category])
#         self.label = Label(id_=22, text='LabelName', project=self.project, label_sets=[self.label_set], threshold=0.5)
#         document_bbox = {
#             '0': {'x0': 0, 'x1': 1, 'y0': 0, 'y1': 1, 'top': 10, 'bottom': 11, 'page_number': 1},
#             '1': {'x0': 2, 'x1': 3, 'y0': 0, 'y1': 1, 'top': 10, 'bottom': 11, 'page_number': 1},
#             '3': {'x0': 3, 'x1': 4, 'y0': 0, 'y1': 1, 'top': 10, 'bottom': 11, 'page_number': 1},
#             '4': {'x0': 4, 'x1': 5, 'y0': 0, 'y1': 1, 'top': 10, 'bottom': 11, 'page_number': 1},
#         }
#         self.document = Document(
#             project=self.project,
#             category=self.category,
#             text='hi ha',
#             bbox=document_bbox,
#             dataset_status=2,
#             pages=[{'original_size': (100, 100)}],
#         )
#
#     def test_get_n_nearest_features(self):
#
#         span_1 = Span(start_offset=0, end_offset=2)
#         span_2 = Span(start_offset=3, end_offset=5)
#         annotation_set_1 = AnnotationSet(id_=1, document=self.document, label_set=self.label_set)
#         annotation_1 = Annotation(
#             document=self.document,
#             is_correct=True,
#             annotation_set=annotation_set_1,
#             label=self.label,
#             label_set=self.label_set,
#             spans=[span_1],
#         )
#         annotation_2 = Annotation(
#             document=self.document,
#             is_correct=True,
#             annotation_set=annotation_set_1,
#             label=self.label,
#             label_set=self.label_set,
#             spans=[span_2],
#         )
#         assert annotation_1.offset_string == ['hi']
#         assert annotation_2.offset_string == ['ha']
#         extraction_ai = DocumentAnnotationMultiClassModel(category=self.category)
#         neighbours = extraction_ai.n_nearest_left + extraction_ai.n_nearest_right
#
#         df, feature_list = extraction_ai.get_n_nearest_features(
#             document=self.document, annotations=self.document.annotations()
#         )
#         assert len(feature_list) == FEATURE_COUNT * neighbours + len(['l_dist0', 'l_dist1', 'r_dist0', 'r_dist1'])
#         assert df.shape == (2, FEATURE_COUNT * neighbours + 4 + 4)
#         for key in ['l0', 'l1', 'r0', 'r1']:
#             assert len([x for x in df if x.startswith(key)]) == FEATURE_COUNT  # We have 49 feature entries.
#
#         assert (df['l_offset_string0'] == ['', 'hi']).all()
#         assert (df['r_offset_string0'] == ['ha', '']).all()
#         assert (df['l_offset_string1'] == ['', '']).all()
#         assert (df['r_offset_string1'] == ['', '']).all()
#         # in df but no feature
#         # {str} 'l_offset_string1'
#         # {str} 'r_offset_string1'
#         # {str} 'l_offset_string0'
#         # {str} 'r_offset_string0'
#
#     def test_get_n_nearest_features_partial(self):
#
#         span_1 = Span(start_offset=0, end_offset=2)
#         span_2 = Span(start_offset=3, end_offset=4)
#         annotation_set_1 = AnnotationSet(id_=1, document=self.document, label_set=self.label_set)
#         annotation_1 = Annotation(
#             document=self.document,
#             is_correct=True,
#             annotation_set=annotation_set_1,
#             label=self.label,
#             label_set=self.label_set,
#             spans=[span_1],
#         )
#         annotation_2 = Annotation(
#             document=self.document,
#             is_correct=True,
#             annotation_set=annotation_set_1,
#             label=self.label,
#             label_set=self.label_set,
#             spans=[span_2],
#         )
#         assert annotation_1.offset_string == ['hi']
#         assert annotation_2.offset_string == ['h']
#         extraction_ai = DocumentAnnotationMultiClassModel(category=self.category)
#         neighbours = extraction_ai.n_nearest_left + extraction_ai.n_nearest_right
#
#         df, feature_list = extraction_ai.get_n_nearest_features(
#             document=self.document, annotations=self.document.annotations()
#         )
#         assert len(feature_list) == FEATURE_COUNT * neighbours + len(['l_dist0', 'l_dist1', 'r_dist0', 'r_dist1'])
#         assert df.shape == (2, FEATURE_COUNT * neighbours + 4 + 4)
#         for key in ['l0', 'l1', 'r0', 'r1']:
#             assert len([x for x in df if x.startswith(key)]) == FEATURE_COUNT  # We have 49 feature entries.
#
#         assert (df['l_offset_string0'] == ['', 'hi']).all()
#         assert (df['r_offset_string0'] == ['ha', '']).all()
#         assert (df['l_offset_string1'] == ['', '']).all()
#         assert (df['r_offset_string1'] == ['', '']).all()
#         # in df but no feature
#         # {str} 'l_offset_string1'
#         # {str} 'r_offset_string1'
#         # {str} 'l_offset_string0'
#         # {str} 'r_offset_string0'
#
#
# FEATURE_COUNT = 49
#
#
# def test_get_span_features():
#     """Test calling get_n_nearest_features with empty annoation document."""
#
#     project = Project(id_=None)
#     category = Category(project=project, id_=1)
#     project.add_category(category)
#     document = Document(project=project, category=category)
#     assert len(project.virtual_documents) == 1
#
#     df, feature_list = get_span_features(
#         document=document,
#         annotations=document.annotations(),
#     )
#     assert df.shape == (0, 49)
#     assert len(feature_list) == FEATURE_COUNT
#
#
# class TestSpanFeatures(unittest.TestCase):
#     def setUp(self):
#         self.project = Project(id_=None)
#         self.category = Category(project=self.project, id_=1)
#         self.project.add_category(self.category)
#
#         self.label_set = LabelSet(id_=33, project=self.project, categories=[self.category])
#         self.label = Label(id_=22, text='LabelName', project=self.project, label_sets=[self.label_set], threshold=0.5)
#         document_bbox = {
#             '0': {'x0': 0, 'x1': 1, 'y0': 0, 'y1': 1, 'top': 10, 'bottom': 11, 'page_number': 1},
#             '1': {'x0': 2, 'x1': 3, 'y0': 0, 'y1': 1, 'top': 10, 'bottom': 11, 'page_number': 1},
#             '3': {'x0': 3, 'x1': 4, 'y0': 0, 'y1': 1, 'top': 10, 'bottom': 11, 'page_number': 1},
#             '4': {'x0': 4, 'x1': 5, 'y0': 0, 'y1': 1, 'top': 10, 'bottom': 11, 'page_number': 1},
#         }
#         self.document = Document(
#             project=self.project,
#             category=self.category,
#             text='hi ha',
#             bbox=document_bbox,
#             dataset_status=2,
#             pages=[{'original_size': (100, 100)}],
#         )
#
#     def test_get_span_features(self):
#         span_1 = Span(start_offset=0, end_offset=2)
#         span_2 = Span(start_offset=3, end_offset=5)
#         annotation_set_1 = AnnotationSet(id_=1, document=self.document, label_set=self.label_set)
#         annotation_1 = Annotation(
#             document=self.document,
#             is_correct=True,
#             annotation_set=annotation_set_1,
#             label=self.label,
#             label_set=self.label_set,
#             spans=[span_1],
#         )
#         annotation_2 = Annotation(
#             document=self.document,
#             is_correct=True,
#             annotation_set=annotation_set_1,
#             label=self.label,
#             label_set=self.label_set,
#             spans=[span_2],
#         )
#         assert annotation_1.offset_string == ['hi']
#         assert annotation_2.offset_string == ['ha']
#
#         [span.bbox() for annotation in self.document.annotations() for span in annotation.spans]
#
#         df, feature_list = get_span_features(
#             document=self.document,
#             annotations=self.document.annotations(),
#         )
#         assert DocumentAnnotationMultiClassModel._SPAN_FEATURE_LIST == feature_list
#         assert list(df) == feature_list
#
#     def test_get_n_nearest_features_partial(self):
#         span_1 = Span(start_offset=0, end_offset=2)
#         span_2 = Span(start_offset=3, end_offset=4)
#         annotation_set_1 = AnnotationSet(id_=1, document=self.document, label_set=self.label_set)
#         annotation_1 = Annotation(
#             document=self.document,
#             is_correct=True,
#             annotation_set=annotation_set_1,
#             label=self.label,
#             label_set=self.label_set,
#             spans=[span_1],
#         )
#         annotation_2 = Annotation(
#             document=self.document,
#             is_correct=True,
#             annotation_set=annotation_set_1,
#             label=self.label,
#             label_set=self.label_set,
#             spans=[span_2],
#         )
#         assert annotation_1.offset_string == ['hi']
#         assert annotation_2.offset_string == ['h']
#
#         [span.bbox() for annotation in self.document.annotations() for span in annotation.spans]
#
#         df, feature_list = get_span_features(
#             document=self.document,
#             annotations=self.document.annotations(),
#         )
#         assert DocumentAnnotationMultiClassModel._SPAN_FEATURE_LIST == feature_list
#         assert list(df) == feature_list
#
#
# def test_get_spatial_features_empty():
#     """Test calling get_n_nearest_features with empty annoation document."""
#
#     project = Project(id_=None)
#     category = Category(project=project, id_=1)
#     project.add_category(category)
#     document = Document(project=project, category=category)
#     assert len(project.virtual_documents) == 1
#
#     abs_pos_feature_list = DocumentAnnotationMultiClassModel._ABS_POS_FEATURE_LIST
#     meta_information_list = DocumentAnnotationMultiClassModel._META_INFORMATION_LIST
#     df, feature_list = get_spatial_features(
#         annotations=document.annotations(),
#         abs_pos_feature_list=abs_pos_feature_list,
#         meta_information_list=meta_information_list,
#     )
#     assert df.empty
#     assert feature_list == []
#
#
# class TestSpanFeatures(unittest.TestCase):
#     def setUp(self):
#         self.project = Project(id_=None)
#         self.category = Category(project=self.project, id_=1)
#         self.project.add_category(self.category)
#
#         self.label_set = LabelSet(id_=33, project=self.project, categories=[self.category])
#         self.label = Label(id_=22, text='LabelName', project=self.project, label_sets=[self.label_set], threshold=0.5)
#         document_bbox = {
#             '0': {'x0': 0, 'x1': 1, 'y0': 0, 'y1': 1, 'top': 10, 'bottom': 11, 'page_number': 1},
#             '1': {'x0': 2, 'x1': 3, 'y0': 0, 'y1': 1, 'top': 10, 'bottom': 11, 'page_number': 1},
#             '3': {'x0': 3, 'x1': 4, 'y0': 0, 'y1': 1, 'top': 10, 'bottom': 11, 'page_number': 1},
#             '4': {'x0': 4, 'x1': 5, 'y0': 0, 'y1': 1, 'top': 10, 'bottom': 11, 'page_number': 1},
#         }
#         self.document = Document(
#             project=self.project,
#             category=self.category,
#             text='hi ha',
#             bbox=document_bbox,
#             dataset_status=2,
#             pages=[{'original_size': (100, 100)}],
#         )
#
#     def test_get_span_features(self):
#         span_1 = Span(start_offset=0, end_offset=2)
#         span_2 = Span(start_offset=3, end_offset=5)
#         annotation_set_1 = AnnotationSet(id_=1, document=self.document, label_set=self.label_set)
#         annotation_1 = Annotation(
#             document=self.document,
#             is_correct=True,
#             annotation_set=annotation_set_1,
#             label=self.label,
#             label_set=self.label_set,
#             spans=[span_1],
#         )
#         annotation_2 = Annotation(
#             document=self.document,
#             is_correct=True,
#             annotation_set=annotation_set_1,
#             label=self.label,
#             label_set=self.label_set,
#             spans=[span_2],
#         )
#         assert annotation_1.offset_string == ['hi']
#         assert annotation_2.offset_string == ['ha']
#
#         abs_pos_feature_list = DocumentAnnotationMultiClassModel._ABS_POS_FEATURE_LIST
#         meta_information_list = DocumentAnnotationMultiClassModel._META_INFORMATION_LIST
#
#         [span.bbox() for annotation in self.document.annotations() for span in annotation.spans]
#
#         df, feature_list = get_spatial_features(
#             annotations=self.document.annotations(),
#             abs_pos_feature_list=abs_pos_feature_list,
#             meta_information_list=meta_information_list,
#         )
#         assert abs_pos_feature_list == feature_list
#         assert list(df) == [
#             'id_',
#             'confidence',
#             'offset_string',
#             'normalized',
#             'start_offset',
#             'end_offset',
#             'is_correct',
#             'revised',
#             'annotation_id',
#             'document_id',
#             'x0',
#             'x1',
#             'y0',
#             'y1',
#             'page_index',
#             'line_index',
#             'x0_relative',
#             'x1_relative',
#             'y0_relative',
#             'y1_relative',
#             'page_index_relative',
#         ]
#
#         assert (df['x0'] == [0, 3]).all()
#         assert (df['x1'] == [3, 5]).all()
#         assert (df['y0'] == [0, 0]).all()
#         assert (df['y1'] == [1, 1]).all()
#
#         assert (df['page_index'] == [0, 0]).all()
#         assert (df['page_index_relative'] == [0, 0]).all()
#         assert (df['line_index'] == [0, 0]).all()
#
#         assert (df['x0_relative'] == [0 / 100, 3 / 100]).all()
#         assert (df['x1_relative'] == [3 / 100, 5 / 100]).all()
#         assert (df['y0_relative'] == [0 / 100, 0 / 100]).all()
#         assert (df['y1_relative'] == [1 / 100, 1 / 100]).all()
#
#     def test_get_n_nearest_features_partial(self):
#         span_1 = Span(start_offset=0, end_offset=2)
#         span_2 = Span(start_offset=3, end_offset=4)
#         annotation_set_1 = AnnotationSet(id_=1, document=self.document, label_set=self.label_set)
#         annotation_1 = Annotation(
#             document=self.document,
#             is_correct=True,
#             annotation_set=annotation_set_1,
#             label=self.label,
#             label_set=self.label_set,
#             spans=[span_1],
#         )
#         annotation_2 = Annotation(
#             document=self.document,
#             is_correct=True,
#             annotation_set=annotation_set_1,
#             label=self.label,
#             label_set=self.label_set,
#             spans=[span_2],
#         )
#         assert annotation_1.offset_string == ['hi']
#         assert annotation_2.offset_string == ['h']
#
#         abs_pos_feature_list = DocumentAnnotationMultiClassModel._ABS_POS_FEATURE_LIST
#         meta_information_list = DocumentAnnotationMultiClassModel._META_INFORMATION_LIST
#
#         [span.bbox() for annotation in self.document.annotations() for span in annotation.spans]
#
#         df, feature_list = get_spatial_features(
#             annotations=self.document.annotations(),
#             abs_pos_feature_list=abs_pos_feature_list,
#             meta_information_list=meta_information_list,
#         )
#         assert abs_pos_feature_list == feature_list
#         assert list(df) == [
#             'id_',
#             'confidence',
#             'offset_string',
#             'normalized',
#             'start_offset',
#             'end_offset',
#             'is_correct',
#             'revised',
#             'annotation_id',
#             'document_id',
#             'x0',
#             'x1',
#             'y0',
#             'y1',
#             'page_index',
#             'line_index',
#             'x0_relative',
#             'x1_relative',
#             'y0_relative',
#             'y1_relative',
#             'page_index_relative',
#         ]
#
#         assert (df['x0'] == [0, 3]).all()
#         assert (df['x1'] == [3, 4]).all()
#         assert (df['y0'] == [0, 0]).all()
#         assert (df['y1'] == [1, 1]).all()
#
#         assert (df['page_index'] == [0, 0]).all()
#         assert (df['page_index_relative'] == [0, 0]).all()
#         assert (df['line_index'] == [0, 0]).all()
#
#         assert (df['x0_relative'] == [0 / 100, 3 / 100]).all()
#         assert (df['x1_relative'] == [3 / 100, 4 / 100]).all()
#         assert (df['y0_relative'] == [0 / 100, 0 / 100]).all()
#         assert (df['y1_relative'] == [1 / 100, 1 / 100]).all()
