# -*- coding: utf-8 -*-
"""Test to train an Extraction AI."""

import logging
import math
import unittest

import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

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
)
from konfuzio_sdk.api import upload_ai_model
from konfuzio_sdk.data import Project
from konfuzio_sdk.tokenizer.regex import WhitespaceTokenizer
from tests.variables import OFFLINE_PROJECT, TEST_DOCUMENT_ID

logger = logging.getLogger(__name__)


class TestSequenceInformationExtraction(unittest.TestCase):
    """Test to train an extraction Model for Documents."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the Data and Pipeline."""
        # cls.project = Project(id_=None, project_folder=OFFLINE_PROJECT)
        cls.project = Project(id_=46)  # todo use offline project
        cls.pipeline = DocumentAnnotationMultiClassModel()

    def test_1_configure_pipeline(self):
        """Make sure the Data and Pipeline is configured."""
        self.pipeline.tokenizer = WhitespaceTokenizer()
        self.pipeline.category = self.project.get_category_by_id(id_=63)
        self.pipeline.documents = self.pipeline.category.documents()[:1]
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
        """Evaluate the model."""
        self.pipeline_path = self.pipeline.save(output_dir=self.project.model_folder)

    def test_5_evaluate_model(self):
        """Evaluate the model."""
        self.pipeline.evaluate()

    def test_6_extract_test_document(self):
        """Extract a randomly selected Test Document."""
        test_document = self.project.get_document_by_id(44823)
        self.pipeline.extract(document=test_document)

    @unittest.skip(reason='Test run offline.')
    def test_7_upload_ai_model(self):
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
        assert 'Features of Document do not match' in str(einfo.value)

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
        pipeline.extract(document)

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
        assert features['revised'].sum() == 1


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
