"""Test searching for the outlier Annotations under a given Label."""
import pytest

from konfuzio_sdk.settings_importer import is_dependency_installed

YOUR_LABEL_NAME = 'Austellungsdatum'


@pytest.mark.skipif(
    not is_dependency_installed('torch'),
    reason='Required dependencies not installed.',
)
def test_outlier_by_regex():
    """Test finding outlier Annotations by regex."""
    from tests.variables import TEST_PROJECT_ID

    YOUR_PROJECT_ID = TEST_PROJECT_ID
    # start project
    from konfuzio_sdk.data import Project

    project = Project(id_=YOUR_PROJECT_ID)

    label = project.get_label_by_name(YOUR_LABEL_NAME)
    outliers = label.get_probable_outliers_by_regex(project.categories)
    # end project
    label = project.get_label_by_name('Bank inkl. IBAN')
    outliers = label.get_probable_outliers_by_regex(project.categories, top_worst_percentage=1.0)
    outlier_spans = [span.offset_string for annotation in outliers for span in annotation.spans]
    assert len(outliers) == 16
    assert 'Deutsche Bank PGK  Nürnbe' in outlier_spans
    assert 'DE73 7607 0024 0568 9745 11' in outlier_spans


@pytest.mark.skipif(
    not is_dependency_installed('torch'),
    reason='Required dependencies not installed.',
)
def get_outlier_by_confidence():
    """Test finding outlier Annotations by confidence."""
    from konfuzio_sdk.trainer.information_extraction import RFExtractionAI
    from konfuzio_sdk.evaluate import ExtractionEvaluation
    from konfuzio_sdk.tokenizer.base import ListTokenizer
    from konfuzio_sdk.tokenizer.regex import RegexTokenizer
    from konfuzio_sdk.data import Project
    from tests.variables import TEST_PROJECT_ID

    YOUR_PROJECT_ID = TEST_PROJECT_ID

    project = Project(id_=YOUR_PROJECT_ID)
    # start get austellungsdatum
    label = project.get_label_by_name(YOUR_LABEL_NAME)
    # end get austellungsdatum
    pipeline = RFExtractionAI()
    pipeline.tokenizer = ListTokenizer(tokenizers=[])
    pipeline.category = label.project.get_category_by_id(id_=63)
    train_doc_ids = {44823, 44834, 44839, 44840, 44841}
    pipeline.documents = [doc for doc in pipeline.category.documents() if doc.id_ in train_doc_ids]
    for cur_label in pipeline.category.labels:
        for regex in cur_label.find_regex(category=pipeline.category):
            pipeline.tokenizer.tokenizers.append(RegexTokenizer(regex=regex))
    pipeline.test_documents = pipeline.category.test_documents()
    pipeline.df_train, pipeline.label_feature_list = pipeline.feature_function(
        documents=pipeline.documents, require_revised_annotations=False
    )
    pipeline.fit()
    predictions = []
    for doc in pipeline.documents:
        predicted_doc = pipeline.extract(document=doc)
        predictions.append(predicted_doc)
    evaluation = ExtractionEvaluation(documents=list(zip(pipeline.documents, predictions)), strict=False)
    # start confidence_outliers
    outliers = label.get_probable_outliers_by_confidence(evaluation)
    # end confidence_outliers
    outliers = label.get_probable_outliers_by_confidence(evaluation, 0.9)
    assert len(outliers) == 2
    outlier_spans = [span.offset_string for annotation in outliers for span in annotation.spans]
    assert '24.05.2018' in outlier_spans


def test_outliers_by_normalization():
    """Test finding outlier Annotations by normalization."""
    from tests.variables import TEST_PROJECT_ID

    YOUR_PROJECT_ID = TEST_PROJECT_ID
    # start normalization
    from konfuzio_sdk.data import Project

    project = Project(id_=YOUR_PROJECT_ID)
    label = project.get_label_by_name(YOUR_LABEL_NAME)
    outliers = label.get_probable_outliers_by_normalization(project.categories)
    # end normalization
    outlier_spans = [span.offset_string for annotation in outliers for span in annotation.spans]
    assert len(outliers) == 1
    assert '328927/10103' in outlier_spans
    assert '22.05.2018' in outlier_spans


@pytest.mark.skipif(
    not is_dependency_installed('torch'),
    reason='Required dependencies not installed.',
)
def test_three_methods():
    """Test combined search for outlier Annotations."""
    from tests.variables import TEST_PROJECT_ID

    YOUR_PROJECT_ID = TEST_PROJECT_ID
    # start combined
    from konfuzio_sdk.data import Project

    project = Project(id_=YOUR_PROJECT_ID)
    label = project.get_label_by_name(YOUR_LABEL_NAME)
    outliers = label.get_probable_outliers(project.categories, confidence_search=False)
    # end combined
    outliers = label.get_probable_outliers(project.categories, confidence_search=False, regex_worst_percentage=1)
    outlier_spans = [span.offset_string for annotation in outliers for span in annotation.spans]
    assert len(outliers) == 1
    assert '328927/10103' in outlier_spans
    assert '22.05.2018' in outlier_spans
