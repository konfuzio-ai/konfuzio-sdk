"""Test searching for the outlier Annotations under a given Label."""
from konfuzio_sdk.data import Project
from variables import YOUR_PROJECT_ID

YOUR_LABEL_NAME = 'Austellungsdatum'

project = Project(id_=YOUR_PROJECT_ID)
label = project.get_label_by_name('Bank inkl. IBAN')
outliers = label.get_probable_outliers_by_regex(project.categories)
outlier_spans = [span.offset_string for annotation in outliers for span in annotation.spans]
assert len(outliers) == 16
assert 'Deutsche Bank PGK  Nürnbe' in outlier_spans
assert 'DE73 7607 0024 0568 9745 11' in outlier_spans
label = project.get_label_by_name('Firmenname')
outliers = label.get_probable_outliers_by_confidence(project.categories, n_outliers=3)
assert len(outliers) == 3
for annotation in outliers:
    assert annotation.confidence < 0.5
    assert annotation.is_correct
label = project.get_label_by_name(YOUR_LABEL_NAME)
outliers = label.get_probable_outliers_by_normalization(project.categories)
outlier_spans = [span.offset_string for annotation in outliers for span in annotation.spans]
assert len(outliers) == 1
assert '328927/10103' in outlier_spans
assert '22.05.2018' in outlier_spans
outliers = label.get_probable_outliers(project.categories, confidence_search=False)
outlier_spans = [span.offset_string for annotation in outliers for span in annotation.spans]
assert len(outliers) == 1
assert '328927/10103' in outlier_spans
assert '22.05.2018' in outlier_spans
