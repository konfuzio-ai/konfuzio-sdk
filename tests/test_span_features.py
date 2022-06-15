import unittest

from konfuzio_sdk.data import Project, Category, Document, LabelSet, Span, AnnotationSet, Label, Annotation
from konfuzio_sdk.pipelines.extraction_ai import DocumentAnnotationMultiClassModel
from konfuzio_sdk.pipelines.features import get_span_features

FEATURE_COUNT = 49


def test_get_span_features():
    """Test calling get_n_nearest_features with empty annoation document."""

    project = Project(id_=None)
    category = Category(project=project, id_=1)
    project.add_category(category)
    document = Document(project=project, category=category)
    assert len(project.virtual_documents) == 1

    df, feature_list = get_span_features(
        document=document,
        annotations=document.annotations(),
    )
    assert df.shape == (0, 49)
    assert len(feature_list) == FEATURE_COUNT


class TestSpanFeatures(unittest.TestCase):

    def setUp(self):
        self.project = Project(id_=None)
        self.category = Category(project=self.project, id_=1)
        self.project.add_category(self.category)

        self.label_set = LabelSet(id_=33, project=self.project, categories=[self.category])
        self.label = Label(id_=22, text='LabelName', project=self.project, label_sets=[self.label_set], threshold=0.5)
        document_bbox = {
            '0': {'x0': 0, 'x1': 1, 'y0': 0, 'y1': 1, 'top': 10, 'bottom': 11, 'page_number': 1},
            '1': {'x0': 2, 'x1': 3, 'y0': 0, 'y1': 1, 'top': 10, 'bottom': 11, 'page_number': 1},
            '3': {'x0': 3, 'x1': 4, 'y0': 0, 'y1': 1, 'top': 10, 'bottom': 11, 'page_number': 1},
            '4': {'x0': 4, 'x1': 5, 'y0': 0, 'y1': 1, 'top': 10, 'bottom': 11, 'page_number': 1}
        }
        self.document = Document(
            project=self.project,
            category=self.category,
            text='hi ha',
            bbox=document_bbox,
            dataset_status=2,
            pages=[{'original_size': (100, 100)}]
        )

    def test_get_span_features(self):
        span_1 = Span(start_offset=0, end_offset=2)
        span_2 = Span(start_offset=3, end_offset=5)
        annotation_set_1 = AnnotationSet(id_=1, document=self.document, label_set=self.label_set)
        annotation_1 = Annotation(
            document=self.document,
            is_correct=True,
            annotation_set=annotation_set_1,
            label=self.label,
            label_set=self.label_set,
            spans=[span_1],
        )
        annotation_2 = Annotation(
            document=self.document,
            is_correct=True,
            annotation_set=annotation_set_1,
            label=self.label,
            label_set=self.label_set,
            spans=[span_2],
        )
        assert annotation_1.offset_string == ['hi']
        assert annotation_2.offset_string == ['ha']

        [span.bbox() for annotation in self.document.annotations() for span in annotation.spans]

        df, feature_list = get_span_features(
            document=self.document,
            annotations=self.document.annotations(),
        )
        assert DocumentAnnotationMultiClassModel._SPAN_FEATURE_LIST == feature_list
        assert list(df) == feature_list

    def test_get_n_nearest_features_partial(self):
        span_1 = Span(start_offset=0, end_offset=2)
        span_2 = Span(start_offset=3, end_offset=4)
        annotation_set_1 = AnnotationSet(id_=1, document=self.document, label_set=self.label_set)
        annotation_1 = Annotation(
            document=self.document,
            is_correct=True,
            annotation_set=annotation_set_1,
            label=self.label,
            label_set=self.label_set,
            spans=[span_1],
        )
        annotation_2 = Annotation(
            document=self.document,
            is_correct=True,
            annotation_set=annotation_set_1,
            label=self.label,
            label_set=self.label_set,
            spans=[span_2],
        )
        assert annotation_1.offset_string == ['hi']
        assert annotation_2.offset_string == ['h']

        [span.bbox() for annotation in self.document.annotations() for span in annotation.spans]

        df, feature_list = get_span_features(
            document=self.document,
            annotations=self.document.annotations(),
        )
        assert DocumentAnnotationMultiClassModel._SPAN_FEATURE_LIST == feature_list
        assert list(df) == feature_list
