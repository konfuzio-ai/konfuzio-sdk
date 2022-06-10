from konfuzio_sdk.data import Project, Category, Document, LabelSet, Span, AnnotationSet, Label, Annotation
from konfuzio_sdk.pipelines.extraction_ai import DocumentAnnotationMultiClassModel

FEATURE_COUNT = 49


def test_get_n_nearest_features_empty():
    """Test calling get_n_nearest_features with empty annoation document."""

    project = Project(id_=None)
    category = Category(project=project, id_=1)
    project.add_category(category)
    document = Document(project=project, category=category)
    assert len(project.virtual_documents) == 1

    extraction_ai = DocumentAnnotationMultiClassModel(category=category)
    extraction_ai.get_n_nearest_features(document=document, annotations=[])


def test_get_n_nearest_features():
    project = Project(id_=None)
    category = Category(project=project, id_=1)
    project.add_category(category)

    label_set = LabelSet(id_=33, project=project, categories=[category])
    label = Label(id_=22, text='LabelName', project=project, label_sets=[label_set], threshold=0.5)
    document_bbox = {
        '0': {'x0': 0, 'x1': 1, 'y0': 0, 'y1': 1, 'top': 10, 'bottom': 11, 'page_number': 1},
        '1': {'x0': 2, 'x1': 3, 'y0': 0, 'y1': 1, 'top': 10, 'bottom': 11, 'page_number': 1},
        '3': {'x0': 3, 'x1': 4, 'y0': 0, 'y1': 1, 'top': 10, 'bottom': 11, 'page_number': 1},
        '4': {'x0': 4, 'x1': 5, 'y0': 0, 'y1': 1, 'top': 10, 'bottom': 11, 'page_number': 1}
    }
    document = Document(
        project=project,
        category=category,
        text='hi ha',
        bbox=document_bbox,
        dataset_status=2,
        pages=[{'original_size': (100, 100)}]
    )
    span_1 = Span(start_offset=0, end_offset=2)
    span_2 = Span(start_offset=3, end_offset=5)
    annotation_set_1 = AnnotationSet(id_=1, document=document, label_set=label_set)
    annotation_1 = Annotation(
        document=document,
        is_correct=True,
        annotation_set=annotation_set_1,
        label=label,
        label_set=label_set,
        spans=[span_1],
    )
    annotation_2 =Annotation(
        document=document,
        is_correct=True,
        annotation_set=annotation_set_1,
        label=label,
        label_set=label_set,
        spans=[span_2],
    )
    assert annotation_1.offset_string == ['hi']
    assert annotation_2.offset_string == ['ha']
    extraction_ai = DocumentAnnotationMultiClassModel(category=category)
    neighbours = extraction_ai.n_nearest_left + extraction_ai.n_nearest_right

    df, feature_list = extraction_ai.get_n_nearest_features(document=document, annotations=document.annotations())
    assert len(feature_list) == FEATURE_COUNT * neighbours + len(['l_dist0', 'l_dist1', 'r_dist0', 'r_dist1'])
    assert df.shape == (2, FEATURE_COUNT * neighbours + 4 + 4)
    for key in ['l0', 'l1', 'r0', 'r1']:
        assert len([x for x in df if x.startswith(key)]) == FEATURE_COUNT  # We have 49 feature entries.

    assert (df['l_offset_string0'] == ['', 'hi']).all()
    assert (df['r_offset_string0'] == ['ha', '']).all()
    assert (df['l_offset_string1'] == ['', '']).all()
    assert (df['r_offset_string1'] == ['', '']).all()
    # in df but no feature
    # {str} 'l_offset_string1'
    # {str} 'r_offset_string1'
    # {str} 'l_offset_string0'
    # {str} 'r_offset_string0'

def test_get_n_nearest_features_partial():
    project = Project(id_=None)
    category = Category(project=project, id_=1)
    project.add_category(category)

    label_set = LabelSet(id_=33, project=project, categories=[category])
    label = Label(id_=22, text='LabelName', project=project, label_sets=[label_set], threshold=0.5)
    document_bbox = {
        '0': {'x0': 0, 'x1': 1, 'y0': 0, 'y1': 1, 'top': 10, 'bottom': 11, 'page_number': 1},
        '1': {'x0': 2, 'x1': 3, 'y0': 0, 'y1': 1, 'top': 10, 'bottom': 11, 'page_number': 1},
        '3': {'x0': 3, 'x1': 4, 'y0': 0, 'y1': 1, 'top': 10, 'bottom': 11, 'page_number': 1},
        '4': {'x0': 4, 'x1': 5, 'y0': 0, 'y1': 1, 'top': 10, 'bottom': 11, 'page_number': 1}
    }
    document = Document(
        project=project,
        category=category,
        text='hi ha',
        bbox=document_bbox,
        dataset_status=2,
        pages=[{'original_size': (100, 100)}]
    )
    span_1 = Span(start_offset=0, end_offset=2)
    span_2 = Span(start_offset=3, end_offset=4)
    annotation_set_1 = AnnotationSet(id_=1, document=document, label_set=label_set)
    annotation_1 = Annotation(
        document=document,
        is_correct=True,
        annotation_set=annotation_set_1,
        label=label,
        label_set=label_set,
        spans=[span_1],
    )
    annotation_2 =Annotation(
        document=document,
        is_correct=True,
        annotation_set=annotation_set_1,
        label=label,
        label_set=label_set,
        spans=[span_2],
    )
    assert annotation_1.offset_string == ['hi']
    assert annotation_2.offset_string == ['h']
    extraction_ai = DocumentAnnotationMultiClassModel(category=category)
    neighbours = extraction_ai.n_nearest_left + extraction_ai.n_nearest_right

    df, feature_list = extraction_ai.get_n_nearest_features(document=document, annotations=document.annotations())
    assert len(feature_list) == FEATURE_COUNT * neighbours + len(['l_dist0', 'l_dist1', 'r_dist0', 'r_dist1'])
    assert df.shape == (2, FEATURE_COUNT * neighbours + 4 + 4)
    for key in ['l0', 'l1', 'r0', 'r1']:
        assert len([x for x in df if x.startswith(key)]) == FEATURE_COUNT  # We have 49 feature entries.

    assert (df['l_offset_string0'] == ['', 'hi']).all()
    assert (df['r_offset_string0'] == ['ha', '']).all()
    assert (df['l_offset_string1'] == ['', '']).all()
    assert (df['r_offset_string1'] == ['', '']).all()
    # in df but no feature
    # {str} 'l_offset_string1'
    # {str} 'r_offset_string1'
    # {str} 'l_offset_string0'
    # {str} 'r_offset_string0'