"""Provide a hardcoded setup for Project data."""
from konfuzio_sdk.data import Category, Project, LabelSet, Document, Label, Annotation, Span, AnnotationSet


class LocalTextProject(Project):
    """A Project without visual information for offline development monitored by tests in TestLocalTextProject."""

    local_training_document: Document = None

    def __init__(self):
        """Create basic structure of a Project."""
        super().__init__(id_=None)
        category = Category(project=self, id_=1)
        category_2 = Category(project=self, id_=2)
        label_set = LabelSet(id_=2, project=self, categories=[category, category_2])
        label = Label(id_=3, text='LabelName', project=self, label_sets=[label_set])
        label_new = Label(id_=4, text='LabelName 2', project=self, label_sets=[label_set])

        self.local_training_document = Document(
            project=self, category=category, text="Hi all, I like fish.", dataset_status=2
        )
        annotation_set = AnnotationSet(id_=4, document=self.local_training_document, label_set=label_set)
        _ = Annotation(
            id_=5,
            document=self.local_training_document,
            is_correct=True,
            confidence=1.0,
            annotation_set=annotation_set,
            label=label,
            label_set=label_set,
            spans=[Span(start_offset=3, end_offset=5)],
        )
        _ = Annotation(
            id_=6,
            document=self.local_training_document,
            is_correct=True,
            confidence=1.0,
            annotation_set=annotation_set,
            label=label_new,
            label_set=label_set,
            spans=[Span(start_offset=7, end_offset=10)],
        )

        document_test_a = Document(project=self, category=category, text="Hi all, I like fish.", dataset_status=3)
        annotation_set_test_a = AnnotationSet(id_=6, document=document_test_a, label_set=label_set)
        _ = Annotation(
            id_=7,
            document=document_test_a,
            is_correct=True,
            annotation_set=annotation_set_test_a,
            label=label,
            label_set=label_set,
            spans=[Span(start_offset=7, end_offset=10)],
        )

        _ = Annotation(
            id_=70,
            document=document_test_a,
            is_correct=True,
            annotation_set=annotation_set_test_a,
            label=label_new,
            label_set=label_set,
            spans=[Span(start_offset=11, end_offset=14)],
        )

        document_test_b = Document(project=self, category=category, text="Hi all,", dataset_status=3)
        annotation_set_test_b = AnnotationSet(id_=8, document=document_test_b, label_set=label_set)
        _ = Annotation(
            id_=9,
            document=document_test_b,
            is_correct=True,
            annotation_set=annotation_set_test_b,
            label=label,
            label_set=label_set,
            spans=[Span(start_offset=3, end_offset=6)],
        )

        # Category 2
        document_2 = Document(project=self, category=category_2, text="Morning.", dataset_status=2)
        annotation_set_2 = AnnotationSet(id_=10, document=document_2, label_set=label_set)
        _ = Annotation(
            id_=11,
            document=document_2,
            is_correct=True,
            annotation_set=annotation_set_2,
            label=label,
            label_set=label_set,
            spans=[Span(start_offset=0, end_offset=7)],
        )

        document_test_2 = Document(project=self, category=category_2, text="Morning.", dataset_status=3)
        annotation_set_test_2 = AnnotationSet(id_=5, document=document_test_2, label_set=label_set)
        _ = Annotation(
            id_=8,
            document=document_test_2,
            is_correct=True,
            annotation_set=annotation_set_test_2,
            label=label,
            label_set=label_set,
            spans=[Span(start_offset=0, end_offset=7)],
        )
