"""Provide a hardcoded setup for Project data."""
from konfuzio_sdk.data import Category, Project, LabelSet, Document, Label, Annotation, Span, AnnotationSet, Page


class LocalTextProject(Project):
    """A Project without visual information for offline development monitored by tests in TestLocalTextProject."""

    local_training_document: Document = None

    def __init__(self):
        """Create basic structure of a Project."""
        super().__init__(id_=None)
        category = Category(project=self, id_=1, name="CategoryName")
        default_label_set = LabelSet(id_=1, project=self, name="CategoryName", categories=[category])
        default_label = Label(id_=6, text='DefaultLabelName', project=self, label_sets=[default_label_set])

        category_2 = Category(project=self, id_=2, name="CategoryName 2")
        default_label_set_2 = LabelSet(id_=2, project=self, name="CategoryName 2", categories=[category_2])
        default_label_2 = Label(id_=7, text='DefaultLabelName 2', project=self, label_sets=[default_label_set_2])

        category_3 = Category(project=self, id_=3, name="CategoryName 3")

        category_4 = Category(project=self, id_=4, name="CategoryName 4")

        label_set = LabelSet(id_=3, project=self, name="LabelSetName", categories=[category, category_2])
        label = Label(id_=4, text='LabelName', project=self, label_sets=[label_set])
        label_new = Label(id_=5, text='LabelName 2', project=self, label_sets=[label_set])

        self.local_none_document = Document(
            project=self, category=category, text="Hi all, I like pizza.", dataset_status=0
        )

        self.local_training_document = Document(
            project=self, category=category, text="Hi all, I like fish.", dataset_status=2
        )
        default_annotation_set = AnnotationSet(
            id_=11, document=self.local_training_document, label_set=default_label_set
        )
        _ = Annotation(
            id_=12,
            document=self.local_training_document,
            is_correct=True,
            confidence=1.0,
            annotation_set=default_annotation_set,
            label=default_label,
            label_set=default_label_set,
            spans=[Span(start_offset=0, end_offset=2)],
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
        default_annotation_set_test_a = AnnotationSet(id_=13, document=document_test_a, label_set=default_label_set)
        _ = Annotation(
            id_=14,
            document=document_test_a,
            is_correct=True,
            confidence=1.0,
            annotation_set=default_annotation_set_test_a,
            label=default_label,
            label_set=default_label_set,
            spans=[Span(start_offset=0, end_offset=3)],
        )

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
        default_annotation_set_test_b = AnnotationSet(id_=15, document=document_test_b, label_set=default_label_set)
        _ = Annotation(
            id_=16,
            document=document_test_b,
            is_correct=True,
            confidence=1.0,
            annotation_set=default_annotation_set_test_b,
            label=default_label,
            label_set=default_label_set,
            spans=[Span(start_offset=0, end_offset=2)],
        )

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
        default_annotation_set_2 = AnnotationSet(id_=17, document=document_2, label_set=default_label_set_2)
        _ = Annotation(
            id_=18,
            document=document_2,
            is_correct=True,
            confidence=1.0,
            annotation_set=default_annotation_set_2,
            label=default_label_2,
            label_set=default_label_set_2,
            spans=[Span(start_offset=7, end_offset=8)],
        )

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
        default_annotation_set_test_2 = AnnotationSet(id_=19, document=document_test_2, label_set=default_label_set_2)
        _ = Annotation(
            id_=20,
            document=document_test_2,
            is_correct=True,
            confidence=1.0,
            annotation_set=default_annotation_set_test_2,
            label=default_label_2,
            label_set=default_label_set_2,
            spans=[Span(start_offset=7, end_offset=8)],
        )

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

        ##########
        # Document with overlapping Annotations to test view_annotation filtering

        doc_3_text = """anno1
date1:08/12/2001   span1
date2: 08/12/2001   span2
uncertain
 last x
"""
        document = Document(project=self, category=category, text=doc_3_text, dataset_status=3)
        label_set_2 = LabelSet(id_=4, project=self, categories=[category])
        view_label = Label(id_=8, text='ViewLabelName', project=self, label_sets=[label_set_2])
        view_label_date = Label(
            id_=9, text='date', project=self, label_sets=[label_set_2], has_multiple_top_candidates=False
        )
        view_label_3 = Label(id_=10, text='ViewLabelName 3', project=self, label_sets=[label_set_2])
        view_label_4 = Label(id_=11, text='ViewLabelName 4', project=self, label_sets=[label_set_2], threshold=0.1)
        view_label_5 = Label(id_=12, text='ViewLabelName 5', project=self, label_sets=[label_set_2])

        view_annotation_set_test = AnnotationSet(id_=6, document=document, label_set=label_set_2)

        # to be displayed in smart view: #16, #18, #19, #24
        _ = Annotation(
            id_=15,
            document=document,
            is_correct=False,
            annotation_set=view_annotation_set_test,
            confidence=0.92,
            label=view_label_3,
            label_set=label_set_2,
            spans=[Span(start_offset=0, end_offset=5)],
        )

        _ = Annotation(
            id_=16,
            document=document,
            is_correct=True,
            annotation_set=view_annotation_set_test,
            label=view_label,
            label_set=label_set_2,
            spans=[Span(start_offset=0, end_offset=5)],
        )
        _ = Annotation(
            id_=17,
            document=document,
            is_correct=False,
            annotation_set=view_annotation_set_test,
            confidence=0.3,
            label=view_label_date,
            label_set=label_set_2,
            spans=[Span(start_offset=12, end_offset=22)],
        )
        _ = Annotation(
            id_=18,
            document=document,
            is_correct=False,
            annotation_set=view_annotation_set_test,
            confidence=0.4,
            label=view_label_date,
            label_set=label_set_2,
            spans=[Span(start_offset=38, end_offset=48)],
        )
        _ = Annotation(
            id_=19,
            document=document,
            is_correct=False,
            annotation_set=view_annotation_set_test,
            confidence=0.9,
            label=view_label_3,
            label_set=label_set_2,
            spans=[Span(start_offset=25, end_offset=30), Span(start_offset=51, end_offset=56)],
        )
        _ = Annotation(
            id_=20,
            document=document,
            is_correct=False,
            annotation_set=view_annotation_set_test,
            confidence=0.6,
            label=view_label_3,
            label_set=label_set_2,
            spans=[Span(start_offset=25, end_offset=30)],
        )
        _ = Annotation(
            id_=21,
            document=document,
            is_correct=False,
            annotation_set=view_annotation_set_test,
            confidence=0.05,
            label=view_label_4,
            label_set=label_set,
            spans=[Span(start_offset=57, end_offset=66)],
        )
        _ = Annotation(
            id_=22,
            document=document,
            is_correct=False,
            revised=True,
            annotation_set=view_annotation_set_test,
            confidence=1,
            label=view_label_4,
            label_set=label_set,
            spans=[Span(start_offset=37, end_offset=49)],
        )
        _ = Annotation(
            id_=23,
            document=document,
            is_correct=False,
            annotation_set=view_annotation_set_test,
            confidence=0.9,
            label=view_label_5,
            label_set=label_set,
            spans=[Span(start_offset=67, end_offset=73)],
        )
        _ = Annotation(
            id_=24,
            document=document,
            is_correct=False,
            annotation_set=view_annotation_set_test,
            confidence=0.99,
            label=view_label_5,
            label_set=label_set,
            spans=[Span(start_offset=68, end_offset=72)],
        )

        # Documents with sub-Documents in them

        text_3 = "Hi all,\nI like bread.\nI hope to get everything done soon.\nMorning,\nI'm glad to see you.\nMorning,"
        document_3 = Document(id_=None, project=self, category=category_3, text=text_3, dataset_status=3)
        _ = Page(
            id_=None,
            original_size=(320, 240),
            document=document_3,
            start_offset=0,
            end_offset=21,
            number=1,
        )

        _ = Page(
            id_=None,
            original_size=(320, 240),
            document=document_3,
            start_offset=22,
            end_offset=57,
            number=2,
        )

        _ = Page(
            id_=None,
            original_size=(320, 240),
            document=document_3,
            start_offset=58,
            end_offset=66,
            number=3,
        )

        _ = Page(
            id_=None,
            original_size=(320, 240),
            document=document_3,
            start_offset=67,
            end_offset=87,
            number=4,
        )

        _ = Page(
            id_=None,
            original_size=(320, 240),
            document=document_3,
            start_offset=88,
            end_offset=97,
            number=5,
        )

        text_4 = "Morning,\nI like bread.\n\fI hope to get everything done soon."
        document_4 = Document(id_=None, project=self, category=category_3, text=text_4, dataset_status=2)
        _ = Page(
            id_=None,
            original_size=(320, 240),
            document=document_4,
            start_offset=0,
            end_offset=22,
            number=1,
        )

        _ = Page(
            id_=None,
            original_size=(320, 240),
            document=document_4,
            start_offset=23,
            end_offset=57,
            number=2,
        )

        text_5 = "Morning,\nI like bread.\n\fWhat are your plans for today?"
        document_5 = Document(id_=None, project=self, category=category_3, text=text_5, dataset_status=2)
        _ = Page(
            id_=None,
            original_size=(320, 240),
            document=document_5,
            start_offset=0,
            end_offset=22,
            number=1,
        )

        _ = Page(
            id_=None,
            original_size=(320, 240),
            document=document_5,
            start_offset=23,
            end_offset=52,
            number=2,
        )

        text_7 = "Evening,\nI like fish.\n\fHow was your day?"
        document_7 = Document(id_=None, project=self, category=category_4, text=text_7, dataset_status=2)
        _ = Page(
            id_=None,
            original_size=(320, 240),
            document=document_7,
            start_offset=0,
            end_offset=21,
            number=1,
        )

        _ = Page(
            id_=None,
            original_size=(320, 240),
            document=document_7,
            start_offset=22,
            end_offset=39,
            number=2,
        )

        text_8 = (
            "Hi all,\nI like bread.\n\fWhat are your plans for today?\n\fEvening,\nI like it.\n\fHow was your week?"
        )
        document_8 = Document(id_=None, project=self, category=category_3, text=text_8, dataset_status=3)
        _ = Page(id_=None, original_size=(320, 240), document=document_8, start_offset=0, end_offset=21, number=1)
        _ = Page(id_=None, original_size=(320, 240), document=document_8, start_offset=22, end_offset=53, number=2)
        _ = Page(id_=None, original_size=(320, 240), document=document_8, start_offset=54, end_offset=74, number=3)
        _ = Page(id_=None, original_size=(320, 240), document=document_8, start_offset=75, end_offset=93, number=4)

        text_9 = text_8 + "\n\fEvening,"
        document_9 = Document(id_=None, project=self, category=category_4, text=text_9, dataset_status=3)
        _ = Page(id_=None, original_size=(320, 240), document=document_9, start_offset=0, end_offset=21, number=1)
        _ = Page(id_=None, original_size=(320, 240), document=document_9, start_offset=22, end_offset=53, number=2)
        _ = Page(id_=None, original_size=(320, 240), document=document_9, start_offset=54, end_offset=74, number=3)
        _ = Page(id_=None, original_size=(320, 240), document=document_9, start_offset=75, end_offset=93, number=4)
        _ = Page(id_=None, original_size=(320, 240), document=document_9, start_offset=94, end_offset=102, number=5)
