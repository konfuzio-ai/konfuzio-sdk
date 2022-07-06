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

        self.local_training_document = Document(project=self, category=category, text="Hi all,", dataset_status=2)
        annotation_set = AnnotationSet(id_=4, document=self.local_training_document, label_set=label_set)
        _ = Annotation(
            id_=5,
            document=self.local_training_document,
            is_correct=True,
            annotation_set=annotation_set,
            label=label,
            label_set=label_set,
            spans=[Span(start_offset=3, end_offset=6)],
        )

        document_test_a = Document(project=self, category=category, text="Hi all,", dataset_status=3)
        annotation_set_test_a = AnnotationSet(id_=6, document=document_test_a, label_set=label_set)
        _ = Annotation(
            id_=7,
            document=document_test_a,
            is_correct=True,
            annotation_set=annotation_set_test_a,
            label=label,
            label_set=label_set,
            spans=[Span(start_offset=3, end_offset=6)],
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

    # def combine_to_one_label(self):
    #     """
    #     Apply a Tokenizer on a list of Document and remove all Spans that can be found.
    #
    #     Use this approach to sequentially work on remaining Spans after a Tokenizer ran on a List of Documents.
    #
    #     :param tokenizer: A Tokenizer that runs on a list of Documents
    #     :param documents: Any list of Documents
    #
    #     :return: A new Project containing all missing Spans contained in a copied version of all Documents.
    #
    #     """
    #     warn('This method is WIP.', FutureWarning, stacklevel=2)
    #     virtual_project = Project(None)
    #     virtual_category = Category(project=virtual_project)
    #     virtual_label_set = virtual_project.no_label_set
    #     virtual_label = virtual_project.no_label
    #     for document in self.documents:
    #         compared = tokenizer.evaluate(document=document)  # todo summarize evaluation, as we are calculating it
    #         # return all Spans that were not found
    #         # todo add the "missing Spans" as a property to Tokenizer as a sideproduct for the evaluation.
    #         missing_spans = compared[(compared['is_correct']) & (compared['is_found_by_tokenizer'] == 0)]
    #         remaining_span_doc = Document(
    #             bbox=document.get_bbox(),
    #             pages=document.pages,
    #             text=document.text,
    #             project=virtual_project,
    #             category=virtual_category,
    #             dataset_status=document.dataset_status,
    #             copy_of_id=document.id_,
    #         )
    #         annotation_set_1 = AnnotationSet(document=remaining_span_doc, label_set=virtual_label_set)
    #         # add Spans to the virtual Document in case the Tokenizer was not able to find them
    #         for index, span_info in missing_spans.iterrows():
    #             # todo: Schema for bbox format https://gitlab.com/konfuzio/objectives/-/issues/8661
    #             new_span = Span(start_offset=span_info['start_offset'], end_offset=span_info['end_offset'])
    #             # todo add Tokenizer used to create Span
    #             _ = Annotation(
    #                 id_=int(span_info['id_']),
    #                 document=remaining_span_doc,
    #                 is_correct=True,
    #                 annotation_set=annotation_set_1,
    #                 label=virtual_label,
    #                 label_set=virtual_label_set,
    #                 spans=[new_span],
    #             )
    #         logger.warning(
    #             f'{len(remaining_span_doc.spans)} of {len(document.spans)} '
    #             f'correct Spans in {document} the abstract Tokenizer did not find.'
    #         )
    #     return virtual_project
