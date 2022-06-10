"""Test the evaluation."""
import unittest

import pandas
from pandas import DataFrame

from konfuzio_sdk.data import Project, Document, AnnotationSet, Annotation, Span, LabelSet, Label, Category
from konfuzio_sdk.evaluate import compare, grouped, Evaluation


TEST_PROJECT_ID = 46


class TestEvaluation(unittest.TestCase):
    """Testing evaluation of the Konfuzio Server.

    Implemented:
        - prediction without complete offsets (e.g missing last character)
        - missing prediction for a Label with multiple=True (https://app.konfuzio.com/a/7344142)
        - evaluation of Annotations with multiple=False in a strict mode, so all must be found
        - missing Annotation Sets (e.g. missing 1st and 3rd Annotation Set in a  Document)
        - missing correct Annotations in annotation-set
        - too many Annotations than correct Annotations in annotation-set
        - too many annotation-sets
        - correct Annotations that are predicted to be in different annotation-sets
        - multiline Annotations with multiple not connected offsets
        - multiple Annotation Sets in 1 line
        - any possible grouping of Annotations into annotation-set(s)
        - if two offsets are correctly grouped into a correct number of Annotations, to evaluate horizontal and vertical
            merging

    Reasoning on how to evaluate them before implementation needed:
        - prediction with incorrect offsets but with correct offset string are no longer possible
        - prediction with incorrect offsets and incorrect offset string
        - prediction of one of multiple Annotations for a Label in one annotation
        - Annotation with custom string

    """

    def test_doc_on_doc_incl_multiline_annotation(self):
        """Test if a Document is 100 % equivalent even it has unrevised Annotations."""
        prj = Project(id_=TEST_PROJECT_ID)
        doc_a = prj.get_document_by_id(44823)
        doc_b = prj.get_document_by_id(44823)  # predicted
        evaluation = compare(doc_a, doc_b)
        assert len(evaluation) == 23  # 24 if considering negative Annotations
        # for an Annotation which is human made, it is nan, so that above threshold is False
        # doc_a 19 + 2 multiline + 2 feedback required + 1 rejected
        assert evaluation["true_positive"].sum() == 21  # 1 multiline with 2 lines = 2 Annotations
        assert evaluation["false_positive"].sum() == 0
        # due to the fact that Konfuzio Server does not save confidence = 100 % if Annotation was not created by a human
        assert evaluation["false_negative"].sum() == 0
        assert evaluation["is_found_by_tokenizer"].sum() == 21
        e = Evaluation(evaluation)
        assert e.accuracy == 1.0
        assert e.f1_score == 1.0
        assert e.precision == 1.0
        assert e.recall == 1.0
        e.label_evaluations()

    def test_empty_evaluation(self):
        e = Evaluation(DataFrame())
        e.label_evaluations(dataset_status=[3])

    def test_doc_where_first_annotation_was_skipped(self):
        """Test if a Document is 100 % equivalent with first Annotation not existing for a certain Label."""
        prj = Project(id_=TEST_PROJECT_ID)
        doc_a = prj.get_document_by_id(44823)
        doc_b = prj.get_document_by_id(44823)  # predicted
        doc_b.annotations()
        doc_b._annotations.pop(0)  # pop an Annotation that is correct in BOTH  Documents
        assert doc_a._annotations == doc_b._annotations  # first Annotation is removed in both  Documents
        evaluation = compare(doc_a, doc_b)
        assert len(evaluation) == 21  # 22 if considering negative Annotations, 2 Annotations are is_correct false
        # doc_a 18 (multiline removed) + 1 multiline + 2 feedback required + 1 rejected
        assert evaluation["true_positive"].sum() == 19
        assert evaluation["false_positive"].sum() == 0
        assert evaluation["false_negative"].sum() == 0
        assert evaluation["is_found_by_tokenizer"].sum() == 19

    def test_doc_where_last_annotation_was_skipped(self):
        """Test if a Document is 100 % equivalent with last Annotation not existing for a certain Label."""
        prj = Project(id_=TEST_PROJECT_ID)
        doc_a = prj.get_document_by_id(44823)
        doc_b = prj.get_document_by_id(44823)  # predicted
        doc_b.annotations()
        doc_b._annotations.pop(11)  # pop an Annotation that is correct in BOTH  Documents
        assert doc_a._annotations == doc_b._annotations  # last Annotation is removed in both  Documents
        evaluation = compare(doc_a, doc_b)
        assert len(evaluation) == 22  # 23 if considering negative Annotations, 2 Annotations are is_correct false
        # doc_a 18 + 2 multiline + 2 feedback required + 1 rejected
        assert evaluation["true_positive"].sum() == 20
        assert evaluation["false_positive"].sum() == 0
        assert evaluation["false_negative"].sum() == 0
        assert evaluation["is_found_by_tokenizer"].sum() == 20

    def test_if_first_multiline_annotation_is_missing_in_b(self):
        """Test if a Document is equivalent if first Annotation is missing."""
        prj = Project(id_=TEST_PROJECT_ID)
        doc_a = prj.get_document_by_id(44823)
        doc_b = Document(project=prj, category=doc_a.category)
        for annotation in doc_a.annotations()[1:]:
            doc_b.add_annotation(annotation)

        assert len(doc_b.annotations()) == len(doc_a.annotations()) - 1
        evaluation = compare(doc_a, doc_b)
        assert len(evaluation) == 23  # 24 if considering negative Annotations, 2 Annotations are false
        # doc_a 19 + 2 multiline + 2 feedback required + 1 rejected
        assert evaluation["true_positive"].sum() == 19  # 1 multiline with 2 lines = 2 Annotations
        assert evaluation["false_positive"].sum() == 0
        assert evaluation["false_negative"].sum() == 2
        assert evaluation["is_found_by_tokenizer"].sum() == 19

    def test_doc_where_first_annotation_is_missing_in_a(self):
        """Test if a Document is equivalent if first Annotation is not present."""
        prj = Project(id_=TEST_PROJECT_ID)
        doc_b = prj.get_document_by_id(44823)
        doc_a = Document(project=prj, category=doc_b.category)
        # use only correct Annotations
        for annotation in doc_b.annotations()[1:]:
            doc_a.add_annotation(annotation)

        # evaluate on doc_b and assume the feedback required ones are correct
        assert len(doc_a.annotations()) == len(doc_b.annotations()) - 1
        evaluation = compare(doc_a, doc_b)
        # 24 if considering negative Annotations, 2 annotations are false and two have feedback required
        assert len(evaluation) == 23
        assert evaluation["true_positive"].sum() == 19
        assert evaluation["false_positive"].sum() == 4  # 1 multiline (2 lines == 2 Annotations) + 2 feedback required
        assert evaluation["false_negative"].sum() == 0
        assert evaluation["is_found_by_tokenizer"].sum() == 19

    def test_only_unrevised_annotations(self):
        """Test to evaluate on a Document that has only unrevised Annotations."""
        prj = Project(id_=TEST_PROJECT_ID)
        doc_a = prj.get_document_by_id(137234)
        doc_b = Document(project=prj, category=doc_a.category)

        assert len(doc_a.annotations()) == len(doc_b.annotations()) == 0
        evaluation = compare(doc_a, doc_b, only_use_correct=True)
        assert len(evaluation) == 1  # placeholder
        assert evaluation["true_positive"].sum() == 0
        assert evaluation["false_positive"].sum() == 0
        assert evaluation["false_negative"].sum() == 0
        assert evaluation["is_found_by_tokenizer"].sum() == 0

    def test_doc_where_first_annotation_from_all_is_missing_in_a(self):
        """Test if a Document is equivalent if all Annotation are not present and feedback required are included."""
        prj = Project(id_=TEST_PROJECT_ID)
        doc_b = prj.get_document_by_id(44823)
        doc_a = Document(project=prj, category=doc_b.category)
        # use correct Annotations and feedback required ones
        for annotation in doc_b.annotations(use_correct=False)[1:]:
            doc_a.add_annotation(annotation)

        assert len(doc_a.annotations()) == len(doc_b.annotations()) - 1
        evaluation = compare(doc_a, doc_b)
        assert len(evaluation) == 23  # 24 if considering negative Annotations, 2 Annotations are false
        # doc_a 18 + 1 multiline + 2 feedback required + 1 rejected
        assert evaluation["true_positive"].sum() == 19
        assert evaluation["false_positive"].sum() == 2  # 1 multiline (2 lines == 2 Annotations)
        assert evaluation["false_negative"].sum() == 0
        assert evaluation["is_found_by_tokenizer"].sum() == 19

    def test_doc_where_last_annotation_is_missing_in_b(self):
        """Test if a Document is equivalent if last Annotation is missing."""
        prj = Project(id_=TEST_PROJECT_ID)
        doc_a = prj.get_document_by_id(44823)
        doc_b = Document(project=prj, category=doc_a.category)
        # use correct Annotations and feedback required ones
        for annotation in doc_a.annotations(use_correct=False)[:-1]:
            doc_b.add_annotation(annotation)

        evaluation = compare(doc_a, doc_b)
        assert len(evaluation) == 23  # 24 if considering negative Annotations, 2 Annotations are false
        # doc_a 19 + 2 multiline
        assert evaluation["true_positive"].sum() == 20  # due to the fact that we find both offsets of the multiline
        assert evaluation["false_positive"].sum() == 0
        assert evaluation["false_negative"].sum() == 1
        assert evaluation["is_found_by_tokenizer"].sum() == 20

    def test_doc_where_last_annotation_is_missing_in_a(self):
        """Test if a Document is equivalent if last Annotation is not present."""
        prj = Project(id_=TEST_PROJECT_ID)
        doc_b = prj.get_document_by_id(44823)
        doc_a = Document(project=prj, category=doc_b.category)
        # use correct Annotations and feedback required ones
        for annotation in doc_b.annotations(use_correct=False)[:-1]:
            doc_a.add_annotation(annotation)

        evaluation = compare(doc_a, doc_b)
        assert len(evaluation) == 23  # 24 if considering negative, 2 Annotations are false
        # doc_a 18 + 2 multiline
        assert evaluation["true_positive"].sum() == 20  # due to the fact that we find both offsets of the multiline
        assert evaluation["false_positive"].sum() == 1
        assert evaluation["false_negative"].sum() == 0
        assert evaluation["is_found_by_tokenizer"].sum() == 20

    def test_nothing_should_be_predicted(self):
        """Support to evaluate that nothing is found in a document."""
        prj = Project(id_=TEST_PROJECT_ID)
        doc_b = prj.get_document_by_id(44823)
        doc_a = Document(project=prj, category=doc_b.category)
        evaluation = compare(doc_a, doc_b)
        assert len(evaluation) == 24  # 25 if considering negative Annotations
        assert evaluation["true_positive"].sum() == 0
        # any Annotation above threshold is a false positive independent if it's correct or revised
        assert len([an for an in doc_b.annotations(use_correct=False) if an.confidence > an.label.threshold]) == 21
        assert evaluation["false_positive"].sum() == 23  # but one annotation is multiline
        assert evaluation["false_negative"].sum() == 0
        assert evaluation["is_found_by_tokenizer"].sum() == 0

    def test_nothing_can_be_predicted(self):
        """Support to evaluate that nothing must be found in a document."""
        prj = Project(id_=TEST_PROJECT_ID)
        doc_a = prj.get_document_by_id(44823)
        doc_b = Document(project=prj, category=doc_a.category)
        evaluation = compare(doc_a, doc_b)
        # 25 if considering negative Annotations, we evaluate on span level an one annotation is multiline
        assert len(evaluation) == 24
        assert evaluation["true_positive"].sum() == 0
        assert evaluation["false_positive"].sum() == 0
        assert evaluation["false_negative"].sum() == 21
        assert evaluation["is_found_by_tokenizer"].sum() == 0

    def test_doc_with_overruled_top_annotations(self):
        """
        Test if a Document is equivalent if prediction follows the top Annotation logic.

        The top Annotation logic considers only 1 Annotation for Labels with multiple=False.
        For example, the "Personalausweis" has multiple=False but several Annotations exist in the document.
        Only 1 is in the prediction.
        """
        # todo: this logic is a view logic on the document: shouldn't this go into the Annotations function
        prj = Project(id_=TEST_PROJECT_ID)
        doc_a = prj.get_document_by_id(44823)
        doc_b = Document(project=prj, category=doc_a.category)

        found = False
        for annotation in doc_a.annotations(use_correct=False):
            if annotation.label.id_ == 12444:
                if found:
                    continue
                found = True

            doc_b.add_annotation(annotation)

        evaluation = compare(doc_a, doc_b)
        assert len(evaluation) == 23  # 24 if considering negative Annotations,
        # Evaluation as it is now: everything needs to be find even if multiple=False
        assert evaluation["true_positive"].sum() == 20
        assert evaluation["false_positive"].sum() == 0
        assert evaluation["false_negative"].sum() == 1
        assert evaluation["is_found_by_tokenizer"].sum() == 20

    def test_doc_with_missing_annotation_set(self):
        """
        Test if we detect an Annotation of a missing Annotation Set.

        Prepare a Project with two Documents, where the first Document has two Annotation Set and the second Document
        has one Annotation Set. However, one the Annotation in one of the Annotation Sets is correct, i.e. contains
        a matching Span, has the correct Label, has the correct Label Set and it's confidence is equal or higher than
        the threshold of the Label.

        Missing Annotation must be evaluated as False Negatives.
        """
        project = Project(id_=None)
        category = Category(project=project)
        label_set = LabelSet(id_=33, project=project, categories=[category])
        label = Label(id_=22, project=project, label_sets=[label_set], threshold=0.5)
        # create a Document A
        document_a = Document(project=project, category=category)
        # first Annotation Set
        span_1 = Span(start_offset=1, end_offset=2)
        annotation_set_a_1 = AnnotationSet(id_=1, document=document_a, label_set=label_set)
        _ = Annotation(
            id_=1,
            document=document_a,
            confidence=0.5,
            is_correct=True,
            annotation_set=annotation_set_a_1,
            label=label,
            label_set=label_set,
            spans=[span_1],
        )
        # second Annotation Set
        span_2 = Span(start_offset=3, end_offset=5)
        annotation_set_a_2 = AnnotationSet(id_=2, document=document_a, label_set=label_set)
        _ = Annotation(
            id_=2,
            document=document_a,
            is_correct=True,
            annotation_set=annotation_set_a_2,
            label=label,
            label_set=label_set,
            spans=[span_2],
        )
        # create a Document B
        span_3 = Span(start_offset=3, end_offset=5)
        document_b = Document(project=project, category=category)
        # with only one Annotation Set, so to say the first one, which then contains the correct Annotation which is in
        # second Annotation Set. This test includes to test if we did not find the first Annotation Set.
        annotation_set_b = AnnotationSet(id_=3, document=document_b, label_set=label_set)
        _ = Annotation(
            id_=3,
            document=document_b,
            confidence=0.5,
            is_correct=True,
            annotation_set=annotation_set_b,
            label=label,
            label_set=label_set,
            spans=[span_3],
        )

        evaluation = compare(document_a, document_b)
        assert len(evaluation) == 2
        assert evaluation["true_positive"].sum() == 1
        assert evaluation["false_positive"].sum() == 0
        assert evaluation["false_negative"].sum() == 1
        assert evaluation["is_found_by_tokenizer"].sum() == 1

    def test_documents_with_different_category(self):
        """Test to not compare two Documents with different Categories."""
        project = Project(id_=None)
        category = Category(project=project)
        document_a = Document(project=project, category=category)
        another_category = Category(project=project)
        document_b = Document(project=project, category=another_category)
        with self.assertRaises(ValueError) as context:
            compare(document_a, document_b)
            assert 'do not match' in context.exception

    def test_doc_with_annotation_with_wrong_offsets(self):
        """
        Test a Document where the Annotation has a Span with wrong offsets.

        It is counted as FP because does not match any correct Annotation and as FN because there was no correct
        prediction for the Annotation in the Document. We are double penalizing here.
        """
        project = Project(id_=None)
        category = Category(project=project)
        label_set = LabelSet(id_=33, project=project, categories=[category])
        label = Label(id_=22, project=project, label_sets=[label_set], threshold=0.5)
        # create a Document A
        document_a = Document(project=project, category=category)
        # first Annotation Set
        span_1 = Span(start_offset=1, end_offset=2)
        annotation_set_a_1 = AnnotationSet(id_=1, document=document_a, label_set=label_set)
        _ = Annotation(
            id_=1,
            document=document_a,
            confidence=0.5,
            is_correct=True,
            annotation_set=annotation_set_a_1,
            label=label,
            label_set=label_set,
            spans=[span_1],
        )
        # create a Document B
        document_b = Document(project=project, category=category)
        span_3 = Span(start_offset=3, end_offset=5)
        annotation_set_b = AnnotationSet(id_=3, document=document_b, label_set=label_set)
        _ = Annotation(
            id_=3,
            document=document_b,
            confidence=0.5,
            annotation_set=annotation_set_b,
            label=label,
            label_set=label_set,
            spans=[span_3],
        )

        evaluation = compare(document_a, document_b)
        assert len(evaluation) == 2
        assert evaluation["true_positive"].sum() == 0
        assert evaluation["false_positive"].sum() == 1
        assert evaluation["false_negative"].sum() == 1
        assert evaluation["is_found_by_tokenizer"].sum() == 0

    def test_doc_with_annotation_with_wrong_label(self):
        """
        Test a Document where the Annotation has a Span with a wrong Label.

        It is counted as FP because the prediction is not correct.
        """
        project = Project(id_=None)
        category = Category(project=project)
        label_set = LabelSet(id_=33, project=project, categories=[category])
        label_1 = Label(id_=22, project=project, label_sets=[label_set], threshold=0.5)
        label_2 = Label(id_=23, project=project, label_sets=[label_set], threshold=0.5)
        # create a Document A
        document_a = Document(project=project, category=category)
        # first Annotation Set
        span_1 = Span(start_offset=1, end_offset=2)
        annotation_set_a_1 = AnnotationSet(id_=1, document=document_a, label_set=label_set)
        _ = Annotation(
            id_=1,
            document=document_a,
            confidence=0.5,
            is_correct=True,
            annotation_set=annotation_set_a_1,
            label=label_1,
            label_set=label_set,
            spans=[span_1],
        )
        # create a Document B
        document_b = Document(project=project, category=category)
        span_2 = Span(start_offset=1, end_offset=2)
        annotation_set_b = AnnotationSet(id_=3, document=document_b, label_set=label_set)
        _ = Annotation(
            id_=3,
            document=document_b,
            confidence=0.5,
            annotation_set=annotation_set_b,
            label=label_2,
            label_set=label_set,
            spans=[span_2],
        )

        evaluation = compare(document_a, document_b)
        assert len(evaluation) == 1
        assert evaluation["true_positive"].sum() == 0
        assert evaluation["false_positive"].sum() == 1
        assert evaluation["false_negative"].sum() == 0
        assert evaluation["is_found_by_tokenizer"].sum() == 1

    def test_doc_with_one_missing_span_of_two_in_one_annotation(self):
        """
        Test a Document where the Annotation has two Spans and one Span has a wrong offsets.

        It is counted as FP because does not match any correct Annotation and as FN because there was no correct
        prediction for the Annotation in the Document. We are double penalizing here.
        """
        project = Project(id_=None)
        category = Category(project=project)
        label_set = LabelSet(id_=33, project=project, categories=[category])
        label = Label(id_=22, project=project, label_sets=[label_set], threshold=0.5)
        # create a Document A
        document_a = Document(project=project, category=category)
        # first Annotation Set
        span_1_1 = Span(start_offset=1, end_offset=2)
        span_1_2 = Span(start_offset=2, end_offset=3)
        annotation_set_a_1 = AnnotationSet(id_=1, document=document_a, label_set=label_set)
        _ = Annotation(
            id_=1,
            document=document_a,
            confidence=0.5,
            is_correct=True,
            annotation_set=annotation_set_a_1,
            label=label,
            label_set=label_set,
            spans=[span_1_1, span_1_2],
        )
        # create a Document B
        document_b = Document(project=project, category=category)
        span_3_1 = Span(start_offset=2, end_offset=5)
        span_3_2 = Span(start_offset=1, end_offset=2)
        annotation_set_b = AnnotationSet(id_=3, document=document_b, label_set=label_set)
        _ = Annotation(
            id_=3,
            document=document_b,
            confidence=0.5,
            is_correct=True,
            annotation_set=annotation_set_b,
            label=label,
            label_set=label_set,
            spans=[span_3_1, span_3_2],
        )

        evaluation = compare(document_a, document_b)
        assert len(evaluation) == 3
        assert evaluation["true_positive"].sum() == 1
        assert evaluation["false_positive"].sum() == 1
        assert evaluation["false_negative"].sum() == 1
        assert evaluation["is_found_by_tokenizer"].sum() == 1

    def test_doc_with_extra_annotation_set(self):
        """
        Test if we detect an Annotation of a missing Annotation Set.

        Prepare a Project with two Documents, where the first Document has two Annotation Set and the second Document
        has tow Annotation Sets. However, only one Annotation in one of the Annotation Sets is correct, i.e. contains
        a matching Span, has the correct Label, has the correct Label Set and it's confidence is equal or higher than
        the threshold of the Label.

        Annotations above threshold in wrong Annotation Set must be considered as False Positives.
        """
        project = Project(id_=None)
        category = Category(project=project)
        label_set = LabelSet(id_=33, project=project, categories=[category])
        label = Label(id_=22, project=project, label_sets=[label_set], threshold=0.5)
        # create a Document A
        document_a = Document(project=project, category=category)
        # first Annotation Set
        span_1 = Span(start_offset=1, end_offset=2)
        annotation_set_a_1 = AnnotationSet(id_=1, document=document_a, label_set=label_set)
        _ = Annotation(
            id_=1,
            document=document_a,
            confidence=0.5,
            is_correct=True,
            annotation_set=annotation_set_a_1,
            label=label,
            label_set=label_set,
            spans=[span_1],
        )
        # second Annotation Set
        span_2 = Span(start_offset=3, end_offset=5)
        annotation_set_a_2 = AnnotationSet(id_=2, document=document_a, label_set=label_set)
        _ = Annotation(
            id_=2,
            document=document_a,
            is_correct=True,
            annotation_set=annotation_set_a_2,
            label=label,
            label_set=label_set,
            spans=[span_2],
        )
        # create a Document B
        document_b = Document(project=project, category=category)
        # first Annotation Set
        span_3 = Span(start_offset=3, end_offset=5)
        annotation_set_b_1 = AnnotationSet(id_=3, document=document_b, label_set=label_set)
        _ = Annotation(
            id_=3,
            document=document_b,
            confidence=0.5,
            is_correct=True,
            annotation_set=annotation_set_b_1,
            label=label,
            label_set=label_set,
            spans=[span_3],
        )
        # second Annotation Set
        span_4 = Span(start_offset=30, end_offset=50)
        annotation_set_b_2 = AnnotationSet(id_=4, document=document_b, label_set=label_set)
        _ = Annotation(
            id_=3,
            document=document_b,
            confidence=0.5,
            is_correct=True,
            annotation_set=annotation_set_b_2,
            label=label,
            label_set=label_set,
            spans=[span_4],
        )

        evaluation = compare(document_a, document_b)
        assert len(evaluation) == 3
        assert evaluation["true_positive"].sum() == 1
        assert evaluation["false_positive"].sum() == 1
        assert evaluation["false_negative"].sum() == 1
        assert evaluation["is_found_by_tokenizer"].sum() == 1

    def test_doc_with_annotations_wrongly_grouped_in_one_annotation_set(self):
        """
        Test to detect that two Annotations are correct but not grouped into the separate Annotation Sets.

        Prepare a Project with two Documents, where the first Document has two Annotation Set and the second Document
        has tow Annotation Sets. However, only one Annotation in one of the Annotation Sets is correct, i.e. contains
        a matching Span, has the correct Label, has the correct Label Set and it's confidence is equal or higher than
        the threshold of the Label.

        Annotations in the wrong Annotation Set must be evaluated as False Positive.
        """
        project = Project(id_=None)
        category = Category(project=project)
        label_set = LabelSet(id_=33, project=project, categories=[category])
        label = Label(id_=22, project=project, label_sets=[label_set], threshold=0.5)
        # create a Document A
        document_a = Document(project=project, category=category)
        # first Annotation Set
        span_1 = Span(start_offset=1, end_offset=2)
        annotation_set_a_1 = AnnotationSet(id_=1, document=document_a, label_set=label_set)
        _ = Annotation(
            id_=1,
            document=document_a,
            confidence=0.5,
            is_correct=True,
            annotation_set=annotation_set_a_1,
            label=label,
            label_set=label_set,
            spans=[span_1],
        )
        # second Annotation Set
        span_2 = Span(start_offset=3, end_offset=5)
        annotation_set_a_2 = AnnotationSet(id_=2, document=document_a, label_set=label_set)
        _ = Annotation(
            id_=2,
            document=document_a,
            is_correct=True,
            annotation_set=annotation_set_a_2,
            label=label,
            label_set=label_set,
            spans=[span_2],
        )
        # create a Document B
        document_b = Document(project=project, category=category)
        # first Annotation Set
        span_3 = Span(start_offset=1, end_offset=2)
        annotation_set_b_1 = AnnotationSet(id_=3, document=document_b, label_set=label_set)
        _ = Annotation(
            id_=3,
            document=document_b,
            confidence=0.5,
            is_correct=True,
            annotation_set=annotation_set_b_1,
            label=label,
            label_set=label_set,
            spans=[span_3],
        )
        # second Annotation added to the same Annotation Set
        span_4 = Span(start_offset=3, end_offset=5)
        _ = Annotation(
            id_=3,
            document=document_b,
            confidence=0.5,
            is_correct=True,
            annotation_set=annotation_set_b_1,
            label=label,
            label_set=label_set,
            spans=[span_4],
        )

        evaluation = compare(document_a, document_b)
        assert len(evaluation) == 2
        assert evaluation["true_positive"].sum() == 1
        assert evaluation["false_positive"].sum() == 1
        assert evaluation["false_negative"].sum() == 0
        assert evaluation["is_found_by_tokenizer"].sum() == 2

    def test_to_evaluate_annotations_in_one_line_belonging_to_two_annotation_sets(self):
        """Test to evaluate two Annotations where each one belongs to a different Annotation Set."""
        project = Project(id_=None)
        category = Category(project=project)
        label_set = LabelSet(id_=33, project=project, categories=[category])
        label = Label(id_=22, project=project, label_sets=[label_set], threshold=0.5)
        # create a Document A
        document_a = Document(project=project, text='ab\n', category=category)
        # first Annotation Set
        span_1 = Span(start_offset=0, end_offset=1)
        annotation_set_a_1 = AnnotationSet(id_=1, document=document_a, label_set=label_set)
        _ = Annotation(
            id_=1,
            document=document_a,
            confidence=0.5,
            is_correct=True,
            annotation_set=annotation_set_a_1,
            label=label,
            label_set=label_set,
            spans=[span_1],
        )
        # second Annotation Set
        span_2 = Span(start_offset=1, end_offset=2)
        annotation_set_a_2 = AnnotationSet(id_=2, document=document_a, label_set=label_set)
        _ = Annotation(
            id_=2,
            document=document_a,
            is_correct=True,
            annotation_set=annotation_set_a_2,
            label=label,
            label_set=label_set,
            spans=[span_2],
        )

        # create a Document B
        document_b = Document(project=project, text='ab\n', category=category)
        # first Annotation Set
        span_3 = Span(start_offset=0, end_offset=1)
        annotation_set_b_1 = AnnotationSet(id_=3, document=document_b, label_set=label_set)
        _ = Annotation(
            id_=3,
            document=document_b,
            confidence=0.5,
            is_correct=True,
            annotation_set=annotation_set_b_1,
            label=label,
            label_set=label_set,
            spans=[span_3],
        )
        # second Annotation added to the same Annotation Set
        annotation_set_b_2 = AnnotationSet(id_=4, document=document_b, label_set=label_set)
        span_4 = Span(start_offset=1, end_offset=2)
        _ = Annotation(
            id_=4,
            document=document_b,
            confidence=0.5,
            is_correct=True,
            annotation_set=annotation_set_b_2,
            label=label,
            label_set=label_set,
            spans=[span_4],
        )

        evaluation = compare(document_a, document_b)
        assert len(evaluation) == 2
        assert evaluation["true_positive"].sum() == 2
        assert evaluation["false_positive"].sum() == 0
        assert evaluation["false_negative"].sum() == 0
        assert evaluation["is_found_by_tokenizer"].sum() == 2

    def test_grouped(self):
        """Test if group catches all relevant errors."""
        grouped(DataFrame([[True, 'a'], [False, 'b']], columns=['is_correct', 'target']), target='target')
        grouped(DataFrame([[False, 'a'], [False, 'b']], columns=['is_correct', 'target']), target='target')
        grouped(DataFrame([[None, 'a'], [None, 'b']], columns=['is_correct', 'target']), target='target')
