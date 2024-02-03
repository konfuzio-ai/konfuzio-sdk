"""Test the evaluation."""
import unittest
from copy import deepcopy
from statistics import mean

import pytest
from pandas import DataFrame

from konfuzio_sdk.data import Annotation, AnnotationSet, Category, Document, Label, LabelSet, Project, Span
from konfuzio_sdk.evaluate import CategorizationEvaluation, EvaluationCalculator, ExtractionEvaluation, compare, grouped
from konfuzio_sdk.samples import LocalTextProject
from konfuzio_sdk.settings_importer import is_dependency_installed
from konfuzio_sdk.tokenizer.regex import ConnectedTextTokenizer
from konfuzio_sdk.trainer.file_splitting import ContextAwareFileSplittingModel, FileSplittingEvaluation, SplittingAI
from tests.variables import OFFLINE_PROJECT, TEST_DOCUMENT_ID


class TestCompare(unittest.TestCase):
    """Testing to compare to Documents.

    Implemented:
        - prediction without complete offsets (e.g. missing last character)
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

    def test_strict_doc_on_doc_incl_multiline_annotation(self):
        """Test if a Document is 100 % equivalent even it has unrevised Annotations."""
        prj = Project(id_=None, project_folder=OFFLINE_PROJECT)
        doc_a = prj.get_document_by_id(TEST_DOCUMENT_ID)
        doc_b = prj.get_document_by_id(TEST_DOCUMENT_ID)  # predicted
        evaluation = compare(doc_a, doc_b)
        assert len(evaluation) == 24  # 23 if not considering negative Annotations
        # for an Annotation which is human made, it is nan, so that above threshold is False
        # doc_a 19 + 2 multiline + 2 feedback required + 1 rejected
        assert evaluation['true_positive'].sum() == 21  # 1 multiline with 2 lines = 2 Annotations
        assert evaluation['false_positive'].sum() == 0
        # due to the fact that Konfuzio Server does not save confidence = 100 % if Annotation was not created by a human
        assert evaluation['false_negative'].sum() == 0
        assert evaluation['tokenizer_true_positive'].sum() == 21

    def test_strict_doc_where_first_annotation_was_skipped(self):
        """Test if a Document is 100 % equivalent with first Annotation not existing for a certain Label."""
        prj = Project(id_=None, project_folder=OFFLINE_PROJECT)
        doc_a = prj.get_document_by_id(TEST_DOCUMENT_ID)
        doc_b = prj.get_document_by_id(TEST_DOCUMENT_ID)  # predicted
        doc_b.annotations()
        k = next(iter(doc_b._annotations))
        doc_b._annotations.pop(k)  # pop an Annotation that is correct in BOTH  Documents
        assert doc_a._annotations == doc_b._annotations  # first Annotation is removed in both  Documents
        evaluation = compare(doc_a, doc_b)
        assert len(evaluation) == 22  # 21 if not considering negative Annotations, 2 Annotations are is_correct false
        # doc_a 18 (multiline removed) + 1 multiline + 2 feedback required + 1 rejected
        assert evaluation['true_positive'].sum() == 19
        assert evaluation['false_positive'].sum() == 0
        assert evaluation['false_negative'].sum() == 0
        assert evaluation['tokenizer_true_positive'].sum() == 19

    def test_strict_doc_where_last_annotation_was_skipped(self):
        """Test if a Document is 100 % equivalent with last Annotation not existing for a certain Label."""
        prj = Project(id_=None, project_folder=OFFLINE_PROJECT)
        doc_a = prj.get_document_by_id(TEST_DOCUMENT_ID)
        doc_b = prj.get_document_by_id(TEST_DOCUMENT_ID)  # predicted
        doc_b.annotations()
        doc_b._annotations.popitem()  # pop an Annotation that is correct in BOTH  Documents
        assert doc_a._annotations == doc_b._annotations  # last Annotation is removed in both  Documents
        evaluation = compare(doc_a, doc_b)
        assert len(evaluation) == 23  # 22 if not considering negative Annotations, 2 Annotations are is_correct false
        # doc_a 18 + 2 multiline + 2 feedback required + 1 rejected
        assert evaluation['true_positive'].sum() == 20
        assert evaluation['false_positive'].sum() == 0
        assert evaluation['false_negative'].sum() == 0
        assert evaluation['tokenizer_true_positive'].sum() == 20

    def test_strict_if_first_multiline_annotation_is_missing_in_b(self):
        """Test if a Document is equivalent if first Annotation is missing."""
        prj = Project(id_=None, project_folder=OFFLINE_PROJECT)
        doc_a = prj.get_document_by_id(TEST_DOCUMENT_ID)
        doc_b = Document(project=prj, category=doc_a.category)
        for annotation in doc_a.annotations()[1:]:
            doc_b.add_annotation(annotation)

        assert len(doc_b.annotations()) == len(doc_a.annotations()) - 1
        evaluation = compare(doc_a, doc_b)
        assert len(evaluation) == 24  # 23 if not considering negative Annotations, 2 Annotations are false
        # doc_a 19 + 2 multiline + 2 feedback required + 1 rejected
        assert evaluation['true_positive'].sum() == 19  # 1 multiline with 2 lines = 2 Annotations
        assert evaluation['false_positive'].sum() == 0
        assert evaluation['false_negative'].sum() == 2
        assert evaluation['tokenizer_true_positive'].sum() == 19

    def test_strict_doc_where_first_annotation_is_missing_in_a(self):
        """Test if a Document is equivalent if first Annotation is not present."""
        prj = Project(id_=None, project_folder=OFFLINE_PROJECT)
        doc_b = prj.get_document_by_id(TEST_DOCUMENT_ID)
        doc_a = Document(project=prj, category=doc_b.category)
        # use only correct Annotations
        for annotation in doc_b.annotations()[1:]:
            doc_a.add_annotation(annotation)

        # evaluate on doc_b and assume the feedback required ones are correct
        assert len(doc_a.annotations()) == len(doc_b.annotations()) - 1
        evaluation = compare(doc_a, doc_b, ignore_below_threshold=False)
        # 24 if considering negative Annotations, 2 annotations are false and two have feedback required
        assert len(evaluation) == 24
        assert evaluation['true_positive'].sum() == 19
        assert evaluation['false_positive'].sum() == 4  # 1 multiline (2 lines == 2 Annotations) + 2 feedback required
        assert evaluation['false_negative'].sum() == 0
        assert evaluation['tokenizer_true_positive'].sum() == 19

    def test_strict_only_unrevised_annotations(self):
        """Test to evaluate on a Document that has only unrevised Annotations."""
        prj = Project(id_=None, project_folder=OFFLINE_PROJECT)
        doc_a = prj.get_document_by_id(137234)
        doc_b = Document(project=prj, category=doc_a.category)

        assert len(doc_a.annotations()) == len(doc_b.annotations()) == 0
        evaluation = compare(doc_a, doc_b, only_use_correct=True)
        assert len(evaluation) == 1  # placeholder
        assert evaluation['true_positive'].sum() == 0
        assert evaluation['false_positive'].sum() == 0
        assert evaluation['false_negative'].sum() == 0
        assert evaluation['tokenizer_true_positive'].sum() == 0
        assert evaluation['tokenizer_false_positive'].sum() == 0
        assert evaluation['tokenizer_false_negative'].sum() == 0

    def test_strict_doc_where_first_annotation_from_all_is_missing_in_a(self):
        """Test if a Document is equivalent if all Annotation are not present and feedback required are included."""
        prj = Project(id_=None, project_folder=OFFLINE_PROJECT)
        doc_b = prj.get_document_by_id(TEST_DOCUMENT_ID)
        doc_a = Document(project=prj, category=doc_b.category)
        # use correct Annotations and feedback required ones
        for annotation in doc_b.annotations(use_correct=False)[1:]:
            doc_a.add_annotation(annotation)

        assert len(doc_a.annotations()) == len(doc_b.annotations()) - 1
        evaluation = compare(doc_a, doc_b, ignore_below_threshold=False)
        assert len(evaluation) == 24  # 23 if not considering negative Annotations, 2 Annotations are false
        # doc_a 18 + 1 multiline + 2 feedback required + 1 rejected
        assert evaluation['true_positive'].sum() == 19
        assert evaluation['false_positive'].sum() == 2  # 1 multiline (2 lines == 2 Annotations)
        assert evaluation['false_negative'].sum() == 0
        assert evaluation['tokenizer_true_positive'].sum() == 19
        assert evaluation['tokenizer_false_positive'].sum() == 5
        assert evaluation['tokenizer_false_negative'].sum() == 0

    def test_strict_doc_where_last_annotation_is_missing_in_b(self):
        """Test if a Document is equivalent if last Annotation is missing."""
        prj = Project(id_=None, project_folder=OFFLINE_PROJECT)
        doc_a = prj.get_document_by_id(TEST_DOCUMENT_ID)
        doc_b = Document(project=prj, category=doc_a.category)
        # use correct Annotations and feedback required ones
        for annotation in doc_a.annotations(use_correct=False)[:-1]:
            doc_b.add_annotation(annotation)

        evaluation = compare(doc_a, doc_b, ignore_below_threshold=False)
        assert len(evaluation) == 24  # 23 if not considering negative Annotations, 2 Annotations are false
        # doc_a 19 + 2 multiline
        assert evaluation['true_positive'].sum() == 20  # due to the fact that we find both offsets of the multiline
        assert evaluation['false_positive'].sum() == 0
        assert evaluation['false_negative'].sum() == 1
        assert evaluation['tokenizer_true_positive'].sum() == 20
        assert evaluation['tokenizer_false_positive'].sum() == 3
        assert evaluation['tokenizer_false_negative'].sum() == 1

    def test_strict_doc_where_last_annotation_is_missing_in_a(self):
        """Test if a Document is equivalent if last Annotation is not present."""
        prj = Project(id_=None, project_folder=OFFLINE_PROJECT)
        doc_b = prj.get_document_by_id(TEST_DOCUMENT_ID)
        doc_a = Document(project=prj, category=doc_b.category)
        # use correct Annotations and feedback required ones
        for annotation in doc_b.annotations(use_correct=False)[:-1]:
            doc_a.add_annotation(annotation)

        evaluation = compare(doc_a, doc_b, ignore_below_threshold=False)
        assert len(evaluation) == 24  # 23 if not considering negative, 2 Annotations are false
        # doc_a 18 + 2 multiline
        assert evaluation['true_positive'].sum() == 20  # due to the fact that we find both offsets of the multiline
        assert evaluation['false_positive'].sum() == 1
        assert evaluation['false_negative'].sum() == 0
        assert evaluation['tokenizer_true_positive'].sum() == 20
        assert evaluation['tokenizer_false_positive'].sum() == 4
        assert evaluation['tokenizer_false_negative'].sum() == 0

    def test_strict_nothing_should_be_predicted(self):
        """Support to evaluate that nothing is found in a document."""
        prj = Project(id_=None, project_folder=OFFLINE_PROJECT)
        doc_b = prj.get_document_by_id(TEST_DOCUMENT_ID)
        doc_a = Document(project=prj, category=doc_b.category)
        evaluation = compare(doc_a, doc_b, ignore_below_threshold=False)
        assert len(evaluation) == 25  # 24 if not considering negative Annotations
        assert evaluation['true_positive'].sum() == 0
        # any Annotation above threshold is a false positive independent if it's correct or revised
        assert len([an for an in doc_b.annotations(use_correct=False) if an.confidence > an.label.threshold]) == 21
        assert evaluation['false_positive'].sum() == 23  # but one annotation is multiline
        assert evaluation['false_negative'].sum() == 0
        assert evaluation['tokenizer_true_positive'].sum() == 0
        assert evaluation['tokenizer_false_positive'].sum() == 24
        assert evaluation['tokenizer_false_negative'].sum() == 0

        # Checking that false positive rows include the document_id of the Document
        false_positives = evaluation[evaluation['false_positive']]
        assert set(false_positives['document_id_predicted']) == {TEST_DOCUMENT_ID}

    def test_strict_nothing_can_be_predicted(self):
        """Support to evaluate that nothing must be found in a document."""
        prj = Project(id_=None, project_folder=OFFLINE_PROJECT)
        doc_a = prj.get_document_by_id(TEST_DOCUMENT_ID)
        doc_b = Document(project=prj, category=doc_a.category)
        evaluation = compare(doc_a, doc_b)
        # 25 if considering negative Annotations, we evaluate on span level and one annotation is multiline
        assert len(evaluation) == 25
        assert evaluation['true_positive'].sum() == 0
        assert evaluation['false_positive'].sum() == 0
        assert evaluation['false_negative'].sum() == 21
        assert evaluation['tokenizer_true_positive'].sum() == 0
        assert evaluation['tokenizer_false_positive'].sum() == 0
        assert evaluation['tokenizer_false_negative'].sum() == 21

    def test_strict_doc_with_overruled_top_annotations(self):
        """
        Test if a Document is equivalent if prediction follows the top Annotation logic.

        The top Annotation logic considers only 1 Annotation for Labels with multiple=False.
        For example, the "Personalausweis" has multiple=False but several Annotations exist in the document.
        Only 1 is in the prediction.
        """
        # todo: this logic is a view logic on the document: shouldn't this go into the Annotations function
        prj = Project(id_=None, project_folder=OFFLINE_PROJECT)
        doc_a = prj.get_document_by_id(TEST_DOCUMENT_ID)
        doc_b = Document(project=prj, category=doc_a.category)

        found = False
        for annotation in doc_a.annotations(use_correct=False):
            if annotation.label.id_ == 12444:
                if found:
                    continue
                found = True

            doc_b.add_annotation(annotation)

        evaluation = compare(doc_a, doc_b, ignore_below_threshold=False)
        assert len(evaluation) == 24  # 23 if not considering negative Annotations,
        # Evaluation as it is now: everything needs to be find even if multiple=False
        assert evaluation['true_positive'].sum() == 20
        assert evaluation['false_positive'].sum() == 0
        assert evaluation['false_negative'].sum() == 1
        assert evaluation['tokenizer_true_positive'].sum() == 20
        assert evaluation['tokenizer_false_positive'].sum() == 3
        assert evaluation['tokenizer_false_negative'].sum() == 1

    def test_strict_doc_with_missing_annotation_set(self):
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
        label_set = LabelSet(id_=33, project=project, categories=[category], has_multiple_annotation_sets=True)
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
        assert evaluation['true_positive'].sum() == 1
        assert evaluation['false_positive'].sum() == 0
        assert evaluation['false_negative'].sum() == 1
        assert evaluation['tokenizer_true_positive'].sum() == 1

    def test_strict_vs_non_strict_doc_with_missing_annotation_set(self):
        """Test if we detect a partially overlapping Span in an Annotation of a missing Annotation Set."""
        project = Project(id_=None)
        category = Category(project=project)
        label_set = LabelSet(id_=33, project=project, categories=[category], has_multiple_annotation_sets=True)
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
        annotation_set_a_2 = AnnotationSet(id_=2, document=document_a, label_set=label_set)
        span_2 = Span(start_offset=3, end_offset=5)
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
        # with only one Annotation Set, so to say the first one, which then contains the correct Annotation which is in
        # second Annotation Set. This test includes to test if we did not find the first Annotation Set.
        annotation_set_b_1 = AnnotationSet(id_=3, document=document_b, label_set=label_set)
        span_3 = Span(start_offset=4, end_offset=5)
        _ = Annotation(
            id_=3,
            document=document_b,
            confidence=0.5,
            # is_correct=True, we don't know this from the prediction
            annotation_set=annotation_set_b_1,
            label=label,
            label_set=label_set,
            spans=[span_3],
        )
        annotation_set_b_2 = AnnotationSet(id_=4, document=document_b, label_set=label_set)
        span_4 = Span(start_offset=7, end_offset=8)
        _ = Annotation(
            id_=4,
            document=document_b,
            confidence=0.5,
            annotation_set=annotation_set_b_2,
            label=label,
            label_set=label_set,
            spans=[span_4],
        )

        evaluation_strict = compare(document_a, document_b)
        assert len(evaluation_strict) == 4
        assert evaluation_strict['true_positive'].sum() == 0
        assert evaluation_strict['false_positive'].sum() == 2
        assert evaluation_strict['false_negative'].sum() == 2
        assert evaluation_strict['tokenizer_true_positive'].sum() == 0

        evaluation = compare(document_a, document_b, strict=False)
        assert len(evaluation) == 2
        assert evaluation['true_positive'].sum() == 1
        assert evaluation['false_positive'].sum() == 1
        assert evaluation['false_negative'].sum() == 0
        assert evaluation['tokenizer_true_positive'].sum() == 0  # we don't find it with the tokenizer but only parts

    def test_non_strict_is_better_than_strict(self):
        """Test if we detect an Annotation of a missing Annotation Set."""
        project = Project(id_=None)
        category = Category(project=project)
        label_set = LabelSet(id_=33, project=project, categories=[category])
        label = Label(id_=22, project=project, label_sets=[label_set], threshold=0.5)
        # create a Document A
        document_a = Document(project=project, category=category)
        # Annotation Set
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
        span_3 = Span(start_offset=4, end_offset=5)
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

        evaluation_strict = compare(document_a, document_b)
        assert len(evaluation_strict) == 2
        assert evaluation_strict['true_positive'].sum() == 0
        assert evaluation_strict['false_positive'].sum() == 1
        assert evaluation_strict['false_negative'].sum() == 1
        assert evaluation_strict['tokenizer_true_positive'].sum() == 0

        evaluation = compare(document_a, document_b, strict=False)
        assert len(evaluation) == 1
        assert evaluation['true_positive'].sum() == 1
        assert evaluation['false_positive'].sum() == 0
        assert evaluation['false_negative'].sum() == 0
        assert evaluation['tokenizer_true_positive'].sum() == 0

    def test_non_strict_filters_out_fps_and_fns(self):
        """
        Test that Non-Strict Evaluation should filter out the case where a Label should only appear once.

        We will create two Documents to demonstrate Non-Strict Evaluation.
        Document A with text: "1234567890"; and Annotations: "34" (ground truth)
        # Document B with text: "1234567890"; and Annotations "3" and "67" (predicted)

        According to Strict Evaluation, there would be 2 FP and 1 FN,
        because 2 Annotations were found that did not correspond to anything in the ground truth,
        and because 1 Annotation in the ground truth was not matched in the predicted Document.

        According to Non-Strict Evaluation, there would be only 1 TP, because "3" is partially overlapping
        with "34", which counts as a TP. And because the Label associated to this Annotation has
        has_multiple_top_candidates=False, which means that as soon as 1 TP is found, FPs and FNs are discarded.
        Thus, the "67" extraction is not considered in the Non-Strict Evaluation.

        For more details see https://help.konfuzio.com/modules/extractions/index.html#evaluation
        """
        project = Project(id_=None)
        category = Category(project=project)
        label_set = LabelSet(id_=33, project=project, categories=[category])
        label = Label(id_=22, project=project, label_sets=[label_set], threshold=0.1, has_multiple_top_candidates=False)
        # create a Document A with text: "1234567890"; and Annotations: "34"
        document_a = Document(project=project, category=category, text='1234567890')
        # Annotation Set
        span_1 = Span(start_offset=2, end_offset=4)
        annotation_set_a = AnnotationSet(id_=2, document=document_a, label_set=label_set)
        _ = Annotation(
            id_=2,
            document=document_a,
            is_correct=True,
            annotation_set=annotation_set_a,
            label=label,
            label_set=label_set,
            spans=[span_1],
        )
        # create a Document B with text: "1234567890"; and Annotations "3" and "67"
        document_b = Document(project=project, category=category, text='1234567890')
        # Annotation Set
        span_2 = Span(start_offset=2, end_offset=3)
        span_3 = Span(start_offset=5, end_offset=7)
        annotation_set_b = AnnotationSet(id_=3, document=document_b, label_set=label_set)
        _ = Annotation(
            id_=4,
            document=document_b,
            confidence=0.5,
            is_correct=False,
            annotation_set=annotation_set_b,
            label=label,
            label_set=label_set,
            spans=[span_2],
        )
        _ = Annotation(
            id_=5,
            document=document_b,
            confidence=0.5,
            is_correct=False,
            annotation_set=annotation_set_b,
            label=label,
            label_set=label_set,
            spans=[span_3],
        )

        evaluation_strict = compare(document_a, document_b)
        assert len(evaluation_strict) == 3
        assert evaluation_strict['true_positive'].sum() == 0
        assert evaluation_strict['false_positive'].sum() == 2
        assert evaluation_strict['false_negative'].sum() == 1
        assert evaluation_strict['tokenizer_true_positive'].sum() == 0

        evaluation = compare(document_a, document_b, strict=False)
        assert len(evaluation) == 1
        assert evaluation['true_positive'].sum() == 1
        assert evaluation['false_positive'].sum() == 0
        assert evaluation['false_negative'].sum() == 0
        assert evaluation['tokenizer_true_positive'].sum() == 0

    def test_strict_documents_with_different_category(self):
        """Test to not compare two Documents with different Categories."""
        project = Project(id_=None)
        category = Category(project=project)
        document_a = Document(project=project, category=category)
        another_category = Category(project=project)
        document_b = Document(project=project, category=another_category)
        with self.assertRaises(ValueError) as context:
            compare(document_a, document_b)
            assert 'do not match' in context.exception

    def test_strict_doc_with_annotation_with_wrong_offsets(self):
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
        assert evaluation['true_positive'].sum() == 0
        assert evaluation['false_positive'].sum() == 1
        assert evaluation['false_negative'].sum() == 1
        assert evaluation['tokenizer_true_positive'].sum() == 0

    def test_strict_doc_with_annotation_with_wrong_label(self):
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
        assert evaluation['true_positive'].sum() == 0
        assert evaluation['false_positive'].sum() == 1
        assert evaluation['false_negative'].sum() == 0
        assert evaluation['tokenizer_true_positive'].sum() == 1

    def test_strict_doc_with_one_missing_span_of_two_in_one_annotation(self):
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
        assert evaluation['true_positive'].sum() == 1
        assert evaluation['false_positive'].sum() == 1
        assert evaluation['false_negative'].sum() == 1
        assert evaluation['tokenizer_true_positive'].sum() == 1

    def test_strict_doc_with_extra_annotation_set(self):
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
        label_set = LabelSet(id_=33, project=project, categories=[category], has_multiple_annotation_sets=True)
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
        assert evaluation['true_positive'].sum() == 1
        assert evaluation['false_positive'].sum() == 1
        assert evaluation['false_negative'].sum() == 1
        assert evaluation['tokenizer_true_positive'].sum() == 1
        assert evaluation['tokenizer_false_positive'].sum() == 1
        assert evaluation['tokenizer_false_negative'].sum() == 1

    def test_strict_doc_with_annotations_wrongly_grouped_in_one_annotation_set(self):
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
        label_set = LabelSet(id_=33, project=project, categories=[category], has_multiple_annotation_sets=True)
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
        assert evaluation['true_positive'].sum() == 1
        assert evaluation['false_positive'].sum() == 1
        assert evaluation['false_negative'].sum() == 0
        assert evaluation['tokenizer_true_positive'].sum() == 2
        assert evaluation['tokenizer_false_positive'].sum() == 0
        assert evaluation['tokenizer_false_negative'].sum() == 0

    def test_strict_to_evaluate_annotations_in_one_line_belonging_to_two_annotation_sets(self):
        """Test to evaluate two Annotations where each one belongs to a different Annotation Set."""
        project = Project(id_=None)
        category = Category(project=project)
        label_set = LabelSet(id_=33, project=project, categories=[category], has_multiple_annotation_sets=True)
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
        assert evaluation['true_positive'].sum() == 2
        assert evaluation['false_positive'].sum() == 0
        assert evaluation['false_negative'].sum() == 0
        assert evaluation['tokenizer_true_positive'].sum() == 2

    def test_strict_grouped_both_above_threshold_both_correct(self):
        """Test grouped for two correct Spans where both are over threshold."""
        result = grouped(
            DataFrame(
                [[True, 3, True, 0.1], [True, 1, True, 0.05]],
                columns=['is_matched', 'target', 'above_predicted_threshold', 'confidence_predicted'],
            ),
            target='target',
        )
        assert result['defined_to_be_correct_target'].to_list() == [3, 3]

    def test_non_strict_grouped_both_above_threshold_both_correct(self):
        """Return second group with target = int(1) as the confidence of 100 % is higher than 99%."""
        result = grouped(
            DataFrame(
                [[True, 3, True, 0.99], [True, 1, True, 1.0]],
                columns=['is_matched', 'target', 'above_predicted_threshold', 'confidence_predicted'],
            ),
            target='target',
        )
        assert result['defined_to_be_correct_target'].to_list() == [1, 1]

    def test_strict_grouped_both_above_threshold_one_correct(self):
        """Test grouped for one correct Span and one incorrect Span over threshold."""
        result = grouped(
            DataFrame(
                [[True, 3, True, 0.5], [False, 1, True, 0.5]],
                columns=['is_matched', 'target', 'above_predicted_threshold', 'confidence_predicted'],
            ),
            target='target',
        )
        assert result['defined_to_be_correct_target'].to_list() == [3, 3]

    def test_strict_grouped_one_above_threshold_both_incorrect(self):
        """Test grouped for incorrect Span over threshold and incorrect Span below threshold."""
        result = grouped(
            DataFrame(
                [[False, 1, False, 0.5], [False, 3, True, 0.5]],
                columns=['is_matched', 'target', 'above_predicted_threshold', 'confidence_predicted'],
            ),
            target='target',
        )
        assert result['defined_to_be_correct_target'].to_list() == [1, 1]  # see reason in commit 4a66394

    def test_strict_grouped_one_above_threshold_none_correct(self):
        """Test grouped for Span below threshold and Span above threshold, while is_correct is empty."""
        result = grouped(
            DataFrame(
                [[None, 3, True, 0.9], [None, 1, False, 0.1]],
                columns=['is_matched', 'target', 'above_predicted_threshold', 'confidence_predicted'],
            ),
            target='target',
        )
        assert result['defined_to_be_correct_target'].to_list() == [3, 3]  # it must be 3, as 3 is above threshold

    def test_strict_grouped_none_above_threshold_none_correct(self):
        """Test grouped for two Spans below threshold, while is_correct is empty."""
        result = grouped(
            DataFrame(
                [[None, 3, False, 0.1], [None, 1, False, 0.2]],
                columns=['is_matched', 'target', 'above_predicted_threshold', 'confidence_predicted'],
            ),
            target='target',
        )
        assert result['defined_to_be_correct_target'].to_list() == [1, 1]

    def test_strict_grouped_but_no_voter_above_threshold_none_confidence(self):
        """Test grouped for two Spans below threshold and the CLF does not provide any confidence."""
        result = grouped(
            DataFrame(
                [[None, 3, False, None], [None, 1, False, None]],
                columns=['is_matched', 'target', 'above_predicted_threshold', 'confidence_predicted'],
            ),
            target='target',
        )
        assert result['defined_to_be_correct_target'].to_list() == [1, 1]


class TestEvaluation(unittest.TestCase):
    """Tes to compare to Documents."""

    def test_project(self):
        """Test that data has not changed."""
        project = LocalTextProject()
        assert len(project.documents) == 5
        assert len(project.test_documents) == 11

    def test_not_strict(self):
        """Test that evaluation can be initialized with strict mode disabled."""
        project = LocalTextProject()
        evaluation = ExtractionEvaluation(documents=list(zip(project.documents, project.documents)), strict=False)
        assert evaluation.strict is False

    def test_not_strict_with_empty_evaluated_document(self):
        """Test that non-strict Evaluation can be used with evaluated Document with no Annotations."""
        project = LocalTextProject()
        document = project.documents[0]
        assert len(document.annotations(use_correct=True)) == 3
        assert len(document.spans(use_correct=True)) == 3
        empty_extracted_document = deepcopy(document)  # no Annotation Document
        assert len(empty_extracted_document.annotations(use_correct=False)) == 0
        assert len(empty_extracted_document.spans(use_correct=False)) == 0
        assert len(empty_extracted_document.eval_dict()) == 1
        place_holder_span_eval_dict = empty_extracted_document.eval_dict()[0]
        assert place_holder_span_eval_dict['start_offset'] == place_holder_span_eval_dict['end_offset'] == 0
        evaluation = ExtractionEvaluation(documents=[(document, empty_extracted_document)], strict=False)
        assert evaluation.strict is False
        assert len(evaluation.data) == 0

    def test_not_strict_with_empty_reference_document(self):
        """Test that non-strict Evaluation can be used with reference Document with no Annotations."""
        project = LocalTextProject()
        document = project.documents[0]
        assert len(document.annotations(use_correct=True)) == 3
        assert len(document.spans(use_correct=True)) == 3
        empty_extracted_document = deepcopy(document)  # no Annotation Document
        assert len(empty_extracted_document.annotations(use_correct=False)) == 0
        assert len(empty_extracted_document.spans(use_correct=False)) == 0
        assert len(empty_extracted_document.eval_dict()) == 1
        place_holder_span_eval_dict = empty_extracted_document.eval_dict()[0]
        assert place_holder_span_eval_dict['start_offset'] == place_holder_span_eval_dict['end_offset'] == 0
        evaluation = ExtractionEvaluation(documents=[(empty_extracted_document, document)], strict=False)
        assert evaluation.strict is False
        assert len(evaluation.data) == 3

    def test_true_positive(self):
        """Count two Spans from two Training Documents."""
        project = LocalTextProject()
        evaluation = ExtractionEvaluation(documents=list(zip(project.documents, project.documents)))
        assert evaluation.tp() == sum([len(doc.spans()) for doc in project.documents])
        evaluation_data = evaluation.get_evaluation_data(search=None)
        assert evaluation_data.tp == sum([len(doc.spans()) for doc in project.documents])

    def test_false_positive(self):
        """Count 3 false positives from one Training Document."""
        project = LocalTextProject()
        true_document = project.documents[0]  # A1(0,2,Label_0) + A2(3,5,Label_1) + A3(7,10,Label_2)
        predicted_document = project.test_documents[0]  # A4(0,3,Label_0) + A5(7,10,Label_1) + A6(11,14,Label_2)
        evaluation = ExtractionEvaluation(
            documents=list(zip([true_document], [predicted_document])), zero_division=None
        )
        assert evaluation.fp() == 3  # A4, A5, A6
        evaluation_data = evaluation.get_evaluation_data(search=None)
        assert evaluation_data.fp == 3

    def test_true_negatives(self):
        """Count zero false negatives from two Training Documents (correctly, nothing is predicted under threshold)."""
        project = LocalTextProject()
        # only those of Categories 1 and 2, because the rest are intended to be used for FileSplitting testing & eval
        documents_test_evaluation = (
            project.get_category_by_id(1).documents() + project.get_category_by_id(2).documents()
        )
        evaluation = ExtractionEvaluation(
            documents=list(zip(documents_test_evaluation, documents_test_evaluation)), zero_division=None
        )
        assert evaluation.tn() == 0
        evaluation_data = evaluation.get_evaluation_data(search=None)
        assert evaluation_data.tn == 0

    def test_f1(self):
        """Test to calculate F1 Score."""
        project = LocalTextProject()
        evaluation = ExtractionEvaluation(documents=list(zip(project.documents, project.documents)), zero_division=None)
        scores = []
        for label in project.labels:
            if label != project.no_label:
                f1 = evaluation.f1(search=label)
                # None would mean that there were no candidate Annotations to check for this label
                if f1 is not None:
                    scores.append(f1)
        assert mean(scores) == 1.0
        evaluation_data = evaluation.get_evaluation_data(search=None)
        assert evaluation_data.f1 == 1.0

    def test_precision(self):
        """Test to calculate Precision."""
        project = LocalTextProject()
        evaluation = ExtractionEvaluation(documents=list(zip(project.documents, project.documents)), zero_division=None)
        scores = []
        for label in project.labels:
            if label != project.no_label:
                precision = evaluation.precision(search=label)
                # None would mean that there were no candidate Annotations to check for this label
                if precision is not None:
                    scores.append(evaluation.precision(search=label))
        assert mean(scores) == 1.0
        evaluation_data = evaluation.get_evaluation_data(search=None)
        assert evaluation_data.precision == 1.0

    def test_recall(self):
        """Test to calculate Recall."""
        project = LocalTextProject()
        evaluation = ExtractionEvaluation(documents=list(zip(project.documents, project.documents)), zero_division=None)
        scores = []
        for label in project.labels:
            if label != project.no_label:
                recall = evaluation.recall(search=label)
                # None would mean that there were no candidate Annotations to check for this label
                if recall is not None:
                    scores.append(evaluation.recall(search=label))
        assert mean(scores) == 1.0
        evaluation_data = evaluation.get_evaluation_data(search=None)
        assert evaluation_data.recall == 1.0

    def test_false_negatives(self):
        """Count zero Annotations from two Training Documents."""
        project = LocalTextProject()
        predicted_document = project.documents[0]  # A1(0,2,Label_0) + A2(3,5,Label_1) + A3(7,10,Label_2)
        true_document = project.test_documents[0]  # A4(0,3,Label_0) + A5(7,10,Label_1) + A6(11,14,Label_2)
        evaluation = ExtractionEvaluation(
            documents=list(zip([true_document], [predicted_document])), zero_division=None
        )
        assert evaluation.tp() == 0  # nothing correctly predicted
        assert evaluation.fp() == 3  # A1, A2, A3
        assert evaluation.fn() == 2  # A4, A6
        assert evaluation.tn() == 0  # nothing to predict under threshold
        evaluation_data = evaluation.get_evaluation_data(search=None)
        assert evaluation_data.fn == 2

    def test_true_positive_label(self):
        """Count two Annotations from two Training Documents and filter by one Label."""
        project = LocalTextProject()
        evaluation = ExtractionEvaluation(documents=list(zip(project.documents, project.documents)), zero_division=None)
        # there is only one Label that is not the NONE_LABEL or from a default LabelSet
        label = project.get_label_by_id(id_=4)
        assert evaluation.tp() == sum([len(doc.spans()) for doc in project.documents])
        assert evaluation.tp(search=label) == 2
        assert evaluation.fp(search=label) == 0
        assert evaluation.fn(search=label) == 0
        assert evaluation.tn(search=label) == 0

    def test_true_positive_document(self):
        """Count zero Annotations from one Training Document that has no ID."""
        project = LocalTextProject()
        evaluation = ExtractionEvaluation(documents=list(zip(project.documents, project.documents)), zero_division=None)
        with pytest.raises(AssertionError) as e:
            evaluation.tp(search=project.documents[0])
            assert 'Document None (None) must have a ID.' in e

    def test_true_positive_label_set(self):
        """Count 3 true positives within a specific label set."""
        project = LocalTextProject()
        evaluation = ExtractionEvaluation(documents=list(zip(project.documents, project.documents)), zero_division=None)
        label_set = project.get_label_set_by_id(id_=3)
        assert evaluation.tp(search=label_set) == 3


class TestEvaluationTwoLabels(unittest.TestCase):
    """Test the calculation two Documents with overlapping Spans and multiple Labels."""

    def setUp(self) -> None:
        """Test evaluation when changing filtered Label and Documents."""
        project = LocalTextProject()
        document_a = project.documents[0]  # A1(0,2,Label_0) + A2(3,5,Label_1) + A3(7,10,Label_2)
        document_b = project.test_documents[0]  # A4(0,3,Label_0) + A5(7,10,Label_1) + A6(11,14,Label_2)
        self.evaluation = ExtractionEvaluation(documents=list(zip([document_b], [document_a])), zero_division=None)

    def test_true_positives(self):
        """Evaluate that all is wrong."""
        assert self.evaluation.tp() == 0

    def test_false_positives(self):
        """Evaluate that Document A predicts three wrong Spans."""
        assert self.evaluation.fp() == 3  # A1, A2 and A3

    def test_false_negatives(self):
        """Evaluate that Document A misses to predict two Spans."""
        assert self.evaluation.fn() == 2  # A1 and A2

    def test_true_negatives(self):
        """Evaluate that nothing is correctly predicted below threshold."""
        assert self.evaluation.tn() == 0


class TestEvaluationFirstLabelDocumentADocumentB(unittest.TestCase):
    """Test the calculation two Documents with overlapping Spans and multiple Labels."""

    def setUp(self) -> None:
        """Test evaluation when changing filtered Label and Documents."""
        project = LocalTextProject()
        document_a = project.documents[0]  # A1(0,2,Label_0) + A2(3,5,Label_1) + A3(7,10,Label_2)
        document_b = project.test_documents[0]  # A4(0,3,Label_0) + A5(7,10,Label_1) + A6(11,14,Label_2)
        self.evaluation = ExtractionEvaluation(documents=list(zip([document_b], [document_a])))
        self.label = project.get_label_by_id(id_=4)

    def test_true_positives(self):
        """Evaluate that all is wrong."""
        assert self.evaluation.tp(search=self.label) == 0

    def test_false_positives(self):
        """Check for overlapping Annotation.

        Based on the example data. We filter all Spans for Label ID 3, while A2 overlaps with A3. Which causes
        A2 to be counted as a FP. When we now filter for Label ID 3, we consider this Annotation as a False Positive.
        One could argue that this should be a False Negative, as A3 should be predicted and is not. As the sum of FP
        and FN is relevant for the F1 Score, and this issue will only happen for exact overlaps, we accept that we
        rather predict 2 FP and 0 FN instead of 1 FP and 1 FN.
        """
        assert self.evaluation.fp(search=self.label) == 2  # todo: it could be 1 with A1, however A1 and A2 are used

    def test_false_negatives(self):
        """Check for overlapping Annotation.

        Based on the example data. We filter all Spans for Label ID 3, while A2 overlaps with A3. Which causes
        A3 to be counted as a FP. When we now filter for Label ID 3, we consider this Annotation as a False Positive.
        One could argue that this should be a False Negative, as A3 should be predicted and is not. As the sum of FP
        and FN is relevant for the F1 Score, and this issue will only happen for exact overlaps, we accept that we
        rather predict 2 FP and 0 FN instead of 1 FP and 1 FN.
        """
        assert self.evaluation.fn(search=self.label) == 0  # todo: it could be 1 with A3, however A2 overrules A3

    def test_true_negatives(self):
        """Evaluate that that nothing is correctly predicted below threshold."""
        assert self.evaluation.tn(search=self.label) == 0


class TestEvaluationFirstLabelDocumentBDocumentA(unittest.TestCase):
    """Test the calculation two Documents with overlapping Spans and multiple Labels."""

    def setUp(self) -> None:
        """Test evaluation when changing filtered Label and Documents."""
        project = LocalTextProject()
        document_a = project.documents[0]  # A1(0,2,Label_0) + A2(3,5,Label_1) + A3(7,10,Label_2)
        document_b = project.test_documents[0]  # A4(0,3,Label_0) + A5(7,10,Label_1) + A6(11,14,Label_2)
        self.evaluation = ExtractionEvaluation(documents=list(zip([document_a], [document_b])))
        self.label = project.get_label_by_id(id_=4)

    def test_true_positives(self):
        """Evaluate that all is wrong."""
        assert self.evaluation.tp(search=self.label) == 0

    def test_false_positives(self):
        """Evaluate that Document B predicts one wrong Span."""
        assert self.evaluation.fp(search=self.label) == 1  # A3

    def test_false_negatives(self):
        """Evaluate that Document B misses to predict one wrong Span."""
        assert self.evaluation.fn(search=self.label) == 1  # A1

    def test_true_negatives(self):
        """Evaluate that that nothing is correctly predicted below threshold."""
        assert self.evaluation.tn(search=self.label) == 0


class TestEvaluationSecondLabelDocumentADocumentB(unittest.TestCase):
    """Test the calculation two Documents with overlapping Spans and multiple Labels."""

    def setUp(self) -> None:
        """Test evaluation when changing filtered Label and Documents."""
        project = LocalTextProject()
        document_a = project.documents[0]  # A1(0,2,Label_0) + A2(3,5,Label_1) + A3(7,10,Label_2)
        document_b = project.test_documents[0]  # A4(0,3,Label_0) + A5(7,10,Label_1) + A6(11,14,Label_2)
        self.evaluation = ExtractionEvaluation(documents=list(zip([document_b], [document_a])))
        self.label = project.get_label_by_id(id_=4)

    def test_true_positives(self):
        """Evaluate that all is wrong."""
        assert self.evaluation.tp(search=self.label) == 0

    def test_false_positives(self):
        """Evaluate that Document A predicts one wrong Spans."""
        assert self.evaluation.fp(search=self.label) == 2  # A2 and A3

    def test_false_negatives(self):
        """Evaluate that Document A doesn't miss any Spans for Label_1."""
        assert self.evaluation.fn(search=self.label) == 0

    def test_true_negatives(self):
        """Evaluate that that nothing is correctly predicted below threshold."""
        assert self.evaluation.tn(search=self.label) == 0


class TestEvaluationSecondLabelDocumentBDocumentA(unittest.TestCase):
    """Test the calculation two Documents with overlapping Spans and multiple Labels."""

    def setUp(self) -> None:
        """Test evaluation when changing filtered Label and Documents."""
        project = LocalTextProject()
        document_a = project.documents[0]  # A1(0,2,Label_0) + A2(3,5,Label_1) + A3(7,10,Label_2)
        document_b = project.test_documents[0]  # A4(0,3,Label_0) + A5(7,10,Label_1) + A6(11,14,Label_2)
        self.evaluation = ExtractionEvaluation(documents=list(zip([document_a], [document_b])))
        self.label = project.get_label_by_id(id_=5)

    def test_true_positives(self):
        """Evaluate that all is wrong."""
        assert self.evaluation.tp(search=self.label) == 0

    def test_false_positives(self):
        """Check for overlapping Annotation.

        Based on the example data. We filter all Spans for Label ID 4, while A3 overlaps with A2. Which causes
        A3 to be counted as a FP. When we now filter for Label ID 4, we consider this Annotation as a False Positive.
        One could argue that this should be a False Negative, as A2 should be predicted and is not. As the sum of FP
        and FN is relevant for the F1 Score, and this issue will only happen for exact overlaps, we accept that we
        rather predict 2 FP and 0 FN instead of 1 FP and 1 FN.
        """
        assert self.evaluation.fp(search=self.label) == 2  # todo: it could be 1 with A4, however A4 and A3 are used

    def test_false_negatives(self):
        """Check for overlapping Annotation.

        Based on the example data. We filter all Spans for Label ID 4, while A3 overlaps with A2. Which causes
        A3 to be counted as a FP. When we now filter for Label ID 4, we consider this Annotation as a False Positive.
        One could argue that this should be a False Negative, as A2 should be predicted and is not. As the sum of FP
        and FN is relevant for the F1 Score, and this issue will only happen for exact overlaps, we accept that we
        rather predict 2 FP and 0 FN instead of 1 FP and 1 FN.
        """
        assert self.evaluation.fn(search=self.label) == 0  # todo: it could be 1 with A2, however A4 and A3 are used

    def test_true_negatives(self):
        """Evaluate that that nothing is correctly predicted below threshold."""
        assert self.evaluation.tn(search=self.label) == 0


class TestCategorizationEvaluation(unittest.TestCase):
    """Test the calculation two Documents with overlapping Spans and multiple Labels."""

    @classmethod
    def setUpClass(cls) -> None:
        """Test evaluation when changing filtered Category and Documents."""
        cls.project = LocalTextProject()
        cls.cat1_doca = cls.project.categories[0].documents()[0]
        cls.cat1_docb = cls.project.categories[0].test_documents()[0]
        cls.cat2_doca = cls.project.categories[1].documents()[0]
        cls.cat2_docb = cls.project.categories[1].test_documents()[0]
        cls.cat_eval = CategorizationEvaluation(
            cls.project.categories,
            documents=[
                (cls.cat1_doca, cls.cat1_docb),
                (cls.cat2_doca, cls.cat2_docb),
                (cls.cat1_doca, cls.cat2_docb),
                (cls.cat1_doca, cls.cat2_docb),
            ],
            zero_division=None,
        )

    def test_get_tp_tn_fp_fn_per_category(self):
        """Test get results per Category for categorization problem."""
        results_per_category = self.cat_eval._get_tp_tn_fp_fn_per_category()
        assert results_per_category[1].tp == 1
        assert results_per_category[1].fp == 0
        assert results_per_category[1].fn == 2
        assert results_per_category[1].tn == 1
        assert results_per_category[2].tp == 1
        assert results_per_category[2].fp == 2
        assert results_per_category[2].fn == 0
        assert results_per_category[2].tn == 1

    def test_global_metrics(self):
        """Test metrics for a categorization problem."""
        assert self.cat_eval.tp(None) == 2
        assert self.cat_eval.fp(None) == 2
        assert self.cat_eval.fn(None) == 2
        assert self.cat_eval.tn(None) == 10

        assert self.cat_eval.precision(None) == 5 / 6
        assert self.cat_eval.recall(None) == 0.5
        assert self.cat_eval.f1(None) == 0.5

    def test_filtered_metrics(self):
        """Test metrics for a categorization problem while filtering for categories."""
        assert self.cat_eval.tp(category=self.project.categories[0]) == 1
        assert self.cat_eval.fp(category=self.project.categories[0]) == 0
        assert self.cat_eval.fn(category=self.project.categories[0]) == 2
        assert self.cat_eval.tn(category=self.project.categories[0]) == 1

        assert self.cat_eval.precision(category=self.project.categories[0]) == 1.0
        assert self.cat_eval.recall(category=self.project.categories[0]) == 1 / 3
        assert self.cat_eval.f1(category=self.project.categories[0]) == 0.5

        assert self.cat_eval.tp(category=self.project.categories[1]) == 1
        assert self.cat_eval.fp(category=self.project.categories[1]) == 2
        assert self.cat_eval.fn(category=self.project.categories[1]) == 0
        assert self.cat_eval.tn(category=self.project.categories[1]) == 1

        assert self.cat_eval.precision(category=self.project.categories[1]) == 1 / 3
        assert self.cat_eval.recall(category=self.project.categories[1]) == 1.0
        assert self.cat_eval.f1(category=self.project.categories[1]) == 0.5


class TestEvaluationCalculator(unittest.TestCase):
    """Test the Evaluation Calculator."""

    def test_evaluation_calculator(self):
        """Test the Evaluation Calculator."""
        evaluation_calculator = EvaluationCalculator(tp=3, fp=22, fn=2)
        assert evaluation_calculator.tn == 0
        assert evaluation_calculator.precision == 0.12  # 3 / (3 + 22)
        assert evaluation_calculator.recall == 0.6  # 3 / (3 + 2)
        assert evaluation_calculator.f1 == 0.2  # 3 / (3 + 0.5 * (22 + 2)) or (2 * 0.12 * 0.6) / (0.12 + 0.6)

    def test_evaluation_calculator_perfect_score_can_be_calculated(self):
        """Check that it's possible to calculate 100% score."""
        evaluation_calculator = EvaluationCalculator(tp=10, fp=0, fn=0)
        assert evaluation_calculator.precision == 1.0
        assert evaluation_calculator.recall == 1.0
        assert evaluation_calculator.f1 == 1.0

    def test_evaluation_calculator_zero_not_allowed(self):
        """Check that the Evaluation Calculator raises ZeroDivisionError when allow_zero==False.

        This should happen in situations where precision or recall calculations would produce a division by zero.
        """
        with pytest.raises(ZeroDivisionError, match='TP and FP are zero'):
            EvaluationCalculator(tp=0, fp=0, fn=100, zero_division='error').precision
        with pytest.raises(ZeroDivisionError, match='TP and FN are zero'):
            EvaluationCalculator(tp=0, fp=100, fn=0, zero_division='error').recall
        with pytest.raises(ZeroDivisionError, match='Precision and recall are zero'):
            EvaluationCalculator(tp=0, fp=0, fn=0, zero_division='error').f1

    def test_evaluation_calculator_none_values(self):
        """Test the Evaluation Calculator when precision or recall are calculated as 0/0."""
        no_precision = EvaluationCalculator(tp=0, fp=0, fn=100, zero_division=None)
        assert no_precision.precision is None
        assert no_precision.recall == 0.0
        assert no_precision.f1 == 0.0
        no_recall = EvaluationCalculator(tp=0, fp=100, fn=0, zero_division=None)
        assert no_recall.precision == 0.0
        assert no_recall.recall is None
        assert no_recall.f1 == 0.0
        no_f1 = EvaluationCalculator(tp=0, fp=0, fn=0, zero_division=None)
        assert no_f1.precision is None
        assert no_f1.recall is None
        assert no_f1.f1 is None

    def test_evaluation_calculator_wrong_value(self):
        """Test passing a wrong value of zero_division into the EvaluationCalculator."""
        with pytest.raises(AssertionError, match=' value of zero_division'):
            EvaluationCalculator(tp=0, fp=0, fn=0, zero_division='hehe')


@pytest.mark.skipif(
    not is_dependency_installed('datasets')
    and not is_dependency_installed('torch')
    and not is_dependency_installed('transformers')
    and not is_dependency_installed('evaluate'),
    reason='Required dependencies not installed.',
)
class TestEvaluationFileSplitting(unittest.TestCase):
    """Test Evaluation class for ContextAwareFileSplitting."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the tested class."""
        cls.project = LocalTextProject()
        #  Categories 3 and 4 are used because they were specifically created for File Splitting testing
        cls.file_splitting_model = ContextAwareFileSplittingModel(
            categories=[cls.project.get_category_by_id(3), cls.project.get_category_by_id(4)],
            tokenizer=ConnectedTextTokenizer(),
        )
        cls.file_splitting_model.test_documents = [
            document for category in cls.file_splitting_model.categories for document in category.test_documents()
        ][:-2]
        cls.test_document = cls.project.get_category_by_id(3).test_documents()[0]  # a multi-file test Document
        cls.wrong_test_document = cls.project.get_category_by_id(4).test_documents()[-2]

    def test_evaluation_input_different_lengths(self):
        """Test having different lengths of input lists of Documnets."""
        with pytest.raises(ValueError, match='must be same length'):
            FileSplittingEvaluation([1, 2], [1])

    def test_evaluation_input_different_projects(self):
        """Test passing Documents from different Projects."""
        self.file_splitting_model.fit(allow_empty_categories=True)
        splitting_ai = SplittingAI(self.file_splitting_model)
        ground_truth = self.test_document
        pred = splitting_ai.propose_split_documents(self.test_document, return_pages=True)[0]
        wrong_project = Project(id_=46)
        wrong_doc = wrong_project.documents[0]
        wrong_pred = splitting_ai.propose_split_documents(wrong_doc, return_pages=True)[0]
        with pytest.raises(ValueError, match='have to belong to the same Project'):
            FileSplittingEvaluation([ground_truth, wrong_doc], [pred, wrong_pred])

    def test_evaluation_input_no_is_first_page_attr(self):
        """Test passing a Document with Page that has is_first_page=None."""
        test_document = deepcopy(self.test_document)
        for page in test_document.pages():
            page.is_first_page = None
        with pytest.raises(ValueError, match='does not have a value of is_first_page'):
            FileSplittingEvaluation([test_document], [test_document])

    def test_evaluation_input_wrong_document_pair(self):
        """Test passing a pair of different Documents."""
        self.file_splitting_model.fit(allow_empty_categories=True)
        splitting_ai = SplittingAI(self.file_splitting_model)
        pred = splitting_ai.propose_split_documents(self.test_document, return_pages=True)[0]
        pred.copy_of_id = 9999999
        with pytest.raises(ValueError, match='Incorrect prediction'):
            FileSplittingEvaluation([self.test_document], [pred])

    def test_evaluation_pred_input_no_is_first_page_attr(self):
        """Test passing a prediction of a Document with Page that has is_first_page=None."""
        self.file_splitting_model.fit(allow_empty_categories=True)
        splitting_ai = SplittingAI(self.file_splitting_model)
        pred = splitting_ai.propose_split_documents(self.test_document, return_pages=True)[0]
        for page in pred.pages():
            page.is_first_page = None
        with pytest.raises(ValueError, match='does not have a value of is_first_page'):
            FileSplittingEvaluation([self.test_document], [pred])

    def test_metrics_calculation(self):
        """Test Evaluation class for ContextAwareFileSplitting."""
        self.file_splitting_model.fit(allow_empty_categories=True)
        splitting_ai = SplittingAI(self.file_splitting_model)
        ground_truth = self.test_document
        pred = splitting_ai.propose_split_documents(self.test_document, return_pages=True)[0]
        evaluation = FileSplittingEvaluation([ground_truth], [pred])
        assert evaluation.tp() == 3
        assert evaluation.fp() == 0
        assert evaluation.fn() == 0
        assert evaluation.tn() == 2
        assert evaluation.precision() == 1.0
        assert evaluation.recall() == 1.0
        assert evaluation.f1() == 1.0

    def test_metrics_calculation_by_category(self):
        """Test Evaluation by Category."""
        splitting_ai = SplittingAI(self.file_splitting_model)
        ground_truth = self.test_document
        pred = splitting_ai.propose_split_documents(self.test_document, return_pages=True)[0]
        evaluation = FileSplittingEvaluation([ground_truth], [pred])
        assert evaluation.tp(search=ground_truth.category) == 3
        assert evaluation.fp(search=ground_truth.category) == 0
        assert evaluation.fn(search=ground_truth.category) == 0
        assert evaluation.tn(search=ground_truth.category) == 2
        assert evaluation.precision(search=ground_truth.category) == 1.0
        assert evaluation.recall(search=ground_truth.category) == 1.0
        assert evaluation.f1(search=ground_truth.category) == 1.0

    def test_wrong_metrics_calculation(self):
        """Test Evaluation on a file that does not return all-100% metrics."""
        splitting_ai = SplittingAI(self.file_splitting_model)
        ground_truth = self.wrong_test_document
        pred = splitting_ai.propose_split_documents(self.wrong_test_document, return_pages=True)[0]
        evaluation = FileSplittingEvaluation([ground_truth], [pred])
        assert evaluation.tp() == 1
        assert evaluation.fp() == 1
        assert evaluation.fn() == 0
        assert evaluation.tn() == 0
        assert evaluation.precision() == 0.5
        assert evaluation.recall() == 1
        assert evaluation.f1() == 0.6666666666666666

    def test_wrong_category_search(self):
        """Test filtering by wrongly input Category."""
        splitting_ai = SplittingAI(self.file_splitting_model)
        ground_truth = self.test_document
        pred = splitting_ai.propose_split_documents(self.test_document, return_pages=True)[0]
        evaluation = FileSplittingEvaluation([ground_truth], [pred])
        wrong_category = Project(id_=46).categories[0]
        with pytest.raises(KeyError, match='Only Categories within a Project can be used for viewing metrics.'):
            evaluation.tp(wrong_category)
            evaluation.fp(wrong_category)
            evaluation.fn(wrong_category)
            evaluation.tn(wrong_category)
            evaluation.precision(wrong_category)
            evaluation.recall(wrong_category)
            evaluation.f1(wrong_category)

    def test_splitting_ai_evaluation(self):
        """Test evaluate_full method of SplittingAI."""
        splitting_ai = SplittingAI(self.file_splitting_model)
        splitting_ai.evaluate_full()
        assert splitting_ai.full_evaluation.evaluation_results['true_positives'] == 9
        assert splitting_ai.full_evaluation.evaluation_results['false_positives'] == 0
        assert splitting_ai.full_evaluation.evaluation_results['false_negatives'] == 0
        assert splitting_ai.full_evaluation.evaluation_results['true_negatives'] == 7
        assert splitting_ai.full_evaluation.evaluation_results['precision'] == 1.0
        assert splitting_ai.full_evaluation.evaluation_results['recall'] == 1.0
        assert splitting_ai.full_evaluation.evaluation_results['f1'] == 1.0

    def test_metrics_get_evaluation_data_no_category(self):
        """Test get_evaluation_data method of the Evaluation class."""
        splitting_ai = SplittingAI(self.file_splitting_model)
        ground_truth = self.test_document
        pred = splitting_ai.propose_split_documents(self.test_document, return_pages=True)[0]
        evaluation = FileSplittingEvaluation([ground_truth], [pred], zero_division=None).get_evaluation_data()
        assert evaluation.tp == 3
        assert evaluation.fp == 0
        assert evaluation.fn == 0
        assert evaluation.tn == 2
        assert evaluation.precision == 1.0
        assert evaluation.recall == 1.0
        assert evaluation.f1 == 1.0

    def test_metrics_get_evaluation_data_by_category(self):
        """Test get_evaluation_data method of the Evaluation class within the specified Category."""
        splitting_ai = SplittingAI(self.file_splitting_model)
        ground_truth = self.test_document
        pred = splitting_ai.propose_split_documents(self.test_document, return_pages=True)[0]
        evaluation = FileSplittingEvaluation([ground_truth], [pred]).get_evaluation_data(search=ground_truth.category)
        assert evaluation.tp == 3
        assert evaluation.fp == 0
        assert evaluation.fn == 0
        assert evaluation.tn == 2
        assert evaluation.precision == 1.0
        assert evaluation.recall == 1.0
        assert evaluation.f1 == 1.0
