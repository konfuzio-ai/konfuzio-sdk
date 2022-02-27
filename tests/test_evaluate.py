"""Test the evaluation."""
import os
import unittest
from copy import deepcopy

from pandas import DataFrame

from konfuzio_sdk.data import Project, Document
from konfuzio_sdk.evaluate import compare, grouped
from konfuzio_sdk.urls import get_document_api_details_url
from konfuzio_sdk.utils import is_file

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
        assert len(evaluation) == 24
        # for an Annotation which is human made, it is nan, so that above threshold is False
        # doc_a 19 + 2 multiline + 2 feedback required + 1 rejected
        assert evaluation["true_positive"].sum() == 21  # 1 multiline with 2 lines = 2 Annotations
        assert evaluation["false_positive"].sum() == 0
        # due to the fact that Konfuzio Server does not save confidence = 100 % if Annotation was not created by a human
        assert evaluation["false_negative"].sum() == 0

    def test_doc_where_first_annotation_was_skipped(self):
        """Test if a Document is 100 % equivalent with first Annotation not existing for a certain Label."""
        prj = Project(id_=TEST_PROJECT_ID)
        doc_a = prj.get_document_by_id(44823)
        doc_b = prj.get_document_by_id(44823)  # predicted
        doc_b.annotations()
        doc_b._annotations.pop(0)  # pop an Annotation that is correct in BOTH  Documents
        assert doc_a._annotations == doc_b._annotations  # first Annotation is removed in both  Documents
        evaluation = compare(doc_a, doc_b)
        assert len(evaluation) == 22  # 2 Annotations are is_correct false
        # doc_a 18 (multiline removed) + 1 multiline + 2 feedback required + 1 rejected
        assert evaluation["true_positive"].sum() == 19
        assert evaluation["false_positive"].sum() == 0
        assert evaluation["false_negative"].sum() == 0

    def test_doc_where_last_annotation_was_skipped(self):
        """Test if a Document is 100 % equivalent with last Annotation not existing for a certain Label."""
        prj = Project(id_=TEST_PROJECT_ID)
        doc_a = prj.get_document_by_id(44823)
        doc_b = prj.get_document_by_id(44823)  # predicted
        doc_b.annotations()
        doc_b._annotations.pop(11)  # pop an Annotation that is correct in BOTH  Documents
        assert doc_a._annotations == doc_b._annotations  # last Annotation is removed in both  Documents
        evaluation = compare(doc_a, doc_b)
        assert len(evaluation) == 23  # 2 Annotations are is_correct false
        # doc_a 18 + 2 multiline + 2 feedback required + 1 rejected
        assert evaluation["true_positive"].sum() == 20
        assert evaluation["false_positive"].sum() == 0
        assert evaluation["false_negative"].sum() == 0

    def test_if_first_multiline_annotation_is_missing_in_b(self):
        """Test if a Document is equivalent if first Annotation is missing."""
        prj = Project(id_=TEST_PROJECT_ID)
        doc_a = prj.get_document_by_id(44823)
        doc_b = Document(project=prj, category=doc_a.category)
        for annotation in doc_a.annotations()[1:]:
            doc_b.add_annotation(annotation)

        assert len(doc_b.annotations()) == len(doc_a.annotations()) - 1
        evaluation = compare(doc_a, doc_b)
        assert len(evaluation) == 24  # 2 Annotations are false
        # doc_a 19 + 2 multiline + 2 feedback required + 1 rejected
        assert evaluation["true_positive"].sum() == 19  # 1 multiline with 2 lines = 2 Annotations
        assert evaluation["false_positive"].sum() == 0
        assert evaluation["false_negative"].sum() == 2

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
        assert len(evaluation) == 24  # 2 annotations are false and two have feedback required
        # TODO: add feedback required again
        assert evaluation["true_positive"].sum() == 19
        assert evaluation["false_positive"].sum() == 4  # 1 multiline (2 lines == 2 Annotations) + 2 feedback required
        assert evaluation["false_negative"].sum() == 0

    def test_only_unrevised_annotations(self):
        """Test to evaluate on a Document that has only unrevised Annotations."""
        prj = Project(id_=TEST_PROJECT_ID)
        for document in prj.no_status_documents:
            if document.id_ == 137234:
                doc_a = document

        doc_b = Document(project=prj)

        assert len(doc_a.annotations()) == len(doc_b.annotations()) == 0
        evaluation = compare(doc_a, doc_b, only_use_correct=True)
        assert len(evaluation) == 1  # placeholder
        # TODO: add feedback required again
        assert evaluation["true_positive"].sum() == 0
        assert evaluation["false_positive"].sum() == 0
        assert evaluation["false_negative"].sum() == 0

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
        assert len(evaluation) == 24  # 2 Annotations are false
        # doc_a 18 + 1 multiline + 2 feedback required + 1 rejected
        assert evaluation["true_positive"].sum() == 19
        assert evaluation["false_positive"].sum() == 2  # 1 multiline (2 lines == 2 Annotations)
        assert evaluation["false_negative"].sum() == 0

    def test_doc_where_last_annotation_is_missing_in_b(self):
        """Test if a Document is equivalent if last Annotation is missing."""
        prj = Project(id_=TEST_PROJECT_ID)
        doc_a = prj.get_document_by_id(44823)
        doc_b = Document(project=prj, category=doc_a.category)
        # use correct Annotations and feedback required ones
        for annotation in doc_a.annotations(use_correct=False)[:-1]:
            doc_b.add_annotation(annotation)

        evaluation = compare(doc_a, doc_b)
        assert len(evaluation) == 24  # 2 Annotations are false
        # doc_a 19 + 2 multiline
        assert evaluation["true_positive"].sum() == 20  # due to the fact that we find both offsets of the multiline
        assert evaluation["false_positive"].sum() == 0
        assert evaluation["false_negative"].sum() == 1

    def test_doc_where_last_annotation_is_missing_in_a(self):
        """Test if a Document is equivalent if last Annotation is not present."""
        prj = Project(id_=TEST_PROJECT_ID)
        doc_b = prj.get_document_by_id(44823)
        doc_a = Document(project=prj, category=doc_b.category)
        # use correct Annotations and feedback required ones
        for annotation in doc_b.annotations(use_correct=False)[:-1]:
            doc_a.add_annotation(annotation)

        evaluation = compare(doc_a, doc_b)
        assert len(evaluation) == 24  # 2 Annotations are false
        # doc_a 18 + 2 multiline
        assert evaluation["true_positive"].sum() == 20  # due to the fact that we find both offsets of the multiline
        assert evaluation["false_positive"].sum() == 1
        assert evaluation["false_negative"].sum() == 0

    def test_nothing_should_be_predicted(self):
        """Support to evaluate that nothing is found in a document."""
        prj = Project(id_=TEST_PROJECT_ID)
        doc_a = Document(project=prj)
        doc_b = prj.get_document_by_id(44823)
        evaluation = compare(doc_a, doc_b)
        assert len(evaluation) == 25
        assert evaluation["true_positive"].sum() == 0
        # any Annotation above threshold is a false positive independent if it's correct or revised
        assert len([an for an in doc_b.annotations(use_correct=False) if an.confidence > an.label.threshold]) == 21
        assert evaluation["false_positive"].sum() == 23  # but one annotation is multiline
        assert evaluation["false_negative"].sum() == 0

    def test_nothing_can_be_predicted(self):
        """Support to evaluate that nothing must be found in a document."""
        prj = Project(id_=TEST_PROJECT_ID)
        doc_a = prj.get_document_by_id(44823)
        doc_b = Document(project=prj)
        evaluation = compare(doc_a, doc_b)
        assert len(evaluation) == 25  # we evaluate on span level an one annotation is multiline
        assert evaluation["true_positive"].sum() == 0
        assert evaluation["false_positive"].sum() == 0
        assert evaluation["false_negative"].sum() == 21

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
        assert len(evaluation) == 24
        # Evaluation as it is now: everything needs to be find even if multiple=False
        assert evaluation["true_positive"].sum() == 20
        assert evaluation["false_positive"].sum() == 0
        assert evaluation["false_negative"].sum() == 1

    @unittest.skip(reason='Add support for doc.annotations(top_candidates=True) filter.')
    def test_doc_with_multiple_top_Annotations(self):
        """Support to evaluate all Annotations of one Label (mult.=T) in one Annotation-Set."""
        # TODO: the test to be skipped should be the one with multiple False because we assume everything behaves
        #  as multiple True
        raise NotImplementedError

    @unittest.skip(reason='Invalid reuse of Annotations across documents.')
    def test_doc_with_missing_annotation_set(self):
        """
        Test if a Document is equivalent if an Annotation Set is missing.

        We create 1 copy of a Document online.
        We copy all Annotation Sets except the last one - doc b.
        Doc b will have a missing Annotation Set compared to doc a.

        All Annotations expected for that Annotation Set will be counted as false negatives.
        """
        prj = Project(id_=TEST_PROJECT_ID)
        doc_a = prj.get_document_by_id(44859)
        doc_b = Document(project=prj, category=doc_a.category)
        doc_a.annotations()

        # TODO: should annotation_sets have the same behaviour as Annotations? issue #8739
        for annotation_set in doc_a.annotation_sets[:-1]:
            doc_b.add_annotation_set(annotation_set)

            # TODO: add Annotations when adding an Annotation Set (issue #8740)
            for annotation in annotation_set.annotations:
                doc_b.add_annotation(annotation)

        evaluation = compare(doc_a, doc_b)
        assert len(evaluation) == 31
        assert evaluation["true_positive"].sum() == 30
        assert evaluation["false_positive"].sum() == 0
        assert evaluation["false_negative"].sum() == 1

    @unittest.skip(reason='Invalid reuse of Annotations across documents.')
    def test_doc_with_extra_annotation_set(self):
        """
        Test if a Document is equivalent if there is one more Annotation Set than the correct ones.

        We create 2 copies of a Document online.
        In copy 1 we keep all Annotation Sets - doc a
        In copy 2 we remove one Annotation Set from the Label Set Brutto Bezug (multiple=True) - doc b
        Doc b will have an extra Annotation Set compared to doc a.

        The Annotations in the extra Annotation Set will be counted as false positives.
        """
        prj = Project(id_=TEST_PROJECT_ID)
        doc_online = prj.get_document_by_id(44859)
        doc_a = Document(project=prj, category=doc_online.category)
        doc_b = Document(project=prj, category=doc_online.category)
        doc_online.annotations()

        # TODO: should annotation_sets have the same behaviour as Annotations? issue #8739
        for annotation_set in doc_online.annotation_sets:
            doc_b.add_annotation_set(annotation_set)

            # TODO: add annotations when adding an Annotation Set (issue #8740)
            for annotation in annotation_set.annotations:
                doc_b.add_annotation(annotation)

        # Last Annotation Set from Brutto Bezug removed
        # TODO: should annotation_sets have the same behaviour as Annotations? issue #8739
        for annotation_set in doc_online.annotation_sets[:-2] + [doc_online.annotation_sets[-1]]:
            doc_a.add_annotation_set(annotation_set)

            # TODO: add annotations when adding an Annotation Set (issue #8740)
            for annotation in annotation_set.annotations:
                doc_a.add_annotation(annotation)

        # TODO:
        #  add "delete" to AnnotationSet that also deletes the annotations that belong to that Annotation Set
        #  add "delete" where the AnnotationSet can be specified by ID

        assert len(doc_a.annotation_sets) == len(doc_b.annotation_sets) - 1
        assert len(doc_b.annotation_sets[-2].annotations) > 0

        evaluation = compare(doc_a, doc_b)
        assert len(evaluation) == 31
        assert evaluation["true_positive"].sum() == 26
        assert evaluation["false_positive"].sum() == 5
        assert evaluation["false_negative"].sum() == 0

    @unittest.skip(reason='Invalid reuse of Annotations across documents.')
    def test_doc_with_annotation_set_with_multiple_label_that_should_be_split(self):
        """
        Test if a Document is equivalent if one Annotation Set is predicted instead of two.

        Annotations that should be from different Annotation Sets, with the same Label Set, are predicted into a single
        one. In this test case, the annotations considered are from a Label with multiple=True.

        Those annotations predicted with the wrong Annotation Set are considered false positives.
        """
        prj = Project(id_=TEST_PROJECT_ID)
        doc_a = prj.get_document_by_id(44841)
        doc_b = Document(project=prj, category=doc_a.category)

        for annotation in doc_a.annotations(use_correct=False):
            # replace 1st Steuer Annotation Set (ID 679457) with 2nd (ID 679458)
            new_annotation = deepcopy(annotation)
            if annotation.annotation_set.id_ == 679457:
                new_annotation.annotation_set.id_ = 679458

            doc_b.add_annotation(new_annotation)

        assert len(doc_b.annotations(use_correct=False)) == len(doc_a.annotations(use_correct=False))

        evaluation = compare(doc_a, doc_b)
        assert len(evaluation) == 32
        assert evaluation["true_positive"].sum() == 27
        assert evaluation["false_positive"].sum() == 5
        # We don't count the annotations for the Annotation Set missing because we would be duplicating the penalization
        # and the user only needs 5 actions to correct it
        assert evaluation["false_negative"].sum() == 0

    @unittest.skip(reason='Invalid reuse of Annotations across documents.')
    def test_doc_with_annotation_set_with_multiple_label_that_should_be_split_and_one_annotation_missing(self):
        """
        Test if a Document is equivalent if 1 Annotation Set is predicted instead of 2 and 1 annotation is missing.

        Annotations that should be from different Annotation Sets, with the same Label Set, are predicted into a single
        one. In this test case, the annotations considered are from a Label with multiple=True and one of them is
        not predicted.

        Those annotations predicted with the wrong Annotation Set are considered false positives and the missing
        annotation is predicted as false negative.
        """
        prj = Project(id_=TEST_PROJECT_ID)
        doc_a = prj.get_document_by_id(44841)
        doc_b = Document(project=prj, category=doc_a.category)

        # skip last multiple annotation from Steuer
        for annotation in doc_a.annotations(use_correct=False)[:-7] + doc_a.annotations(use_correct=False)[-6:]:
            # replace 1st Steuer Annotation Set (ID 679457) with 2nd (ID 679458)
            new_annotation = deepcopy(annotation)
            if annotation.annotation_set.id_ == 679457:
                new_annotation.annotation_set.id_ = 679458

            doc_b.add_annotation(new_annotation)

        assert len(doc_b.annotations()) == len(doc_a.annotations()) - 1

        evaluation = compare(doc_a, doc_b)
        assert len(evaluation) == 32
        assert evaluation["true_positive"].sum() == 27
        assert evaluation["false_positive"].sum() == 4
        assert evaluation["false_negative"].sum() == 1

    def test_doc_with_annotation_with_wrong_offsets(self):
        """
        Test a Document with an annotation that has the correct offset string and classification but the wrong offsets.

        This means that the AI got it from the wrong position in the document.
        The offset string, Annotation Set,Labeland Label Set are correct but the start and end offsets are wrong.

        It is counted as FP because does not match any correct annotation and as FN because there was no correct
        prediction for the annotation in the document.

        We are double penalizing the AI, however to correct this, the user needs to reject the predicted annotation and
        create a new one (2 actions).
        TODO: review
        """
        prj = Project(id_=TEST_PROJECT_ID)
        doc_a = prj.get_document_by_id(44841)
        doc_b = Document(project=prj, category=doc_a.category)

        new_annotation = deepcopy(doc_a.annotations(use_correct=False)[1])
        # keep the offset string but change the start and end offsets
        assert new_annotation.offset_string == ['00600']
        new_annotation.spans[0].start_offset = 79
        new_annotation.spans[0].end_offset = 84
        doc_b.add_annotation(new_annotation)

        for annotation in [doc_a.annotations(use_correct=False)[0]] + doc_a.annotations(use_correct=False)[2:]:
            doc_b.add_annotation(annotation)

        assert len(doc_b.annotations()) == len(doc_a.annotations())

        evaluation = compare(doc_a, doc_b)
        assert len(evaluation) == 33
        assert evaluation["true_positive"].sum() == 31
        assert evaluation["false_positive"].sum() == 1
        # We get 1 false negative. However, it's already counted as FP. Nevertheless, the user needs to reject the
        # the predicted one and create a new one even if the Label, Label Set and Annotation Set are correct
        assert evaluation["false_negative"].sum() == 1

    def test_doc_with_annotation_with_incomplete_offsets(self):
        """
        Test a Document with an annotation that has incomplete offsets (last character missing).

        It is counted as FP because does not match any correct annotation and as FN because there was no correct
        prediction for the annotation in the document.

        We are double penalizing the AI, however to correct this, the user only needs to adjust the offsets.
        TODO: review
        """
        prj = Project(id_=TEST_PROJECT_ID)
        doc_a = prj.get_document_by_id(44841)
        doc_b = Document(project=prj, category=doc_a.category)

        new_annotation = deepcopy(doc_a.annotations(use_correct=False)[0])
        # keep the offset string but change the start and end offsets
        assert new_annotation.offset_string == ['03.01.2018']
        new_annotation.spans[0].end_offset = new_annotation.spans[0].end_offset - 1
        doc_b.add_annotation(new_annotation)

        for annotation in doc_a.annotations(use_correct=False)[1:]:
            doc_b.add_annotation(annotation)

        assert len(doc_b.annotations()) == len(doc_a.annotations())

        evaluation = compare(doc_a, doc_b)
        assert len(evaluation) == 33
        assert evaluation["true_positive"].sum() == 31
        assert evaluation["false_positive"].sum() == 1
        # We get 1 false negative. However, it's already counted as FP. Nevertheless, the user needs to adjust the
        # offsets
        assert evaluation["false_negative"].sum() == 1

    def test_doc_with_annotation_with_wrong_offsets_and_wrong_classification(self):
        """
        Test a Document with a wrong annotation (wrong offsets + wrong classification).

        It is counted as FP because does not match any correct annotation and as FN because there was no correct
        prediction for the annotation in the document.

        We are double penalizing the AI, however to correct this, the user needs to reject the predicted annotation and
        create a new one (2 actions).
        TODO: review
        """
        prj = Project(id_=TEST_PROJECT_ID)
        doc_a = prj.get_document_by_id(44841)
        doc_b = Document(project=prj, category=doc_a.category)

        new_annotation = deepcopy(doc_a.annotations(use_correct=False)[0])
        # keep the offset string but change the start and end offsets
        assert new_annotation.offset_string == ['03.01.2018']
        new_annotation.spans[0].start_offset = 79
        new_annotation.spans[0].end_offset = 84
        doc_b.add_annotation(new_annotation)

        for annotation in doc_a.annotations(use_correct=False)[1:]:
            doc_b.add_annotation(annotation)

        assert len(doc_b.annotations()) == len(doc_a.annotations())

        evaluation = compare(doc_a, doc_b)
        assert len(evaluation) == 33
        assert evaluation["true_positive"].sum() == 31
        assert evaluation["false_positive"].sum() == 1
        # We get 1 false negative. However, it's already counted as FP. Nevertheless, the user needs to adjust the
        # offsets
        assert evaluation["false_negative"].sum() == 1

    @unittest.skip(reason='Invalid reuse of Annotations across documents.')
    def test_doc_with_annotation_set_with_each_annotation_belonging_to_different_annotation_sets(self):
        """
        Test a Document where each annotation from a certain Annotation Set is predicted with different Annotation Sets.

        Each annotation in a certain Annotation Set is predicted as belonging to a different Annotation Set.
        The Annotation Set cannot be determined by the mode of the Annotation Set IDs.
        """
        prj = Project(id_=TEST_PROJECT_ID)
        doc_a = prj.get_document_by_id(44841)
        doc_b = Document(project=prj, category=doc_a.category)

        # 1st annotation from 1st Annotation Set Brutto-Bezug belonging to the 2nd Annotation Set
        new_annotation_1 = deepcopy(doc_a.annotations(use_correct=False)[6])
        # keep the annotation but change the Annotation Set ID
        assert new_annotation_1.offset_string == ['2020']
        new_annotation_1.annotation_set.id_ = 79165 + 10
        doc_b.add_annotation(new_annotation_1)

        # 3rd annotation from 1st Annotation Set Brutto-Bezug belonging to the 3rd Annotation Set
        new_annotation_2 = deepcopy(doc_a.annotations(use_correct=False)[8])
        # keep the annotation but change the Annotation Set ID
        assert new_annotation_2.offset_string == ['2.285,50']
        new_annotation_2.annotation_set.id_ = 79166 + 10
        doc_b.add_annotation(new_annotation_2)

        for annotation in (
            doc_a.annotations(use_correct=False)[:6]
            + [doc_a.annotations(use_correct=False)[7]]
            + doc_a.annotations(use_correct=False)[9:]
        ):
            new_annotation = deepcopy(annotation)
            new_annotation.annotation_set.id_ = new_annotation.annotation_set.id_ + 10
            doc_b.add_annotation(new_annotation)

        assert len(doc_b.annotations()) == len(doc_a.annotations())
        evaluation = compare(doc_a, doc_b)
        assert len(evaluation) == 32
        assert evaluation["true_positive"].sum() == 30
        assert evaluation["false_positive"].sum() == 2
        assert evaluation["false_negative"].sum() == 0

    @unittest.skip(reason='Invalid reuse of Annotations across documents.')
    def test_doc_with_missing_annotation_in_a_line_with_multiple_annotation_sets(self):
        """
        Test is a Document is equivalent if an annotation is missing in a line where exists another Annotation Set.

        In this test case, there are two annotations where each one belongs to an Annotation Set from a different Label
        Set.
        """
        prj = Project(id_=TEST_PROJECT_ID)
        doc_a = prj.get_document_by_id(44841)
        doc_b = Document(project=prj, category=doc_a.category)

        # skip 1 annotation from an Annotation Set that is in the same line as other Annotation Set
        for annotation in doc_a.annotations(use_correct=False)[:28] + doc_a.annotations(use_correct=False)[29:]:
            doc_b.add_annotation(annotation)

        assert len(doc_b.annotations()) == len(doc_a.annotations()) - 1

        evaluation = compare(doc_a, doc_b)
        assert len(evaluation) == 32
        assert evaluation["true_positive"].sum() == 31
        assert evaluation["false_positive"].sum() == 0
        assert evaluation["false_negative"].sum() == 1

    @unittest.skip(reason="Cannot load Project with custom strings with current changes.")
    def test_doc_with_custom_offset_string(self):
        """
        Test a Document with an annotation that has a custom offset string.

        These annotations are not used for training so they should not be counted for evaluation.
        How would they be handled in the evaluation?
        TODO: review
        """
        prj = Project(id_=TEST_PROJECT_ID)
        _ = prj.get_document_by_id(44823)
        _ = Document(project=prj)

    @unittest.skip(reason='Waiting to be able toe specify the Extraction AI model version.')
    def test_compare_extractions_available_online_to_the_current_version(self):
        """Compare a version of the online available extraction to the status quo of the human doc."""
        prj = Project(id_=TEST_PROJECT_ID)
        doc_human = prj.documents[-1]
        online = get_document_api_details_url(doc_human.id_)
        doc_online = Document(project=prj)
        doc_online = doc_online.get_from_online(online)
        compare(doc_human, doc_online)

    @unittest.skip(reason='Waiting for new trainer ai model build.')
    def test_compare_extractions_to_a_real_doc(self):
        """Test to compare the results of an extraction model to the human annotations."""
        prj = Project(id_=TEST_PROJECT_ID)
        human_doc = prj.get_document_by_id(44823)
        path_to_model = os.path.join(os.getcwd(), "lohnabrechnung.pkl")
        if is_file(file_path=path_to_model, raise_exception=False):
            evaluation = human_doc.evaluate_extraction_model(path_to_model)
            self.assertEqual(10, evaluation["true_positive"].sum())
            self.assertEqual(1, evaluation["false_positive"].sum())
            self.assertEqual(9, evaluation["false_negative"].sum())
            self.assertEqual(1519, evaluation['start_offset'][6])
            self.assertEqual(1552, evaluation['end_offset'][18])


def test_grouped():
    """Test if group catches all relevant errors."""
    grouped(DataFrame([[True, 'a'], [False, 'b']], columns=['is_correct', 'target']), target='target')
    grouped(DataFrame([[False, 'a'], [False, 'b']], columns=['is_correct', 'target']), target='target')
    grouped(DataFrame([[None, 'a'], [None, 'b']], columns=['is_correct', 'target']), target='target')
