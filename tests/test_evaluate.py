"""Test the evaluation."""
import unittest
import os

from konfuzio_sdk.data import Project, Document
from konfuzio_sdk.evaluate import compare
from konfuzio_sdk.urls import get_document_api_details_url

from settings import BASE_DIR


class TestEvaluation(unittest.TestCase):
    """Testing evaluation of the Konfuzio Server.

    Implemented:
        - prediction without complete offsets (e.g missing last character)
        - missing prediction for a label with multiple=True (https://app.konfuzio.com/a/7344142)
        - evaluation of Annotations with multiple=False in a strict mode, so all must be found
        - missing annotation sets (e.g. missing 1st and 3rd annotation set in a document)
        - missing correct annotations in annotation-set
        - too many annotations than correct annotations in annotation-set
        - too many annotation-sets
        - correct annotations that are predicted to be in different annotation-sets
        - multiline annotations with multiple not connected offsets
        - multiple annotation sets in 1 line
        - any possible grouping of annotations into annotation-set(s)

    Unsupported:
        - if two offsets are correctly grouped into a correct number of Annotations
            todo: this would be needed to evaluate how well a horizontal and vertical merging is working
                the approach would be quite similar to grouping Annotations (mult.=T) into Annotation-Sets

    Reasoning on how to evaluate them before implementation needed:
        - prediction with incorrect offsets but with correct offset string are no longer possible
        - prediction with incorrect offsets and incorrect offset string
        - prediction of one of multiple annotations for a label in one annotation
        - annotation with custom string

    """

    def test_doc_on_doc_incl_multiline_annotation(self):
        """Test if a document is 100 % equivalent even it has unrevised Annotations."""
        prj = Project()
        doc_a = prj.documents[0]
        doc_b = prj.documents[0]
        evaluation = compare(doc_a, doc_b)
        assert len(evaluation) == 13  # 2 annotations are false
        # for an annotation which is human made, it is nan, so that above threshold is False
        assert evaluation["true_positive"].sum() == 11
        assert evaluation["false_positive"].sum() == 0
        # due to the fact that Konfuzio Server does not save confidence = 100 % if annotation was not created by a human
        assert evaluation["false_negative"].sum() == 0

    def test_doc_on_a_doc_where_first_annotation_was_skipped(self):
        """Test if a document is 100 % equivalent even it has unrevised Annotations."""
        prj = Project()
        doc_a = prj.documents[0]
        doc_b = prj.documents[0]
        doc_b._annotations.pop(0)  # pop an annotation that is correct
        evaluation = compare(doc_a, doc_b)
        assert len(evaluation) == 11  # 2 annotations are false
        assert evaluation["true_positive"].sum() == 9
        assert evaluation["false_positive"].sum() == 0
        assert evaluation["false_negative"].sum() == 0

    def test_doc_where_first_annotation_was_skipped(self):
        """Test if a document is 100 % equivalent even it has unrevised Annotations."""
        prj = Project()
        doc_a = prj.documents[0]
        doc_b = prj.documents[0]
        doc_a._annotations.pop(0)  # pop an annotaiton that is correct
        evaluation = compare(doc_a, doc_b)
        assert len(evaluation) == 11  # 2 annotations are false
        assert evaluation["true_positive"].sum() == 9  # todo check
        assert evaluation["false_positive"].sum() == 0
        assert evaluation["false_negative"].sum() == 0

    def test_doc_on_a_doc_where_last_annotation_was_skipped(self):
        """Test if a document is 100 % equivalent even it has unrevised Annotations."""
        prj = Project()
        doc_a = prj.documents[0]
        doc_b = prj.documents[0]
        doc_b._annotations.pop(11)  # pop an annotation that is correct
        evaluation = compare(doc_a, doc_b)
        assert len(evaluation) == 12  # 2 annotations are false
        assert evaluation["true_positive"].sum() == 10  # due to the fact that we find both offsets of the multiline
        assert evaluation["false_positive"].sum() == 0
        assert evaluation["false_negative"].sum() == 0

    def test_doc_where_last_annotation_was_skipped(self):
        """Test if a document is 100 % equivalent even it has unrevised Annotations."""
        prj = Project()
        doc_a = prj.documents[0]
        doc_b = prj.documents[0]
        doc_a._annotations.pop(11)  # pop an annotaiton that is correct
        evaluation = compare(doc_a, doc_b)
        assert len(evaluation) == 12  # 2 annotations are false
        assert evaluation["true_positive"].sum() == 10  # due to the fact that we find both offsets of the multiline
        assert evaluation["false_positive"].sum() == 0
        assert evaluation["false_negative"].sum() == 0

    def test_nothing_should_be_predicted(self):
        """Support to evaluate that nothing is found in a document."""
        prj = Project()
        doc_a = Document()
        doc_b = prj.documents[0]
        evaluation = compare(doc_a, doc_b)
        assert len(evaluation) == 14
        assert evaluation["true_positive"].sum() == 0
        # any annotation above threshold is a false positive independent if it's correct or revised
        assert len([an for an in doc_b.annotations(use_correct=False) if an.accuracy > an.label.threshold]) == 12
        assert evaluation["false_positive"].sum() == 13
        assert evaluation["false_negative"].sum() == 0

    def test_nothing_can_be_predicted(self):
        """Support to evaluate that nothing must be found in a document."""
        prj = Project()
        doc_a = prj.documents[0]
        doc_b = Document()
        evaluation = compare(doc_a, doc_b)
        assert len(evaluation) == 14
        assert evaluation["true_positive"].sum() == 0
        assert evaluation["false_positive"].sum() == 0
        assert evaluation["false_negative"].sum() == 11

    def test_doc_with_overruled_top_annotations(self):
        """Support to evaluate only one of x Annotations of one Label (mult.=F) in one Annotation-Set."""
        # todo: this logic is a view logic on the document: shouldn't this go into the annotations function
        prj = Project()
        doc_a = prj.documents[0]
        doc_b = prj.documents[0]
        evaluation = compare(doc_a, doc_b)
        assert len(evaluation) == 13  # 2 annotations are false
        assert evaluation["true_positive"].sum() == 11  # due to the fact that we find both offsets of the multiline
        assert evaluation["false_positive"].sum() == 0
        assert evaluation["false_negative"].sum() == 0
        return False

    @unittest.skip(reason='Add support for doc.annotations(top_candidates=True) filter.')
    def test_doc_with_multiple_top_Annotations(self):
        """Support to evaluate all Annotations of one Label (mult.=T) in one Annotation-Set."""
        raise NotImplementedError

    @unittest.skip(reason='Waiting to be able toe specify the Extraction AI model version.')
    def test_compare_extractions_available_online_to_the_current_version(self):
        """Compare a version of the online available extraction to the status quo of the human doc."""
        prj = Project()
        doc_human = prj.documents[-1]
        online = get_document_api_details_url(doc_human.id, include_extractions=True)
        doc_online = Document()
        doc_online = doc_online.get_from_online(online)
        compare(doc_human, doc_online)

    @unittest.skip(reason="Request access via info@konfuzio.com to run this test.")
    def test_compare_extractions_to_a_real_doc(self):
        """Test to compare the results of an extraction model to the human annotations."""
        prj = Project()
        human_doc = prj.documents[0]
        self.assertEqual(44823, human_doc.id)
        path_to_model = os.path.join(BASE_DIR, "lohnabrechnung.pkl")
        evaluation = human_doc.evaluate_extraction_model(path_to_model)
        self.assertEqual(10, evaluation["true_positive"].sum())
        self.assertEqual(1, evaluation["false_positive"].sum())
        self.assertEqual(1, evaluation["false_negative"].sum())
        self.assertEqual(1519, evaluation['start_offset'][5])
        self.assertEqual(1552, evaluation['end_offset'][13])
