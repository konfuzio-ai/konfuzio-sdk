"""Validate data functions."""
import glob
import logging
import os
import unittest

import pytest
from konfuzio_sdk.data import Project, Annotation
from konfuzio_sdk.utils import is_file

logger = logging.getLogger(__name__)


@pytest.mark.serial
class TestAPIDataSetup(unittest.TestCase):
    """Test handle data."""

    document_count = 24
    test_document_count = 4
    correct_document_count = 24

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the test project."""
        cls.prj = Project()
        assert len(cls.prj.documents) == cls.document_count
        assert len(cls.prj.test_documents) == cls.test_document_count
        assert len(cls.prj.labels[0].correct_annotations) == cls.correct_document_count

    def test_label_sets(self):
        """Test label sets in the test project."""
        assert self.prj.label_sets.__len__() == 2
        assert self.prj.label_sets[0].name == 'Lohnabrechnung'
        assert self.prj.label_sets[0].categories.__len__() == 0
        assert self.prj.label_sets[0].labels.__len__() == 8
        assert not self.prj.label_sets[0].has_multiple_annotation_sets
        assert self.prj.label_sets[1].name == 'Brutto-Bezug'
        assert self.prj.label_sets[1].categories.__len__() == 1
        assert self.prj.label_sets[1].categories[0].name == 'Lohnabrechnung'
        assert self.prj.label_sets[1].labels.__len__() == 3
        assert self.prj.label_sets[1].has_multiple_annotation_sets

    def test_categories(self):
        """Test get labels in the project."""
        assert self.prj.categories.__len__() == 1
        assert self.prj.categories[0].name == 'Lohnabrechnung'
        assert self.prj.categories[0].is_default
        assert not self.prj.categories[0].has_multiple_annotation_sets

    def test_get_images(self):
        """Test get paths to the images of the first training document."""
        self.prj.documents[0].get_images()
        assert len(self.prj.documents[0].image_paths) == len(self.prj.documents[0].pages)

    def test_get_file(self):
        """Test get path to the file of the first training document."""
        self.prj.documents[0].get_file()
        assert self.prj.documents[0].ocr_file_path

    def test_labels(self):
        """Test get labels in the project."""
        assert len(self.prj.labels) == 11
        assert self.prj.labels[0].name == 'Auszahlungsbetrag'
        assert not self.prj.labels[0].has_multiple_top_candidates

    def test_project(self):
        """Test basic properties of the project object."""
        assert is_file(self.prj.meta_file_path)
        assert self.prj.documents[1].id > self.prj.documents[0].id
        assert len(self.prj.documents)
        # check if we can initialize a new project object, which will use the same data
        assert len(self.prj.documents) == self.document_count
        new_project = Project()
        assert len(new_project.documents) == self.correct_document_count
        assert new_project.meta_file_path == self.prj.meta_file_path

    def test_update_prj(self):
        """Test number of documents after updating a project."""
        assert len(self.prj.documents) == self.document_count
        self.prj.update()
        assert len(self.prj.documents) == self.correct_document_count
        is_file(self.prj.meta_file_path)

    def test_document(self):
        """Test properties of a specific documents in the test project."""
        doc = self.prj.labels[0].documents[5]  # one doc before doc without annotations
        assert doc.id == 44842
        assert doc.category.name == 'Lohnabrechnung'
        assert len(self.prj.labels[0].correct_annotations) == self.correct_document_count
        doc.update()
        self.assertEqual(len(self.prj.labels[0].correct_annotations), self.document_count)
        assert len(doc.text) == 4793
        assert len(glob.glob(os.path.join(doc.root, '*.*'))) == 4

        # existing annotation
        assert len(doc.annotations(use_correct=False)) == 13
        assert doc.annotations()[0].offset_string == '22.05.2018'  # start_offset=465, start_offset=466
        assert len(doc.annotations()) == 13
        assert doc.annotations()[0].is_online
        assert not doc.annotations()[0].save()  # Save returns False because Annotation is already online.

    def test_document_with_multiline_annotation(self):
        """Test properties of a specific documents in the test project."""
        doc = self.prj.labels[0].documents[0]  # one doc before doc without annotations
        assert doc.id == 44823
        assert doc.category.name == 'Lohnabrechnung'
        self.assertEqual(len(self.prj.labels[0].correct_annotations), self.document_count)
        doc.update()
        self.assertEqual(self.document_count, len(self.prj.labels[0].correct_annotations))
        self.assertEqual(len(doc.text), 4537)
        self.assertEqual(len(glob.glob(os.path.join(doc.root, '*.*'))), 4)

        # existing annotation
        self.assertEqual(len(doc.annotations(use_correct=False)), 12)
        # a multiline annotation in the top right corner, see https://app.konfuzio.com/a/4419937
        # todo improve multiline support
        self.assertEqual(66, doc.annotations()[0]._spans[0].start_offset)
        self.assertEqual(78, doc.annotations()[0]._spans[0].end_offset)
        self.assertEqual(159, doc.annotations()[0]._spans[1].start_offset)
        self.assertEqual(169, doc.annotations()[0]._spans[1].end_offset)
        self.assertEqual(len(doc.annotations()), 10)
        self.assertTrue(doc.annotations()[0].is_online)
        self.assertTrue(not doc.annotations()[0].save())  # Save returns False because Annotation is already online.

    def test_add_document_twice(self):
        """Test adding same document twice."""
        old_doc = self.prj.documents[1]
        assert len(self.prj.documents) == self.document_count
        self.prj.add_document(old_doc)
        assert len(self.prj.documents) == self.document_count

    def test_correct_annotations(self):
        """Test correct annotations of a certain label in a specific document."""
        for doc in self.prj.documents:
            if doc.id == 26608:
                assert len(doc.annotations(self.prj.get_label_by_id(579))) == 1

    def test_annotation_start_offset_zero_filter(self):
        """Test annotations with start offset equal to zero."""
        doc = self.prj.labels[0].documents[5]  # one doc before doc without annotations
        assert len(doc.annotations()) == 13
        assert doc.annotations()[0].start_offset == 188
        assert len(doc.annotations()) == 13

    def test_multiline_annotation(self):
        """Test to convert a multiline span Annotation to a dict."""
        assert len(self.prj.documents[0].annotations()[0].eval_dict) == 2

    def test_annotation_to_dict(self):
        """Test to convert a Annotation to a dict."""
        anno = self.prj.documents[0].annotations()[1].eval_dict[0]
        assert anno["id"] == 4420022
        assert anno["accuracy"] == 1.0
        # assert anno["created_by"] ==  todo: support this variable provided via API in the Annotation
        # assert anno["custom_offset_string"] ==   todo: support this variable provided via API in the Annotation
        assert anno["end_offset"] == 366  # todo support multiline Annotations
        # assert anno["get_created_by"] == "ana@konfuzio.com"  todo: support this variable provided via API
        # assert anno["get_revised_by"] == "n/a" todo: support this variable provided via API in the Annotation
        assert anno["is_correct"]
        assert anno["label_id"] == 860  # original REST API calls it "label" however means label_id
        # assert anno["label_data_type"] == "Text"  # todo add to evaluation
        # assert anno["label_text"] ==  not supported in SDK but in REST API
        assert anno["label_threshold"] == 0.1
        # assert anno["normalized"] == '1'  # todo normalized is not really normalized data, e.g. for dates
        # assert anno["offset_string"] ==  # todo not supported by SDK but REST API
        # assert anno["offset_string_original"] ==  # todo not supported by SDK but REST API
        assert not anno["revised"]
        # self.assertIsNone(anno["revised_by"])   # todo not supported by SDK but REST API
        assert anno["annotation_set_id"] == 78730  # v2 REST API calls it still section
        assert anno["label_set_id"] == 63  # v2 REST API call it still section_label_id
        # assert anno["annotation_set_text"] == "Lohnabrechnung"  # v2 REST API call it still section_label_text
        assert anno["start_offset"] == 365
        # self.assertIsNone(anno["translated_string"])  # todo: how to translate null in a meaningful way

    def test_document_annotations_filter(self):
        """Test annotations filter."""
        doc = self.prj.labels[0].documents[5]  # one doc before doc without annotations
        self.assertEqual(len(doc.annotations()), 13)
        assert len(doc.annotations(label=self.prj.labels[0])) == 1
        assert len(doc.annotations(use_correct=False)) == 13

    def test_document_offset(self):
        """Test document offsets."""
        doc = self.prj.labels[0].documents[5]  # one doc before doc without annotations
        self.assertEqual(44842, doc.id)
        assert doc.text[395:396] == '4'
        annotations = doc.annotations()

        self.assertEqual(13, len(annotations))
        assert annotations[1].offset_string == '4'

    def test_document_add_new_annotation(self):
        """Test adding a new annotation."""
        doc = self.prj.labels[0].documents[5]  # the latest document
        # we create a revised annotations, as only revised annotation can be deleted
        # if we would delete an unrevised annotation, we would provide feedback and thereby keep the
        # annotation as "wrong" but "revised"
        assert len(doc.annotations(use_correct=False)) == 13
        label = self.prj.labels[0]
        new_anno = Annotation(
            start_offset=225,
            end_offset=237,
            label=label.id,
            label_set_id=label.label_sets[0].id,  # hand selected document section label
            revised=True,
            is_correct=True,
            accuracy=0.98765431,
            document=doc,
        )
        # make sure document annotations are updated too
        assert len(doc.annotations(use_correct=False)) == 14
        self.assertEqual(self.document_count + 1, len(self.prj.labels[0].correct_annotations))
        assert new_anno.id is None
        new_anno.save()
        assert new_anno.id
        new_anno.delete()
        assert new_anno.id is None
        assert len(doc.annotations(use_correct=False)) == 13
        self.assertEqual(self.document_count, len(self.prj.labels[0].correct_annotations))

    @unittest.skip(reason="Issue https://gitlab.com/konfuzio/objectives/-/issues/8664.")
    def test_get_text_in_bio_scheme(self):
        """Test getting document in the BIO scheme."""
        doc = self.prj.documents[0]
        bio_annotations = doc.get_text_in_bio_scheme()
        assert len(bio_annotations) == 398
        # check for multiline support in bio shema
        assert bio_annotations[1][0] == '328927/10103/00104'
        assert bio_annotations[1][1] == 'B-Austellungsdatum'
        assert bio_annotations[8][0] == '22.05.2018'
        assert bio_annotations[8][1] == 'B-Austellungsdatum'

    @classmethod
    def tearDownClass(cls) -> None:
        """Test if the project remains the same as in the beginning."""
        assert len(cls.prj.documents) == cls.document_count
        assert len(cls.prj.labels[0].correct_annotations) == cls.correct_document_count
