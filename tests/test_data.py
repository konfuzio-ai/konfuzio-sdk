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

    document_count = 26
    test_document_count = 3
    correct_document_count = 26

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the test project."""
        cls.prj = Project()
        assert len(cls.prj.documents) == cls.document_count
        assert len(cls.prj.test_documents) == cls.test_document_count
        assert len(cls.prj.labels[0].correct_annotations) == cls.correct_document_count

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
        assert len(self.prj.labels[0].correct_annotations) == self.correct_document_count
        doc.update()
        assert len(self.prj.labels[0].correct_annotations) == 26
        assert len(doc.text) == 4793
        assert len(glob.glob(os.path.join(doc.root, '*.*'))) == 4

        # existing annotation
        assert len(doc.annotations(use_correct=False)) == 13
        assert doc.annotations()[0].offset_string == '22.05.2018'  # start_offset=465, start_offset=466
        assert len(doc.annotations()) == 13
        assert doc.annotations()[0].is_online
        assert not doc.annotations()[0].save()  # Save returns False because Annotation is already online.

    def test_add_document_twice(self):
        """Test adding same document twice."""
        old_doc = self.prj.documents[1]
        assert len(self.prj.documents) == self.document_count
        self.prj.add_document(old_doc)
        assert len(self.prj.documents) == self.document_count

    def test_sections(self):
        """Test section labels in the test project."""
        assert self.prj.templates.__len__() == 2
        assert self.prj.templates[0].labels.__len__() == 8
        assert self.prj.templates[1].labels.__len__() == 3

    def test_correct_annotations(self):
        """Test correct annotations of a certain label in a specific document."""
        for doc in self.prj.documents:
            if doc.id == 26608:
                assert len(doc.annotations(self.prj.get_label_by_id(579))) == 1

    def test_annotation_start_offset_zero_filter(self):
        """Test annotations with start offset equal to zero."""
        doc = self.prj.labels[0].documents[5]  # one doc before doc without annotations
        assert len(doc.annotations()) == 13
        doc.annotations()[0].start_offset = 0
        assert len(doc.annotations()) == 13

    def test_document_annotations_filter(self):
        """Test annotations filter."""
        doc = self.prj.labels[0].documents[5]  # one doc before doc without annotations
        assert len(doc.annotations()) == 13
        assert len(doc.annotations(label=self.prj.labels[0])) == 1
        assert len(doc.annotations(use_correct=False)) == 13

        assert len(doc.annotations(start_offset=395, end_offset=396)) == 1
        assert len(doc.annotations(start_offset=394, end_offset=396)) == 1
        assert len(doc.annotations(start_offset=396, end_offset=397)) == 1
        assert len(doc.annotations(start_offset=396, end_offset=4793)) == 12
        assert len(doc.annotations(start_offset=0, end_offset=4793)) == 13

    def test_document_offset(self):
        """Test document offsets."""
        doc = self.prj.labels[0].documents[5]  # one doc before doc without annotations
        assert doc.text[395:396] == '4'
        logger.info(doc.annotations())
        assert len(doc.offset(395, 396)) == 1
        assert len(doc.offset(394, 396)) == 2
        annotations = doc.offset(394, 397)

        assert len(annotations) == 3
        assert annotations[0].offset_string == ' '
        assert annotations[1].offset_string == '4'
        assert annotations[2].offset_string == ' '

        assert len(doc.offset(465, 3659)) == 21
        assert len(doc.offset(0, 3660)) == 26

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
            template_id=label.templates[0].id,  # hand selected document section label
            revised=True,
            is_correct=True,
            accuracy=0.98765431,
            document=doc,
        )
        # make sure document annotations are updated too
        assert len(doc.annotations(use_correct=False)) == 14
        assert len(self.prj.labels[0].correct_annotations) == 27
        assert new_anno.id is None
        new_anno.save()
        assert new_anno.id
        new_anno.delete()
        assert new_anno.id is None
        assert len(doc.annotations(use_correct=False)) == 13
        assert len(self.prj.labels[0].correct_annotations) == 26

    def test_get_text_in_bio_scheme(self):
        """Test getting document in the BIO scheme."""
        doc = self.prj.documents[0]
        bio_annotations = doc.get_text_in_bio_scheme()
        assert len(bio_annotations) == 391
        assert bio_annotations[8][0] == '22.05.2018'
        assert bio_annotations[8][1] == 'B-Austellungsdatum'

    @classmethod
    def tearDownClass(cls) -> None:
        """Test if the project remains the same as in the beginning."""
        assert len(cls.prj.documents) == cls.document_count
        assert len(cls.prj.labels[0].correct_annotations) == cls.correct_document_count
