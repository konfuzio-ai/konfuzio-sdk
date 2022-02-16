"""Validate data functions."""
import glob
import logging
import os
import unittest

import pytest
from konfuzio_sdk.data import Project, Annotation, Document, Label, AnnotationSet, LabelSet, Data
from konfuzio_sdk.utils import is_file, get_default_label_set_documents, separate_labels

logger = logging.getLogger(__name__)

TEST_PROJECT_ID = 46
TEST_DOCUMENT_ID = 44823


@pytest.mark.serial
class TestKonfuzioDataSetup(unittest.TestCase):
    """Test handle data."""

    document_count = 26
    test_document_count = 3
    annotations_correct = 25

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the test project."""
        cls.prj = Project(id_=46)

    def test_number_training_documents(self):
        """Test the number of documents in data set status test."""
        assert len(self.prj.documents) == self.document_count

    def test_number_test_documents(self):
        """Test the number of documents in data set status test."""
        assert len(self.prj.test_documents) == self.test_document_count

    def test_annotation_of_label(self):
        """Test the number of annotations across all documents in training."""
        label = self.prj.get_label_by_id(867)
        assert len(label.correct_annotations) == self.annotations_correct

    def test_number_of_label_sets(self):
        """Test label sets numbers."""
        assert self.prj.label_sets.__len__() == 5

    def test_has_multiple_annotation_sets(self):
        """Test label sets in the test project."""
        assert self.prj.get_label_set_by_name('Brutto-Bezug').has_multiple_annotation_sets

    def test_has_not_multiple_annotation_sets(self):
        """Test label sets in the test project."""
        assert not self.prj.get_label_set_by_name('Lohnabrechnung').has_multiple_annotation_sets

    def test_default_label_set(self):
        """Test the main label set incl. it's labels."""
        default_label_set = self.prj.get_label_set_by_name('Lohnabrechnung')
        assert default_label_set.labels.__len__() == 10

    def test_to_filter_annotations_by_label(self):
        """Test to get correct annotations of a label."""
        label = self.prj.get_label_by_id(858)
        self.assertEqual(len(label.correct_annotations), self.annotations_correct + 1)

    def test_category(self):
        """Test if category of main label set is initialized correctly."""
        assert len(self.prj.categories) == 1
        assert self.prj.categories[0].id_ == 63
        assert self.prj.label_sets[0].categories[0].id_ == 63

    def test_label_set_multiple(self):
        """Test label set config that is set to multiple."""
        label_set = self.prj.get_label_set_by_name('Brutto-Bezug')
        assert label_set.categories.__len__() == 1

    def test_number_of_labels_of_label_set(self):
        """Test the number of labels of the default label set."""
        label_set = self.prj.get_label_set_by_name('Lohnabrechnung')
        assert label_set.categories[0] is None  # defines a category
        assert label_set.labels.__len__() == 10

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

    def test_category_of_document(self):
        """Test to download a file which includes a whitespace in the name."""
        category = Project(id_=46).get_document_by_id(44860).category
        self.assertEqual(category.name, 'Lohnabrechnung')

    def test_category_of_document_without_category(self):
        """Test the category of a document without category."""
        category = Project(id_=46).get_document_by_id(44864).category
        self.assertIsNone(category)

    def test_get_file_with_white_colon_name(self):
        """Test to download a file which includes a whitespace in the name."""
        doc = Project(id_=46).get_document_by_id(44860)
        doc.get_file()

    def test_labels(self):
        """Test get labels in the project."""
        assert len(self.prj.labels) == 18

    def test_project(self):
        """Test basic properties of the project object."""
        assert is_file(self.prj.meta_file_path)
        assert self.prj.documents[1].id_ > self.prj.documents[0].id_
        assert len(self.prj.documents)
        # check if we can initialize a new project object, which will use the same data
        assert len(self.prj.documents) == self.document_count
        new_project = Project(id_=TEST_PROJECT_ID)
        assert len(new_project.documents) == self.document_count
        assert new_project.meta_file_path == self.prj.meta_file_path

    def test_update_prj(self):
        """Test number of documents after updating a project."""
        assert len(self.prj.documents) == self.document_count
        self.prj.get(update=True)
        assert len(self.prj.documents) == self.document_count
        is_file(self.prj.meta_file_path)

    def test_document(self):
        """Test properties of a specific documents in the test project."""
        doc = self.prj.get_document_by_id(44842)
        assert doc.category.name == 'Lohnabrechnung'
        assert len(self.prj.labels[0].correct_annotations) == self.annotations_correct
        doc.update()
        self.assertEqual(len(self.prj.labels[0].correct_annotations), self.annotations_correct)
        assert len(doc.text) == 4793
        assert len(glob.glob(os.path.join(doc.document_folder, '*.*'))) == 4

    def test_annotations_in_document(self):
        """Test number and value of annotations."""
        doc = self.prj.get_document_by_id(44842)
        assert len(doc.annotations(use_correct=False)) == 24
        assert doc.annotations()[0].offset_string == ['22.05.2018']  # start_offset=465, start_offset=466
        assert len(doc.annotations()) == 24
        assert doc.annotations()[0].is_online
        assert not doc.annotations()[0].save()  # Save returns False because Annotation is already online.

    def test_annotation_sets_in_document(self):
        """Test number of annotation sets in a specific document in the test project."""
        doc = self.prj.get_document_by_id(44842)
        assert len(doc.annotation_sets) == 5

    def test_document_with_multiline_annotation(self):
        """Test properties of a specific documents in the test project."""
        doc = self.prj.get_document_by_id(TEST_DOCUMENT_ID)
        assert doc.category.name == 'Lohnabrechnung'
        label = self.prj.get_label_by_id(867)
        self.assertEqual(len(label.correct_annotations), self.annotations_correct)
        doc.update()
        self.assertEqual(len(label.correct_annotations), self.annotations_correct)
        self.assertEqual(len(doc.text), 4537)
        self.assertEqual(len(glob.glob(os.path.join(doc.document_folder, '*.*'))), 4)

        # existing annotation
        # https://app.konfuzio.com/admin/server/sequenceannotation/?document_id=44823&project=46
        self.assertEqual(len(doc.annotations(use_correct=False)), 22)
        # a multiline annotation in the top right corner, see https://app.konfuzio.com/a/4419937
        # todo improve multiline support
        self.assertEqual(66, doc.annotations()[0]._spans[0].start_offset)
        self.assertEqual(78, doc.annotations()[0]._spans[0].end_offset)
        self.assertEqual(159, doc.annotations()[0]._spans[1].start_offset)
        self.assertEqual(169, doc.annotations()[0]._spans[1].end_offset)
        self.assertEqual(len(doc.annotations()), 19)
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
        doc = self.prj.get_document_by_id(44842)
        label = self.prj.get_label_by_id(867)
        assert len(doc.annotations(label=label)) == 1

    def test_annotation_start_offset_zero_filter(self):
        """Test annotations with start offset equal to zero."""
        doc = self.prj.get_document_by_id(44842)
        assert len(doc.annotations()) == 24
        assert doc.annotations()[0].start_offset == 188
        assert len(doc.annotations()) == 24

    def test_multiline_annotation(self):
        """Test to convert a multiline span Annotation to a dict."""
        doc = self.prj.get_document_by_id(TEST_DOCUMENT_ID)
        assert len(doc.annotations()[0].eval_dict) == 2

    def test_annotation_to_dict(self):
        """Test to convert a Annotation to a dict."""
        for annotation in self.prj.documents[0].annotations():
            if annotation.id_ == 4420022:
                anno = annotation.eval_dict[0]

        assert anno["confidence"] == 1.0
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
        doc = self.prj.get_document_by_id(TEST_DOCUMENT_ID)
        self.assertEqual(len(doc.annotations()), 19)
        assert len(doc.annotations(label=self.prj.get_label_by_id(858))) == 1
        assert len(doc.annotations(use_correct=False)) == 22

    def test_document_offset(self):
        """Test document offsets."""
        doc = self.prj.get_document_by_id(44842)
        assert doc.text[395:396] == '4'
        annotations = doc.annotations()
        self.assertEqual(24, len(annotations))
        assert annotations[2].offset_string == ['4']

    @unittest.skip(reason='Waiting for API to support to add to default annotation set')
    def test_document_add_new_annotation(self):
        """Test adding a new annotation."""
        doc = self.prj.labels[0].documents[5]  # the latest document
        # we create a revised annotations, as only revised annotation can be deleted
        # if we would delete an unrevised annotation, we would provide feedback and thereby keep the
        # annotation as "wrong" but "revised"
        assert len(doc.annotations(use_correct=False)) == 23
        label = self.prj.labels[0]
        new_anno = Annotation(
            start_offset=225,
            end_offset=237,
            label=label.id_,
            label_set_id=None,  # hand selected document section label
            revised=True,
            is_correct=True,
            accuracy=0.98765431,
            document=doc,
        )
        # make sure document annotations are updated too
        assert len(doc.annotations(use_correct=False)) == 24
        self.assertEqual(self.document_count + 1, len(self.prj.labels[0].correct_annotations))
        assert new_anno.id_ is None
        new_anno.save()
        assert new_anno.id_
        new_anno.delete()
        assert new_anno.id_ is None
        assert len(doc.annotations(use_correct=False)) == 13
        self.assertEqual(self.document_count, len(self.prj.labels[0].correct_annotations))

    @unittest.skip(reason="Skip: Changes in Trainer Annotation needed to require a Label for every Annotation init.")
    def test_document_add_new_annotation_without_label(self):
        """Test adding a new annotation."""
        with self.assertRaises(AttributeError) as _:
            _ = Annotation(
                start_offset=225,
                end_offset=237,
                label=None,
                label_set_id=0,  # hand selected document section label
                revised=True,
                is_correct=True,
                accuracy=0.98765431,
                document=Document(),
            )
        # TODO: expand assert to check for specific error message

    @unittest.skip(reason="Skip: Changes in Trainer Annotation needed to require a Document for every Annotation init.")
    def test_init_annotation_without_document(self):
        """Test adding a new annotation."""
        with self.assertRaises(AttributeError) as _:
            _ = Annotation(
                start_offset=225,
                end_offset=237,
                label=None,
                label_set_id=0,
                revised=True,
                is_correct=True,
                accuracy=0.98765431,
                document=None,
            )

        # TODO: expand assert to check for specific error message

    @pytest.mark.xfail(reason='We cannot define the annotation set automatically.')
    def test_init_annotation_with_default_annotation_set(self):
        """Test adding a new annotation without providing the annotation set."""
        prj = Project(id_=TEST_PROJECT_ID)
        doc = Document(project=prj)
        annotation = Annotation(
            start_offset=225,
            end_offset=237,
            label=prj.labels[0],
            label_set_id=0,
            revised=True,
            is_correct=True,
            accuracy=0.98765431,
            document=doc,
            annotation_set=None,
        )

        # an Annotation set needs to be created or retrieved after the annotation is saved
        assert annotation.annotation_set.id_ == 78730

    @unittest.skip(reason="Issue https://gitlab.com/konfuzio/objectives/-/issues/8664.")
    def test_get_text_in_bio_scheme(self):
        """Test getting document in the BIO scheme."""
        doc = self.prj.documents[0]
        bio_annotations = doc.get_text_in_bio_scheme()
        assert len(bio_annotations) == 398
        # check for multiline support in bio schema
        assert bio_annotations[1][0] == '328927/10103/00104'
        assert bio_annotations[1][1] == 'B-Austellungsdatum'
        assert bio_annotations[8][0] == '22.05.2018'
        assert bio_annotations[8][1] == 'B-Austellungsdatum'

    def test_create_empty_annotation(self):
        """Create an empty Annotation and get the start offset."""
        prj = Project(id_=TEST_PROJECT_ID)
        label = Label(project=prj)
        doc = Document(text='', project=prj)
        label_set = LabelSet(project=prj)
        annotation_set = AnnotationSet(document=doc, label_set=label_set)
        _ = Annotation(label=label, annotation_set=annotation_set, label_set=label_set, document=doc)

    def test_get_annotations_for_offset_of_first_and_last_name(self):
        """Get Annotations for all offsets in the document."""
        doc = self.prj.get_document_by_id(TEST_DOCUMENT_ID)
        filtered_annotations = doc.annotations(start_offset=1500, end_offset=1530)
        self.assertEqual(len(filtered_annotations), 3)  # 3 is correct even 4 Spans!
        text = '198,34\n  Erna-Muster Eiermann                         KiSt      15,83   Solz        10,89\n  '
        self.assertEqual(doc.text[1498:1590], text)

    def test_create_list_of_regex_for_label_without_annotations(self):
        """Check regex build for empty Labels."""
        try:
            label = next(x for x in self.prj.labels if len(x.annotations) == 0)
            automated_regex_for_label = label.regex()
            # There is no regex available.
            assert len(automated_regex_for_label) == 0
            is_file(label.regex_file_path)
        except StopIteration:
            pass

    @pytest.mark.xfail(reason='Refactoring via Pair Programming needed to clean up code and clarify test cases.')
    def test_get_default_label_set_documents(self):
        """Check get documents and labels associated to a default annotation_set label."""
        default_label_sets = [x for x in self.prj.label_sets if x.is_default]

        default_label_set_documents_dict, default_labels_dict = get_default_label_set_documents(
            documents=self.prj.documents,
            selected_default_label_sets=default_label_sets,
            project_label_sets=self.prj.label_sets,
            merge_multi_default=False,
        )

        assert list(default_label_set_documents_dict.keys()) == [63]
        self.assertEqual(len(default_label_set_documents_dict[63]), self.correct_document_count)
        assert default_label_set_documents_dict.keys() == default_labels_dict.keys()
        assert len(default_labels_dict[63]) == 17

    @pytest.mark.xfail(reason='Refactoring via Pair Programming needed to clean up code and clarify test cases.')
    def test_separate_labels(self):
        """Test LabelSets and Labels can be combined correctly."""
        self.assertEqual(len(self.prj.label_sets), 5)
        assert len(self.prj.labels) == 18

        labels_with_annotations = {}

        for label_set in [x for x in self.prj.label_sets if not x.is_default]:
            labels_names = [label.name for label in label_set.labels if len(label.annotations) > 0]
            labels_with_annotations[label_set.name] = labels_names

        n_labels_with_annotations = 0

        for _, labels_names in labels_with_annotations.items():
            n_labels_with_annotations += len(labels_names)

        prj = separate_labels(project=self.prj)
        assert len(prj.label_sets) == 5

        # separate labels changes the labels of the annotations that belong to non default label_sets in the project
        separated_labels_with_annotations = [
            label.name
            for label in prj.labels
            if len([s for s in label.label_sets if not s.is_default]) > 0 and len(label.annotations) > 0
        ]

        assert len(separated_labels_with_annotations) == n_labels_with_annotations
        assert len(prj.documents) == self.document_count  # only 29 documents have been added to the dataset

        # verify if new labels are indeed added
        for label_set in [x for x in prj.label_sets if not x.is_default]:
            new_labels = [label.name for label in label_set.labels if len(label.annotations) > 0 and "__" in label.name]

            assert len(new_labels) > 0
            original_labels = labels_with_annotations[label_set.name]

            for label in new_labels:
                original_label = label.split("__")[1]
                assert original_label in original_labels

        self.prj = Project(id_=46)
        assert len(self.prj.label_sets) == 5
        assert len(self.prj.labels) == 18
        assert len(self.prj.labels[0].correct_annotations) == self.correct_document_count

    def test_annotation_keywordtoken_with_linebreak(self):
        """Check if number replacement of regexed is correctly preferred, see _N_."""
        label = self.prj.get_label_by_id(858)
        # 26 Annotations can be represented by 3 Regex
        assert label.tokens() == [
            '(?P<Auszahlungsbetrag_N_4420351_3777>\\d\\.\\d\\d\\d\\,\\d\\d)',
            '(?P<Auszahlungsbetrag_N_671698_3433>\\d\\d\\d\\,\\d\\d)',
            '(?P<Auszahlungsbetrag_N_673143_4074>\\d\\d\\,[ ]+\\d\\d[-])',
        ]

    @unittest.skip(reason='Waiting for Text-Annotation Documentation.')
    def test_to_change_an_annotation_online(self):
        """Test to update an Annotation from revised to not revised and back to revised."""
        doc = self.prj.get_document_by_id(44864)
        annotations = doc.annotations(start_offset=10, end_offset=200)
        first_annotation = annotations[0]
        first_annotation.revised = False
        first_annotation.save()

    @classmethod
    def tearDownClass(cls) -> None:
        """Test if the project remains the same as in the beginning."""
        assert len(cls.prj.documents) == cls.document_count
        assert len(cls.prj.test_documents) == cls.test_document_count
        assert len(cls.prj.labels[0].annotations) == cls.annotations_correct


@pytest.mark.local
class TestData(unittest.TestCase):
    """Test functions that don't require data."""

    def test_compare_none_and_id(self):
        """Test to compare an instance to None."""
        a = Data()
        a.id_ = 5
        self.assertNotEqual(a, None)

    def test_compare_nones(self):
        """Test to compare an instance with None ID to None."""
        a = Data()
        self.assertNotEqual(a, None)

    def test_compare_id_with_instance_without(self):
        """Test to compare an instance with ID to an instance with None ID."""
        a = Data()
        a.id_ = 5
        b = Data()
        self.assertNotEqual(a, b)
