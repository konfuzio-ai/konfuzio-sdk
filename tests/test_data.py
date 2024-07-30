"""Validate data functions."""
import json
import logging
import os
import time
import unittest
from copy import copy, deepcopy

import PIL
import pytest
from PIL.PngImagePlugin import PngImageFile
from requests import ConnectionError, HTTPError

from konfuzio_sdk.api import delete_project, restore_snapshot
from konfuzio_sdk.data import (
    Annotation,
    AnnotationSet,
    Bbox,
    BboxValidationTypes,
    Category,
    CategoryAnnotation,
    Data,
    Document,
    Label,
    LabelSet,
    Page,
    Project,
    Span,
)
from konfuzio_sdk.samples import LocalTextProject
from konfuzio_sdk.settings_importer import is_dependency_installed
from konfuzio_sdk.tokenizer.base import ListTokenizer
from konfuzio_sdk.tokenizer.regex import ConnectedTextTokenizer, RegexTokenizer, WhitespaceTokenizer
from konfuzio_sdk.utils import get_spans_from_bbox, is_file
from tests.variables import (
    OFFLINE_PROJECT,
    TEST_DOCUMENT_ID,
    TEST_FALLBACK_DOCUMENT_ID,
    TEST_FALLBACK_PROJECT_ID,
    TEST_PAYSLIPS_CATEGORY_ID,
    TEST_PROJECT_ID,
    TEST_RECEIPTS_CATEGORY_ID,
)

logger = logging.getLogger(__name__)
RESTORED_PROJECT_ID = restore_snapshot(snapshot_id=65)


class TestOnlineProject(unittest.TestCase):
    """Use this class only to test data.py operations that need an online Project."""

    annotations_correct = 24

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the test Project."""
        cls.project = Project(id_=RESTORED_PROJECT_ID, update=True)
        original_document_text = Project(id_=46).get_document_by_id(TEST_DOCUMENT_ID).text
        cls.test_document_id = [
            document for document in cls.project.documents if document.text == original_document_text
        ][0].id_
        cls.test_category_id = cls.project.categories[0].id_

    def test_document(self):
        """Test properties of a specific Documents in the test Project."""
        doc = self.project.get_document_by_id(self.test_document_id)
        assert doc.ocr_ready is True
        assert doc.category.name == 'Lohnabrechnung'
        label = self.project.labels[0]
        annotations = label.annotations(categories=[self.project.get_category_by_id(self.test_category_id)])
        assert len(annotations) == self.annotations_correct
        doc.update()
        annotations = label.annotations(categories=[self.project.get_category_by_id(self.test_category_id)])
        self.assertEqual(len(annotations), self.annotations_correct)
        assert len(doc.text) == 4537
        assert is_file(doc.txt_file_path)
        # assert is_file(doc.bbox_file_path) bbox is not loaded at this point.
        assert is_file(doc.annotation_file_path)
        assert is_file(doc.annotation_set_file_path)

    def test_document_no_label_annotations_after_update(self):
        """Test that Annotations in the no_label_annotation_set of the Document are removed after update."""
        document = self.project.get_document_by_id(self.test_document_id)
        span = Span(start_offset=0, end_offset=1)
        _ = Annotation(
            document=document,
            # annotation_set=document.no_label_annotation_set,
            label=self.project.no_label,
            label_set=self.project.no_label_set,
            spans=[span],
        )
        with pytest.raises(ValueError, match='save Annotations with Label NO_LABEL'):
            _.save()
        assert len(document.annotations(use_correct=False, label=self.project.no_label)) == 1
        document.update()
        assert len(document.annotations(use_correct=False, label=self.project.no_label)) == 0

    def test_document_with_multiline_annotation(self):
        """Test properties of a specific Documents in the test Project."""
        doc = self.project.get_document_by_id(self.test_document_id)
        label = self.project.get_label_by_name('Austellungsdatum')
        annotations = label.annotations(categories=[self.project.get_category_by_id(self.test_category_id)])
        self.assertEqual(len(annotations), self.annotations_correct)
        doc.update()
        annotations = label.annotations(categories=[self.project.get_category_by_id(self.test_category_id)])
        self.assertEqual(len(annotations), self.annotations_correct)
        self.assertEqual(len(doc.text), 4537)
        # self.assertEqual(len(glob.glob(os.path.join(doc.document_folder, '*.*'))), 4)

        # existing annotation
        # https://app.konfuzio.com/admin/server/sequenceannotation/?document_id=44823&project=46
        # we are no longer filtering out the rejected Annotations so it's 21
        self.assertEqual(21, len(doc.annotations(use_correct=False)))
        # a multiline Annotation in the top right corner, see https://app.konfuzio.com/a/4419937
        self.assertEqual(66, doc.annotations()[0].spans[0].start_offset)
        self.assertEqual(78, doc.annotations()[0].spans[0].end_offset)
        self.assertEqual(159, doc.annotations()[0].spans[1].start_offset)
        self.assertEqual(169, doc.annotations()[0].spans[1].end_offset)
        self.assertEqual(len(doc.annotations()), 21)
        # helm: 21.06.2022 changed from 21 to 19 as someone added (?) two annotations?
        # todo check this number, the offline project was still working fine for all evaluation tests
        # 15.01.2024 reverted to 21 because of changes in testing api.py - creation of multiline annotations
        self.assertTrue(doc.annotations()[0].is_online)
        with self.assertRaises(ValueError) as context:
            doc.annotations()[0].save()
            assert 'cannot update Annotations once saved online' in context.exception

    def test_get_pages_files(self):
        """Test to download page files."""
        doc = self.project.get_document_by_id(self.test_document_id)
        assert len(doc.pages()) == 1
        assert doc.pages()[0].category == doc.category

    def test_load_image_in_memory(self):
        """Test to download page files."""
        doc = self.project.get_document_by_id(self.test_document_id)
        for page in doc.pages():
            image = page.get_image(update=True)
            assert type(image) is PngImageFile

    def test_load_externally_provided_image(self):
        """Test loading a Page image provided from an external source rather than loaded from the Project's folder."""
        # Why this testcase? Because if you need to retrieve a Page image from a blob storage, there is no image path.
        import numpy
        from PIL import Image

        external_image = Image.fromarray(numpy.zeros((5, 5)))
        doc = self.project.get_document_by_id(self.test_document_id)
        page = doc.pages()[0]
        page.image = external_image  # provide an image for the Page ad-hoc
        image = page.get_image()
        assert image is external_image

    def test_load_image_from_bytes(self):
        """Test loading a Page image provided as bytes rather than loaded from the Project's folder."""
        doc = self.project.get_document_by_id(self.test_document_id)
        page = doc.pages()[0]
        original_image = page.get_image(update=True)  # Pillow loads from page.image_path file
        assert type(original_image) is PngImageFile
        image_in_bytes_format = open(page.image_path, 'rb').read()
        # reset image data
        page.image = None
        page.image_bytes = image_in_bytes_format
        # page.get_image() will bypass Pillow loading page.image_path file and instead use the provided bytes
        image = page.get_image()
        # check correspondence between the two loading methods
        assert type(image) is PngImageFile

    def test_get_annotation_by_id(self):
        """Test to find an online Annotation by its ID."""
        doc = self.project.get_document_by_id(self.test_document_id)
        searched_id = doc.annotations()[5].id_
        annotation = doc.get_annotation_by_id(searched_id)
        assert annotation.start_offset == 1507
        assert annotation.end_offset == 1518
        assert annotation.offset_string == ['Erna-Muster']

    def test_get_nonexistent_annotation_by_id(self):
        """Test to find an online Annotation that does not exist by its ID, should raise an IndexError."""
        doc = self.project.get_document_by_id(self.test_document_id)
        with pytest.raises(IndexError, match='is not a part of'):
            _ = doc.get_annotation_by_id(999999)

    def test_create_annotation_offline(self):
        """Test to add an Annotation to the Document offline, and that it does not persist after updating the doc."""
        doc = self.project.get_document_by_id(self.test_document_id)
        doc.update()
        assert Span(start_offset=1590, end_offset=1602) not in doc.spans()
        label = self.project.get_label_by_name('Lohnart')
        label_set = label.label_sets[0]
        annotation_set = AnnotationSet(label_set=label_set, document=doc)
        annotation = Annotation(
            document=doc,
            spans=[Span(start_offset=1590, end_offset=1602)],
            label=label,
            annotation_set=annotation_set,
            accuracy=1.0,
            is_correct=True,
        )
        assert annotation in doc.annotations()
        doc.update()  # redownload Document information to check that the Annotation was not added online
        assert annotation not in doc.annotations()

    def test_create_annotation_then_delete_annotation(self):
        """Test to add an Annotation to the document online, then to delete it offline and online as well."""
        # We do 3 tests in 1 here since unit tests should be independent,
        # we don't want to refer to an Annotation created by a previous test

        # Test1: add an Annotation to the document online
        doc = self.project.get_document_by_id(self.test_document_id)
        assert Span(start_offset=1590, end_offset=1602) not in doc.spans()
        label = self.project.get_label_by_name('Vorname')

        default_annotation_set = doc.default_annotation_set
        assert default_annotation_set.label_set.is_default

        annotation = Annotation(
            document=doc,
            annotation_set=default_annotation_set,
            spans=[Span(start_offset=1590, end_offset=1602)],
            label=label,
            accuracy=1.0,
            is_correct=True,
        )
        annotation.save(annotation_set_id=annotation.annotation_set.id_)
        assert annotation in doc.annotations()
        doc.update()  # redownload Document information to check that the Annotation was saved online
        assert annotation in doc.annotations()

        # Test2: delete the Annotation from the Document offline
        annotation.delete(delete_online=False)
        assert annotation not in doc.get_annotations()
        doc.update()  # redownload Document information to check that the Annotation was not deleted online
        assert annotation in doc.get_annotations()

        # Test3: delete the Annotation from the Document online.
        annotation.delete()  # doc.update() performed internally when delete_online=True, which is default
        assert annotation not in doc.get_annotations()

    def test_create_bbox_annotation(self):
        """Test creating a Bbox-based Annotation."""
        doc = self.project.get_document_by_id(self.test_document_id)
        doc.status = 2
        doc.get_bbox()
        label = self.project.get_label_by_name('Bezeichnung')
        label_set = self.project.get_label_set_by_name('Brutto-Bezug')
        bbox = {'page_index': 0, 'x0': 198, 'x1': 300, 'y0': 508, 'y1': 517}
        annotation_set = AnnotationSet(document=doc, label_set=label_set)
        annotation = Annotation(
            document=doc,
            annotation_set=annotation_set,
            label=label,
            label_set_id=label_set.id_,
            accuracy=1.0,
            is_correct=True,
            bboxes=[bbox],
        )
        annotation.save(label_set_id=label_set.id_)
        assert annotation in doc.annotations()
        doc.update()
        assert annotation in doc.annotations()
        assert round(annotation.bbox().x0) == 199
        assert round(annotation.bbox().x1) == 287
        assert round(annotation.bbox().y0) == 509
        assert round(annotation.bbox().y1) == 517
        annotation.delete(delete_online=True)

    def test_create_empty_bbox_annotation(self):
        """Test creating an empty Annotation using empty Bbox is impossible."""
        doc = self.project.get_document_by_id(self.test_document_id)
        label = self.project.get_label_by_name('Bezeichnung')
        label_set = self.project.get_label_set_by_name('Brutto-Bezug')
        bbox = {'page_index': 0, 'x0': 1, 'x1': 4, 'y0': 1, 'y1': 4}
        annotation_set = AnnotationSet(document=doc, label_set=label_set)
        with pytest.raises(NotImplementedError):
            Annotation(
                document=doc,
                annotation_set=annotation_set,
                label=label,
                label_set_id=label_set.id_,
                accuracy=1.0,
                is_correct=True,
                bboxes=[bbox],
            )

    def test_get_sentence_spans_from_bbox(self):
        """Test to get sentence Spans in a bounding box."""
        document = Document.from_file(path='tests/test_data/textposition.pdf', project=self.project)
        document = WhitespaceTokenizer().tokenize(deepcopy(document))
        page = document.get_page_by_index(0)

        bbox = Bbox(x0=39, y0=728, x1=539, y1=742, page=page)

        assert bbox.document is document

        spans = get_spans_from_bbox(selection_bbox=bbox)

        sentences_spans = Span.get_sentence_from_spans(spans=spans)

        assert len(sentences_spans) == 2
        first_sentence = sentences_spans[0]
        assert len(first_sentence) == 1
        assert first_sentence[0].offset_string == 'Hi, my name is LeftTop.'

    def test_merge_documents(self):
        """Merge documents into a new document."""
        test_documents = self.project.test_documents
        all_pages = [page for doc in test_documents for page in doc.pages()]
        pages_text = '\f'.join([doc.text for doc in test_documents])
        new_doc = Document(project=self.project, id_=None, text=pages_text)
        i = 1
        running_start_offset = 0
        running_end_offset = 0
        for page in all_pages:
            running_end_offset += page.end_offset
            _ = Page(
                id_=i,
                original_size=(1500, 2400),
                document=new_doc,
                start_offset=running_start_offset,
                end_offset=running_end_offset,
                number=i,
            )
            i += 1
            running_start_offset += page.end_offset + 1
            running_end_offset += 1

        for i, page in enumerate(all_pages):
            assert page.text == new_doc.pages()[i].text

    def test_modify_document_metadata(self):
        """Test modification of meta-data of test document."""
        doc = self.project.get_document_by_id(self.test_document_id)

        doc.assignee = 42
        doc.dataset_status = 1

        with pytest.raises(HTTPError, match='Invalid user'):
            doc.save_meta_data()

        doc.assignee = None
        doc.dataset_status = 2
        doc.save_meta_data()

        self.project.init_or_update_document(from_online=True)

    def test_get_segmentation(self):
        """Test getting the detectron segmentation of a Document."""
        document = self.project.get_document_by_id(self.test_document_id)

        page = document.get_page_by_index(0)
        assert page._segmentation is None

        with pytest.raises(ConnectionError, match='Max retries exceeded with url: .* timed out'):
            segmentation = document.get_segmentation(timeout=0.1, num_retries=1)
        assert page._segmentation is None

        segmentation = document.get_segmentation()
        assert len(segmentation) == 1
        assert len(segmentation[0]) == 12
        assert len(page._segmentation) == 12

        virtual_document = deepcopy(document)

        # retrieving from original Document so no ConnectionError should be raised
        virtual_document_segmentation = virtual_document.get_segmentation(timeout=0.1, num_retries=1)

        assert len(virtual_document_segmentation) == 1
        assert len(virtual_document_segmentation[0]) == 12

        virtual_document_page = virtual_document.get_page_by_index(0)
        assert virtual_document_page._segmentation is None

    def test_create_invalid_file_type_document(self):
        """Test the creation of an invalid pdf Document. File should be checked and raise error before upload."""
        with pytest.raises(NotImplementedError, match='We do not support file'):
            Document.from_file('tests/test_data/invalid_pdf.pdf', self.project)

    def test_create_modify_and_delete_document(self):
        """Test the creation of an online Document from a file, modification, and then deletion of the Document."""
        # Test Document creation
        doc = Document.from_file('tests/test_data/pdf.pdf', self.project)
        doc_id = doc.id_
        doc.dataset_status = 1
        doc.save_meta_data()

        assert doc in self.project.preparation_documents
        assert doc.name == 'pdf.pdf'
        time.sleep(5)  # for ocr processing completion
        assert doc.get_file(ocr_version=True).split('/')[-1] == 'pdf_ocr.pdf'

        # Test Document modification
        assert doc.dataset_status == 1
        assert doc.assignee is None

        with pytest.raises(HTTPError, match='documents which are part of a dataset'):
            # Cannot delete Document with dataset_status != 0
            doc.delete(delete_online=True)

        doc.dataset_status = 0
        doc.save_meta_data()

        doc.update()

        assert doc.dataset_status == 0

        doc.delete(delete_online=False)

        with pytest.raises(IndexError, match='was not found in'):
            doc = self.project.get_document_by_id(doc_id)

        self.project.init_or_update_document(from_online=True)  # retrieve online version of the Document

        doc = self.project.get_document_by_id(doc_id)

        doc.delete(delete_online=True)
        self.project.init_or_update_document()

        with pytest.raises(IndexError, match='was not found in'):
            self.project.get_document_by_id(doc_id)

    def test_no_category(self):
        """Test that NO_CATEGORY is present in the Project."""
        assert self.project.no_category

    def test_no_category_document(self):
        """Test that a categoriless Document gets NO_CATEGORY assigned upon creation."""
        _ = Document(project=self.project)
        assert _.category == self.project.no_category
        _.delete()

    def test_none_category_document_property(self):
        """Test the return of a Document with Category == None."""
        _ = Document(project=self.project, category=None)
        assert _.category == self.project.no_category
        assert _._category == self.project.no_category
        _.delete()

    def test_set_none_category(self):
        """Test that setting Category to None gives the Document NO_CATEGORY."""
        test_document = self.project.get_document_by_id(self.test_document_id)
        test_document.set_category(None)
        assert test_document.category == self.project.no_category
        assert test_document._category == self.project.no_category

    def test_get_image_png(self):
        """Test that a PNG image file can be obtained via page.get_image() method."""
        document = Document.from_file(path='tests/test_data/png.png', project=self.project)
        image = document.pages()[0].get_image()
        assert isinstance(image, PIL.PngImagePlugin.PngImageFile)
        document.delete(delete_online=True)

    def test_get_image_pdf(self):
        """Test that a PDF image file can be obtained via page.get_image() method."""
        document = Document.from_file(path='tests/test_data/pdf.pdf', project=self.project)
        image = document.pages()[0].get_image()
        assert isinstance(image, PIL.PngImagePlugin.PngImageFile)
        document.delete(delete_online=True)

    def test_get_image_jpeg(self):
        """Test that a JPEG image file can be obtained via page.get_image() method."""
        document = Document.from_file(path='tests/test_data/jpg.jpg', project=self.project)
        image = document.pages()[0].get_image()
        assert isinstance(image, PIL.PngImagePlugin.PngImageFile)
        document.delete(delete_online=True)

    def test_add_category_with_repeated_name(self):
        """Test that it is impossible to create a Category with the same name as one of already existing in a Project."""
        with pytest.raises(ValueError, match='another name'):
            _ = Category(project=self.project, name='Lohnabrechnung')

    def test_prohibit_creating_multiple_annotation_sets(self):
        """Test that it is not possible to create multiple Annotation Sets for a Document."""
        document = self.project.get_document_by_id(self.test_document_id)
        # tokenize the document to get the spans
        document = WhitespaceTokenizer().tokenize(document)
        span = Span(document.spans()[0].start_offset, document.spans()[0].end_offset)
        # we make sure that the document has multiple Annotation Sets attribute set to True
        document.category.label_sets[0].has_multiple_annotation_sets = True
        # we create a second Annotation Set in the Document
        _ = AnnotationSet(document=document, label_set=document.category.label_sets[0])
        # we set the has_multiple_annotation_sets attribute to False back again
        # to force the ValueError to be raised
        document.category.label_sets[0].has_multiple_annotation_sets = False
        # initialize a new Annotation and make sure that the ValueError is raised
        with pytest.raises(ValueError, match='multiple Annotation Sets'):
            _ = Annotation(
                document=document,
                label_set=document.category.label_sets[0],
                label=document.annotations()[0].label,
                spans=[span],
            )

    def test_ignore_empty_annotation_sets(self):
        """Test that empty Annotation Sets are not loaded."""
        # use the fallback project to avoid issues where annotation_set.json5 files
        # of old projects do not store the labels and their annotations data
        project = Project(id_=TEST_FALLBACK_PROJECT_ID)
        document = project.get_document_by_id(TEST_FALLBACK_DOCUMENT_ID)
        _ = document.download_document_details()
        # modify the annotation_set.json5 file to intentionally have an empty Annotation Set
        with open(document.annotation_set_file_path, 'r') as f:
            annotation_set_data = json.load(f)
            annotation_set_data.append({'labels': [{'annotations': []}], 'id': 0})
        with open(document.annotation_set_file_path, 'w') as f:
            json.dump(annotation_set_data, f)
        # load the document again and make sure no error due to multiple Annotation Sets is raised
        _ = document.annotations()

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove any files that might have been left from the test pipeline."""
        project = Project(id_=TEST_PROJECT_ID, update=True)
        for document in project._documents:
            if document.name in os.listdir('tests/test_data/'):
                document.delete(delete_online=True)


class TestOfflineExampleData(unittest.TestCase):
    """Test data features without real data."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the test Project."""
        cls.project = Project(id_=None, project_folder=OFFLINE_PROJECT)
        cls.payslips_category = cls.project.get_category_by_id(TEST_PAYSLIPS_CATEGORY_ID)
        cls.receipts_category = cls.project.get_category_by_id(TEST_RECEIPTS_CATEGORY_ID)

    @classmethod
    def tearDownClass(cls) -> None:
        """Control the number of Documents created in the Test."""
        assert len(cls.payslips_category.documents()) == 25
        assert len(cls.receipts_category.documents()) == 23
        assert cls.project.get_document_by_id(44864).category.name == cls.project.no_category.name
        assert len(cls.project.documents) == 25 + 23 + 1

    def test_copy(self):
        """Test that copy is not allowed as it needs to be implemented for every SDK concept."""
        data = Data()
        with pytest.raises(NotImplementedError):
            copy(data)

    def test_receipts_category_annotations(self):
        """Test retrieving the Annotations of receipts Category."""
        for document in self.receipts_category.documents():
            document.get_annotations()

        for document in self.receipts_category.test_documents():
            document.get_annotations()

    def test_deepcopy(self):
        """Test that deepcopy is not allowed as it needs to be implemented for every SDK concept."""
        data = Data()
        with pytest.raises(NotImplementedError):
            deepcopy(data)

    def test_document_copy(self) -> None:
        """Test to create a new Document instance."""
        document = self.project.get_document_by_id(TEST_DOCUMENT_ID)
        document.get_page_by_index(0).image_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00'
        new_document = deepcopy(document)
        assert new_document != document
        assert new_document.get_page_by_index(0).width == 595.2
        assert new_document.get_page_by_index(0).image_bytes == b'\x89PNG\r\n\x1a\n\x00\x00\x00'
        assert new_document._annotations is None  # for now the implementation just copies the bbox and text

    def test_project_num_label(self):
        """Test that no_label exists in the Labels of the Project and has the expected name."""
        self.assertEqual(19, len(self.payslips_category.labels))
        self.assertEqual(30, len(self.receipts_category.labels))
        self.assertEqual(19 + 30 - 1, len(self.project.labels))  # subtract one to avoid double counting the NO_LABEL

    def test_no_label(self):
        """Test if NO_LABEL is available."""
        assert self.project.no_label.name == 'NO_LABEL'
        self.assertIn(self.project.no_label, self.project.labels)

    def test_annotation_bbox(self):
        """Create a Span and calculate it's bbox."""
        span = Span(start_offset=1764, end_offset=1769)  # the correct Annotation spans 1763 to 1769
        document = self.project.get_document_by_id(TEST_DOCUMENT_ID)
        _ = Annotation(
            id_=None,
            document=document,
            is_correct=True,
            annotation_set=document.no_label_annotation_set,
            label=self.project.no_label,
            spans=[span],
        )
        box = span.bbox()  # verify if we can calculate valid bounding boxes from a given Text offset.
        assert box.x1 == 113.28
        assert box.x0 == 84.28
        assert box.y0 == 532.592
        assert box.y1 == 540.592
        assert box.top == 301.088

    def test_get_category_name_for_fallback_prediction(self):
        """Test turn a category name to lowercase, remove parentheses along with their contents, and trim spaces."""
        assert self.payslips_category.fallback_name == 'lohnabrechnung'
        assert self.receipts_category.fallback_name == 'quittung'
        test_category = Category(project=self.project, id_=1, name='Te(s)t Category Name (content content)')
        assert test_category.fallback_name == 'tet category name'

    def test_document_with_no_category_has_category_annotations_with_zero_confidence(self):
        """Test that a Document with no Category has only Category Annotations with zero confidence."""
        document = deepcopy(self.project.get_document_by_id(89928))
        document.set_category(self.project.no_category)
        for page in document.pages():
            assert page.category_annotations == []
        assert len(document.category_annotations) == len(self.project.categories)
        assert document.category_annotations[0].category == self.payslips_category
        assert document.category_annotations[0].confidence == 0.0
        assert document.category_annotations[1].category == self.receipts_category
        assert document.category_annotations[1].confidence == 0.0
        assert document.maximum_confidence_category_annotation is None
        assert document.maximum_confidence_category == self.project.no_category
        # test that no annotations are attached to the Pages
        for page in document.pages():
            assert page.category_annotations == []
            assert page.category == self.project.no_category

    def test_category_annotations_no_predictions(self):
        """Test Category Annotations for a Document with a user defined Category but with no AI Category predictions."""
        document = deepcopy(self.project.get_document_by_id(89928))
        assert document.category == self.receipts_category
        for page in document.pages():
            assert page.category_annotations == []
        assert len(document.category_annotations) == len(self.project.categories)
        assert document.category_annotations[0].category == self.payslips_category
        assert document.category_annotations[0].confidence == 0.0
        assert document.category_annotations[1].category == self.receipts_category
        assert document.category_annotations[1].confidence == 1.0
        assert document.maximum_confidence_category_annotation.category == self.receipts_category
        assert document.maximum_confidence_category == self.receipts_category
        # test that no annotations are attached to the Pages while still having their Category defined
        for page in document.pages():
            assert page.category_annotations == []
            assert page.category == self.receipts_category

    def test_category_annotations_with_predictions(self):
        """Test Category Annotations for a Document with no user defined Category but with AI Category predictions."""
        document = deepcopy(self.project.get_document_by_id(89928))
        document.set_category(self.project.no_category)
        for page in document.pages():  # this Document has 2 Pages
            assert page.category_annotations == []
            # simulate the prediction of a Categorization AI by adding Category Annotations to the Pages
            CategoryAnnotation(category=self.payslips_category, confidence=0.2 * page.number, page=page)  # 0.2+0.4=0.6
            CategoryAnnotation(category=self.receipts_category, confidence=0.3 * page.number, page=page)  # 0.3+0.6=0.9
            assert page.maximum_confidence_category_annotation.category == self.receipts_category
            assert page.category == self.receipts_category
        assert len(document.category_annotations) == len(self.project.categories)
        assert document.category_annotations[0].category == self.payslips_category
        assert round(document.category_annotations[0].confidence, 2) == 0.3  # 0.6/2
        assert document.category_annotations[1].category == self.receipts_category
        assert round(document.category_annotations[1].confidence, 2) == 0.45  # 0.9/2
        assert document.maximum_confidence_category_annotation.category == self.receipts_category
        assert document.maximum_confidence_category == self.receipts_category

    def test_category_annotations_with_predictions_and_user_revised_category(self):
        """Test Category Annotations for a Document with both user defined Category and AI Category predictions."""
        document = deepcopy(self.project.get_document_by_id(89928))
        document.set_category(self.project.no_category)
        for page in document.pages():  # this Document has 2 Pages
            assert page.category_annotations == []
            # simulate the prediction of a Categorization AI by adding Category Annotations to the Pages
            CategoryAnnotation(category=self.payslips_category, confidence=0.2 * page.number, page=page)  # 0.2+0.4=0.6
            CategoryAnnotation(category=self.receipts_category, confidence=0.3 * page.number, page=page)  # 0.3+0.6=0.9
            assert page.maximum_confidence_category_annotation.category == self.receipts_category
            assert page.category == self.receipts_category
        # test a user defined Category that is different from the maximum confidence predicted Category will override
        document.set_category(self.payslips_category)
        assert len(document.category_annotations) == len(self.project.categories)
        assert document.category_annotations[0].category == self.payslips_category
        assert round(document.category_annotations[0].confidence, 2) == 0.3  # 0.6/2
        assert document.category_annotations[1].category == self.receipts_category
        # Test that a user revised Category overrides predictions
        assert round(document.category_annotations[1].confidence, 2) == 0.45  # 0.9/2
        assert document.maximum_confidence_category_annotation.category == self.payslips_category
        assert round(document.maximum_confidence_category_annotation.confidence, 2) == 0.3
        assert document.maximum_confidence_category == self.payslips_category

    def test_no_category(self):
        """Test that NO_CATEGORY is present in the offline Project."""
        assert self.project.no_category

    def test_find_outlier_annotations_by_regex(self):
        """Test finding the possibly incorrect Annotations of a Label."""
        project_regex = Project(id_=TEST_PROJECT_ID)
        label = project_regex.get_label_by_name('Bank inkl. IBAN')
        train_doc_ids = {44823, 44834, 44839, 44840, 44841}
        for doc in project_regex.documents:
            if doc.id_ not in train_doc_ids:
                doc.dataset_status = 1
        outliers = label.get_probable_outliers_by_regex(project_regex.categories, top_worst_percentage=1.0)
        outlier_spans = [span.offset_string for annotation in outliers for span in annotation.spans]
        assert len(outliers) == 3
        assert 'DE47 7001 0500 0000 2XxXX XX' in outlier_spans
        outliers_with_test = label.get_probable_outliers_by_regex(
            project_regex.categories, use_test_docs=True, top_worst_percentage=1.0
        )
        outlier_test_spans = [span.offset_string for annotation in outliers_with_test for span in annotation.spans]
        assert len(outlier_test_spans) == 6
        assert 'DE38 7609 0900 0001 2XXX XX' in outlier_test_spans

    @pytest.mark.skipif(
        not is_dependency_installed('torch'),
        reason='Required dependencies not installed.',
    )
    def test_find_outlier_annotations_by_confidence(self):
        """Test finding the Annotations with the least confidence."""
        from konfuzio_sdk.trainer.information_extraction import RFExtractionAI

        label = self.project.get_label_by_name('Austellungsdatum')
        pipeline = RFExtractionAI()
        pipeline.tokenizer = ListTokenizer(tokenizers=[])
        pipeline.category = self.project.get_category_by_id(id_=63)
        train_doc_ids = {44823, 44834, 44839, 44840, 44841}
        pipeline.documents = [doc for doc in pipeline.category.documents() if doc.id_ in train_doc_ids]
        for cur_label in pipeline.category.labels:
            for regex in cur_label.find_regex(category=pipeline.category):
                pipeline.tokenizer.tokenizers.append(RegexTokenizer(regex=regex))
        pipeline.test_documents = pipeline.category.test_documents()
        pipeline.df_train, pipeline.label_feature_list = pipeline.feature_function(
            documents=pipeline.documents, require_revised_annotations=False
        )
        pipeline.fit()
        evaluation = pipeline.evaluate_full(strict=False, use_training_docs=True)
        outliers = label.get_probable_outliers_by_confidence(evaluation, 0.9)
        assert len(outliers) >= 2
        outlier_spans = [span.offset_string for annotation in outliers for span in annotation.spans]
        assert '24.05.2018' in outlier_spans

    def test_find_outlier_annotations_by_normalization(self):
        """Test finding the Annotations that do not correspond the Label's data type."""
        project = Project(id_=TEST_PROJECT_ID)
        label = project.get_label_by_name('Austellungsdatum')
        outliers = label.get_probable_outliers_by_normalization(project.categories)
        assert len(outliers) == 0

    def test_find_outlier_annotations(self):
        """Test finding the Annotations that are deemed outliers by several methods of search."""
        label = self.project.get_label_by_name('Austellungsdatum')
        outliers = label.get_probable_outliers(
            self.project.categories, regex_worst_percentage=1.0, confidence_search=False
        )
        outlier_spans = [span.offset_string for annotation in outliers for span in annotation.spans]
        assert len(outliers) == 1
        assert '328927/10103' in outlier_spans
        assert '22.05.2018' in outlier_spans

    def test_find_outlier_annotations_error(self):
        """Test impossibility of running outlier Annotation search with all modes disabled."""
        label = self.project.get_label_by_name('Austellungsdatum')
        with pytest.raises(ValueError, match='search modes disabled'):
            label.get_probable_outliers(
                self.project.categories, regex_search=False, confidence_search=False, normalization_search=False
            )

    def test_get_original_page_from_copy(self):
        """Test getting an original Page from a copy of a Page."""
        document = self.project.get_document_by_id(44823)
        copied_document = deepcopy(document)
        copied_page = copied_document.pages()[0]
        page = copied_page.get_original_page()
        assert page == document.pages()[0]

    def test_get_page_by_id(self):
        """Test getting a Page from the Document by the ID."""
        document = self.project.get_document_by_id(44823)
        page = document.get_page_by_id(1923)
        assert page == document.pages()[0]

    def test_page_annotation_sets(self):
        """Test viewing Annotation Sets of Annotations present at the Page."""
        document = self.project.get_document_by_id(TEST_DOCUMENT_ID)
        page = document.pages()[0]
        annotation_sets = page.annotation_sets()
        assert len(annotation_sets) == 5

    def test_page_lines(self):
        """Test grouping Spans of a Page into lines."""
        document = self.project.get_document_by_id(TEST_DOCUMENT_ID)
        page = document.get_page_by_index(0)
        lined_spans = page.lines()
        assert len(lined_spans) == 53
        assert lined_spans[0].offset_string == 'x02   328927/10103/00104'

    def test_create_project_metadata_json(self):
        """Test creating a JSON with a Project's metadata."""
        metadata_dict = self.project.create_project_metadata_dict()
        assert len(metadata_dict['categories']) == 3
        assert len(metadata_dict['categories'][0]['schema']) == 6
        assert len(metadata_dict['categories'][0]['schema'][0]['labels']) == 10

    def test_create_category_wrong_name(self):
        """Test that it's impossible to create a Category with a name that contains a special character."""
        wrong_name = Category(project=self.project, name='Category/name', name_clean='Category/name')
        assert wrong_name.name == 'Category/name'
        assert wrong_name.name_clean == 'Categoryname'

    def test_get_category_by_name(self):
        """Test that Categories can be searched by name."""
        assert (
            self.project.get_category_by_id(63).name
            == self.project.get_category_by_name(category_name='Lohnabrechnung').name
        )
        assert (
            self.project.get_category_by_id(63).name_clean
            == self.project.get_category_by_name(category_name='Lohnabrechnung').name_clean
        )

    def test_delete_empty_annotation_set(self):
        """Test that an Annotation Set is deleted from a Document when its last Annotation is deleted."""
        document = self.project.get_document_by_id(TEST_DOCUMENT_ID)
        annotation_set = document.get_annotation_set_by_id(687572)
        annotation = annotation_set.annotations(use_correct=False)[0]
        annotation.delete(delete_online=False)
        assert annotation_set not in document.annotation_sets()


class TestEqualityAnnotation(unittest.TestCase):
    """Test the equality of Annotations."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the test Project."""
        cls.project = Project(id_=None)
        cls.category = Category(project=cls.project, id_=1)
        cls.label_set = LabelSet(project=cls.project, categories=[cls.category], id_=421)
        cls.label_one = Label(project=cls.project, text='First', label_sets=[cls.label_set])
        cls.label_two = Label(project=cls.project, text='Second', label_sets=[cls.label_set])
        cls.document = Document(project=cls.project, category=cls.category)
        # cls.label_set.add_label(cls.label)
        cls.annotation_set = AnnotationSet(document=cls.document, label_set=cls.label_set)
        assert len(cls.project.virtual_documents) == 1

    def test_overlapping_correct_same_label(self):
        """Reject to add Annotations that are identical."""
        document = Document(project=self.project, category=self.category)
        first_span = Span(start_offset=1, end_offset=2)
        second_span = Span(start_offset=1, end_offset=2)
        _ = Annotation(
            document=document, spans=[first_span], label_set=self.label_set, label=self.label_one, is_correct=True
        )

        with pytest.raises(ValueError) as e:
            _ = Annotation(
                document=document, spans=[second_span], label_set=self.label_set, label=self.label_one, is_correct=True
            )
            assert 'is a duplicate of' in str(e)

    def test_partially_overlapping_correct_same_label(self):
        """Accept to add Annotation with the same Label if parts of their Spans differ."""
        document = Document(project=self.project, category=self.category)
        first_span = Span(start_offset=1, end_offset=2)
        second_span = Span(start_offset=1, end_offset=2)
        third_span = Span(start_offset=2, end_offset=3)
        _ = Annotation(
            document=document, spans=[first_span], label_set=self.label_set, label=self.label_one, is_correct=True
        )

        _ = Annotation(
            document=document,
            spans=[second_span, third_span],
            label_set=self.label_set,
            label=self.label_one,
            is_correct=True,
        )

    def test_overlapping_wrong_same_label(self):
        """Accept to add Annotation with the same Label if both are not correct."""
        document = Document(project=self.project, category=self.category)
        first_span = Span(start_offset=1, end_offset=2)
        second_span = Span(start_offset=1, end_offset=2)
        _ = Annotation(
            document=document, spans=[first_span], label_set=self.label_set, label=self.label_one, is_correct=False
        )

        with pytest.raises(ValueError) as e:
            _ = Annotation(
                document=document, spans=[second_span], label_set=self.label_set, label=self.label_one, is_correct=False
            )
            assert 'is a duplicate of' in str(e)

    def test_partially_overlapping_wrong_same_label(self):
        """Accept to add Annotation with the same Label if parts of their Spans differ and one is not correct."""
        document = Document(project=self.project, category=self.category)
        first_span = Span(start_offset=1, end_offset=2)
        second_span = Span(start_offset=1, end_offset=2)
        third_span = Span(start_offset=2, end_offset=3)
        _ = Annotation(
            document=document, spans=[first_span], label_set=self.label_set, label=self.label_one, is_correct=False
        )

        _ = Annotation(
            document=document,
            spans=[second_span, third_span],
            label_set=self.label_set,
            label=self.label_one,
            is_correct=False,
        )

    def test_overlapping_partially_correct_same_label(self):
        """Accept to add Annotation with the same Label if one Annotation is not correct."""
        document = Document(project=self.project, category=self.category)
        first_span = Span(start_offset=1, end_offset=2)
        second_span = Span(start_offset=1, end_offset=2)
        _ = Annotation(
            document=document, spans=[first_span], label_set=self.label_set, label=self.label_one, is_correct=True
        )

        with pytest.raises(ValueError) as e:
            _ = Annotation(
                document=document, spans=[second_span], label_set=self.label_set, label=self.label_one, is_correct=False
            )
            assert 'is a duplicate of' in str(e)

    def test_partially_overlapping_partially_correct_same_label(self):
        """Accept to add Annotation with the same Label if parts of their Spans differ and one is not correct."""
        document = Document(project=self.project, category=self.category)
        first_span = Span(start_offset=1, end_offset=2)
        second_span = Span(start_offset=1, end_offset=2)
        third_span = Span(start_offset=2, end_offset=3)
        _ = Annotation(
            document=document, spans=[first_span], label_set=self.label_set, label=self.label_one, is_correct=True
        )

        _ = Annotation(
            document=document,
            spans=[second_span, third_span],
            label_set=self.label_set,
            label=self.label_one,
            is_correct=False,
        )

    def test_overlapping_correct_other_label(self):
        """Accept to add Annotation with different Labels."""
        document = Document(project=self.project, category=self.category)
        first_span = Span(start_offset=1, end_offset=2)
        second_span = Span(start_offset=1, end_offset=2)
        _ = Annotation(
            document=document, spans=[first_span], label_set=self.label_set, label=self.label_two, is_correct=True
        )

        _ = Annotation(
            document=document, spans=[second_span], label_set=self.label_set, label=self.label_one, is_correct=True
        )

    def test_overlapping_wrong_other_label(self):
        """Accept to add Annotation with different Labels if both are not correct."""
        document = Document(project=self.project, category=self.category)
        first_span = Span(start_offset=1, end_offset=2)
        second_span = Span(start_offset=1, end_offset=2)
        _ = Annotation(
            document=document, spans=[first_span], label_set=self.label_set, label=self.label_one, is_correct=False
        )

        _ = Annotation(
            document=document, spans=[second_span], label_set=self.label_set, label=self.label_two, is_correct=False
        )

    def test_partially_overlapping_partially_correct_other_label(self):
        """Accept to add Annotation with different Labels if one is not correct and one is only some Spans overlap."""
        document = Document(project=self.project, category=self.category)
        first_span = Span(start_offset=1, end_offset=2)
        second_span = Span(start_offset=1, end_offset=2)
        third_span = Span(start_offset=2, end_offset=3)
        _ = Annotation(
            document=document, spans=[first_span], label_set=self.label_set, label=self.label_one, is_correct=True
        )

        _ = Annotation(
            document=document,
            spans=[second_span, third_span],
            label_set=self.label_set,
            label=self.label_two,
            is_correct=False,
        )


class TestOfflineDataSetup(unittest.TestCase):
    """Test data features on programmatically constructed Project."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the test Project."""
        cls.project = Project(id_=None)
        cls.label = Label(project=cls.project, text='First Offline Label')
        cls.category = Category(project=cls.project, id_=2)
        cls.category2 = Category(project=cls.project, id_=3)
        cls.document = Document(project=cls.project, category=cls.category, text='Hello.')
        cls.label_set = LabelSet(project=cls.project, categories=[cls.category], id_=421)
        cls.label_set.add_label(cls.label)
        cls.annotation_set = AnnotationSet(document=cls.document, label_set=cls.label_set)
        assert len(cls.project.virtual_documents) == 1

    @classmethod
    def tearDownClass(cls) -> None:
        """Control the number of Documents created in the Test."""
        assert len(cls.project.virtual_documents) == 64

    def test_document_only_needs_project(self):
        """Test that a Document can be created without Category."""
        _ = Document(project=self.project)

    def test_project_no_label(self):
        """Test that no_label exists in the Labels of the Project and has the expected name."""
        assert self.project.no_label in self.project.labels
        assert self.project.no_label.name == 'NO_LABEL'

    def test_project_no_label_set(self):
        """Test that no_label_set exists in the Label Sets of the Project."""
        assert self.project.no_label_set in self.project.label_sets

    def test_project_has_category(self):
        """Test that no_label_set exists in the Label Sets of the Categories of the Project."""
        assert self.category in self.project.categories

    def test_project_no_label_set_in_all_categories(self):
        """Test that no_label_set exists in the Label Sets of the Categories of the Project."""
        for category in self.project.categories:
            assert self.project.no_label_set in category.project.label_sets

    def test_project_credentials(self):
        """Test that a Project can be initialized with credentials and they are stored as an attribute."""
        assert hasattr(self.project, 'credentials')
        assert self.project.credentials == {}
        credentials = {'EXAMPLE_KEY_1': 'EXAMPLE_VALUE_1', 'EXAMPLE_KEY_2': 'EXAMPLE_VALUE_2'}
        project = Project(id_=None, credentials=credentials)
        assert project.credentials == credentials
        assert project.get_credentials('EXAMPLE_KEY_1') == 'EXAMPLE_VALUE_1'
        assert project.get_credentials('EXAMPLE_KEY_2') == 'EXAMPLE_VALUE_2'
        assert project.get_credentials('EXAMPLE_NONEXISTING_KEY') is None

    def test_document_no_label_annotation_set_label_set(self):
        """Test that Label Set of the no_label_annotation_set of the Document has the no_label_set of the Project."""
        assert self.document.no_label_annotation_set.label_set.id_ == self.project.no_label_set.id_ == 0
        assert self.project.no_label_set.name == 'NO_LABEL_SET'
        assert self.document.no_label_annotation_set.label_set == self.project.no_label_set

    def test_category_of_document(self):
        """Test if setup worked."""
        assert self.document.category == self.category
        assert self.document.maximum_confidence_category == self.category
        for page in self.document.pages():
            assert page.category == self.category

    def test_categorize_when_all_pages_have_same_category(self):
        """Test categorizing a Document when all Pages have the same Category."""
        document = Document(project=self.project, text='hello')
        for i in range(2):
            page = Page(
                id_=None,
                document=document,
                start_offset=0,
                end_offset=0,
                number=i + 1,
                original_size=(0, 0),
            )
            page.set_category(self.category)
            assert page.maximum_confidence_category_annotation.category == self.category
            assert page.maximum_confidence_category_annotation.confidence == 1.0
            assert len(page.category_annotations) == 1
        assert document.maximum_confidence_category == self.category
        assert document.category == self.category

    def test_categorize_when_all_pages_have_no_category(self):
        """Test categorizing a Document when all Pages have no Category."""
        document = Document(project=self.project, text='hello')
        for i in range(2):
            page = Page(id_=None, document=document, start_offset=0, end_offset=0, number=i + 1, original_size=(0, 0))
            assert page.category == self.project.no_category
            assert page.maximum_confidence_category_annotation is None
            assert len(page.category_annotations) == 0
        assert document.maximum_confidence_category is document.project.no_category
        assert document.category is document.project.no_category

    def test_categorize_when_pages_have_different_categories(self):
        """Test categorizing a Document when Pages have different Category."""
        document = Document(project=self.project, text='hello')
        for i in range(2):
            page = Page(
                id_=None,
                document=document,
                start_offset=0,
                end_offset=0,
                number=i + 1,
                original_size=(0, 0),
            )
            page_category = self.category if i else self.category2
            page.set_category(page_category)
            assert page.maximum_confidence_category_annotation.category == page_category
            assert page.maximum_confidence_category_annotation.confidence == 1.0
            assert len(page.category_annotations) == 1
        assert len(document.category_annotations) == 2
        assert document.category == document.project.no_category
        # as each page got assigned a different Category with confidence all equal to 1,
        # the maximum confidence Category of the Document will be a random one
        assert document.maximum_confidence_category in [self.category, self.category2]
        # if the user revises it, it will be consistently updated
        document.set_category(self.category)
        assert document.maximum_confidence_category == self.category
        assert document.category == self.category

    def test_categorize_when_pages_have_mixed_categories_or_no_category(self):
        """Test categorizing a Document when Pages have different Category or no Category."""
        document = Document(project=self.project, text='hello')
        for i in range(3):
            page = Page(
                id_=None,
                document=document,
                start_offset=0,
                end_offset=0,
                number=i + 1,
                original_size=(0, 0),
            )
            page_category = [self.category, self.category2, self.project.no_category][i]
            page.set_category(page_category)
            if page_category != self.project.no_category:
                assert page.maximum_confidence_category_annotation.category == page_category
                assert page.maximum_confidence_category_annotation.confidence == 1.0
                assert len(page.category_annotations) == 1
            else:
                assert page.category is self.project.no_category
                assert page.maximum_confidence_category_annotation is None
                assert len(page.category_annotations) == 0
        assert len(document.category_annotations) == 2
        assert document.category == document.project.no_category

    def test_categorize_with_no_pages(self):
        """Test categorizing a Document with no Pages."""
        document = Document(project=self.project, text='hello')
        assert document.category == document.project.no_category
        assert document.pages() == []

    def test_categorize_when_pages_have_same_category_or_no_category(self):
        """Test categorizing a Document where some Pages are of the same Category and others are blank."""
        document = Document(project=self.project, text='hello')
        for i in range(3):
            page = Page(
                id_=None,
                document=document,
                start_offset=0,
                end_offset=0,
                number=i + 1,
                original_size=(0, 0),
            )
            page_category = [
                self.category,
                self.project.no_category,
                self.category,
            ][i]
            page.set_category(page_category)
            if page_category != self.project.no_category:
                assert page.maximum_confidence_category_annotation.category == page_category
                assert page.maximum_confidence_category_annotation.confidence == 1.0
                assert len(page.category_annotations) == 1
            else:
                assert page.category is self.project.no_category
                assert page.maximum_confidence_category_annotation is None
                assert len(page.category_annotations) == 0
            assert len(document.category_annotations) == 2
            assert document.category == self.category

    def test_span_negative_offset(self):
        """Negative Span creation should not be possible."""
        project = Project(id_=None)
        category = Category(project=project)
        label_set = LabelSet(id_=33, project=project, categories=[category])
        label = Label(id_=22, text='LabelName', project=project, label_sets=[label_set], threshold=0.5)
        document = Document(project=project, category=category, text='From 14.12.2021 to 1.1.2022.', dataset_status=2)
        with self.assertRaises(ValueError):
            span_1 = Span(start_offset=-1, end_offset=2)
            annotation_set_1 = AnnotationSet(id_=1, document=document, label_set=label_set)
            _ = Annotation(
                document=document,
                is_correct=True,
                annotation_set=annotation_set_1,
                label=label,
                label_set=label_set,
                spans=[span_1],
            )

    def test_span_negative_offset_force_allow(self):
        """Negative Span creation should only be possible by force disabling validation rules."""
        project = Project(id_=None, strict_data_validation=False)
        category = Category(project=project)
        label_set = LabelSet(id_=33, project=project, categories=[category])
        label = Label(id_=22, text='LabelName', project=project, label_sets=[label_set], threshold=0.5)
        document = Document(project=project, category=category, text='From 14.12.2021 to 1.1.2022.', dataset_status=2)
        span_1 = Span(start_offset=-1, end_offset=2, strict_validation=False)
        annotation_set_1 = AnnotationSet(id_=1, document=document, label_set=label_set)
        _ = Annotation(
            document=document,
            is_correct=True,
            annotation_set=annotation_set_1,
            label=label,
            label_set=label_set,
            spans=[span_1],
        )

    def test_training_document_annotations_are_available(self):
        """Test if the Label can access the new Annotation."""
        project = Project(id_=None)
        category = Category(project=project)
        label_set = LabelSet(id_=33, project=project, categories=[category])
        label = Label(id_=22, text='LabelName', project=project, label_sets=[label_set], threshold=0.5)
        document = Document(project=project, category=category, text='From 14.12.2021 to 1.1.2022.', dataset_status=2)
        span_1 = Span(start_offset=5, end_offset=15)
        annotation_set_1 = AnnotationSet(id_=1, document=document, label_set=label_set)
        annotation = Annotation(
            document=document,
            is_correct=True,
            annotation_set=annotation_set_1,
            label=label,
            label_set=label_set,
            spans=[span_1],
        )
        assert label.annotations(categories=[category]) == [annotation]

    def test_add_annotation_with_complete_bbox_data(self):
        """Test to add an Annotation via complete bboxes param."""
        document = Document(
            project=self.project, category=self.category, text='hello', bbox_validation_type=BboxValidationTypes.STRICT
        )
        page = Page(id_=None, document=document, start_offset=0, end_offset=4, number=1, original_size=(12, 6))
        document_bbox = {'1': Bbox(x0=0, x1=1, y0=0, y1=1, page=page)}
        document.set_bboxes(document_bbox)
        ann_bbox = {
            'bottom': 1,
            'end_offset': 2,
            'line_number': 0,
            'offset_string': 'he',
            'offset_string_original': 'he',
            'page_index': 0,
            'start_offset': 0,
            'top': 0,
            'x0': 0,
            'x1': 2,
            'y0': 0,
            'y1': 1,
        }
        annotation = Annotation(
            document=document,
            label=self.label,
            label_set=self.label_set,
            bboxes=[ann_bbox],
        )
        assert annotation.start_offset == ann_bbox['start_offset']
        assert annotation.end_offset == ann_bbox['end_offset']

    def test_add_annotation_with_incomplete_bbox_data(self):
        """Test to add an Annotation via bboxes param that is missing offset information."""
        document = Document(
            project=self.project, category=self.category, text='hello', bbox_validation_type=BboxValidationTypes.STRICT
        )
        page = Page(id_=None, document=document, start_offset=0, end_offset=4, number=1, original_size=(12, 6))
        document_bbox = {'1': Bbox(x0=0, x1=1, y0=0, y1=1, page=page)}
        document.set_bboxes(document_bbox)
        # An Annotation can be created by providing a list of Spans or a list of bboxes.
        # In the latter case, the minimum information required is the start and end offsets corresponding
        # to the characters of each bbox.
        annotation_bboxes = [{'start_offset': 0, 'end_offset': 1}, {'start_offset': 3}]
        with pytest.raises(ValueError, match='cannot read Bbox'):
            Annotation(document=document, bboxes=annotation_bboxes, label=self.label, label_set=self.label_set)

    def test_add_annotation_with_label_set_none(self):
        """Test to add an Annotation to a Document where the LabelSet is None."""
        project = Project(id_=None)
        category = Category(project=project)
        label_set = LabelSet(project=project, categories=[category])
        label = Label(project=project, label_sets=[label_set])
        # create a document A
        document_a = Document(project=project, category=category)
        span = Span(start_offset=1, end_offset=2)
        annotation_set_a = AnnotationSet(document=document_a, label_set=label_set)

        annotation = Annotation(document=document_a, annotation_set=annotation_set_a, label=label, spans=[span])

        assert annotation.label_set is annotation_set_a.label_set

    def test_add_annotation_with_annotation_set_and_label_set_none_w_multiple_false(self):
        """
        Test to add an Annotation to a Document where the LabelSet and AnnotationSet are None.

        With LabelSet.has_multiple_annotation_sets True.
        """
        project = Project(id_=None)
        category = Category(project=project)
        label_set = LabelSet(id_=93710, project=project, categories=[category], has_multiple_annotation_sets=False)
        label = Label(project=project, label_sets=[label_set])

        document = Document(project=project, category=category)
        span = Span(start_offset=1, end_offset=2)

        annotation = Annotation(document=document, label=label, spans=[span])

        assert isinstance(annotation.annotation_set, AnnotationSet)
        assert annotation.label_set is annotation.annotation_set.label_set

    def test_add_annotation_with_annotation_set_and_label_set_none_w_multiple_true(self):
        """
        Test to add an Annotation to a Document where the LabelSet and AnnotationSet are None.

        With LabelSet.has_multiple_annotation_sets True.
        """
        project = Project(id_=None)
        category = Category(project=project)
        label_set = LabelSet(id_=93711, project=project, categories=[category], has_multiple_annotation_sets=True)
        label = Label(project=project, label_sets=[label_set])

        document = Document(project=project, category=category)
        span = Span(start_offset=1, end_offset=2)

        with pytest.raises(ValueError, match='Cannot assign .* to AnnotationSet.* can have multiple Annotation Sets'):
            _ = Annotation(document=document, label=label, spans=[span])

    @pytest.mark.xfail(
        strict=True,
        reason='To not interrupt server workflows, we log an error instead of raising a ValueError for now.',
    )
    def test_add_annotation_set_w_multiple_false(self):
        """Test to add a second AnnotationSet to a Document where LabelSet.has_multiple_annotation_sets is False."""
        project = Project(id_=None)
        category = Category(project=project)
        label_set = LabelSet(id_=93712, project=project, categories=[category], has_multiple_annotation_sets=False)

        document = Document(project=project, category=category)
        _ = AnnotationSet(id_=1, document=document, label_set=label_set)

        with pytest.raises(ValueError, match='is already used by another Annotation Set'):
            _ = AnnotationSet(id_=2, document=document, label_set=label_set)

    def test_add_annotation_set_w_multiple_true(self):
        """Test to add a second AnnotationSet to a Document where LabelSet.has_multiple_annotation_sets is True."""
        project = Project(id_=None)
        category = Category(project=project)
        label_set = LabelSet(id_=93713, project=project, categories=[category], has_multiple_annotation_sets=True)

        document = Document(project=project, category=category)
        annotation_set_1 = AnnotationSet(document=document, label_set=label_set)
        annotation_set_2 = AnnotationSet(document=document, label_set=label_set)  # no error
        assert annotation_set_1 != annotation_set_2  # both ids are None, but different local_ids

    def test_get_default_label_set_and_annotation_set(self):
        """Test to get the default AnnotationSet of a Document."""
        project = Project(id_=None)
        category = Category(id_=143, name='Category143', project=project)

        document = Document(project=project, category=category)

        label_set = category.default_label_set

        assert label_set.id_ == category.id_
        assert label_set.has_multiple_annotation_sets is False
        assert label_set.is_default is True
        assert label_set.name == category.name

        annotation_set = document.default_annotation_set

        assert annotation_set.is_default is True
        assert annotation_set.label_set is label_set

        # with pytest.raises(ValueError, match='is already used by another Annotation Set'):
        #     _ = AnnotationSet(document=document, label_set=label_set)

    def test_add_none_label_annotation(self):
        """Test to add an Annotation with none Label."""
        project = Project(id_=None)
        category = Category(project=project)
        print(project.no_label_set.categories)
        document = Document(project=project, category=category)

        span = Span(start_offset=1, end_offset=2)

        annotation = Annotation(document=document, label=None, spans=[span])

        assert annotation.label is project.no_label
        assert annotation.label_set is project.no_label_set
        assert annotation.annotation_set is document.no_label_annotation_set

    def test_to_get_threshold(self):
        """Define fallback threshold for a Label."""
        project = Project(id_=None)
        label = Label(project=project, text='Third Offline Label')
        assert label.threshold == 0.1

    def test_to_add_label_to_project(self):
        """Add one Label to a Project."""
        _ = Label(project=self.project, text='Third Offline Label')
        assert sorted([label.name for label in self.project.labels]) == [
            'First Offline Label',
            'NO_LABEL',
            'Second Offline Label',
            'Third Offline Label',
        ]

    def test_label_has_label_sets(self):
        """Pass and store Label Sets."""
        project = Project(id_=None)
        label = Label(project=project, label_sets=[self.label_set], text='Second Offline Label')
        assert [ls.id_ for ls in label.label_sets] == [421]

    def test_to_add_label_to_project_twice(self):
        """Add an existing Label to a Project."""
        with self.assertRaises(ValueError):
            self.project.add_label(self.label)

    def test_get_labels_of_category(self):
        """Return only related Labels as Information Extraction can be trained per Category."""
        assert self.category.labels.__len__() == 1

    def test_to_add_spans_to_annotation(self):
        """Add one Span to one Annotation."""
        document = Document(project=self.project, category=self.category)
        span = Span(start_offset=1, end_offset=2)
        annotation = Annotation(document=document, spans=[span], label=self.label, label_set=self.label_set)
        self.assertEqual([span], annotation.spans)

    def test_span_reference_to_annotation(self):
        """Test Span reference to Annotation."""
        document = Document(project=self.project, category=self.category)
        span = Span(start_offset=1, end_offset=2)
        annotation = Annotation(document=document, spans=[span], label=self.label, label_set=self.label_set)
        assert annotation.spans[0].annotation is not None
        assert annotation.spans[0].bbox() is None  # Span bboxes must be explicitly loaded using span.bbox
        # Here this would be failing even when calling Span.bbox as the test document does not have a bbox.

    def test_get_span_bbox_with_characters_without_height_allowed(self):
        """
        Test get the bbox of a Span where the characters do not have height (OCR problem).

        Without specifying strict validation, we allow such bboxes.
        """
        document_bbox = {'1': {'text': 'e', 'x0': 0, 'x1': 1, 'y0': 1, 'y1': 1, 'page_number': 1}}
        document = Document(project=self.project, category=self.category, text='hello', bbox=document_bbox)
        span = Span(start_offset=1, end_offset=2)
        _ = Annotation(document=document, spans=[span], label=self.label, label_set=self.label_set)
        _ = Page(id_=1, number=1, original_size=(595.2, 300.0), document=document, start_offset=0, end_offset=1)
        self.assertTrue(span.bbox())

    @pytest.mark.xfail(reason='We now only log a warning because Azure OCR sometimes returns bboxes without height.')
    def test_get_span_bbox_with_characters_without_height_strict_validation(self):
        """
        Test get the bbox of a Span where the characters do not have height (OCR problem).

        With strict validation specified, we don't allow such bboxes.
        """
        document_bbox = {'1': {'text': 'e', 'x0': 0, 'x1': 1, 'y0': 1, 'y1': 1, 'page_number': 1}}
        document = Document(
            project=self.project,
            category=self.category,
            text='hello',
            bbox=document_bbox,
            bbox_validation_type=BboxValidationTypes.STRICT,
        )
        span = Span(start_offset=1, end_offset=2)
        _ = Annotation(document=document, spans=[span], label=self.label, label_set=self.label_set)
        _ = Page(id_=1, number=1, original_size=(595.2, 300.0), document=document, start_offset=0, end_offset=1)
        with pytest.raises(ValueError, match='has no height in Page 0.'):
            span.bbox()

    def test_get_span_bbox_with_characters_without_width_missing_bbox(self):
        """Test get the bbox of a Span where the characters do not have width (OCR problem)."""
        document_bbox = {'1': {'x0': 1, 'x1': 1, 'y0': 0, 'y1': 1, 'page_number': 1}}
        document = Document(project=self.project, category=self.category, text='hello', bbox=document_bbox)
        span = Span(start_offset=1, end_offset=2)
        _ = Annotation(document=document, spans=[span], label=self.label, label_set=self.label_set)
        _ = Page(id_=1, number=1, original_size=(595.2, 300.0), document=document, start_offset=0, end_offset=1)
        with pytest.raises(ValueError, match='provides Character "None" Document text refers to "e"'):
            span.bbox()

    def test_get_span_bbox_with_characters_without_width_allowed(self):
        """
        Test get the bbox of a Span where the characters do not have width (OCR problem).

        Without strict validation specified, we allow such bboxes.
        """
        document_bbox = {'0': {'x0': 1, 'x1': 1, 'y0': 0, 'y1': 1, 'page_number': 1, 'text': 'h'}}
        document = Document(project=self.project, category=self.category, text='hello', bbox=document_bbox)
        span = Span(start_offset=0, end_offset=1)
        _ = Annotation(document=document, spans=[span], label=self.label, label_set=self.label_set)
        _ = Page(id_=1, number=1, original_size=(595.2, 300.0), document=document, start_offset=0, end_offset=1)
        self.assertTrue(span.bbox())

    @pytest.mark.xfail(reason='We now only log a warning because Azure OCR sometimes returns bboxes without width.')
    def test_get_span_bbox_with_characters_without_width_strict_validation(self):
        """
        Test get the bbox of a Span where the characters do not have width (OCR problem).

        With strict validation specified, we allow such bboxes.
        """
        document_bbox = {'0': {'x0': 1, 'x1': 1, 'y0': 0, 'y1': 1, 'page_number': 1, 'text': 'h'}}
        document = Document(
            project=self.project,
            category=self.category,
            text='hello',
            bbox=document_bbox,
            bbox_validation_type=BboxValidationTypes.STRICT,
        )
        span = Span(start_offset=0, end_offset=1)
        _ = Annotation(document=document, spans=[span], label=self.label, label_set=self.label_set)
        _ = Page(id_=1, number=1, original_size=(595.2, 300.0), document=document, start_offset=0, end_offset=1)
        with pytest.raises(ValueError, match='has no width in Page 0'):
            span.bbox()

    def test_get_span_bbox_with_characters_with_negative_x_coord(self):
        """Test get the bbox of a Span where the characters have negative x coordinates (OCR problem)."""
        document_bbox = {'1': {'text': 'e', 'x0': -1, 'x1': 1, 'y0': 0, 'y1': 1, 'page_number': 1}}
        document = Document(project=self.project, category=self.category, text='hello', bbox=document_bbox)
        span = Span(start_offset=1, end_offset=2)
        _ = Annotation(document=document, spans=[span], label=self.label, label_set=self.label_set)
        _ = Page(id_=1, number=1, original_size=(595.2, 300.0), document=document, start_offset=0, end_offset=3)
        with pytest.raises(ValueError, match='negative x coordinate'):
            span.bbox()

    def test_get_span_bbox_with_characters_with_negative_y_coord(self):
        """Test get the bbox of a Span where the characters have negative x coordinates (OCR problem)."""
        document_bbox = {'1': {'text': 'e', 'x0': 0, 'x1': 1, 'y0': -1, 'y1': 1, 'page_number': 1}}
        document = Document(project=self.project, category=self.category, text='hello', bbox=document_bbox)
        span = Span(start_offset=1, end_offset=2)
        _ = Annotation(document=document, spans=[span], label=self.label, label_set=self.label_set)
        _ = Page(id_=1, number=1, original_size=(595.2, 300.0), document=document, start_offset=0, end_offset=3)
        with pytest.raises(ValueError, match='negative y coordinate'):
            span.bbox()

    def test_get_span_bbox_with_characters_with_x_coord_outside_page_width(self):
        """Test get the bbox of a Span where the characters have negative x coordinates (OCR problem)."""
        document_bbox = {'1': {'text': 'e', 'x0': 596, 'x1': 597, 'y0': 0, 'y1': 1, 'page_number': 1}}
        document = Document(project=self.project, category=self.category, text='hello', bbox=document_bbox)
        span = Span(start_offset=1, end_offset=2)
        _ = Annotation(document=document, spans=[span], label=self.label, label_set=self.label_set)
        _ = Page(id_=1, number=1, original_size=(595.2, 300.0), document=document, start_offset=0, end_offset=3)
        with pytest.raises(ValueError, match='exceeds width of Page 0'):
            span.bbox()

    def test_get_span_bbox_with_characters_with_x_coord_outside_page_width_disable_validations(self):
        """Test disable validations of Bbox where the characters have negative x coordinates (OCR problem)."""
        document_bbox = {'1': {'text': 'e', 'x0': 596, 'x1': 597, 'y0': 0, 'y1': 1, 'page_number': 1}}
        document = Document(
            project=self.project,
            category=self.category,
            text='hello',
            bbox=document_bbox,
            bbox_validation_type=BboxValidationTypes.DISABLED,
        )
        span = Span(start_offset=1, end_offset=2)
        _ = Annotation(document=document, spans=[span], label=self.label, label_set=self.label_set)
        _ = Page(id_=1, number=1, original_size=(595.2, 300.0), document=document, start_offset=0, end_offset=3)
        span.bbox()

    def test_get_span_bbox_with_characters_with_y_coord_outside_page_height(self):
        """Test get the bbox of a Span where the characters have negative y coordinates (OCR problem)."""
        document_bbox = {'1': {'text': 'e', 'x0': 0, 'x1': 1, 'y0': 301, 'y1': 302, 'page_number': 1}}
        document = Document(project=self.project, category=self.category, text='hello', bbox=document_bbox)
        span = Span(start_offset=1, end_offset=2)
        _ = Annotation(document=document, spans=[span], label=self.label, label_set=self.label_set)
        _ = Page(id_=1, number=1, original_size=(595.2, 300.0), document=document, start_offset=0, end_offset=3)
        with pytest.raises(ValueError, match='exceeds height of Page 0'):
            span.bbox()

    def test_get_span_bbox_with_characters_with_y_coord_outside_page_height_disable_validations(self):
        """Test disable validations of Bbox where the characters have negative y coordinates (OCR problem)."""
        document_bbox = {'1': {'text': 'e', 'x0': 0, 'x1': 1, 'y0': 301, 'y1': 302, 'page_number': 1}}
        document = Document(
            project=self.project,
            category=self.category,
            text='hello',
            bbox=document_bbox,
            bbox_validation_type=BboxValidationTypes.DISABLED,
        )
        span = Span(start_offset=1, end_offset=2)
        _ = Annotation(document=document, spans=[span], label=self.label, label_set=self.label_set)
        _ = Page(id_=1, number=1, original_size=(595.2, 300.0), document=document, start_offset=0, end_offset=3)
        span.bbox()

    def test_get_span_bbox_with_unavailable_characters(self):
        """Test get the bbox of a Span where the characters are unavailable."""
        document_bbox = {
            '0': {'x0': 0, 'x1': 1, 'y0': 0, 'y1': 1, 'top': 10, 'bottom': 11, 'page_number': 1},
            '2': {'x0': 1, 'x1': 2, 'y0': 0, 'y1': 1, 'top': 10, 'bottom': 11, 'page_number': 1},
        }
        document = Document(project=self.project, category=self.category, text='hello', bbox=document_bbox)
        span = Span(start_offset=1, end_offset=2)
        _ = Annotation(document=document, spans=[span], label=self.label, label_set=self.label_set)

        span.bbox()
        # with self.assertRaises(ValueError) as context:
        # raise ValueError
        # todo find a way to raise a value error for characters, but ignore special Characters that
        #  do not provide a Bbox
        # assert 'does not have available characters bounding boxes.' in context.exception

    def test_document_check_bbox_coordinates(self):
        """Test bbox check for coordinates with valid coordinates."""
        document_bbox = {
            '0': {'x0': 0, 'x1': 1, 'y0': 0, 'y1': 1, 'top': 10, 'bottom': 11, 'page_number': 1, 'text': 'h'}
        }
        document = Document(project=self.project, category=self.category, text='h', bbox=document_bbox)
        _ = Page(id_=1, number=1, original_size=(595.2, 841.68), document=document, start_offset=0, end_offset=1)
        self.assertTrue(document.bboxes)

    @pytest.mark.xfail(reason='We now only log a warning because Azure OCR sometimes returns bboxes without height.')
    def test_document_check_bbox_zero_height_allowed(self):
        """Test bbox check with zero height without strict validation."""
        document_bbox = {
            '0': {'x0': 0, 'x1': 2, 'y0': 0, 'y1': 0, 'top': 10, 'bottom': 11, 'page_number': 1, 'text': 'h'}
        }
        document = Document(project=self.project, category=self.category, text='h', bbox=document_bbox)
        _ = Page(id_=1, number=1, original_size=(595.2, 841.68), document=document, start_offset=0, end_offset=1)
        self.assertTrue(document.bboxes)

    @pytest.mark.xfail(reason='We now only log a warning because Azure OCR sometimes returns bboxes without height.')
    def test_document_check_bbox_zero_height_strict_validation(self):
        """Test bbox check with zero height with strict validation, which does not allow it."""
        document_bbox = {
            '0': {'x0': 0, 'x1': 2, 'y0': 0, 'y1': 0, 'top': 10, 'bottom': 11, 'page_number': 1, 'text': 'h'}
        }
        document = Document(
            project=self.project,
            category=self.category,
            text='h',
            bbox=document_bbox,
            bbox_validation_type=BboxValidationTypes.STRICT,
        )
        _ = Page(id_=1, number=1, original_size=(595.2, 841.68), document=document, start_offset=0, end_offset=1)
        with pytest.raises(ValueError, match='has no height'):
            document.bboxes

    def test_document_check_bbox_zero_width_allowed(self):
        """Test bbox check with zero width without strict validation."""
        document_bbox = {
            '0': {'x0': 0, 'x1': 0, 'y0': 0, 'y1': 2, 'top': 10, 'bottom': 11, 'page_number': 1, 'text': 'h'}
        }
        document = Document(project=self.project, category=self.category, text='h', bbox=document_bbox)
        _ = Page(id_=1, number=1, original_size=(595.2, 841.68), document=document, start_offset=0, end_offset=1)
        self.assertTrue(document.bboxes)

    @pytest.mark.xfail(reason='We now only log a warning because Azure OCR sometimes returns bboxes without width.')
    def test_document_check_bbox_zero_width_strict_validation(self):
        """Test bbox check with zero width with strict validation, which does not allow it."""
        document_bbox = {
            '0': {'x0': 0, 'x1': 0, 'y0': 0, 'y1': 2, 'top': 10, 'bottom': 11, 'page_number': 1, 'text': 'h'}
        }
        document = Document(
            project=self.project,
            category=self.category,
            text='h',
            bbox=document_bbox,
            bbox_validation_type=BboxValidationTypes.STRICT,
        )
        _ = Page(id_=1, number=1, original_size=(595.2, 841.68), document=document, start_offset=0, end_offset=1)
        with pytest.raises(ValueError, match='has no width'):
            document.bboxes

    def test_docs_with_same_bbox_hash(self):
        """Test that bbox insertion order doesn't change the hash of the bboxes in a document."""
        document1_bbox = {
            '0': {'x0': 0, 'x1': 1, 'y0': 0, 'y1': 2, 'top': 10, 'bottom': 11, 'page_number': 1, 'text': 'h'},
            '1': {'x0': 1, 'x1': 2, 'y0': 1, 'y1': 3, 'top': 10, 'bottom': 11, 'page_number': 1, 'text': 'e'},
        }
        document1 = Document(project=self.project, category=self.category, text='hello', bbox=document1_bbox)
        _ = Page(id_=1, number=1, original_size=(595.2, 841.68), document=document1, start_offset=0, end_offset=1)
        document1.set_text_bbox_hashes()
        document2_bbox = {
            '1': {'x0': 1, 'x1': 2, 'y0': 1, 'y1': 3, 'top': 10, 'bottom': 11, 'page_number': 1, 'text': 'e'},
            '0': {'x0': 0, 'x1': 1, 'y0': 0, 'y1': 2, 'top': 10, 'bottom': 11, 'page_number': 1, 'text': 'h'},
        }
        document2 = Document(project=self.project, category=self.category, text='hello', bbox=document2_bbox)
        _ = Page(id_=1, number=1, original_size=(595.2, 841.68), document=document2, start_offset=0, end_offset=1)
        document2.set_text_bbox_hashes()
        assert document1._bbox_hash == document2._bbox_hash

    def test_document_text_modified(self):
        """Test that we can detect changes in the text of a document."""
        document_bbox = {
            '0': {'x0': 0, 'x1': 1, 'y0': 0, 'y1': 2, 'top': 10, 'bottom': 11, 'page_number': 1, 'text': 'h'}
        }
        document = Document(
            project=self.project,
            category=self.category,
            text='hello',
            bbox=document_bbox,
            bbox_validation_type=BboxValidationTypes.STRICT,
        )
        _ = Page(id_=1, number=1, original_size=(595.2, 841.68), document=document, start_offset=0, end_offset=1)
        self.assertTrue(document.text)
        document.set_text_bbox_hashes()
        self.assertFalse(document._check_text_or_bbox_modified())
        document._text = '123' + document.text
        self.assertTrue(document._check_text_or_bbox_modified())

    def test_document_bbox_modified(self):
        """Test that we can detect changes in the bboxes of a Document."""
        document_bbox = {
            '0': {'x0': 0, 'x1': 1, 'y0': 0, 'y1': 2, 'top': 10, 'bottom': 11, 'page_number': 1, 'text': 'h'}
        }
        document = Document(
            project=self.project,
            category=self.category,
            text='hello',
            bbox=document_bbox,
            bbox_validation_type=BboxValidationTypes.STRICT,
        )
        page = Page(id_=1, number=1, original_size=(595.2, 841.68), document=document, start_offset=0, end_offset=1)
        self.assertTrue(document.bboxes)
        document.set_text_bbox_hashes()
        self.assertFalse(document._check_text_or_bbox_modified())
        document._characters[1] = Bbox(x0=1, x1=2, y0=1, y1=3, page=page, validation=BboxValidationTypes.STRICT)
        self.assertTrue(document._check_text_or_bbox_modified())

    def test_document_spans(self):
        """Test getting spans from a Document."""
        document = Document(project=self.project, category=self.category, text='p\n1\fnap2')
        span1 = Span(start_offset=0, end_offset=1)
        span2 = Span(start_offset=2, end_offset=3)
        span3 = Span(start_offset=4, end_offset=5)
        span4 = Span(start_offset=6, end_offset=8)

        _ = Annotation(
            document=document, is_correct=True, label=self.label, label_set=self.label_set, spans=[span1, span2]
        )
        _ = Annotation(document=document, is_correct=False, label=self.label, label_set=self.label_set, spans=[span3])
        _ = Annotation(document=document, is_correct=True, label=self.label, label_set=self.label_set, spans=[span4])

        assert len(document.spans()) == 4
        assert len(document.spans(use_correct=True)) == 3
        assert len(document.spans(start_offset=0, end_offset=4)) == 2
        assert len(document.spans(fill=True)) == 7
        assert len(document.spans(start_offset=4, end_offset=8, fill=True)) == 3

    def test_page_width(self):
        """Test width of Page."""
        document_bbox = {
            '0': {'x0': 0, 'x1': 1, 'y0': 0, 'y1': 2, 'top': 10, 'bottom': 11, 'page_number': 1, 'text': 'h'}
        }
        document = Document(project=self.project, category=self.category, text='h', bbox=document_bbox)
        _ = Page(id_=1, number=1, original_size=(595.2, 841.68), document=document, start_offset=0, end_offset=1)
        assert document.get_page_by_index(0).width == 595.2

    def test_page_height(self):
        """Test height of Page."""
        document_bbox = {
            '0': {'x0': 0, 'x1': 1, 'y0': 0, 'y1': 2, 'top': 10, 'bottom': 11, 'page_number': 1, 'text': 'h'}
        }
        document = Document(project=self.project, category=self.category, text='h', bbox=document_bbox)
        _ = Page(id_=1, number=1, original_size=(595.2, 841.68), document=document, start_offset=0, end_offset=1)
        assert document.get_page_by_index(0).height == 841.68

    def test_page_text(self):
        """Test text Page."""
        document_bbox = {
            '0': {'x0': 0, 'x1': 1, 'y0': 0, 'y1': 2, 'top': 10, 'bottom': 11, 'page_number': 1, 'text': 'p'}
        }
        document = Document(project=self.project, category=self.category, text='page1\fpage2', bbox=document_bbox)
        _ = Page(id_=1, number=1, original_size=(595.2, 841.68), document=document, start_offset=0, end_offset=5)
        _ = Page(id_=2, number=2, original_size=(595.2, 841.68), document=document, start_offset=6, end_offset=11)
        assert document.get_page_by_index(0).text == 'page1'
        assert document.get_page_by_index(1).text == 'page2'

    def test_page_text_without_specifying_offsets(self):
        """Test text Page when start and end offsets are implicitly calculated from the Document's text page breaks."""
        document_bbox = {
            '0': {'x0': 0, 'x1': 1, 'y0': 0, 'y1': 2, 'top': 10, 'bottom': 11, 'page_number': 1, 'text': 'p'}
        }
        document = Document(
            project=self.project, category=self.category, text='page1\fpage2\fpage3', bbox=document_bbox
        )
        _ = Page(id_=1, number=1, original_size=(595.2, 841.68), document=document)
        _ = Page(id_=2, number=2, original_size=(595.2, 841.68), document=document)
        _ = Page(id_=3, number=3, original_size=(595.2, 841.68), document=document)
        assert document.get_page_by_index(0).text == 'page1'
        assert document.get_page_by_index(1).text == 'page2'
        assert document.get_page_by_index(2).text == 'page3'

    def test_page_text_offsets(self):
        """Test text Page offsets."""
        document_bbox = {
            '0': {'x0': 0, 'x1': 1, 'y0': 0, 'y1': 2, 'top': 10, 'bottom': 11, 'page_number': 1, 'text': 'p'}
        }
        document = Document(project=self.project, category=self.category, text='page1\fpage2', bbox=document_bbox)
        page1 = Page(id_=1, number=1, original_size=(595.2, 841.68), document=document, start_offset=0, end_offset=5)
        page2 = Page(id_=2, number=2, original_size=(595.2, 841.68), document=document, start_offset=6, end_offset=11)
        assert page1.text == document.text[page1.start_offset : page1.end_offset]
        assert page2.text == document.text[page2.start_offset : page2.end_offset]

    def test_page_text_offsets_without_specifying_offsets(self):
        """Test Page offsets when implicitly calculated from the Document's text page breaks."""
        document_bbox = {
            '0': {'x0': 0, 'x1': 1, 'y0': 0, 'y1': 2, 'top': 10, 'bottom': 11, 'page_number': 1, 'text': 'p'}
        }
        document = Document(
            project=self.project, category=self.category, text='page1\fpage2\fpage3', bbox=document_bbox
        )
        page1 = Page(id_=1, number=1, original_size=(595.2, 841.68), document=document)
        page2 = Page(id_=2, number=2, original_size=(595.2, 841.68), document=document)
        page3 = Page(id_=3, number=3, original_size=(595.2, 841.68), document=document)
        assert page1.text == document.text[page1.start_offset : page1.end_offset]
        assert page2.text == document.text[page2.start_offset : page2.end_offset]
        assert page3.text == document.text[page3.start_offset : page3.end_offset]

    def test_page_get_bbox(self):
        """Test getting bbox for Page."""
        document_bbox = {
            '0': {'x0': 0, 'x1': 1, 'y0': 0, 'y1': 2, 'top': 10, 'bottom': 11, 'page_number': 1, 'text': 'p'},
            '2': {'x0': 1, 'x1': 0, 'y0': 0, 'y1': 2, 'top': 10, 'bottom': 11, 'page_number': 1, 'text': '1'},
            '8': {'x0': 0, 'x1': 1, 'y0': 10, 'y1': 12, 'top': 10, 'bottom': 11, 'page_number': 2, 'text': 'p'},
            '10': {'x0': 1, 'x1': 0, 'y0': 10, 'y1': 12, 'top': 10, 'bottom': 11, 'page_number': 2, 'text': '2'},
        }
        document = Document(project=self.project, category=self.category, text='p1\fp2', bbox=document_bbox)
        page1 = Page(id_=1, number=1, original_size=(595.2, 841.68), document=document, start_offset=0, end_offset=2)
        page2 = Page(id_=2, number=2, original_size=(595.2, 841.68), document=document, start_offset=3, end_offset=5)
        assert '0' in page1.get_bbox() and '2' in page1.get_bbox()
        assert '8' in page2.get_bbox() and '10' in page2.get_bbox()
        assert '0' not in page2.get_bbox() and '2' not in page2.get_bbox()
        assert '8' not in page1.get_bbox() and '10' not in page1.get_bbox()

    def test_page_get_bbox_without_specifying_offsets(self):
        """Test getting bbox for Page when offsets are implicitly calculated from the Document's text page breaks."""
        document_bbox = {
            '0': {'x0': 0, 'x1': 1, 'y0': 0, 'y1': 2, 'top': 10, 'bottom': 11, 'page_number': 1, 'text': 'p'},
            '2': {'x0': 1, 'x1': 0, 'y0': 0, 'y1': 2, 'top': 10, 'bottom': 11, 'page_number': 1, 'text': '1'},
            '8': {'x0': 0, 'x1': 1, 'y0': 10, 'y1': 12, 'top': 10, 'bottom': 11, 'page_number': 2, 'text': 'p'},
            '10': {'x0': 1, 'x1': 0, 'y0': 10, 'y1': 12, 'top': 10, 'bottom': 11, 'page_number': 2, 'text': '2'},
        }
        document = Document(project=self.project, category=self.category, text='p1\fp2', bbox=document_bbox)
        page1 = Page(id_=1, number=1, original_size=(595.2, 841.68), document=document)
        page2 = Page(id_=2, number=2, original_size=(595.2, 841.68), document=document)
        assert '0' in page1.get_bbox() and '2' in page1.get_bbox()
        assert '8' in page2.get_bbox() and '10' in page2.get_bbox()
        assert '0' not in page2.get_bbox() and '2' not in page2.get_bbox()
        assert '8' not in page1.get_bbox() and '10' not in page1.get_bbox()

    def test_page_annotations(self):
        """Test getting Annotations of a Page."""
        document = Document(project=self.project, category=self.category, text='p\n1\fnap2')
        span1 = Span(start_offset=0, end_offset=1)
        span2 = Span(start_offset=2, end_offset=3)
        span3 = Span(start_offset=7, end_offset=9)

        page1 = Page(id_=1, number=1, original_size=(595.2, 841.68), document=document, start_offset=0, end_offset=3)
        page2 = Page(id_=2, number=2, original_size=(595.2, 841.68), document=document, start_offset=4, end_offset=8)

        annotation1 = Annotation(
            document=document, is_correct=True, label=self.label, label_set=self.label_set, spans=[span1, span2]
        )
        annotation2 = Annotation(
            document=document, is_correct=True, label=self.label, label_set=self.label_set, spans=[span3]
        )
        assert document.get_page_by_index(0).text == 'p\n1'
        assert document.get_page_by_index(1).text == 'nap2'
        assert annotation1 in document.annotations()
        assert annotation2 in document.annotations()
        assert annotation1 in page1.annotations()
        assert annotation2 in page2.annotations()
        assert annotation1 not in page2.annotations()
        assert annotation2 not in page1.annotations()
        assert page2.annotations(start_offset=4, end_offset=6) == []
        assert len(page2.annotations(start_offset=4, end_offset=6, fill=True)) == 1

    def test_page_spans(self):
        """Test getting spans from a Page."""
        document = Document(project=self.project, category=self.category, text='p\n1\fnap2')
        span1 = Span(start_offset=0, end_offset=1)
        span2 = Span(start_offset=2, end_offset=3)
        span3 = Span(start_offset=4, end_offset=5)
        span4 = Span(start_offset=6, end_offset=8)

        page1 = Page(id_=1, number=1, original_size=(595.2, 841.68), document=document, start_offset=0, end_offset=3)
        page2 = Page(id_=2, number=2, original_size=(595.2, 841.68), document=document, start_offset=4, end_offset=8)

        _ = Annotation(
            document=document, is_correct=True, label=self.label, label_set=self.label_set, spans=[span1, span2]
        )
        _ = Annotation(document=document, is_correct=False, label=self.label, label_set=self.label_set, spans=[span3])
        _ = Annotation(document=document, is_correct=True, label=self.label, label_set=self.label_set, spans=[span4])

        assert len(page1.spans()) == 2
        assert len(page2.spans()) == 2
        assert len(page2.spans(start_offset=7, end_offset=8)) == 1
        assert len(page2.spans(use_correct=True)) == 1
        page_2_spans = page2.spans(fill=True)
        assert len(page_2_spans) == 3
        filled_span = page_2_spans[1]
        assert filled_span.annotation.label.name == 'NO_LABEL'
        assert document.text[filled_span.start_offset : filled_span.end_offset] == 'a'

    def test_document_check_bbox_invalid_height_coordinates(self):
        """Test bbox check with invalid x coordinates regarding the Page height."""
        document_bbox = {
            '0': {'x0': 1, 'x1': 0, 'y0': 0, 'y1': 2, 'top': 10, 'bottom': 11, 'page_number': 1, 'text': 'h'}
        }
        document = Document(project=self.project, category=self.category, text='h', bbox=document_bbox)
        _ = Page(id_=1, number=1, original_size=(595.2, 841.68), document=document, start_offset=0, end_offset=1)
        with pytest.raises(ValueError, match='has negative width'):
            document.bboxes

    def test_bypass_document_check_bbox_invalid_height_coordinates(self):
        """Test bypassing bbox check with invalid x coordinates regarding the page height."""
        document_bbox = {
            '0': {'x0': 1, 'x1': 0, 'y0': 0, 'y1': 2, 'top': 10, 'bottom': 11, 'page_number': 1, 'text': 'h'}
        }
        document = Document(
            project=self.project,
            category=self.category,
            text='h',
            bbox=document_bbox,
            bbox_validation_type=BboxValidationTypes.DISABLED,
        )
        _ = Page(id_=1, number=1, original_size=(595.2, 841.68), document=document, start_offset=0, end_offset=1)
        document.bboxes

    def test_document_check_duplicated_annotations(self):
        """Test Annotations check when an error is raised due to duplicated Annotations by get_annotations."""

        # overwriting get_annotations for test
        class DocumentDuplicatedAnnotations(Document):
            def get_annotations(self):
                raise ValueError('is a duplicate of.')

        document = DocumentDuplicatedAnnotations(project=self.project, category=self.category, text='hello')
        self.assertFalse(document.check_annotations())

    def test_document_check_category_annotations(self):
        """Test Annotations check when an error is raised due to an incorrect Category by get_annotations."""

        # overwriting get_annotations for test
        class DocumentIncorrectCategoryAnnotations(Document):
            def get_annotations(self):
                raise ValueError('related to.')

        document = DocumentIncorrectCategoryAnnotations(project=self.project, category=self.category, text='hello')
        self.assertFalse(document.check_annotations())

    def test_to_there_must_not_be_a_folder(self):
        """Check that a virtual Document has no folder."""
        assert not os.path.isdir(self.document.document_folder)

    def test_new_annotation_in_annotation_set_of_document_of_add_foreign_annotation_set(self):
        """Add new Annotation to a Document, when the AnnotationSet is not part of the same Document."""
        project = Project(id_=None)
        document = Document(project=project, category=self.category)
        span = Span(start_offset=1, end_offset=2)

        with self.assertRaises(IndexError) as context:
            _ = Annotation(
                document=document,
                is_correct=True,
                label=self.label,
                annotation_set=self.annotation_set,
                label_set=self.label_set,
                spans=[span],
            )
            assert 'Annotation Set None is not part of Document None' in context.exception

    def test_new_annotation_in_document(self):
        """Add new Annotation to a Document."""
        project = Project(id_=None)
        document = Document(project=project, category=self.category)
        span = Span(start_offset=1, end_offset=2)
        annotation_set = AnnotationSet(document=document, label_set=self.label_set)

        annotation = Annotation(
            document=document,
            is_correct=True,
            label=self.label,
            annotation_set=annotation_set,
            label_set=self.label_set,
            spans=[span],
        )

        assert annotation in document.annotations()

    def test_new_annotation_in_document_with_confidence_zero(self):
        """Add new Annotation to a Document with confidence of 0.0."""
        project = Project(id_=None)
        document = Document(project=project, category=self.category)
        span = Span(start_offset=1, end_offset=2)
        annotation_set = AnnotationSet(document=document, label_set=self.label_set)

        annotation = Annotation(
            document=document,
            label=self.label,
            annotation_set=annotation_set,
            label_set=self.label_set,
            spans=[span],
            confidence=0.0,
        )

        assert annotation in document.annotations(use_correct=False)

    def test_new_annotation_in_annotation_set_of_document(self):
        """Add new Annotation to a Document."""
        project = Project(id_=None)
        document = Document(project=project, category=self.category)
        span = Span(start_offset=1, end_offset=2)
        annotation_set = AnnotationSet(document=document, label_set=self.label_set)

        annotation = Annotation(
            document=document,
            is_correct=True,
            label=self.label,
            annotation_set=annotation_set,
            label_set=self.label_set,
            spans=[span],
        )

        assert annotation in annotation_set.annotations()

    def test_create_document_with_page_object(self):
        """Create a Document with Pages information from a Page object."""
        document = Document(project=self.project, category=self.category, text='a')
        page_list = [{'id_': 1, 'number': 1, 'original_size': [595.2, 841.68]}]
        page = Page(**page_list[0], document=document, start_offset=0, end_offset=1)

        assert len(document.pages()) == len(page_list)
        assert page.image is None
        assert page.number == 1
        assert page.width == 595.2
        assert page.category == self.category

    def test_create_new_annotation_set_in_document(self):
        """Add new Annotation Set to a Document."""
        document = Document(project=self.project, category=self.category)
        annotation_set = AnnotationSet(document=document, label_set=self.label_set)
        assert annotation_set in document.annotation_sets()
        assert annotation_set in document.annotation_sets(label_set=self.label_set)

    def test_annotation_set_start_end_offset_and_line_index(self):
        """Test AnnotationSet info methods."""
        project = Project(id_=None)
        document = Document(project=project, category=self.category, text='l1\nl2\nl3\nl4')
        annotation_set = AnnotationSet(document=document, label_set=self.label_set)

        span1 = Span(start_offset=3, end_offset=5)
        annotation1 = Annotation(
            document=document,
            is_correct=True,
            label=self.label,
            annotation_set=annotation_set,
            label_set=self.label_set,
            spans=[span1],
        )
        assert span1.offset_string == 'l2'

        span2 = Span(start_offset=6, end_offset=8)
        annotation2 = Annotation(
            document=document,
            is_correct=False,
            confidence=0.2,
            label=self.label,
            annotation_set=annotation_set,
            label_set=self.label_set,
            spans=[span2],
        )
        assert span2.offset_string == 'l3'

        span3 = Span(start_offset=9, end_offset=11)
        annotation3 = Annotation(
            document=document,
            is_correct=False,
            confidence=0.05,
            label=self.label,
            annotation_set=annotation_set,
            label_set=self.label_set,
            spans=[span3],
        )
        assert span3.offset_string == 'l4'

        assert annotation1 in annotation_set.annotations()
        assert annotation2 not in annotation_set.annotations()
        assert annotation2 in annotation_set.annotations(use_correct=False)
        assert annotation3 not in annotation_set.annotations()
        assert annotation3 in annotation_set.annotations(use_correct=False, ignore_below_threshold=False)
        assert annotation_set.start_line_index == 1
        assert annotation_set.end_line_index == 2
        assert annotation_set.start_offset == 3
        assert annotation_set.end_offset == 8

    def test_to_add_two_spans_to_annotation(self):
        """Add one Span to one Annotation."""
        document = Document(project=self.project, category=self.category)
        span = Span(start_offset=1, end_offset=2)
        with self.assertRaises(ValueError) as context:
            Annotation(document=document, spans=[span, span], label=self.label, label_set=self.label_set)
            assert 'is a duplicate and will not be added' in context.exception

    def test_to_add_annotation_set_of_another_document(self):
        """One Annotation Set must only belong to one Document."""
        document = Document(project=self.project, category=self.category)
        with self.assertRaises(ValueError):
            document.add_annotation_set(self.annotation_set)

    def test_to_add_annotation_to_none_category_document(self):
        """A Document with Category NO_CATEGORY must not contain Annotations."""
        document = Document(project=self.project)
        annotation_set = AnnotationSet(document=document, label_set=self.label_set)

        # Add annotation for the first time
        span = Span(start_offset=1, end_offset=2)
        with pytest.raises(ValueError, match='where the Category is'):
            _ = Annotation(
                document=document,
                is_correct=True,
                label=self.label,
                annotation_set=annotation_set,
                label_set=self.label_set,
                spans=[span],
            )

    def test_add_overlapping_virtual_annotations(self):
        """Add one Span as Annotation multiple times when document.id_ is None."""
        document = Document(project=self.project, category=self.category, data_file_name='add_twice.pdf')
        span = Span(start_offset=1, end_offset=2)
        annotation_set = AnnotationSet(document=document, label_set=self.label_set)
        Annotation(
            document=document,
            is_correct=True,
            label=self.label,
            annotation_set=annotation_set,
            label_set=self.label_set,
            spans=[span],
        )

        # Add annotation for the second time, heere it should be skipped.
        span = Span(start_offset=1, end_offset=2)
        with pytest.raises(ValueError, match='is a duplicate of'):
            Annotation(
                document=document,
                is_correct=True,
                label=self.label,
                annotation_set=annotation_set,
                label_set=self.label_set,
                spans=[span],
            )

    def test_force_add_overlapping_virtual_annotations(self):
        """Add one Span as Annotation multiple times by disabling Project level data validations."""
        self.project._strict_data_validation = False
        document = Document(project=self.project, category=self.category, data_file_name='add_twice.pdf')
        span = Span(start_offset=1, end_offset=2)
        annotation_set = AnnotationSet(document=document, label_set=self.label_set)
        Annotation(
            document=document,
            is_correct=True,
            label=self.label,
            annotation_set=annotation_set,
            label_set=self.label_set,
            spans=[span],
        )

        # Force add annotation for the second time
        span = Span(start_offset=1, end_offset=2)
        Annotation(
            document=document,
            is_correct=True,
            label=self.label,
            annotation_set=annotation_set,
            label_set=self.label_set,
            spans=[span],
        )
        self.project._strict_data_validation = True  # restore validation rules to not interfere with other tests

    def test_to_add_an_annotation_twice_to_a_document(self):
        """Test to add the same Annotation twice to a Document."""
        document = Document(project=self.project, category=self.category)
        span = Span(start_offset=1, end_offset=2)
        annotation = Annotation(document=document, spans=[span], label=self.label, label_set=self.label_set)
        with self.assertRaises(ValueError):
            document.add_annotation(annotation)
        self.assertEqual([annotation], document.annotations(use_correct=False))

    def test_to_add_annotation_with_same_span_offsets_and_label_to_a_document(self):
        """Test to add Annotation with a Span with the same offsets and same Label and Label Set to a Document."""
        document = Document(project=self.project, category=self.category)
        span_1 = Span(start_offset=1, end_offset=2)
        _ = Annotation(id_=1, document=document, spans=[span_1], label=self.label, label_set=self.label_set)
        span_2 = Span(start_offset=1, end_offset=2)
        assert span_1 == span_2
        with self.assertRaises(ValueError):
            _ = Annotation(id_=2, document=document, spans=[span_2], label=self.label, label_set=self.label_set)

    def test_to_add_annotation_with_same_span_offsets_but_different_label_to_a_document(self):
        """
        Test to add Annotation with a Span with the same offsets but different Label to a Document.

        Both Annotations have is_correct=False.
        """
        document = Document(project=self.project, category=self.category)
        label = Label(project=self.project, text='Second Offline Label', label_sets=[self.label_set])
        span_1 = Span(start_offset=1, end_offset=2)
        _ = Annotation(id_=1, document=document, spans=[span_1], label=self.label, label_set=self.label_set)
        span_2 = Span(start_offset=1, end_offset=2)
        _ = Annotation(id_=2, document=document, spans=[span_2], label=label, label_set=self.label_set)

    def test_to_add_two_annotations_to_a_document(self):
        """Test to add the same Annotation twice to a Document."""
        document = Document(project=self.project, category=self.category)
        first_span = Span(start_offset=1, end_offset=2)
        second_span = Span(start_offset=1, end_offset=2)
        third_span = Span(start_offset=2, end_offset=3)
        first_annotation = Annotation(document=document, spans=[first_span], label_set=self.label_set, label=self.label)
        second_annotation = Annotation(
            document=document, spans=[second_span, third_span], label_set=self.label_set, label=self.label
        )
        self.assertEqual([first_annotation, second_annotation], document.annotations(use_correct=False))

    def test_to_return_a_custom_offset_string(self):
        """Test to return a offset string which was human edited on the Server."""
        document = Document(project=self.project, category=self.category)
        first_span = Span(start_offset=1, end_offset=2)
        annotation = Annotation(
            document=document,
            spans=[first_span],
            label_set=self.label_set,
            label=self.label,
            is_correct=True,
            custom_offset_string=True,
            offset_string='custom_string',
        )
        assert annotation.offset_string == 'custom_string'

    def test_to_add_a_correct_annotation_with_a_duplicated_span_to_a_document(self):
        """Test to Span that has the same start and end offsets to a correct Annotation.

        24.06.2022: It's now allowed to have this operation. As one Annotation spanning only one Span is not
        identical with another Annotation with the same label but one additional Span.

        Example:
            A Document contains the text "My name is Manfred Meister": It should be possible to have one Annotation
            with Name: Span: "Manfred" and one Annotation with Name: Span "Manfred" and Span "Müller" as both
            Annotation should have a different confidence.

        """
        document = Document(project=self.project, category=self.category)
        first_span = Span(start_offset=1, end_offset=2)
        second_span = Span(start_offset=1, end_offset=2)
        third_span = Span(start_offset=2, end_offset=3)
        _ = Annotation(
            document=document, spans=[first_span], label_set=self.label_set, label=self.label, is_correct=True
        )

        _ = Annotation(
            document=document,
            spans=[second_span, third_span],
            label_set=self.label_set,
            label=self.label,
            is_correct=True,
        )

        # todo: check if Spans are related to the Document and Annotations are just linked where one Span can relate to
        #    many Annotations.
        # with self.assertRaises(ValueError) as context:
        #    assert 'Span can relate to multiple Annotations but is unique in a Document' in context.exception

    def test_to_reuse_spans_across_correct_annotations(self):
        """Test if we find inconsistencies when one Span is assigned to a new correct Annotation."""
        document = Document(project=self.project, category=self.category)
        first_span = Span(start_offset=1, end_offset=2)
        second_span = Span(start_offset=2, end_offset=3)
        Annotation(document=document, spans=[first_span], label_set=self.label_set, label=self.label, is_correct=True)

        Annotation(
            document=document,
            spans=[first_span, second_span],
            label_set=self.label_set,
            label=self.label,
            is_correct=True,
        )

    def test_to_reuse_spans_across_annotations(self):
        """Test if we find inconsistencies when one Span is assigned to a new Annotation."""
        document = Document(project=self.project, category=self.category)
        first_span = Span(start_offset=1, end_offset=2)
        second_span = Span(start_offset=2, end_offset=3)
        Annotation(document=document, spans=[first_span], label_set=self.label_set, label=self.label)
        Annotation(document=document, spans=[first_span, second_span], label_set=self.label_set, label=self.label)
        assert len(document.annotations(use_correct=False)) == 2

    def test_lose_weight(self):
        """Lose weight should remove session and Documents."""
        project = Project(id_=None)
        _ = Category(project=project)
        label_set = LabelSet(project=project)
        Label(project=project, label_sets=[label_set])
        project.lose_weight()
        assert project.session is None
        assert project.categories[0].session is None
        assert project.label_sets[0].session is None
        assert project.labels[0].session is None
        assert project.labels[0]._evaluations == {}
        assert project.labels[0]._tokens == {}
        assert project.labels[0]._regex == {}
        assert project._documents == []
        assert project.virtual_documents == []
        assert project.documents == []
        assert project.test_documents == []
        assert project._meta_data == []

    def test_create_subdocument_from_page_range(self):
        """Test creating a smaller Document from original one within a Page range."""
        project = LocalTextProject()
        test_document = project.get_document_by_id(9)
        # Set page categories so we can check if they are copied correctly
        for page in test_document.pages():
            page.set_category(test_document.category)
        new_doc = test_document.create_subdocument_from_page_range(
            test_document.pages()[0], test_document.pages()[1], include=True
        )
        assert len(new_doc.pages()) == 2
        for page in new_doc.pages():
            assert page.category == test_document.category
            assert len(page.category_annotations) == 1
        assert new_doc.text == 'Hi all,\nI like bread.\n\fI hope to get everything done soon.\n'

    def test_page_is_first_attribute(self):
        """Test correct setting of Page's is_first_page attribute."""
        project = LocalTextProject()
        text = 'Sample text.'
        document = Document(id=None, project=project, text=text, dataset_status=1)
        _ = Page(
            id_=None,
            original_size=(320, 240),
            document=document,
            start_offset=0,
            end_offset=12,
            number=1,
        )
        assert _.is_first_page is True
        text = 'Sample text.\n\fSome more.'
        document = Document(id=None, project=project, text=text, dataset_status=2)
        _ = Page(
            id_=None,
            original_size=(320, 240),
            document=document,
            start_offset=0,
            end_offset=12,
            number=1,
        )
        assert _.is_first_page
        _ = Page(
            id_=None,
            original_size=(320, 240),
            document=document,
            start_offset=13,
            end_offset=23,
            number=2,
        )
        assert not _.is_first_page

    def test_bbox_rounding(self):
        """Test that Bbox coordinates are rounded correctly in the `_valid` method."""
        # Initialize a Page with a width and height of 1000
        page = Page(
            id_=1,
            number=1,
            original_size=(1000, 1000),
            document=self.document,
            start_offset=0,
            end_offset=1,
        )

        # Test a Bbox with coordinates that exceed the height and width of the document, unless rounded.
        valid_height_width = 1000.005  # round(n, 2) = 1000.0
        bbox_valid = Bbox(
            x0=valid_height_width,
            x1=valid_height_width,
            y0=valid_height_width,
            y1=valid_height_width,
            page=page,
        )
        assert bbox_valid.document is self.document
        # Validate that the `_valid` method returns None, indicating that the Bbox coordinates were correctly rounded
        self.assertIsNone(bbox_valid._valid())

        # Test a Bbox with coordinates that exceed the height and width of the document, even when rounded
        invalid_height_width = 1000.006  # round(n, 2) = 1000.01
        with self.assertRaises(ValueError):
            bbox_invalid = Bbox(
                x0=invalid_height_width,
                x1=invalid_height_width,
                y0=invalid_height_width,
                y1=invalid_height_width,
                page=page,
            )
            bbox_invalid._valid()

    def test_page_none_category(self):
        """Test that Page always has a Category and never can have a Category = None."""
        document = Document(project=self.project, text='text')
        for i in range(2):
            page = Page(id_=None, document=document, start_offset=0, end_offset=0, number=i + 1, original_size=(0, 0))
            assert page.category == self.project.no_category
            with pytest.raises(ValueError, match='forbid'):
                page.set_category(None)


class TestSeparateLabels(unittest.TestCase):
    """Test the feature create separated Labels per Label Set."""

    @unittest.skip(reason='Feature needed')
    def test_normal_setup(self):
        """Labels are initialized by the Project can be reused by Label Sets."""
        raise NotImplementedError

    @unittest.skip(reason='Feature needed')
    def test_separat_setup(self):
        """Labels are initialized by the Project cannot be reused by Label Sets."""
        raise NotImplementedError


class TestKonfuzioDataCustomPath(unittest.TestCase):
    """Test handle data."""

    def test_get_text_for_doc_needing_update(self):
        """Test to load the Project into a custom folder and only get one Document."""
        prj = Project(id_=TEST_PROJECT_ID, project_folder='my_own_data')
        doc = prj.get_document_by_id(214414)
        doc.download_document_details()
        self.assertTrue(is_file(doc.txt_file_path))
        for document in prj.documents:
            if document.id_ != doc.id_:
                self.assertTrue(not is_file(document.txt_file_path, raise_exception=False))
        self.assertTrue(doc.text)
        prj.delete()

    def test_make_sure_text_is_downloaded_automatically(self):
        """Test if a text is downloaded automatically."""
        prj = Project(id_=TEST_PROJECT_ID, project_folder='my_own_data')
        doc = prj.get_document_by_id(214414)
        self.assertFalse(is_file(doc.txt_file_path, raise_exception=False))
        self.assertEqual(None, doc._text)
        self.assertTrue(doc.text)
        self.assertTrue(is_file(doc.txt_file_path))
        prj.delete()


class TestKonfuzioOneVirtualTwoRealCategories(unittest.TestCase):
    """Test handle data."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the test Project."""
        cls.project = Project(id_=None, project_folder=OFFLINE_PROJECT)
        category = Category(project=cls.project, name_clean='Virtual Category')
        label = Label(name='Only virtual Category Label', project=cls.project)
        _ = LabelSet(project=cls.project, is_default=False, labels=[label], categories=[category])

    def test_get_labels_of_virtual_category(self):
        """Return only related Labels as Information Extraction can be trained per Category."""
        assert len(self.project.categories[-1].labels) == 1  # virtual created Categories have no NO_LABEL

    def test_get_labels_of_category(self):
        """Return only related Labels as Information Extraction can be trained per Category."""
        real_category1 = self.project.get_category_by_id(TEST_PAYSLIPS_CATEGORY_ID)
        real_category2 = self.project.get_category_by_id(TEST_RECEIPTS_CATEGORY_ID)
        # we calculate the set to avoid double counting the NO_LABEL
        assert len(set(real_category1.labels + real_category2.labels)) == len(self.project.labels) - 1


class TestKonfuzioDataSetup(unittest.TestCase):
    """Test handle data."""

    document_count = 49
    test_document_count = 6
    annotations_correct = 24
    # 24 created by human
    # https://app.konfuzio.com/admin/server/sequenceannotation/?
    # document__dataset_status__exact=2&label__id__exact=867&project=46&status=3
    # 1 Created by human and revised by human, but on a document that has no category
    # https://app.konfuzio.com/admin/server/sequenceannotation/?
    # document__dataset_status__exact=2&label__id__exact=867&project=46&status=1

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the test Project."""
        cls.prj = Project(id_=None, project_folder=OFFLINE_PROJECT)
        cls.project = Project(id_=RESTORED_PROJECT_ID, update=True)
        original_document_text = cls.prj.get_document_by_id(TEST_DOCUMENT_ID).text
        cls.test_document_id = [
            document for document in cls.project.documents if document.text == original_document_text
        ][0].id_

    def test_number_training_documents(self):
        """Test the number of Documents in dataset status Training."""
        assert len(self.prj.documents) == self.document_count

    def test_get_labels_of_category(self):
        """Return only related Labels as Information Extraction can be trained per Category."""
        # we calculate the set to avoid double counting the NO_LABEL
        assert len(set(self.prj.categories[0].labels + self.prj.categories[1].labels)) == len(self.prj.labels)

    def test_no_category_after_update(self):
        """Test that NO_CATEGORY is not lost after updating a Project."""
        self.prj = Project(id_=None, project_folder=OFFLINE_PROJECT, update=True)
        assert self.prj.no_category

    def test_document_with_no_category_has_only_no_label_annotations(self):
        """Test if we skip Annotations except for NO_LABEL in no Category Documents."""
        document = self.prj.get_document_by_id(44864)
        assert document.category.name == self.prj.no_category.name
        assert document.annotations() == []

    def test_number_test_documents(self):
        """Test the number of Documents in dataset status Test."""
        assert len(self.prj.test_documents) == self.test_document_count

    def test_number_excluded_documents(self):
        """Test the number of Documents in dataset status Excluded."""
        assert len(self.prj.excluded_documents) == 1

    def test_all_labels_have_threshold(self):
        """Test that all Labels have the attribute threshold."""
        for label in self.prj.labels:
            assert hasattr(label, 'threshold')

    def test_number_preparation_documents(self):
        """Test the number of Documents in dataset status Preparation."""
        assert len(self.prj.preparation_documents) == 0

    def test_annotation_of_label(self):
        """Test the number of Annotations across all Documents in Training."""
        label = self.prj.get_label_by_id(867)
        annotations = label.annotations(categories=[self.prj.get_category_by_id(63)])
        assert len(annotations) == self.annotations_correct

    def test_annotation_hashable(self):
        """Test if an Annotation can be hashed."""
        set(self.prj.get_document_by_id(TEST_DOCUMENT_ID).annotations())

    def test_get_all_spans_of_a_document(self):
        """Test to get all Spans in a Document."""
        # Before we had 21 Spans after the a code change to allow overlapping Annotations we have 23 Spans
        # due to the fact that one Span is not identical, so one Annotation relates to one Span.
        # One more for a total of 24 since we are not filtering out the rejected Annotations.
        assert len(self.prj.get_document_by_id(TEST_DOCUMENT_ID).spans()) == 24

    def test_span_hashable(self):
        """Test if a Span can be hashed."""
        annotation = self.prj.get_document_by_id(TEST_DOCUMENT_ID).annotations()[0]
        set(annotation.spans)

    def test_number_of_label_sets(self):
        """Test Label Sets numbers."""
        # Online Label Sets + added during tests +  no_label_set
        assert len(self.prj.label_sets) == 12

    # def test_check_tokens(self):
    #     """Test to find not matched Annotations."""
    #     category = self.prj.get_category_by_id(63)
    #     spans = self.prj.get_label_by_id(867).check_tokens(categories=[category])
    #     assert len(spans) == 25

    def test_has_multiple_annotation_sets(self):
        """Test Label Sets in the test Project."""
        assert self.prj.get_label_set_by_name('Brutto-Bezug').has_multiple_annotation_sets

    def test_has_not_multiple_annotation_sets(self):
        """Test Label Sets in the test Project."""
        assert not self.prj.get_label_set_by_name('Lohnabrechnung').has_multiple_annotation_sets

    def test_default_label_set(self):
        """Test the main Label Set incl. its Labels."""
        default_label_set = self.prj.get_label_set_by_name('Lohnabrechnung')
        assert default_label_set.labels.__len__() == 10

    def test_to_filter_annotations_by_label(self):
        """Test to get correct Annotations of a Label."""
        label = self.prj.get_label_by_id(858)
        annotations = label.annotations(categories=[self.prj.get_category_by_id(63)])
        self.assertEqual(len(annotations), self.annotations_correct + 1)

    def test_category(self):
        """Test if Category of main Label Set is initialized correctly."""
        assert len(self.prj.categories) == 2
        assert self.prj.categories[0].id_ == 63
        assert self.prj.label_sets[0].categories[0].id_ == 63

    def test_category_documents(self):
        """Test Category of Documents associated to a Category."""
        category = self.prj.get_category_by_id(63)
        category_documents = category.documents()

        assert len(category_documents) == 25
        for document in category_documents:
            assert document.category == category
            for page in document.pages():
                assert page.category == category

    def test_category_test_documents(self):
        """Test Category of Test Documents associated to a Category."""
        category = self.prj.get_category_by_id(63)
        category_test_documents = category.test_documents()

        assert len(category_test_documents) == 3
        for document in category_test_documents:
            assert document.category == category
            for page in document.pages():
                assert page.category == category

    def test_category_annotations_by_label(self):
        """Test getting Annotations of a Category by Labels."""
        category = self.prj.get_category_by_id(63)
        category_label_sets = category.label_sets
        label = category_label_sets[0].labels[0]
        for annotation in label.annotations(categories=[category]):
            if annotation.document.category is not None:
                assert annotation.document.category == category
                for page in annotation.document.pages():
                    assert page.category == category

    def test_category_annotations_by_document(self):
        """Test getting Annotations of a Category by Documents."""
        category = self.prj.get_category_by_id(63)
        for document in category.documents():
            for annotation in document.annotations():
                if not annotation.label_set.is_default:
                    assert annotation.label_set in category.label_sets

    def test_label_sets_of_category(self):
        """Test Label Sets of a Category."""
        category = self.prj.get_category_by_id(63)
        category_label_sets = category.label_sets

        assert len(category_label_sets) > 0
        for label_set in category_label_sets:
            assert category in label_set.categories

    def test_labels_of_category(self):
        """Test Labels of a Category."""
        category = self.prj.get_category_by_id(63)
        with self.assertRaises(AttributeError) as context:
            category.labels
            assert "'Category' object has no attribute 'labels'" in context.exception

    def test_label_sets_of_label(self):
        """Test Label Sets of a Label."""
        label: Label = self.prj.get_label_by_id(861)  # Lohnart
        self.assertEqual(2, len(label.label_sets))

    def test_label_set_multiple(self):
        """Test Label Set config that is set to multiple."""
        label_set = self.prj.get_label_set_by_name('Brutto-Bezug')
        assert label_set.categories.__len__() == 1

    def test_number_of_labels_of_label_set(self):
        """Test the number of Labels of the default Label Set."""
        label_set = self.prj.get_label_set_by_name('Lohnabrechnung')
        # assert label_set.categories == [self.prj.get_category_by_id(label_set.id_)]  # defines a category
        assert label_set.labels.__len__() == 10

    def test_categories(self):
        """Test get Labels in the Project."""
        assert self.prj.categories.__len__() == 2
        payslips_category = self.prj.get_category_by_id(TEST_PAYSLIPS_CATEGORY_ID)
        assert payslips_category.name == 'Lohnabrechnung'
        # We have 5 Label Sets, Lohnabrechnung is Category and a Label Set as it hold labels, however a Category
        # cannot hold Labels
        assert sorted([label_set.name for label_set in self.prj.categories[0].label_sets]) == [
            'Brutto-Bezug',
            'Lohnabrechnung',
            'NO_LABEL_SET',
            'Netto-Bezug',
            'Steuer',
            'Verdiensibescheinigung',
        ]
        receipts_category = self.prj.get_category_by_id(TEST_RECEIPTS_CATEGORY_ID)
        assert receipts_category.name == 'Quittung (GERMAN)'
        # We have 5 Label Sets, Quittung is Category and a Label Set as it hold labels, however a Category
        # cannot hold Labels
        assert sorted([label_set.name for label_set in self.prj.categories[0].label_sets]) == [
            'Brutto-Bezug',
            'Lohnabrechnung',
            'NO_LABEL_SET',
            'Netto-Bezug',
            'Steuer',
            'Verdiensibescheinigung',
        ]

    def test_label_spans(self):
        """Test get Label Spans in the Project."""
        category = self.prj.get_category_by_id(TEST_PAYSLIPS_CATEGORY_ID)
        label = self.prj.get_label_by_name('Austellungsdatum')

        assert len(label.annotations(categories=[category])) == 24
        assert len(label.spans(categories=[category])) == 25

    def test_get_images(self):
        """Test get paths to the images of the first Training Document."""
        document = self.prj.get_document_by_id(TEST_DOCUMENT_ID)
        assert len(document.pages()) == 1

    def test_get_file(self):
        """Test get path to the file of the first Training Document."""
        self.prj.documents[0].get_file()
        assert self.prj.documents[0].ocr_file_path

    def test_get_file_without_ocr(self):
        """Download file without OCR."""
        doc = self.prj.get_document_by_id(TEST_DOCUMENT_ID)
        doc.get_file(ocr_version=False)
        is_file(doc.file_path)

    def test_get_file_with_ocr(self):
        """Download file without OCR."""
        doc = self.prj.get_document_by_id(TEST_DOCUMENT_ID)
        doc.get_file()
        is_file(doc.ocr_file_path)

    def test_make_sure_annotations_are_downloaded_automatically(self):
        """Test if Annotations are downloaded automatically."""
        prj = Project(id_=RESTORED_PROJECT_ID, project_folder='another')
        doc = prj.get_document_by_id(self.test_document_id)
        self.assertFalse(is_file(doc.annotation_file_path, raise_exception=False))
        self.assertEqual(None, doc._annotations)
        self.assertTrue(doc.annotations())
        self.assertEqual(21, len(doc._annotations))
        self.assertTrue(is_file(doc.annotation_file_path))
        prj.delete()

    def test_make_sure_annotation_sets_are_downloaded_automatically(self):
        """Test if Annotation Sets are downloaded automatically."""
        prj = Project(id_=RESTORED_PROJECT_ID, project_folder='another2')
        doc = prj.get_document_by_id(self.test_document_id)
        self.assertFalse(is_file(doc.annotation_set_file_path, raise_exception=False))
        self.assertEqual(None, doc._annotation_sets)
        self.assertTrue(doc.annotation_sets())
        assert doc._annotation_sets
        self.assertTrue(is_file(doc.annotation_set_file_path))
        prj.delete()

    def test_make_sure_pages_are_downloaded_automatically(self):
        """Test if Pages are downloaded automatically."""
        prj = Project(id_=RESTORED_PROJECT_ID, project_folder='another33')
        doc = prj.get_document_by_id(self.test_document_id)
        self.assertFalse(is_file(doc.pages_file_path, raise_exception=False))
        self.assertEqual([], doc._pages)
        self.assertTrue(doc.pages())
        self.assertTrue(is_file(doc.pages_file_path))
        prj.delete()

    def test_add_label_set_without_category_to_document_with_category(self):
        """Test to add a Label Set without Category to a Document with a Category."""
        prj = Project(id_=RESTORED_PROJECT_ID)  # new init to not add data to self.prj
        doc = prj.get_document_by_id(self.test_document_id)
        label_set = LabelSet(project=prj)
        label = Label(project=prj, label_sets=[label_set])
        with self.assertRaises(ValueError) as context:
            Annotation(document=doc, label_set=label_set, label=label)
            assert 'uses Label Set without Category' in context.exception

    def test_get_annotations_set_without_category_to_document_with_category(self):
        """Test to add a Label Set without Category to a Document with a Category."""
        document = Document.from_file(path='tests/test_data/textposition.pdf', project=self.project)
        assert document.annotations() == []

    def test_get_bbox(self):
        """Test to get BoundingBox of Text offset."""
        prj = Project(id_=RESTORED_PROJECT_ID)  # new init to not add data to self.prj
        doc = prj.get_document_by_id(self.test_document_id)
        doc.update()
        assert doc.category
        label_set = LabelSet(project=prj, categories=[doc.category])
        label = Label(project=prj, label_sets=[label_set])
        span = Span(start_offset=1, end_offset=2)
        annotation = Annotation(document=doc, label_set=label_set, label=label, spans=[span])
        span = Span(start_offset=44, end_offset=65, annotation=annotation)
        # only Character 60, 61 and 62 provide bounding boxes, all others are None
        span.bbox()
        self.assertEqual(span.page.index, 0)
        self.assertEqual(span.line_index, 0)
        self.assertEqual(span.bbox().x0, 426.0)
        self.assertEqual(span.bbox().x1, 442.8)
        self.assertEqual(span.bbox().y0, 808.831)
        self.assertEqual(span.bbox().y1, 817.831)
        self.assertEqual(span.bbox().area, 151.2)

    def test_size_of_project(self):
        """Test size of Project and compare it to the size after Documents have been loaded."""
        import sys
        from gc import get_referents
        from types import FunctionType, ModuleType

        # Custom objects know their class.
        # Function objects seem to know way too much, including modules.
        # Exclude modules as well.
        BLACKLIST = type, ModuleType, FunctionType

        def _getsize(obj):
            """Sum size of object & members. From https://stackoverflow.com/a/30316760."""
            if isinstance(obj, BLACKLIST):
                raise TypeError('getsize() does not take argument of type: ' + str(type(obj)))
            seen_ids = set()
            size = 0
            objects = [obj]
            while objects:
                need_referents = []
                for obj in objects:
                    if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                        seen_ids.add(id(obj))
                        size += sys.getsizeof(obj)
                        need_referents.append(obj)
                objects = get_referents(*need_referents)
            return size

        # start of test
        prj = Project(id_=RESTORED_PROJECT_ID)
        before = _getsize(prj)
        for document in prj.documents:
            document.text
        after = _getsize(prj)
        assert 1.6 < after / before < 5
        assert after < 610000

        # strings in prj take slightly less space than in a list
        assert _getsize([doc.text for doc in prj.documents]) + before < after + 500

        # the text of the document is the only thing causing the size difference
        for document in prj.documents:
            document._text = None
        assert _getsize(prj) == before

    def test_online_project_document_default_update_setting(self):
        """Test update setting of Document when online Project is initialized."""
        project = Project(id_=RESTORED_PROJECT_ID)
        document = project.get_document_by_id(self.test_document_id)

        assert document._update is False

        document.get_annotations()

        assert is_file(document.annotation_file_path, raise_exception=False)
        assert is_file(document.annotation_set_file_path, raise_exception=False)

        assert document._update is False
        document._update = True

        document.get_annotations()  # download again

        assert document._update is False

    def test_create_new_doc_via_text_and_bbox(self):
        """Test to create a new Document which by a text and a bbox."""
        doc = Project(id_=None, project_folder=OFFLINE_PROJECT).get_document_by_id(TEST_DOCUMENT_ID)
        new_doc = Document(project=doc.project, text=doc.text, bbox=doc.get_bbox())
        assert new_doc.text
        assert new_doc.get_bbox()
        assert new_doc.number_of_pages == 1
        assert new_doc.number_of_lines == 70

    def test_category_of_document(self):
        """Test to download a file which includes a whitespace in the name."""
        category = Project(id_=None, project_folder=OFFLINE_PROJECT).get_document_by_id(44860).category
        self.assertEqual(category.name, 'Lohnabrechnung')

    def test_category_of_document_without_category(self):
        """Test the Category of a Document without Category."""
        category = Project(id_=None, project_folder=OFFLINE_PROJECT).get_document_by_id(44864).category
        assert category.name == 'NO_CATEGORY'

    def test_get_file_with_white_colon_name(self):
        """Test to download a file which includes a whitespace in the name."""
        doc = Project(id_=None, project_folder=OFFLINE_PROJECT).get_document_by_id(44860)
        doc.get_file()

    def test_labels(self):
        """Test get Labels in the Project."""
        assert [label.name for label in sorted(self.prj.get_category_by_id(TEST_PAYSLIPS_CATEGORY_ID).labels)] == [
            'Austellungsdatum',
            'Auszahlungsbetrag',
            'Bank inkl. IBAN',
            'Betrag',
            'Bezeichnung',
            'EMPTY_LABEL',
            'Faktor',
            'Gesamt-Brutto',
            'Lohnart',
            'Menge',
            'NO_LABEL',  # Added for the Tokenizer
            'Nachname',
            'Netto-Verdienst',
            'Personalausweis',
            'Sozialversicherung',
            'Steuer-Brutto',
            'Steuerklasse',
            'Steuerrechtliche Abzüge',
            'Vorname',
        ]
        assert [label.name for label in sorted(self.prj.get_category_by_id(TEST_RECEIPTS_CATEGORY_ID).labels)] == [
            'Ansprechpartner',
            'Anzahl / Menge',
            'Artikelbezeichnung',
            'Artikelnummer',
            'Ausstelldatum',
            'Bedienung Nr',
            'BonNr',
            'Brutto (Ergebnis der MwSt. Berechnung)',
            'Einheit',
            'Einzelpreis (Brutto)',
            'Filial-/Markt-Nummer',
            'Firmenname',
            'Gesamtpreis (Brutto)',
            'Hausnummer',
            'Kassennummer',
            'Mehrwertsteuerbetrag',
            'Mehrwertsteuersatz',
            'NO_LABEL',
            'Netto (Basis der MwSt. Berechnung)',
            'Ort',
            'Postleitzahl',
            'Rechnungsnummer',
            'Referenz auf MwSt',
            'Steuernummer',
            'Straße',
            'Telefonnummer',
            'Uhrzeit',
            'Umsatzsteuer-Identifikationsnummer',
            'Währung',
            'Zahlungsmethode',
        ]

    def test_project(self):
        """Test basic properties of the Project object."""
        assert is_file(self.prj.meta_file_path)
        assert self.prj.documents[1].id_ > self.prj.documents[0].id_
        assert len(self.prj.documents)
        # check if we can initialize a new project object, which will use the same data
        assert len(self.prj.documents) == self.document_count
        new_project = Project(id_=None, project_folder=OFFLINE_PROJECT)
        assert len(new_project.documents) == self.document_count
        assert new_project.meta_file_path == self.prj.meta_file_path

    def test_update_prj(self):
        """Test number of Documents after updating a Project."""
        assert len(self.prj.documents) == self.document_count
        self.prj.get(update=True)
        assert len(self.prj.documents) == self.document_count
        is_file(self.prj.meta_file_path)

    @unittest.skip(reason='No update logic of project about new Annotation.')
    def test_annotations_in_document(self):
        """Test number and value of Annotations."""
        doc = self.prj.get_document_by_id(TEST_DOCUMENT_ID)
        assert len(doc.annotations(use_correct=False)) == 24
        assert doc.annotations()[0].offset_string == ['22.05.2018']  # start_offset=465, start_offset=466
        assert len(doc.annotations()) == 24
        assert doc.annotations()[0].is_online
        assert not doc.annotations()[0].save()  # Save returns False because Annotation is already online.

    def test_span_line_index_in_document(self):
        """Test line_index of span."""
        doc = self.prj.get_document_by_id(TEST_DOCUMENT_ID)
        label_set = self.prj.get_label_set_by_name('Lohnabrechnung')
        label = label_set.labels[0]
        span = Span(start_offset=1000, end_offset=1002)
        _ = Annotation(document=doc, label_set=label_set, label=label, spans=[span])
        assert span.page.index == 0
        assert span.line_index == 13

    def test_annotation_sets_in_document(self):
        """Test number of Annotation Sets in a specific Document in the test Project."""
        doc = self.prj.get_document_by_id(TEST_DOCUMENT_ID)
        assert len(doc.annotation_sets()) == 24  # After Update to use the TEST_DOCUMENT_ID
        default_label_set = self.prj.get_label_set_by_name('Lohnabrechnung')
        assert len(doc.annotation_sets(label_set=default_label_set)) == 1
        brutto_bezug_label_set = self.prj.get_label_set_by_name('Brutto-Bezug')
        assert len(doc.annotation_sets(label_set=brutto_bezug_label_set)) == 21  # ??

    def test_get_annotation_set_after_removal(self):
        """Test get an Annotation Set that no longer exists."""
        with self.assertRaises(IndexError) as _:
            # create annotation for a certain annotation set in a document
            doc = self.prj.get_document_by_id(TEST_DOCUMENT_ID)

            # get the annotation set ID of the first annotation
            annotations = doc.annotations()
            annotation_set_id = annotations[0].annotation_set.id_

            assert isinstance(annotation_set_id, int)

            # delete annotation set
            doc._annotation_sets = []

            # trying to get an annotation set that no longer exists
            _ = doc.get_annotation_set_by_id(annotation_set_id)

    def test_name_of_category(self):
        """Test the name of the Category."""
        doc = self.prj.get_document_by_id(TEST_DOCUMENT_ID)
        assert doc.category.name == 'Lohnabrechnung'

    def test_assignee_of_document(self):
        """Test assignee of a Document."""
        doc = self.prj.get_document_by_id(TEST_DOCUMENT_ID)
        assert doc.assignee == 1043  # Document has Assignee ch+test@konfuzio.com with user ID 1043

    def test_add_document_twice(self):
        """Test adding same Document twice."""
        old_doc = self.prj.get_document_by_id(44834)
        with self.assertRaises(ValueError):
            self.prj.add_document(old_doc)
        assert len(self.prj.documents) == self.document_count

    def test_correct_annotations(self):
        """Test correct Annotations of a certain Label in a specific Document."""
        doc = self.prj.get_document_by_id(TEST_DOCUMENT_ID)
        label = self.prj.get_label_by_id(867)
        assert len(doc.annotations(label=label)) == 1

    def test_annotation_start_offset_zero_filter(self):
        """Test Annotations with start offset equal to zero."""
        doc = self.prj.get_document_by_id(TEST_DOCUMENT_ID)
        assert len(doc.annotations()) == 19
        assert doc.annotations()[0].start_offset == 66

    def test_multiline_annotation(self):
        """Test to convert a multiline Span Annotation to a dict."""
        doc = self.prj.get_document_by_id(TEST_DOCUMENT_ID)
        assert len(doc.annotations()[0].eval_dict) == 2

    def test_compare_dicts(self):
        """Test to convert a Annotation to a dict."""
        annotations = self.prj.documents[0].annotations()
        for annotation in annotations:
            if annotation.id_ == 4420022:
                span = annotation.spans[0]

        empty_span = Span(start_offset=0, end_offset=0)

        assert empty_span.eval_dict().keys() == span.eval_dict().keys()

    def test_annotation_to_dict(self):
        """Test to convert a Annotation to a dict."""
        anno = None
        annotations = self.prj.documents[0].annotations()
        for annotation in annotations:
            if annotation.id_ == 4420022:
                anno = annotation.eval_dict[0]

        assert anno is not None

        assert anno['confidence'] == 1.0
        assert anno['created_by'] == 59
        assert not anno['custom_offset_string']
        assert anno['end_offset'] == 366
        assert anno['is_correct']
        assert anno['label_id'] == 860  # original REST API calls it "label" however means label_id
        assert anno['label_threshold'] == 0.1
        assert anno['custom_offset_string'] is None
        assert not anno['revised']
        assert anno['revised_by'] is None
        assert anno['annotation_set_id'] == 78730  # v2 REST API calls it still section
        assert anno['label_set_id'] == 63  # v2 REST API call it still section_label_id
        assert anno['start_offset'] == 365

        assert anno['page_width'] == 595.2
        assert anno['page_height'] == 841.68
        assert anno['x0'] == 126.96
        assert anno['x1'] == 131.04
        assert anno['y0'] == 772.589
        assert anno['y1'] == 783.589
        assert anno['x0_relative'] == 0.2133064516129032
        assert anno['x1_relative'] == 0.2201612903225806
        assert anno['y0_relative'] == 0.9179129835566963
        assert anno['y1_relative'] == 0.9309820834521435
        assert anno['line_index'] == 4
        assert anno['page_index'] == 0
        assert anno['page_index_relative'] == 0

    def test_document_annotations_filter(self):
        """Test Annotations filter."""
        doc = self.prj.get_document_by_id(TEST_DOCUMENT_ID)
        self.assertEqual(len(doc.annotations()), 19)
        assert len(doc.annotations(label=self.prj.get_label_by_id(858))) == 1
        assert len(doc.annotations(use_correct=False)) == 22  # 21 if not considering negative ones

    def test_document_offset(self):
        """Test Document offsets."""
        doc = self.prj.get_document_by_id(TEST_DOCUMENT_ID)
        assert doc.annotations()[0].offset_string == ['328927/10103', '22.05.2018']

    def test_document_id_when_loading_from_disk(self):
        """Test ID of Document."""
        doc = self.prj.get_document_by_id(TEST_DOCUMENT_ID)
        assert doc.id_ == TEST_DOCUMENT_ID

    def test_document_check_bbox_available(self):
        """Test deepcopy will copy over Bbox."""
        doc = self.prj.get_document_by_id(TEST_DOCUMENT_ID)
        virtual_doc = deepcopy(doc)
        assert virtual_doc.bboxes

    def test_document_check_bbox(self):
        """Test Bbox check."""
        doc = self.prj.get_document_by_id(TEST_DOCUMENT_ID)
        virtual_doc = deepcopy(doc)
        self.assertTrue(virtual_doc.bboxes)
        virtual_doc.set_text_bbox_hashes()
        virtual_doc._text = '123' + doc.text  # Change text to bring bbox out of sync.
        with pytest.raises(ValueError, match='Bbox provides Character "n" Document text refers to "l"'):
            virtual_doc.check_bbox()

    def test_document_check_bbox_without_validations(self):
        """Test bbox check when force disabling validation rules."""
        self.prj._strict_data_validation = False
        doc = self.prj.get_document_by_id(TEST_DOCUMENT_ID)
        virtual_doc = deepcopy(doc)
        self.assertTrue(virtual_doc.bboxes)
        virtual_doc.set_text_bbox_hashes()
        virtual_doc._text = '123' + doc.text  # Change text to bring bbox out of sync.
        virtual_doc.check_bbox()  # no exception is raised
        self.prj._strict_data_validation = True  # restore data validations to not interfere with other tests

    def test_hashing_bboxes_faster_than_recalculation(self):
        """Test that it's 100x faster to compare hashes of text and bboxes rathar than force recalculation of bboxes."""
        import time

        doc = self.prj.get_document_by_id(TEST_DOCUMENT_ID)
        virtual_doc = deepcopy(doc)
        virtual_doc.bboxes

        t0 = time.monotonic()
        for _ in range(100):
            virtual_doc._check_text_or_bbox_modified()
            virtual_doc.bboxes
        t_hash = time.monotonic() - t0

        t0 = time.monotonic()
        for _ in range(100):
            virtual_doc.check_bbox()
        t_recalculate = time.monotonic() - t0

        assert t_hash / t_recalculate < 1 / 100

    @unittest.skip(reason='Waiting for API to support to add to default Annotation Set')
    def test_document_add_new_annotation(self):
        """Test adding a new annotation."""
        doc = self.prj.labels[0].documents[5]  # the latest Document
        # we create a revised Annotations, as only revised Annotation can be deleted
        # if we would delete an unrevised Annotation, we would provide feedback and thereby keep the
        # Annotation as "wrong" but "revised"
        assert len(doc.annotations(use_correct=False)) == 23
        label = self.prj.labels[0]
        new_anno = Annotation(
            start_offset=225,
            end_offset=237,
            label=label.id_,
            label_set_id=None,  # hand selected Document Label Set
            revised=True,
            is_correct=True,
            accuracy=0.98765431,
            document=doc,
        )
        # make sure Document Annotations are updated too
        assert len(doc.annotations(use_correct=False)) == 24
        label = self.prj.labels[0]
        annotations = label.annotations(categories=[self.prj.get_category_by_id(63)])
        self.assertEqual(self.document_count + 1, len(annotations))
        assert new_anno.id_ is None
        new_anno.save()
        assert new_anno.id_
        new_anno.delete()
        assert new_anno.id_ is None
        assert len(doc.annotations(use_correct=False)) == 13
        annotations = label.annotations(categories=[self.prj.get_category_by_id(63)])
        self.assertEqual(self.document_count, len(annotations))

    @unittest.skip(
        reason='Skip: Changes in AbstractExtractionAI Annotation needed to require a Label for every Annotation init.'
    )
    def test_document_add_new_annotation_without_label(self):
        """Test adding a new Annotation."""
        with self.assertRaises(AttributeError) as _:
            _ = Annotation(
                start_offset=225,
                end_offset=237,
                label=None,
                label_set_id=0,  # hand selected Document section Label
                revised=True,
                is_correct=True,
                accuracy=0.98765431,
                document=Document(),
            )
        # TODO: expand assert to check for specific error message

    @unittest.skip(
        reason='Skip: Changes in AbstractExtractionAI Annotation needed to require a Document for every '
        'Annotation init.'
    )
    def test_init_annotation_without_document(self):
        """Test adding a new Annotation."""
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

    @pytest.mark.xfail(reason='We cannot define the Annotation Set automatically.')
    def test_init_annotation_with_default_annotation_set(self):
        """Test adding a new Annotation without providing the Annotation Set."""
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

        # an Annotation Set needs to be created or retrieved after the Annotation is saved
        assert annotation.annotation_set.id_ == 78730

    def test_bio_scheme_saving_and_loading(self):
        """Test if generated bio scheme list is identical to loaded from file."""
        doc = self.prj.documents[0]
        bio_annotations1 = doc.get_text_in_bio_scheme(update=True)
        bio_annotations2 = doc.get_text_in_bio_scheme(update=False)

        assert bio_annotations1 == bio_annotations2

    @unittest.skip(reason='Issue https://gitlab.com/konfuzio/objectives/-/issues/8664.')
    def test_get_text_in_bio_scheme(self):
        """Test getting Document in the BIO scheme."""
        doc = self.prj.documents[0]
        bio_annotations = doc.get_text_in_bio_scheme()
        assert len(bio_annotations) == 398
        # check for multiline support in bio schema
        assert bio_annotations[1][0] == '328927/10103/00104'
        assert bio_annotations[1][1] == 'B-Austellungsdatum'
        assert bio_annotations[8][0] == '22.05.2018'
        assert bio_annotations[8][1] == 'B-Austellungsdatum'

    def test_number_of_all_documents(self):
        """Count the number of all available Documents online."""
        project = Project(id_=None, project_folder=OFFLINE_PROJECT)
        assert len(project._documents) == 57

    def test_create_empty_annotation(self):
        """
        Create an empty Annotation and get the start offset.

        The empty Annotation should be added to the Document as this represents the way the tokenizer
        creates empty Annotations.
        """
        prj = Project(id_=RESTORED_PROJECT_ID)
        test_category_id = prj.categories[0].id_
        doc = Document(text='', project=prj, category=prj.get_category_by_id(test_category_id))
        label_set = doc.category.default_label_set
        label = label_set.labels[0]
        span = Span(start_offset=1, end_offset=2)
        annotation_set = AnnotationSet(document=doc, label_set=label_set)
        _ = Annotation(label=label, annotation_set=annotation_set, label_set=label_set, document=doc, spans=[span])

    def test_get_annotations_for_offset_of_first_and_last_name(self):
        """Get Annotations for all offsets in the Document."""
        doc = self.prj.get_document_by_id(TEST_DOCUMENT_ID)
        filtered_annotations = doc.annotations(start_offset=1500, end_offset=1530)
        self.assertEqual(len(filtered_annotations), 3)  # 3 is correct even 4 Spans!
        text = '198,34\n  Erna-Muster Eiermann                         KiSt      15,83   Solz        10,89\n  '
        self.assertEqual(doc.text[1498:1590], text)

    def test_create_list_of_regex_for_label_without_annotations(self):
        """Check regex build for empty Labels."""
        category = self.prj.get_category_by_id(63)
        label = next(x for x in self.prj.labels if len(x.annotations(categories=[category])) == 0)
        automated_regex_for_label = label.regex(categories=[category])[category.id_]
        # There is no regex available.
        assert len(automated_regex_for_label) == 0

    def test_add_annotation_for_no_category_document(self):
        """Test adding Annotation error for no-category Document."""
        project = LocalTextProject()
        test_document = project.get_document_by_id(19)
        test_document._category = project.get_category_by_id(3)
        label_set = LabelSet(project=project, categories=[test_document.category])
        label = Label(project=project, label_sets=[label_set])
        span = Span(start_offset=1, end_offset=2)
        annotation_set = AnnotationSet(document=test_document, label_set=label_set)
        test_document._category = project.no_category
        with pytest.raises(ValueError, match='We cannot add'):
            _ = Annotation(
                label=label, annotation_set=annotation_set, label_set=label_set, document=test_document, spans=[span]
            )

    @unittest.skip(reason='Patch not supported by Text-Annotation Server.')
    def test_to_change_an_annotation_online(self):
        """Test to update an Annotation from revised to not revised and back to revised."""
        doc = self.prj.get_document_by_id(44864)
        annotations = doc.annotations(start_offset=10, end_offset=200)
        first_annotation = annotations[0]
        first_annotation.revised = False
        first_annotation.save()

    @classmethod
    def tearDownClass(cls) -> None:
        """Test if the Project remains the same as in the beginning."""
        assert len(cls.prj.documents) == cls.document_count
        assert len(cls.prj.test_documents) == cls.test_document_count
        category = cls.prj.get_category_by_id(63)
        assert len(cls.prj.labels[0].annotations(categories=[category])) == cls.annotations_correct


class TestKonfuzioForceOfflineData(unittest.TestCase):
    """Test handle data forced offline."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the test Project."""
        cls.project = Project(id_=RESTORED_PROJECT_ID, update=True)
        original_document_text = Project(id_=TEST_PROJECT_ID).get_document_by_id(TEST_DOCUMENT_ID).text
        cls.test_document_id = [
            document for document in cls.project.documents if document.text == original_document_text
        ][0].id_

    def test_force_offline_project(self):
        """Test that a Project with an ID can be forced offline."""
        prj = Project(id_=RESTORED_PROJECT_ID)
        prj.set_offline()
        self.assertFalse(prj.is_online)
        # all Data belonging to that Project should be offline without setting individual instances offline
        category = Category(prj, id_=1)
        self.assertFalse(category.is_online)
        label_set = LabelSet(prj, categories=[category], id_=1)
        self.assertFalse(label_set.is_online)
        doc = Document(prj, category=category, id_=1)
        self.assertFalse(doc.is_online)
        annotation_set = AnnotationSet(doc, label_set, id_=1)
        self.assertFalse(annotation_set.is_online)
        label = Label(prj, label_sets=[label_set], id_=1)
        self.assertFalse(label.is_online)
        annotation = Annotation(
            doc, annotation_set=annotation_set, label_set=label_set, label=label, id_=1, spans=[Span(0, 1)]
        )
        self.assertFalse(annotation.is_online)
        prj.delete()

    def test_make_sure_annotations_are_not_downloaded_automatically(self):
        """Test that Annotations are not downloaded automatically."""
        prj = Project(id_=RESTORED_PROJECT_ID, project_folder='another')
        doc = prj.get_document_by_id(self.test_document_id)
        doc.set_offline()
        self.assertFalse(is_file(doc.annotation_file_path, raise_exception=False))
        self.assertEqual(None, doc._annotations)
        self.assertFalse(doc.annotations())
        self.assertEqual(0, len(doc._annotations))
        self.assertFalse(is_file(doc.annotation_file_path, raise_exception=False))
        with self.assertRaises(NotImplementedError):
            doc.download_document_details()
        prj.delete()

    def test_annotations_are_loadable_for_offline_project_with_id_forced_offline(self):
        """Test that Annotations are loadable for OFFLINE_PROJECT if it's given an ID and forced offline."""
        prj = Project(id_=TEST_PROJECT_ID, project_folder=OFFLINE_PROJECT)
        doc = prj.get_document_by_id(TEST_DOCUMENT_ID)
        doc.set_offline()
        self.assertTrue(is_file(doc.annotation_file_path, raise_exception=False))
        self.assertEqual(None, doc._annotations)
        self.assertTrue(doc.annotations())
        self.assertEqual(22, len(doc._annotations))
        with self.assertRaises(NotImplementedError):
            doc.download_document_details()

    def test_make_sure_annotation_sets_are_not_downloaded_automatically(self):
        """Test that Annotation Sets are not downloaded automatically."""
        prj = Project(id_=RESTORED_PROJECT_ID, project_folder='another2')
        doc = prj.get_document_by_id(self.test_document_id)
        doc.set_offline()
        self.assertFalse(is_file(doc.annotation_set_file_path, raise_exception=False))
        self.assertEqual(None, doc._annotation_sets)
        self.assertFalse(doc.annotation_sets())
        self.assertEqual(0, len(doc._annotation_sets))
        self.assertFalse(is_file(doc.annotation_set_file_path, raise_exception=False))
        with self.assertRaises(NotImplementedError):
            doc.download_document_details()
        prj.delete()

    def test_annotations_sets_are_loadable_for_offline_project_with_id_forced_offline(self):
        """Test that AnnotationSets are loadable for OFFLINE_PROJECT if it's given an ID and forced offline."""
        prj = Project(id_=TEST_PROJECT_ID, project_folder=OFFLINE_PROJECT)
        doc = prj.get_document_by_id(TEST_DOCUMENT_ID)
        doc.set_offline()
        self.assertTrue(is_file(doc.annotation_set_file_path, raise_exception=False))
        self.assertEqual(None, doc._annotation_sets)
        self.assertTrue(doc.annotation_sets())
        self.assertEqual(24, len(doc._annotation_sets))
        with self.assertRaises(NotImplementedError):
            doc.download_document_details()

    def test_make_sure_pages_are_not_downloaded_automatically(self):
        """Test that Pages are not downloaded automatically."""
        prj = Project(id_=RESTORED_PROJECT_ID, project_folder='another33')
        doc = prj.get_document_by_id(self.test_document_id)
        doc.set_offline()
        self.assertFalse(is_file(doc.pages_file_path, raise_exception=False))
        self.assertEqual([], doc._pages)
        self.assertFalse(doc.pages())
        self.assertFalse(is_file(doc.pages_file_path, raise_exception=False))
        with self.assertRaises(NotImplementedError):
            doc.download_document_details()
        prj.delete()

    def test_view_annotations(self):
        """Test that Document.view_annotations() gets all the right Annotations."""
        project = LocalTextProject()
        document = project.get_document_by_id(7)
        annotations = document.view_annotations()
        assert len(annotations) == 4  # 4 if top_annotations filter is used
        assert sorted([ann.id_ for ann in annotations]) == [16, 18, 19, 24]

    def test_document_lose_weight(self):
        """Test that Document.lose_weight() removes all the right Annotations."""
        project = LocalTextProject()
        document = project.get_document_by_id(7)

        assert len(document.annotations(use_correct=False, ignore_below_threshold=False)) == 11

        document.lose_weight()

        assert len(document.annotations(use_correct=False, ignore_below_threshold=False)) == 8

    def test_annotationset_annotations(self):
        """Test AnnotationSet.annotations method."""
        project = LocalTextProject()
        document = project.get_document_by_id(7)

        annotation_set = document.annotation_sets()[0]

        assert len(annotation_set.annotations()) == 1
        assert len(annotation_set.annotations(use_correct=False)) == 10
        assert len(annotation_set.annotations(use_correct=False, ignore_below_threshold=True)) == 9

    def test_annotationset_start_end_offset_and_start_line_index(self):
        """Test AnnotationSet start and end offset methods and start_line_index."""
        project = LocalTextProject()
        document = project.get_document_by_id(7)

        annotation_set = document.annotation_sets()[0]

        assert annotation_set.start_offset == 0
        assert annotation_set.start_line_index == 0
        assert annotation_set.end_offset == 73

    def test_label_spans_not_found_by_tokenizer(self):
        """Test Label spans_not_found_by_tokenizer method."""
        project = LocalTextProject()

        whitespace_tokenizer = WhitespaceTokenizer()
        al_tokenizer = RegexTokenizer('al')

        category = project.get_category_by_id(1)
        label = project.get_label_by_id(4)
        label_span = label.annotations(categories=[category])[0].spans[0]

        whitespace_spans = label.spans_not_found_by_tokenizer(whitespace_tokenizer, categories=[category])
        assert len(whitespace_spans) == 1
        assert whitespace_spans not in label_span.regex_matching

        al_spans = label.spans_not_found_by_tokenizer(al_tokenizer, categories=[category])
        assert len(al_spans) == 0
        assert al_tokenizer in label_span.regex_matching

    def test_offline_project_creates_no_files(self):
        """Test that an offline Project does not create any files, even if Documents have IDs."""
        virtual_project = Project(id_=None)
        virtual_project.set_offline()
        assert not os.path.isdir(virtual_project.project_folder)

        virtual_category = Category(project=virtual_project)
        virtual_document = Document(
            project=virtual_project,
            id_=999999999,
            category=virtual_category,
            dataset_status=2,
            copy_of_id=999999999,
        )
        assert not os.path.isdir(virtual_document.document_folder)

    def test_category_collect_exclusive_first_page_strings(self):
        """Test collecting exclusive first-page strings within the Documents of a Category."""
        project = LocalTextProject()
        category = project.get_category_by_id(3)
        tokenizer = ConnectedTextTokenizer()
        first_page_strings = category.exclusive_first_page_strings(tokenizer)
        assert len(first_page_strings) == 2
        assert 'I like bread.' in first_page_strings
        assert 'Morning,' in first_page_strings

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove the project created specifically for this test pipeline."""
        cls.project = Project(id_=RESTORED_PROJECT_ID, update=True)
        for document in cls.project.documents + cls.project.test_documents:
            document.dataset_status = 0
            document.save_meta_data()
            document.delete(delete_online=True)
        response = delete_project(project_id=RESTORED_PROJECT_ID)
        assert response.status_code == 204


class TestFillOperation(unittest.TestCase):
    """Separate Test as we add non Labels to the Project."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the test: https://app.konfuzio.com/projects/46/docs/44823/bbox-annotations/."""
        cls.prj = Project(id_=None, project_folder=OFFLINE_PROJECT)
        cls.doc = cls.prj.get_document_by_id(TEST_DOCUMENT_ID)
        default_label_set = cls.prj.get_label_set_by_name('Lohnabrechnung')
        assert default_label_set.labels.__len__() == 10
        cls.annotations = cls.doc.annotations(start_offset=1498, end_offset=1590, fill=True)
        cls.sorted_spans = sorted([span for annotation in cls.annotations for span in annotation.spans])
        cls.text = '198,34\n  Erna-Muster Eiermann                         KiSt      15,83   Solz        10,89\n  '
        assert cls.doc.text[1498:1590] == cls.text

    def test_number_of_annotations(self):
        """Get Annotations for all offsets in the Document."""
        self.assertEqual(len(self.annotations), 7)  # 2 single line Annotation, one multiline with two spans

    def test_number_of_spans(self):
        """Get Annotations for all offsets in the Document."""
        self.assertEqual(len([span for annotation in self.annotations for span in annotation.spans]), 10)

    @unittest.skip(reason='Documents without Category cannot be processed.')
    def test_fill_doc_without_category(self):
        """Try to fill a Document without Category."""
        self.prj.get_document_by_id(44864).annotations(fill=True)

    def test_fill_full_document_with_category(self):
        """Try to fill a Document with Category."""
        # Failing because the Document already has the Annotations created by fill (from the tests setup)
        with self.assertRaises(ValueError) as context:
            self.prj.get_document_by_id(TEST_DOCUMENT_ID).annotations(fill=True)
            assert 'is a duplicate of' in context.exception

    def test_correct_text_offset(self):
        """Test if the sorted Spans can create the offset text."""
        offsets = [sorted_span.offset_string for sorted_span in self.sorted_spans]
        span_text = ''.join(offsets)
        self.assertEqual(self.doc.text[1498:1590], span_text)

    def test_span_start_and_end(self):
        """Test if the Spans have the correct offsets."""
        spa = [(span.start_offset, span.end_offset) for span in self.sorted_spans]
        assert self.doc.text[slice(spa[0][0], spa[0][1])] == self.doc.text[1498:1504] == '198,34'
        assert self.doc.text[slice(spa[1][0], spa[1][1])] == self.doc.text[1504:1505] == '\n'
        assert self.doc.text[slice(spa[2][0], spa[2][1])] == self.doc.text[1505:1507] == '  '
        assert self.doc.text[slice(spa[3][0], spa[3][1])] == self.doc.text[1507:1518] == 'Erna-Muster'
        assert self.doc.text[slice(spa[4][0], spa[4][1])] == self.doc.text[1518:1519] == ' '
        assert self.doc.text[slice(spa[5][0], spa[5][1])] == self.doc.text[1519:1527] == 'Eiermann'
        unlabeled = '                         KiSt      15,83   Solz        '
        assert self.doc.text[slice(spa[6][0], spa[6][1])] == self.doc.text[1527:1582] == unlabeled
        assert self.doc.text[slice(spa[7][0], spa[7][1])] == self.doc.text[1582:1587] == '10,89'
        assert self.doc.text[slice(spa[8][0], spa[8][1])] == self.doc.text[1587:1588] == '\n'
        assert self.doc.text[slice(spa[9][0], spa[9][1])] == self.doc.text[1588:1590] == '  '


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

    def test_not_online(self):
        """Test that data with a None ID is not online."""
        a = Data()
        self.assertFalse(a.is_online)

    def test_is_online(self):
        """Test that data with an ID is online."""
        a = Data()
        a.id_ = 0
        self.assertTrue(a.is_online)

    def test_force_offline_data(self):
        """Test that data with an ID can be forced offline."""
        a = Data()
        a.id_ = 1
        a.set_offline()
        self.assertFalse(a.is_online)


def test_export_project_data():
    """Test downloading of data from training and test documents."""
    project = Project(id_=1249, update=True)
    category_id = project.categories[0].id_
    project.export_project_data(include_ais=True, training_and_test_documents=True, category_id=category_id)


def test_to_init_prj_from_folder():
    """Load Project from folder."""
    prj = Project(id_=46, project_folder='data_46')
    assert len(prj.documents) == 26
