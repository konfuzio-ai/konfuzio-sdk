"""Test Paragraph Tokenizer."""
import logging
import unittest
from copy import deepcopy

import pytest

from konfuzio_sdk.data import Annotation, Project, Span
from konfuzio_sdk.settings_importer import is_dependency_installed
from konfuzio_sdk.tokenizer.paragraph_and_sentence import ParagraphTokenizer
from konfuzio_sdk.trainer.information_extraction import RFExtractionAI

logger = logging.getLogger(__name__)

@unittest.skip(reason='Project 458 is under maintentance now and switching to another Project requires major changes.')
class TestDetectronParagraphTokenizer(unittest.TestCase):
    """Test Detectron Paragraph Tokenizer."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the tokenizer and test setup."""
        cls.project = Project(id_=458)
        cls.tokenizer = ParagraphTokenizer(mode='detectron')

        cls.document_1 = cls.project.get_document_by_id(601418)  # Lorem ipsum test document
        cls.document_2 = cls.project.get_document_by_id(601419)  # Two column paper

    def test_paragraph_document_1(self):
        """Test detectron Paragraph tokenizer on Lorem ipsum Document."""
        virtual_doc = deepcopy(self.document_1)

        virtual_doc = self.tokenizer.tokenize(virtual_doc)

        assert len(virtual_doc.annotations(use_correct=False)) == 30
        assert len(virtual_doc.spans(use_correct=False)) == 99

        assert len(virtual_doc.annotations(use_correct=False)[10].spans) == 7
        assert virtual_doc.annotations(use_correct=False)[10].spans[0].start_offset == 3145
        assert virtual_doc.annotations(use_correct=False)[10].spans[0].end_offset == 3233

        pages = virtual_doc.pages()

        assert len(pages) == 3

        assert len(pages[0].annotations(use_correct=False)) == 8
        assert len(pages[1].annotations(use_correct=False)) == 12
        assert len(pages[2].annotations(use_correct=False)) == 10

    def test_paragraph_document_1_use_detectron_labels(self):
        """Test detectron Paragraph tokenizer on Lorem ipsum Document with create_detectron_labels option."""
        virtual_doc = deepcopy(self.document_1)

        tokenizer = ParagraphTokenizer(mode='detectron', create_detectron_labels=True)

        virtual_doc = tokenizer(virtual_doc)

        assert len(virtual_doc.annotations(use_correct=False)) == 31
        assert len(virtual_doc.spans(use_correct=False)) == 99

        pages = virtual_doc.pages()

        assert len(pages) == 3

        assert len(pages[0].annotations(use_correct=False)) == 8
        assert len(pages[1].annotations(use_correct=False)) == 12

        page_2_annotations = pages[2].annotations(use_correct=False)
        assert len(page_2_annotations) == 11
        assert page_2_annotations[0].spans == page_2_annotations[1].spans
        page_2_annotations[0].label.name == 'title'
        page_2_annotations[1].label.name == 'text'

        assert virtual_doc.annotations(use_correct=False)[0].label.name == 'title'
        assert virtual_doc.annotations(use_correct=False)[1].label.name == 'text'

    def test_paragraph_document_2(self):
        """Test detectron Paragraph tokenizer on two column paper."""
        virtual_doc = deepcopy(self.document_2)

        doc = self.tokenizer.tokenize(virtual_doc)
        assert len(doc.annotations(use_correct=False)) == 79
        assert len(doc.spans(use_correct=False)) == 600

        assert len(doc.annotations(use_correct=False)[20].spans) == 10
        assert doc.annotations(use_correct=False)[20].spans[0].start_offset == 8157
        assert doc.annotations(use_correct=False)[20].spans[0].end_offset == 8217

        pages = doc.pages()
        assert len(pages) == 7

        assert len(pages[0].annotations(use_correct=False)) == 16
        assert len(pages[1].annotations(use_correct=False)) == 10
        assert len(pages[2].annotations(use_correct=False)) == 12
        assert len(pages[3].annotations(use_correct=False)) == 15
        assert len(pages[4].annotations(use_correct=False)) == 12
        assert len(pages[5].annotations(use_correct=False)) == 10
        assert len(pages[6].annotations(use_correct=False)) == 4

@unittest.skip(reason='Project 458 is under maintentance now and switching to another Project requires major changes.')
class TestLineDistanceParagraphTokenizer(unittest.TestCase):
    """Test Line Distance Paragraph Tokenizer."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the tokenizer and test setup."""
        cls.project = Project(id_=458)
        cls.tokenizer = ParagraphTokenizer(mode='line_distance')

        cls.document_1 = cls.project.get_document_by_id(601418)  # Lorem ipsum test document

    def test_paragraph_document_1(self):
        """Test detectron Paragraph tokenizer on Lorem ipsum Document."""
        virtual_doc = deepcopy(self.document_1)

        doc = self.tokenizer.tokenize(virtual_doc)

        assert len(doc.annotations(use_correct=False)) == 29

        pages = doc.pages()

        assert len(pages) == 3

        assert len(pages[0].annotations(use_correct=False)) == 8
        assert len(pages[1].annotations(use_correct=False)) == 12
        assert len(pages[2].annotations(use_correct=False)) == 9

    @pytest.mark.skipif(not is_dependency_installed('cloudpickle'), reason='Required dependency not installed.')
    def test_paragraph_document_2_merge_vertical_like(self):
        """Test vertical_merge_like to merge Annotations like another Document."""
        virtual_doc = deepcopy(self.document_1)
        self.tokenizer.tokenize(virtual_doc)
        assert len(virtual_doc.annotations(use_correct=False)) == 29
        virtual_doc_pages = virtual_doc.pages()

        assert len(virtual_doc_pages) == 3
        assert len(virtual_doc_pages[0].annotations(use_correct=False)) == 8
        assert len(virtual_doc_pages[1].annotations(use_correct=False)) == 12
        assert len(virtual_doc_pages[2].annotations(use_correct=False)) == 9

        new_virtual_document = deepcopy(virtual_doc)

        for span in virtual_doc.spans(use_correct=False):
            spans = [Span(start_offset=span.start_offset, end_offset=span.end_offset)]
            _ = Annotation(
                document=new_virtual_document,
                annotation_set=new_virtual_document.no_label_annotation_set,
                label=new_virtual_document.project.no_label,
                label_set=new_virtual_document.project.no_label_set,
                category=new_virtual_document.category,
                spans=spans,
            )

        pages = new_virtual_document.pages()
        assert len(pages) == 3

        assert len(pages[0].annotations(use_correct=False)) == len(virtual_doc_pages[0].spans(use_correct=False))
        assert len(pages[0].annotations(use_correct=False)) == 32
        assert len(pages[1].annotations(use_correct=False)) == len(virtual_doc_pages[1].spans(use_correct=False))
        assert len(pages[1].annotations(use_correct=False)) == 32
        assert len(pages[2].annotations(use_correct=False)) == len(virtual_doc_pages[2].spans(use_correct=False))
        assert len(pages[2].annotations(use_correct=False)) == 35

        RFExtractionAI().merge_vertical_like(document=new_virtual_document, template_document=virtual_doc)

        assert len(pages[0].annotations(use_correct=False)) == len(virtual_doc_pages[0].annotations(use_correct=False))
        assert len(pages[0].annotations(use_correct=False)) == 8
        assert len(pages[1].annotations(use_correct=False)) == len(virtual_doc_pages[1].annotations(use_correct=False))
        assert len(pages[1].annotations(use_correct=False)) == 12
        assert len(pages[2].annotations(use_correct=False)) == len(virtual_doc_pages[2].annotations(use_correct=False))
        assert len(pages[2].annotations(use_correct=False)) == 9

@unittest.skip(reason='Project 458 is under maintentance now and switching to another Project requires major changes.')
def test_no_bbox_document(caplog):
    """Test processing a Document with no Bboxes."""
    project = Project(id_=458)
    tokenizer = ParagraphTokenizer(mode='detectron')
    document = project.get_document_by_id(601418)
    virtual_doc = deepcopy(document)
    virtual_doc.bboxes_available = False
    with caplog.at_level(logging.WARNING):
        tokenizer.tokenize(virtual_doc)
    assert 'Cannot tokenize Document' in caplog.text
