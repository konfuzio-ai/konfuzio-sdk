"""Test paragraph tokenizers."""
import logging
import unittest

from copy import deepcopy
from konfuzio_sdk.data import Project, Span, Annotation

from konfuzio_sdk.tokenizer.paragraph_and_sentence import ParagraphTokenizer


logger = logging.getLogger(__name__)


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

        doc = self.tokenizer.tokenize(virtual_doc)

        assert len(doc.annotations(use_correct=False)) == 26
        assert len(doc.spans(use_correct=False)) == 99

        assert len(doc.annotations(use_correct=False)[10].spans) == 7
        assert doc.annotations(use_correct=False)[10].spans[0].start_offset == 3145
        assert doc.annotations(use_correct=False)[10].spans[0].end_offset == 3233

        pages = doc.pages()

        assert len(pages) == 3

        assert len(pages[0].annotations(use_correct=False)) == 8
        assert len(pages[1].annotations(use_correct=False)) == 10
        assert len(pages[2].annotations(use_correct=False)) == 8

    def test_paragraph_document_2(self):
        """Test detectron Paragraph tokenizer on two column paper."""
        virtual_doc = deepcopy(self.document_2)

        doc = self.tokenizer.tokenize(virtual_doc)
        assert len(doc.annotations(use_correct=False)) == 78
        assert len(doc.spans(use_correct=False)) == 600

        assert len(doc.annotations(use_correct=False)[20].spans) == 10
        assert doc.annotations(use_correct=False)[20].spans[0].start_offset == 8157
        assert doc.annotations(use_correct=False)[20].spans[0].end_offset == 8217

        pages = doc.pages()
        assert len(pages) == 7

        assert len(pages[0].annotations(use_correct=False)) == 16
        assert len(pages[1].annotations(use_correct=False)) == 10
        assert len(pages[2].annotations(use_correct=False)) == 12
        assert len(pages[3].annotations(use_correct=False)) == 14
        assert len(pages[4].annotations(use_correct=False)) == 12
        assert len(pages[5].annotations(use_correct=False)) == 10
        assert len(pages[6].annotations(use_correct=False)) == 4


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

        new_virtual_document.merge_vertical_like(virtual_doc)

        assert len(pages[0].annotations(use_correct=False)) == len(virtual_doc_pages[0].annotations(use_correct=False))
        assert len(pages[0].annotations(use_correct=False)) == 8
        assert len(pages[1].annotations(use_correct=False)) == len(virtual_doc_pages[1].annotations(use_correct=False))
        assert len(pages[1].annotations(use_correct=False)) == 12
        assert len(pages[2].annotations(use_correct=False)) == len(virtual_doc_pages[2].annotations(use_correct=False))
        assert len(pages[2].annotations(use_correct=False)) == 9
